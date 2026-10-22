"""Read-only Home Assistant, Open-Meteo, and Australian calendar inputs."""
import json, re, sqlite3, time
from datetime import date, timedelta
from urllib.parse import urlparse
import httpx, numpy as np, pandas as pd, holidays
from websockets.sync.client import connect
from .config import DATA,read_json
from .forecast_policy import archived_run,INTERVAL_POLICY
from .ha_inputs import numeric_value, history_values, history_intervals, normalized_value

class HomeAssistant:
    def __init__(self, settings):
        self.s=settings
        self.url=settings.ha_url.rstrip('/')
        self.headers={'Authorization':'Bearer '+settings.ha_token}

    def get(self, path, params=None):
        with httpx.Client(timeout=120, headers=self.headers) as c:
            r=c.get(self.url+path, params=params); r.raise_for_status(); return r.json()

    def states(self):
        return self.get('/api/states')

    def ws(self, message):
        u=urlparse(self.url); wsurl=('wss' if u.scheme=='https' else 'ws')+'://'+u.netloc+u.path+'/api/websocket'
        with connect(wsurl, open_timeout=20, max_size=100*1024*1024) as ws:
            ws.recv(); ws.send(json.dumps({'type':'auth','access_token':self.s.ha_token}))
            if json.loads(ws.recv())['type']!='auth_ok': raise ValueError('Home Assistant rejected the API token')
            ws.send(json.dumps({'id':1, **message}))
            while True:
                result=json.loads(ws.recv(timeout=120))
                if result.get('id')==1:
                    if not result.get('success'): raise ValueError(str(result.get('error')))
                    return result['result']

    def statistics(self, entities, start, end, period='hour'):
        try:return self.ws({'type':'recorder/statistics_during_period','start_time':pd.Timestamp(start).isoformat(),'end_time':pd.Timestamp(end).isoformat(),'statistic_ids':entities,'period':period,'types':['mean','sum','min','max'],'units':{'power':'kW','energy':'kWh'}})
        except ValueError:
            return self.ws({'type':'recorder/statistics_during_period','start_time':pd.Timestamp(start).isoformat(),'end_time':pd.Timestamp(end).isoformat(),'statistic_ids':entities,'period':period,'types':['mean','sum'],'units':{'power':'kW','energy':'kWh'}})

    def history(self, entity, start, end):
        return self.get('/api/history/period/'+pd.Timestamp(start).isoformat(), {'filter_entity_id':entity, 'end_time':pd.Timestamp(end).isoformat()})

    def frame(self, entities, start, end, period='hour', progress=lambda s:None, units=None, corrections=None, quality_cutoff=None, measurement_kind=None, max_power_kw=None):
        """Statistics first; fill individual retained-history gaps with explicit provenance."""
        from .quality import clean_sensor,physical,apply_corrections
        states={x['entity_id']:x for x in self.states()}
        try: metadata={x['statistic_id']:x for x in self.ws({'type':'recorder/list_statistic_ids'})}
        except Exception:metadata={}
        start=pd.Timestamp(start);end=pd.Timestamp(end);step=pd.Timedelta(minutes=5) if period=='5minute' else pd.Timedelta(hours=1)
        grid=pd.date_range(start.floor(step),end.ceil(step),freq=step,inclusive='left');records={e:[] for e in entities};info={};cols={};raw_cols={}
        for left in pd.date_range(start-step,end,freq='30D',inclusive='left'):
            right=min(left+pd.Timedelta(days=30),end);progress(f'Home Assistant: {left.date()} to {right.date()}')
            raw=self.statistics(entities,left,right,period)
            for entity in entities:
                for row in raw.get(entity,[]):
                    t=pd.to_datetime(row['start'],unit=('ms' if row['start']>1e11 else 's'),utc=True) if isinstance(row['start'],(float,int)) else pd.to_datetime(row['start'],utc=True)
                    records[entity].append((t,row.get('mean'),row.get('sum'),row.get('min'),row.get('max')))
        for entity in entities:
            attrs=states.get(entity,{}).get('attributes',{});meta=metadata.get(entity,{})
            override=(units or {}).get(entity)
            metadata_unit=attrs.get('unit_of_measurement') or meta.get('unit_of_measurement') or meta.get('statistics_unit_of_measurement')
            known_unit=metadata_unit in ('W','kW','Wh','kWh','%')
            source_unit=metadata_unit if known_unit else override
            kind=measurement_kind or ('soc' if entity==self.s.battery_soc_entity else 'battery_power' if entity==self.s.battery_power_entity else 'load' if entity in [self.s.consumption_entity,*self.s.historical_consumption_entities] else 'solar')
            normalized_value(1,source_unit,kind)
            # Explicit WS units describe returned values, never the stored unit.
            unit=('kW' if source_unit in ('W','kW') else 'kWh' if source_unit in ('Wh','kWh') else source_unit) if known_unit else source_unit
            history_units=[];duration_frames=[];history_errors=[]
            if unit not in ('W','kW','Wh','kWh','%'):raise ValueError(f'Unknown units for {entity}; choose an explicit dataset unit')
            records_frame=pd.DataFrame(records[entity],columns=['time','mean','sum','min','max']).drop_duplicates('time',keep='last').set_index('time').sort_index()
            records_frame.index=pd.DatetimeIndex(pd.to_datetime(records_frame.index,utc=True))
            energy_flags=pd.Series('',index=grid)
            if unit in ('Wh','kWh'):
                total=pd.to_numeric(records_frame['sum'],errors='coerce');duration=total.index.to_series().diff().dt.total_seconds()/3600;delta=total.diff()
                negative=delta<0;gap=duration!=step.total_seconds()/3600
                for reason,mask in [('negative_energy_delta',negative),('recorder_interval_gap',gap)]:
                    ix=delta.index[mask].intersection(grid);energy_flags.loc[ix]=reason
                s=(delta/duration/(1000 if unit=='Wh' else 1)).where((delta>=0)&~gap)
            else:s=pd.to_numeric(records_frame['mean'],errors='coerce')/(1000 if unit=='W' else 1)
            s=s.reindex(grid);coverage=pd.Series('statistics',index=grid).where(s.notna(),'missing')
            # Query all requested gaps, including gaps inside an otherwise populated series.
            if s.isna().any():
                for left in pd.date_range(start,end,freq='7D',inclusive='left'):
                    right=min(left+pd.Timedelta(days=7),end)
                    if not s.loc[(s.index>=left)&(s.index<right)].isna().any():continue
                    pieces=[]
                    try:
                        for group in self.history(entity,left,right):
                            for row in group:
                                t=row.get('last_updated') or row.get('last_changed')
                                if t is not None:
                                    raw_unit=row.get('attributes',{}).get('unit_of_measurement') or override
                                    if numeric_value(row) is None:
                                        pieces.append((pd.to_datetime(t,utc=True),None));continue
                                    if raw_unit is None:
                                        history_errors.append({'time':str(t),'reason':'Numeric raw history lacks a unit; unavailable until an entity-specific override is provided'})
                                        pieces.append((pd.to_datetime(t,utc=True),None));continue
                                    normalized_value(1,raw_unit,kind)
                                    if (raw_unit in ('Wh','kWh'))!=(unit in ('Wh','kWh')):raise ValueError('Historical measurement dimension changed')
                                    if not history_units or history_units[-1]['unit']!=raw_unit:
                                        history_units.append({'time':str(t),'unit':raw_unit,'basis':'history attribute' if row.get('attributes',{}).get('unit_of_measurement') else 'explicit override'})
                                    pieces.append((pd.to_datetime(t,utc=True),normalized_value(numeric_value(row),raw_unit,kind)))
                    except ValueError:raise
                    except Exception as exc:
                        history_errors.append({'start':left.isoformat(),'end':right.isoformat(),'reason':str(exc)});continue
                    if pieces:
                        intervals=history_intervals(pieces,left,right,step,cumulative=unit in ('Wh','kWh'))
                        held=intervals['mean'];duration_frames.append(intervals.drop(columns='mean'))
                        unknown=intervals.index[intervals.unknown_seconds>0].intersection(grid)
                        energy_flags.loc[unknown]='raw_history_unavailable'
                        fill=s.isna()&held.reindex(grid).notna();s.loc[fill]=held.reindex(grid).loc[fill];coverage.loc[fill]='raw_cumulative_delta' if unit in ('Wh','kWh') else 'raw_history'
            raw_cols[entity]=s.copy()
            limit=100 if kind=='soc' else max(self.s.battery_charge_limit_kw,self.s.battery_discharge_limit_kw)*1.2 if kind=='battery_power' else (max_power_kw if max_power_kw is not None else self.s.load_max_power_kw if kind=='load' else self.s.solar_max_power_kw)
            clean,flags,details=clean_sensor(s,kind,limit,self.s,minmax=records_frame[['min','max']]/(1000 if unit=='W' else 1) if unit in ('W','kW') else None,cutoff=quality_cutoff)
            flags.loc[energy_flags.ne('')&clean.isna()]=energy_flags.loc[energy_flags.ne('')&clean.isna()]
            details['energy_exclusions']={reason:int((flags==reason).sum()) for reason in ('negative_energy_delta','recorder_interval_gap') if (flags==reason).any()}
            original_flags=flags.copy()
            if kind=='load':
                edits=corrections if corrections is not None else read_json('dataset-load',{}).get('request',{}).get('corrections',[])
                clean,flags=apply_corrections(s,clean,flags,self.s,kind,limit,edits,original_flags)
            cols[entity]=clean
            info[entity]={'unit':unit,'stored_unit':meta.get('statistics_unit_of_measurement'),'source_unit':source_unit,'normalized_unit':'%' if kind=='soc' else 'kW','unit_override':override,'unit_basis':'explicit statistics API units' if known_unit else 'explicit entity override for returned statistics','history_units':history_units,'history_errors':history_errors,'original_quality_flags':original_flags,'durations':pd.concat(duration_frames) if duration_frames else None,'raw_values':s.copy(),'coverage':coverage,'quality_flags':flags,'statistics':records_frame,'details':details}
        result=pd.DataFrame(cols,index=grid);result.attrs={'sensors':info,'raw':pd.DataFrame(raw_cols,index=grid),'evidence_limit':'Raw history only where retained by recorder; expired gaps remain missing'}
        return result

class Weather:
    def __init__(self, settings): self.s=settings

    def fetch(self, start, end, historical=False):
        start=pd.Timestamp(start); end=pd.Timestamp(end)
        key=f'weather-v3-{self.s.latitude:.4f}-{self.s.longitude:.4f}-{start.date()}-{end.date()}-{historical}'
        file=DATA/(key+'.pkl')
        if file.exists() and (historical or time.time()-file.stat().st_mtime<1200): return pd.read_pickle(file)
        params={'latitude':self.s.latitude,'longitude':self.s.longitude,'hourly':'temperature_2m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation','timezone':'UTC','start_date':str(start.date()),'end_date':str(end.date())}
        # Reanalysis through two days ago; use the forecast API's past_days coverage for the tail.
        chunks=[]
        cutoff=pd.Timestamp.now(tz='UTC').normalize()-pd.Timedelta(days=2)
        ranges=[(start,min(end,cutoff-pd.Timedelta(days=1)),True),(max(start,cutoff),end,False)] if historical else [(start,end,False)]
        for left,right,archive in ranges:
            if left>right: continue
            q={**params,'start_date':str(left.date()),'end_date':str((right+pd.Timedelta(days=1)).date())}
            url='https://archive-api.open-meteo.com/v1/archive' if archive else 'https://api.open-meteo.com/v1/forecast'
            with httpx.Client(timeout=120) as c:
                for attempt in range(3):
                    r=c.get(url,params=q)
                    if r.status_code!=429: break
                    time.sleep(3*(attempt+1))
                r.raise_for_status(); h=r.json()['hourly']
            f=pd.DataFrame(h); f.index=pd.to_datetime(f.pop('time'),utc=True); chunks.append(f)
        if not chunks: raise ValueError('No weather data for that period')
        f=pd.concat(chunks).sort_index(); f=f[~f.index.duplicated(keep='last')]; f=weather_intervals(f); f.to_pickle(file)
        return f

SCHOOL_ROOT='https://raw.githubusercontent.com/pmcau/AustralianHolidays/main/src/AustralianHolidays/'
class Calendar:
    """Public holidays from python-holidays; school terms from a maintained state calendar."""
    def __init__(self, settings): self.s=settings; self.terms={}; self.events=[]; self.warnings=[]

    def load(self):
        file=DATA/('calendar-'+self.s.state+'.json')
        if file.exists() and time.time()-file.stat().st_mtime<86400*7:
            self.terms=json.loads(file.read_text())
        else:
            try:
                with httpx.Client(timeout=30) as c:
                    r=c.get(SCHOOL_ROOT+'SchoolHolidays_'+self.s.state+'.cs'); r.raise_for_status()
                for year,body in re.findall(r'\[(\d{4})\]\s*=\s*\[(.*?)\],',r.text,re.S):
                    dates=re.findall(r'new\((\d{4}),\s*(\d+),\s*(\d+)\)',body)
                    self.terms[year]=[date(*map(int,d)).isoformat() for d in dates]
                if not self.terms: raise ValueError('School calendar contained no term dates')
                file.write_text(json.dumps(self.terms))
            except Exception as exc:
                if file.exists(): self.terms=json.loads(file.read_text())
                self.warnings.append('School calendar refresh failed: '+str(exc))
        bundled=__import__('pathlib').Path(__file__).parent/'calendar_data'/(self.s.state+'.json')
        if bundled.exists():
            self.terms={**json.loads(bundled.read_text())['terms'], **self.terms}
        if self.s.school_calendar_url:
            # Optional holiday ICS feed supports historical/custom school calendars.
            from icalendar import Calendar as ICS
            with httpx.Client(timeout=30,follow_redirects=True) as c:
                r=c.get(self.s.school_calendar_url); r.raise_for_status()
            for event in ICS.from_ical(r.content).walk('VEVENT'):
                a=event.decoded('DTSTART'); b=event.decoded('DTEND') if event.get('DTEND') else a+timedelta(days=1)
                self.events.append((a.date() if hasattr(a,'date') else a,b.date() if hasattr(b,'date') else b))
        return self

    def features(self, index):
        local=index.tz_convert(self.s.timezone); years=sorted(set(local.year))
        public=holidays.Australia(subdiv=self.s.state,years=years)
        school=[]; known=[]
        for t in local:
            d=t.date(); terms=self.terms.get(str(d.year),[])
            covered=bool(terms) or bool(self.events)
            holiday=not any(date.fromisoformat(terms[i])<=d<=date.fromisoformat(terms[i+1]) for i in range(0,len(terms),2)) if terms else False
            if self.events: holiday=any(a<=d<b for a,b in self.events)
            school.append(int(holiday)); known.append(int(covered))
        return pd.DataFrame({'public_holiday':[int(t.date() in public) for t in local], 'school_holiday':school,'school_calendar_known':known},index=index)

    def summary(self, index):
        f=self.features(index)
        return {'source':'python-holidays + AustralianHolidays state school calendar','state':self.s.state,'covered_years':list(self.terms),'missing_school_hours':int((f.school_calendar_known==0).sum()),'warnings':self.warnings}

def store_live(states):
    with sqlite3.connect(DATA/'history.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS samples (time TEXT, entity TEXT, value REAL, unit TEXT, PRIMARY KEY(time,entity))')
        now=pd.Timestamp.now(tz='UTC').isoformat()
        for s in states:
            value=numeric_value(s)
            if value is None:continue
            db.execute('INSERT OR IGNORE INTO samples VALUES (?,?,?,?)',(now,s['entity_id'],value,s.get('attributes',{}).get('unit_of_measurement','')))
        db.execute("DELETE FROM samples WHERE time < ?",((pd.Timestamp.now(tz='UTC')-pd.Timedelta(days=120)).isoformat(),))


def forecast_weather(settings,start,end,model='ecmwf_ifs'):
    """Fetch the selected weather model for the live forecast."""
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    key=f'weather-live-v2-{settings.latitude:.4f}-{settings.longitude:.4f}-{model}-{start.strftime("%Y%m%d%H")}-{end.strftime("%Y%m%d%H")}'
    file=DATA/(key+'.pkl')
    if file.exists() and time.time()-file.stat().st_mtime<1200: return pd.read_pickle(file)
    params={'latitude':settings.latitude,'longitude':settings.longitude,'models':model,'hourly':'temperature_2m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation','timezone':'UTC','start_date':str(start.date()),'end_date':str((end+pd.Timedelta(days=1)).date())}
    with httpx.Client(timeout=120) as c:
        r=c.get('https://api.open-meteo.com/v1/forecast',params=params); r.raise_for_status(); h=r.json()['hourly']
    f=pd.DataFrame(h); f.index=pd.to_datetime(f.pop('time'),utc=True)
    columns=['temperature_2m','shortwave_radiation','direct_normal_irradiance','diffuse_radiation']
    for col in columns:
        if col not in f and col+'_'+model in f: f[col]=f[col+'_'+model]
    f=weather_intervals(f[columns]); f.attrs.update(run=None,run_basis='Stitched operational response; initialization unknown'); f.to_pickle(file); return f


def solar_weather(settings,bank,start,end,historical=False,issued_at=None,cached_only=False):
    """Open-Meteo GTI for a bank or panel-count-weighted combined layout."""
    if bank.id == '__combined':
        frames=[(b,solar_weather(settings,b,start,end,historical=historical,issued_at=issued_at,cached_only=cached_only)) for b in settings.banks]
        result=frames[0][1].copy()
        count=sum(b.panel_count for b,_ in frames)
        for i,(b,f) in enumerate(frames):result[f'bank_irradiance_{i}']=f['tilted_irradiance']*b.panel_count/count
        result['tilted_irradiance']=result[[f'bank_irradiance_{i}' for i in range(len(frames))]].sum(axis=1,min_count=len(frames))
        result.attrs={**frames[0][1].attrs,'irradiance_source':'Panel-count-weighted Open-Meteo tilted irradiance'}
        return result
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    # Allow for global-model publication after initialization, rather than using
    # a run whose results would not yet have been available at the issue time.
    run=archived_run(issued_at) if issued_at is not None else None
    mode=run.strftime('%Y%m%d%H') if run is not None else ('history' if historical else 'live')
    model=settings.weather_model
    key=f'solar-weather-v3-{settings.latitude:.6f}-{settings.longitude:.6f}-{bank.tilt:g}-{bank.azimuth:g}-{model}-{mode}-{start.strftime("%Y%m%d%H")}-{end.strftime("%Y%m%d%H")}'
    path=DATA/(key+'.pkl')
    if path.exists() and (historical or run is not None or time.time()-path.stat().st_mtime<1200):
        return pd.read_pickle(path)
    if run is not None:
        for version in ('v3','v2'):
            prefix=f'solar-weather-{version}-{settings.latitude:.6f}-{settings.longitude:.6f}-{bank.tilt:g}-{bank.azimuth:g}-{model}-{mode}-'
            for cached in DATA.glob(prefix+'*.pkl'):
                frame=pd.read_pickle(cached)
                if frame.index.min()<=start and frame.index.max()>=end-pd.Timedelta(hours=1) and frame.attrs.get('run')==run.isoformat():return frame
    if cached_only:raise ValueError('Archived solar weather is not cached')
    params={'latitude':settings.latitude,'longitude':settings.longitude,'hourly':'temperature_2m,shortwave_radiation,global_tilted_irradiance','timezone':'UTC','tilt':bank.tilt,'azimuth':(bank.azimuth%360)-180}
    if run is not None:
        ranges=[(start,end,'https://single-runs-api.open-meteo.com/v1/forecast')]
    elif historical:
        cutoff=pd.Timestamp.now(tz='UTC').normalize()-pd.Timedelta(days=2)
        ranges=[(start,min(end,cutoff-pd.Timedelta(days=1)),'https://archive-api.open-meteo.com/v1/archive'),(max(start,cutoff),end,'https://api.open-meteo.com/v1/forecast')]
    else:
        ranges=[(start,end,'https://api.open-meteo.com/v1/forecast')]
    chunks=[]
    with httpx.Client(timeout=30 if run is not None else 120) as client:
        for left,right,url in ranges:
            if left>right: continue
            q={**params,'start_date':str(left.date()),'end_date':str((right+pd.Timedelta(days=1)).date())}
            if run is not None:
                q.pop('start_date'); q.pop('end_date')
                q.update(models=model,run=run.strftime('%Y-%m-%dT%H:%M'),forecast_days=7)
            elif 'archive-api' not in url: q['models']=model
            for attempt in range(4):
                response=client.get(url,params=q)
                if response.status_code not in (429,500,502,503,504): break
                time.sleep(min(15,3*(attempt+1)))
            response.raise_for_status(); hourly=response.json()['hourly']
            frame=pd.DataFrame(hourly)
            frame.index=pd.to_datetime(frame.pop('time'),utc=True)
            for column in ('temperature_2m','shortwave_radiation','global_tilted_irradiance'):
                if column not in frame and column+'_'+model in frame:
                    frame[column]=frame[column+'_'+model]
            chunks.append(frame[['temperature_2m','shortwave_radiation','global_tilted_irradiance']].rename(columns={'global_tilted_irradiance':'tilted_irradiance'}))
    if not chunks: raise ValueError('No Open-Meteo solar weather in the requested period')
    result=pd.concat(chunks).sort_index(); result=result[~result.index.duplicated(keep='last')]
    result=weather_intervals(result)
    result.attrs={'interval_policy':INTERVAL_POLICY,'run_basis':'Conservative archived initialization' if run is not None else 'Historical proxy' if historical else 'Stitched operational response; initialization unknown','irradiance_source':'Open-Meteo global_tilted_irradiance','weather_model':model,'run':run.isoformat() if run is not None else None,'publication_buffer_hours':12 if run is not None else None}
    result.to_pickle(path)
    return result


def consumption_weather(settings,start,end,historical=False,issued_at=None,cached_only=False):
    """Selected-model weather runs without any panel geometry dependency."""
    if issued_at is None:
        return Weather(settings).fetch(start,end,historical=True) if historical else forecast_weather(settings,start,end,settings.weather_model)
    run=archived_run(issued_at)
    path=DATA/f'load-weather-v2-{settings.latitude:.6f}-{settings.longitude:.6f}-{settings.weather_model}-{run.strftime("%Y%m%d%H")}.pkl'
    if path.exists():return pd.read_pickle(path)
    if cached_only:raise ValueError('Archived consumption weather is not cached')
    q={'latitude':settings.latitude,'longitude':settings.longitude,'models':settings.weather_model,'run':run.strftime('%Y-%m-%dT%H:%M'),'forecast_days':7,'hourly':'temperature_2m,shortwave_radiation','timezone':'UTC'}
    with httpx.Client(timeout=30) as client:
        response=client.get('https://single-runs-api.open-meteo.com/v1/forecast',params=q);response.raise_for_status();f=pd.DataFrame(response.json()['hourly'])
    f.index=pd.to_datetime(f.pop('time'),utc=True)
    for col in ('temperature_2m','shortwave_radiation'):
        if col not in f and col+'_'+settings.weather_model in f:f[col]=f[col+'_'+settings.weather_model]
    f=weather_intervals(f[['temperature_2m','shortwave_radiation']]);f.attrs={'interval_policy':INTERVAL_POLICY,'run':run.isoformat(),'publication_buffer_hours':12,'run_basis':'Conservative archived initialization'};f.to_pickle(path);return f


def weather_intervals(frame):
    """Preceding-hour radiation and endpoint-mean temperature at interval start."""
    f=frame.sort_index().copy()
    endpoint=f.reindex(f.index+pd.Timedelta(hours=1));endpoint.index=f.index
    for col in f:
        f[col]=(f[col]+endpoint[col])/2 if col=='temperature_2m' else endpoint[col]
    f.attrs['interval_policy']=INTERVAL_POLICY
    return f

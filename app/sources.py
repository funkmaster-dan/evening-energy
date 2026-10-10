"""Read-only Home Assistant, Open-Meteo, and Australian calendar inputs."""
import json, re, sqlite3, time
from datetime import date, timedelta
from urllib.parse import urlparse
import httpx, numpy as np, pandas as pd, holidays
from websockets.sync.client import connect
from .config import DATA
from .ha_inputs import numeric_value, history_values

class HomeAssistant:
    def __init__(self, settings):
        if not settings.ha_token:
            raise ValueError('Enter a Home Assistant long-lived access token and save the connection first.')
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
        return self.ws({'type':'recorder/statistics_during_period','start_time':pd.Timestamp(start).isoformat(),'end_time':pd.Timestamp(end).isoformat(),'statistic_ids':entities,'period':period,'types':['mean','sum']})

    def history(self, entity, start, end):
        return self.get('/api/history/period/'+pd.Timestamp(start).isoformat(), {'filter_entity_id':entity, 'end_time':pd.Timestamp(end).isoformat(),'minimal_response':'true','no_attributes':'true','significant_changes_only':'false'})

    def frame(self, entities, start, end, period='hour', progress=lambda s:None):
        """Use retained recorder statistics, with REST history for unrecorded sensors."""
        states={x['entity_id']:x for x in self.states()}
        start=pd.Timestamp(start); end=pd.Timestamp(end)
        result={}; step='5min' if period=='5minute' else '1h'
        # Chunking avoids a large recorder query for long datasets.
        for left in pd.date_range(start,end,freq='30D',inclusive='left'):
            right=min(left+pd.Timedelta(days=30),end)
            progress(f'Home Assistant: {left.date()} to {right.date()}')
            raw=self.statistics(entities,left,right,period)
            for entity in entities:
                rows=raw.get(entity,[])
                points=[]
                for row in rows:
                    t=pd.to_datetime(row['start'],unit=('ms' if row['start']>1e11 else 's'),utc=True) if isinstance(row['start'],(float,int)) else pd.to_datetime(row['start'],utc=True)
                    points.append((t,row.get('mean'),row.get('sum')))
                result.setdefault(entity,[]).extend(points)
        cols={}
        for entity in entities:
            attrs=states.get(entity,{}).get('attributes',{})
            unit=attrs.get('unit_of_measurement','W')
            rows=result[entity]
            if rows:
                f=pd.DataFrame(rows,columns=['time','mean','sum']).drop_duplicates('time').set_index('time').sort_index()
                if attrs.get('device_class')=='energy' or unit in ('kWh','Wh'):
                    # Hourly sum is cumulative; reset/negative deltas are discarded.
                    s=pd.to_numeric(f['sum'],errors='coerce').diff().clip(lower=0)
                    s=s/(1000 if unit=='Wh' else 1)/(1/12 if period=='5minute' else 1)
                else:
                    s=pd.to_numeric(f['mean'],errors='coerce')/(1000 if unit=='W' else 1)
            else:
                pieces=[]
                for left in pd.date_range(max(start,end-pd.Timedelta(days=30)),end,freq='7D',inclusive='left'):
                    history=self.history(entity,left,min(left+pd.Timedelta(days=7),end))
                    for group in history:
                        for r in group:
                            t=r.get('last_updated',r.get('last_changed'))
                            if t is not None:pieces.append((pd.to_datetime(t,utc=True),numeric_value(r)))
                s=pd.Series(dtype=float)
                if pieces:
                    s=history_values(pieces,max(start,end-pd.Timedelta(days=30)),end,step)/(1000 if unit=='W' else 1)
            cols[entity]=s
        frame=pd.DataFrame(cols).sort_index()
        # No recorder/history rows produces a RangeIndex; keep the time-index contract even when empty.
        frame.index=pd.to_datetime(frame.index,utc=True)
        return frame.loc[(frame.index>=start)&(frame.index<end)]

class Weather:
    def __init__(self, settings): self.s=settings

    def fetch(self, start, end, historical=False):
        start=pd.Timestamp(start); end=pd.Timestamp(end)
        key=f'weather-v2-{self.s.latitude:.4f}-{self.s.longitude:.4f}-{start.date()}-{end.date()}-{historical}'
        file=DATA/(key+'.pkl')
        if file.exists() and (historical or time.time()-file.stat().st_mtime<1200): return pd.read_pickle(file)
        params={'latitude':self.s.latitude,'longitude':self.s.longitude,'hourly':'temperature_2m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation','timezone':'UTC','start_date':str(start.date()),'end_date':str(end.date())}
        # Reanalysis through two days ago; use the forecast API's past_days coverage for the tail.
        chunks=[]
        cutoff=pd.Timestamp.now(tz='UTC').normalize()-pd.Timedelta(days=2)
        ranges=[(start,min(end,cutoff-pd.Timedelta(days=1)),True),(max(start,cutoff),end,False)] if historical else [(start,end,False)]
        for left,right,archive in ranges:
            if left>right: continue
            q={**params,'start_date':str(left.date()),'end_date':str((right+pd.Timedelta(days=0 if archive else 1)).date())}
            url='https://archive-api.open-meteo.com/v1/archive' if archive else 'https://api.open-meteo.com/v1/forecast'
            with httpx.Client(timeout=120) as c:
                for attempt in range(3):
                    r=c.get(url,params=q)
                    if r.status_code!=429: break
                    time.sleep(3*(attempt+1))
                r.raise_for_status(); h=r.json()['hourly']
            f=pd.DataFrame(h); f.index=pd.to_datetime(f.pop('time'),utc=True)-pd.Timedelta(hours=1); chunks.append(f)
        if not chunks: raise ValueError('No weather data for that period')
        f=pd.concat(chunks).sort_index(); f=f[~f.index.duplicated(keep='last')]; f.to_pickle(file)
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


def forecast_weather(settings,start,end,model='ecmwf_ifs',issued_at=None):
    """Single archived run with a conservative 12h publication buffer, or live model."""
    start=pd.Timestamp(start);end=pd.Timestamp(end)
    run=(pd.Timestamp(issued_at)-pd.Timedelta(hours=12)).floor('6h') if issued_at is not None else None
    key=f'weather-run-{settings.latitude:.4f}-{settings.longitude:.4f}-{model}-{run.strftime("%Y%m%d%H") if run is not None else start.strftime("%Y%m%d%H")}-{end.strftime("%Y%m%d%H")}'
    file=DATA/(key+'.pkl')
    if file.exists() and (run is not None or time.time()-file.stat().st_mtime<1200):return pd.read_pickle(file)
    params={'latitude':settings.latitude,'longitude':settings.longitude,'models':model,'hourly':'temperature_2m,shortwave_radiation,direct_normal_irradiance,diffuse_radiation','timezone':'UTC','forecast_days':7}
    if run is not None: params['run']=run.strftime('%Y-%m-%dT%H:%M')
    else:
        params.pop('forecast_days')
        params.update(start_date=str(start.date()),end_date=str((end+pd.Timedelta(days=1)).date()))
    url='https://single-runs-api.open-meteo.com/v1/forecast' if run is not None else 'https://api.open-meteo.com/v1/forecast'
    with httpx.Client(timeout=120) as c:
        r=c.get(url,params=params);r.raise_for_status();h=r.json()['hourly']
    f=pd.DataFrame(h);f.index=pd.to_datetime(f.pop('time'),utc=True)-pd.Timedelta(hours=1)
    for col in ['temperature_2m','shortwave_radiation','direct_normal_irradiance','diffuse_radiation']:
        if col not in f and col+'_'+model in f:f[col]=f[col+'_'+model]
    f=f[['temperature_2m','shortwave_radiation','direct_normal_irradiance','diffuse_radiation']]
    f.attrs={'model':model,'run':run.isoformat() if run is not None else None,'retrieved_at':pd.Timestamp.now(tz='UTC').isoformat(),'publication_buffer_hours':12 if run is not None else None}
    f.to_pickle(file);return f

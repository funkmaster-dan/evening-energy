"""Immutable issued forecasts, scored only once complete actual intervals exist."""
import json, sqlite3, hashlib
import numpy as np, pandas as pd
from .config import DATA, read_json, write_json
from .sources import HomeAssistant


def archive_issue(settings,result,weather=None,alternatives=None):
    with sqlite3.connect(DATA/'history.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS forecast_issues (issued TEXT PRIMARY KEY, payload TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS forecast_weather_inputs (id TEXT PRIMARY KEY, payload TEXT)')
        raw=json.dumps(weather or {},allow_nan=False); weather_id=hashlib.sha256(raw.encode()).hexdigest()
        db.execute('INSERT OR IGNORE INTO forecast_weather_inputs VALUES (?,?)',(weather_id,raw))
        payload={'timezone':settings.timezone,'load_entity':settings.consumption_entity,'solar_entity':settings.solar_ct_entity,'peak_start':settings.export_start,'free_windows':[w.model_dump() for w in settings.charging_windows], 'forecast':[{k:r[k] for k in ('time','hours','load_middle','solar_middle','load_low','load_high','solar_low','solar_high','known_loads') if k in r} for r in result['forecast']],'model_updated_at':result['model_updated_at'],'solar_correction':result.get('solar_correction'), 'alternatives':alternatives or {}, 'weather_input_id':weather_id}
        db.execute('INSERT OR IGNORE INTO forecast_issues VALUES (?,?)',(result['updated_at'],json.dumps(payload,allow_nan=False)))


def issue_rows():
    with sqlite3.connect(DATA/'history.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS forecast_issues (issued TEXT PRIMARY KEY, payload TEXT)')
        return [(pd.Timestamp(t),json.loads(p)) for t,p in db.execute('SELECT issued,payload FROM forecast_issues ORDER BY issued')]


def score_issues(settings,progress=lambda s:None):
    issues=issue_rows(); now=pd.Timestamp.now(tz='UTC').floor('h')
    if not issues: return {'daily':{},'overnight':{'samples':0},'note':'Recording issued forecasts. Scores appear after complete actual days and nights.'}
    groups={}
    # Choose the latest forecast issued BEFORE each local midnight or configured peak.
    for issued,p in issues:
        times=pd.DatetimeIndex([r['time'] for r in p['forecast']]); tz=p['timezone']
        for date in sorted(set(times.tz_convert(tz).date)):
            start=pd.Timestamp(str(date),tz=tz).tz_convert('UTC'); end=(pd.Timestamp(str(date),tz=tz)+pd.DateOffset(days=1)).tz_convert('UTC')
            for kind,a,b in [('daily',start,end),('overnight',pd.Timestamp(f'{date} {p["peak_start"]}',tz=tz).tz_convert('UTC'),None)]:
                if kind=='overnight':
                    from .load_training import next_free_start
                    # Snapshot tariff assumptions, not today's edited windows.
                    from .config import Window
                    old=settings.model_copy(update={'timezone':tz,'charging_windows':[Window(**w) for w in p['free_windows']]})
                    b=next_free_start(a,old)
                if b is None or issued>a or b>now or times[0]>a or pd.Timestamp(p['forecast'][-1]['time'])+pd.Timedelta(hours=p['forecast'][-1]['hours'])<b: continue
                key=(kind,str(date),p['load_entity'],p['solar_entity'],tz)
                groups[key]=(issued,p,a,b)
    scores={'daily':{'load':[],'solar':[]},'overnight':{'load':[]},'weather_models':{},'comparisons':[]}
    actual_cache={}
    for (kind,date,load,solar,tz),(issued,p,a,b) in groups.items():
        key=(load,solar)
        if key not in actual_cache:
            left=min(v[2] for k,v in groups.items() if k[2:4]==key);right=max(v[3] for k,v in groups.items() if k[2:4]==key)
            progress('Fetching actual load and solar for issued-forecast scoring')
            actual_cache[key]=HomeAssistant(settings).frame(list(key),left,right,'5minute')
        actual=actual_cache[key]; grid=pd.date_range(a,b,freq='5min',inclusive='left'); observed=actual.reindex(grid)
        rows=p['forecast']
        for name,entity in [('load',load),('solar',solar)] if kind=='daily' else [('load',load)]:
            if observed[entity].isna().any(): continue
            value=float(observed[entity].sum()/12)
            predicted=sum(r[name+'_middle']*max(0,(min(b,pd.Timestamp(r['time'])+pd.Timedelta(hours=r['hours']))-max(a,pd.Timestamp(r['time']))).total_seconds()/3600) for r in rows)
            error=value-predicted;scores[kind][name].append(error)
            scores['comparisons'].append({'kind':kind,'date':date,'quantity':name,'issued_at':issued.isoformat(),'actual_kwh':round(value,3),'predicted_kwh':round(predicted,3),'error_kwh':round(error,3)})
            if kind=='daily' and name=='solar':
                for model,points in p.get('alternatives',{}).items():
                    series=pd.Series({pd.Timestamp(t):v for t,v in points.items()});vals=series.reindex(grid.floor('h'))
                    if not vals.isna().any(): scores['weather_models'].setdefault(model,[]).append(value-float(vals.sum()/12))
    def summary(errors):
        return {'samples':len(errors),'energy_mae_kwh':round(float(np.mean(np.abs(errors))),3),'energy_bias_kwh':round(float(np.mean(errors)),3),'underprediction_kwh':round(float(np.mean(np.maximum(errors,0))),3)} if errors else {'samples':0}
    result={'updated_at':now.isoformat(),'issues':len(issues),'daily':{k:summary(v) for k,v in scores['daily'].items()},'overnight':summary(scores['overnight']['load']),'weather_models':{k:summary(v) for k,v in scores['weather_models'].items()},'comparisons':scores['comparisons'][-120:],'note':'Actual issued forecasts, including weather and corrections available at issue time. One issue per target day/night; only complete measured intervals are scored. Positive bias means underprediction. Small sample counts remain preliminary.'}
    write_json('forecast-measurement',result);return result


def solar_correction(settings,now,solar,live=None):
    """Use a prediction issued before the observation window, never a hindsight fit."""
    info={'applied':False,'reason':'Not enough recent daylight measurements'}
    if live is not None:
        from .ha_inputs import snapshot_age
        if live.get('solar_kw') is None or snapshot_age(live,now)>=900:return solar,{'applied':False,'reason':'Current solar reading or HA snapshot unavailable'}
    if not settings.solar_nowcast: return solar,{'applied':False,'reason':'Disabled'}
    start=now-pd.Timedelta(minutes=60)
    with sqlite3.connect(DATA/'history.sqlite') as db:
        db.execute('CREATE TABLE IF NOT EXISTS forecast_issues (issued TEXT PRIMARY KEY, payload TEXT)')
        row=db.execute('SELECT issued,payload FROM forecast_issues WHERE issued<=? ORDER BY issued DESC LIMIT 1',(start.isoformat(),)).fetchone()
    previous=(pd.Timestamp(row[0]),json.loads(row[1])) if row else None
    if previous and previous[1]['solar_entity']!=settings.solar_ct_entity:previous=None
    if previous is None: return solar,info
    with sqlite3.connect(DATA/'history.sqlite') as db:
        rows=db.execute('SELECT time,value,unit FROM samples WHERE entity=? AND time>=? AND time<? ORDER BY time',(settings.solar_ct_entity,start.isoformat(),now.isoformat())).fetchall()
    if len(rows)<6: return solar,info
    times=pd.DatetimeIndex([r[0] for r in rows]); values=np.array([r[1]/(1000 if r[2]=='W' else 1) for r in rows])
    if (times[-1]-times[0]).total_seconds()<1800 or (now-times[-1]).total_seconds()>180: return solar,info
    prior=previous[1]['forecast'];index=pd.DatetimeIndex([r['time'] for r in prior]);idx=index.get_indexer(times,method='pad');valid=idx>=0
    expected=np.array([prior[i]['solar_middle'] if i>=0 else np.nan for i in idx])
    if not valid.all() or not np.isfinite(values).all() or expected.mean()<.3: return solar,info
    # Half-strength ratio, capped at +/-25%, zero influence after four hours and at next local date.
    factor=float(np.clip(1+.5*(values.mean()/expected.mean()-1),.75,1.25))
    lead=np.maximum(0,(solar.index-now).total_seconds()/3600);fade=np.maximum(0,1-lead/4)
    fade=np.where(solar.index.tz_convert(settings.timezone).date==now.tz_convert(settings.timezone).date(),fade,0)
    corrected=solar.mul(1+(factor-1)*fade,axis=0)
    return corrected,{'applied':True,'factor':round(factor,3),'samples':len(rows),'measured_kw':round(float(values.mean()),3),'predicted_kw':round(float(expected.mean()),3),'fade_hours':4,'reference_issued_at':previous[0].isoformat()}

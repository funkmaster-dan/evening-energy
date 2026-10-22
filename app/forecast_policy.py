"""Shared availability, postprocessing and private issue records."""
import hashlib,json,sqlite3
import numpy as np
import pandas as pd
from .config import DATA

FEATURE_SCHEMA='issue-weather-energy-v3'
INTERVAL_POLICY='radiation-preceding-temperature-endpoints-v2'
PUBLICATION_HOURS=12
SETTLEMENT_HOURS=1
LEAD_GROUPS=((0,24),(24,48),(48,73))

def archived_run(origin):
    """Conservative replay assumption, not a claim of actual publication time."""
    return (pd.Timestamp(origin).tz_convert('UTC')-pd.Timedelta(hours=PUBLICATION_HOURS)).floor('6h')

def lead_groups(leads):
    return np.minimum(np.nan_to_num(np.asarray(leads,float),nan=0).astype(int)//24,2)

def model_columns(model,default):
    return list(getattr(model,'feature_name_',default))

def ordered(raw,leads=None,calibration=None,dark=None):
    values=np.column_stack([np.asarray(raw[k],float) for k in ('low','middle','high')])
    if calibration:
        group=lead_groups(leads if leads is not None else np.zeros(len(values)))
        for i,key in enumerate(('low','middle','high')):
            values[:,i]+=np.array([calibration.get(str(g),calibration.get('pooled',{})).get(key,0.) for g in group])
    values=np.maximum(0,np.sort(values,axis=1))
    if dark is not None:values[np.asarray(dark,bool)]=0.
    return dict(zip(('low','middle','high'),values.T))

def diagnostics(actual,predicted,point=None):
    y=np.asarray(actual,float)
    if not len(y):return {'samples':0}
    q={k:np.asarray(predicted[k],float) for k in ('low','middle','high')}
    result={'samples':len(y),'coverage':float(np.mean((y>=q['low'])&(y<=q['high']))),'width_kw':float(np.mean(q['high']-q['low'])),
            'lower_tail':float(np.mean(y<q['low'])),'upper_tail':float(np.mean(y>q['high'])),
            'median_residual_kw':float(np.median(y-q['middle'])),'zero_actual_share':float(np.mean(y==0)),
            'point_bias_kw':float(np.mean(np.asarray(point if point is not None else q['middle'])-y)),
            'bias_direction':'predicted minus actual; positive means overprediction'}
    for key,a in zip(('low','middle','high'),(.1,.5,.9)):
        e=y-q[key]
        result[key]={'cdf_strict':float(np.mean(y<q[key])),'cdf_inclusive':float(np.mean(y<=q[key])),
                     'pinball':float(np.mean(np.maximum(a*e,(a-1)*e))),'zero_share':float(np.mean(q[key]==0))}
    result['mean_pinball']=float(np.mean([result[k]['pinball'] for k in q]))
    return result

def fit_calibration(records,grouped=False,solar=False):
    """Earlier held-out residuals; solar counts independent daylight targets."""
    rows=[r for r in records if r.get('scored',True)]
    def distinct(selected):
        # Latest issue per target prevents overlapping horizons inflating confidence.
        chosen={}
        for r in selected:
            if r['time'] not in chosen or r['origin']>chosen[r['time']]['origin']:chosen[r['time']]=r
        return list(chosen.values())
    pooled_rows=distinct(rows) if solar else rows
    minimum=24 if solar else 180
    days={pd.Timestamp(r['time']).date() for r in pooled_rows} if solar else {r['origin'][:10] for r in pooled_rows}
    if len(pooled_rows)<minimum or len(days)<3:return {}
    def offsets(selected):
        n=len(selected);shrink=n/(n+(24 if solar else 100))
        return {k:float(np.quantile([r['actual']-r[k] for r in selected],a)*shrink) for k,a in zip(('low','middle','high'),(.1,.5,.9))}
    pooled=offsets(pooled_rows);result={'pooled':pooled}
    if grouped:
        for g in range(3):
            part=[r for r in rows if min(int(r['lead'])//24,2)==g]
            if solar:part=distinct(part)
            if len(part)>=(24 if solar else 180):
                local=offsets(part);w=len(part)/(len(part)+(48 if solar else 300))
                result[str(g)]={k:w*local[k]+(1-w)*pooled[k] for k in pooled}
    return result


def recent_correction(records,origin,leads,policy,solar=False):
    """Settled previous issues only; one latest issue per target, bounded/shrunk."""
    identity=1. if solar else 0.
    if not policy or not records:return np.full(len(leads),identity)
    origin=pd.Timestamp(origin);stop=origin-pd.Timedelta(hours=SETTLEMENT_HOURS)
    left=stop-pd.Timedelta(days=policy['days']);latest={}
    origin_text=origin.isoformat();left_text=left.isoformat();right_text=(stop-pd.Timedelta(hours=1)).isoformat()
    for r in records:
        target=r['time'];issued=r['origin']
        if issued>=origin_text or target<left_text or target>right_text or not r.get('scored',True):continue
        if target not in latest or issued>latest[target]['origin']:latest[target]=r
    rows=list(latest.values())
    if len(rows)<24 or max(pd.Timestamp(r['time']) for r in rows)<stop-pd.Timedelta(days=2):return np.full(len(leads),identity)
    actual=sum(r['actual'] for r in rows);pred=sum(r.get('base_point',r.get('point',r['middle'])) for r in rows)
    if solar and pred<2:return np.full(len(leads),identity)
    delta=actual/pred-1 if solar else (actual-pred)/len(rows)
    delta=float(np.clip(delta,-policy['bound'],policy['bound']))*len(rows)/(len(rows)+48)
    return identity+delta*np.exp(-np.asarray(leads)/policy['taper'])

def issue_store():
    db=sqlite3.connect(DATA/'forecast-issues.sqlite',timeout=30)
    db.execute('CREATE TABLE IF NOT EXISTS issues (id TEXT PRIMARY KEY, kind TEXT, origin TEXT, payload TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS outcomes (issue_id TEXT, target TEXT, actual REAL, PRIMARY KEY(issue_id,target))')
    return db

def record_issue(kind,origin,hours,x,predicted,weather,model_identity,**extra):
    """First issue wins; repeated refreshes cannot replace original inputs."""
    forecast_origin=pd.Timestamp(origin).isoformat();origin=pd.Timestamp(extra.pop('issued_at',origin)).isoformat();identity=hashlib.sha256((kind+origin+model_identity).encode()).hexdigest()
    payload={'schema':FEATURE_SCHEMA,'forecast_origin_hour':forecast_origin,'origin':origin,'kind':kind,'model_identity':model_identity,'hours':[t.isoformat() for t in hours],
             'features':json.loads(x.to_json(orient='split',date_format='iso')),'weather':json.loads(weather.to_json(orient='split',date_format='iso')),
             'weather_metadata':weather.attrs,'predictions':{k:np.asarray(v).tolist() for k,v in predicted.items()},**extra}
    with issue_store() as db:db.execute('INSERT OR IGNORE INTO issues VALUES (?,?,?,?)',(identity,kind,origin,json.dumps(payload,default=str,allow_nan=False)))
    return identity

def settled_records(kind,origin,days=14,model_identity=None):
    start=(pd.Timestamp(origin)-pd.Timedelta(days=days+4)).isoformat();rows=[]
    with issue_store() as db:
        for payload,target,actual in db.execute('SELECT i.payload,o.target,o.actual FROM issues i JOIN outcomes o ON i.id=o.issue_id WHERE i.kind=? AND i.origin>=? AND i.origin<?',(kind,start,pd.Timestamp(origin).isoformat())):
            p=json.loads(payload)
            if model_identity and p['model_identity']!=model_identity:continue
            j=p['hours'].index(target)
            features=p.get('features',{});columns=features.get('columns',[]);dark=bool(features['data'][j][columns.index('full_dark')]) if 'full_dark' in columns else False
            rows.append({'origin':p['origin'],'time':target,'actual':actual,'lead':j,**{k:v[j] for k,v in p['predictions'].items()},'scored':not dark})
    return rows

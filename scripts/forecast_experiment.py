"""Private calibration/comparison and resumable prospective collection.

Run inside the application container with its normal ENERGY_DATA_DIR. All output
remains in private data storage; the command never publishes household results.
"""
import argparse,json,time
import numpy as np,pandas as pd
from app.config import load_settings,DATA,read_json,write_json
from app.forecast_policy import issue_store,SETTLEMENT_HOURS

def settle(settings):
    from app.sources import HomeAssistant
    from app.quality import effective_policy,quality_frame,dark_hours
    from app.datasets import splice
    now=pd.Timestamp.now(tz='UTC')-pd.Timedelta(hours=SETTLEMENT_HOURS);count=0
    with issue_store() as db:issues=list(db.execute('SELECT id,kind,origin,payload FROM issues'))
    for kind in sorted({row[1] for row in issues}):
        selected=[r for r in issues if r[1]==kind];targets=sorted({t for r in selected for t in json.loads(r[3])['hours'] if pd.Timestamp(t)+pd.Timedelta(hours=1)<=now})
        if not targets:continue
        bank=next((b for b in settings.banks if kind=='solar-'+b.id),None);entities=[bank.entity] if bank else settings.historical_consumption_entities
        policy=effective_policy(settings,'solar' if bank else 'load',read_json('dataset-'+kind,{}).get('request',{}))
        f=HomeAssistant(settings).frame(entities,pd.Timestamp(targets[0]),pd.Timestamp(targets[-1])+pd.Timedelta(hours=1),measurement_kind='solar' if bank else 'load',max_power_kw=policy['max_power_kw'],units=policy['units'],corrections=policy['corrections'])
        raw=splice(f,entities,read_json('dataset-'+kind,{}).get('request',{}).get('combine','splice'))
        q,_=quality_frame(raw,settings,'solar' if bank else 'load',policy['max_power_kw'],corrections=policy['corrections'])
        values=q.target.where(q.validation_eligible)
        if bank:
            # Deterministic dark zeros only for ordinary missing values, never rejected faults.
            dark=dark_hours(settings,values.index);values=values.mask(dark&q.quality_flags.eq('missing'),0.)
        with issue_store() as db:
            for identity,_,_,payload in selected:
                p=json.loads(payload)
                for t in p['hours']:
                    actual=values.get(pd.Timestamp(t),np.nan)
                    if pd.Timestamp(t)+pd.Timedelta(hours=1)<=now and np.isfinite(actual):
                        cur=db.execute('INSERT OR IGNORE INTO outcomes VALUES (?,?,?)',(identity,t,float(actual)));count+=cur.rowcount
    return count

def score(settings):
    from app.forecast_policy import diagnostics
    result={}
    with issue_store() as db:
        for kind, in db.execute('SELECT DISTINCT kind FROM issues'):
            rows=[];windows={str(h):[] for h in (24,48,72)};days=set()
            for identity,origin,payload in db.execute('SELECT id,origin,payload FROM issues WHERE kind=?',(kind,)):
                p=json.loads(payload);truth=dict(db.execute('SELECT target,actual FROM outcomes WHERE issue_id=?',(identity,)));q=p['predictions']
                for i,t in enumerate(p['hours']):
                    if t in truth:rows.append({'actual':truth[t],**{k:v[i] for k,v in q.items()}});days.add(pd.Timestamp(origin).tz_convert(settings.timezone).date())
                for h in (24,48,72):
                    origin_time=pd.Timestamp(p['origin']);until=origin_time+pd.Timedelta(hours=h)
                    weighted=[]
                    for i,t in enumerate(p['hours']):
                        target=pd.Timestamp(t);duration=max(0,(min(target+pd.Timedelta(hours=1),until)-max(target,origin_time)).total_seconds()/3600)
                        if duration:weighted.append((i,t,duration))
                    if abs(sum(w for i,t,w in weighted)-h)<1e-8 and all(t in truth for i,t,w in weighted):windows[str(h)].append(sum((q.get('point',q['middle'])[i]-truth[t])*w for i,t,w in weighted))
            if not rows:result[kind]={'status':'Pending settled outcomes'};continue
            result[kind]={'status':'Prospective interim; assess after 28–56 days','distinct_issue_days':len(days),'quantile':diagnostics([r['actual'] for r in rows],{k:[r[k] for r in rows] for k in ('low','middle','high')},[r.get('point',r['middle']) for r in rows]),'horizons':{h:{'windows':len(v),'mae_kwh':float(np.mean(np.abs(v))) if v else None,'bias_kwh':float(np.mean(v)) if v else None} for h,v in windows.items()}}
    write_json('prospective-assessment',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','watch','score','compare','calibrate']);p.add_argument('--once',action='store_true');p.add_argument('--hours',type=float,default=24*28);p.add_argument('--promote',action='store_true');p.add_argument('--models',nargs='*');args=p.parse_args();settings=load_settings()
    if args.mode=='compare':
        from app.assessment import aggregate
        result=aggregate(settings);print(json.dumps({k:v for k,v in result.items() if k not in ('candidate','existing')},indent=2));return
    if args.mode=='calibrate':
        from app.models import train,promote_candidate
        names=args.models or ['load',*['solar-'+b.id for b in settings.banks]]
        for name in names:
            train(settings,name,lambda message:print(message,flush=True))
            if args.promote:promote_candidate(settings,name)
        return
    if args.mode=='score':print(json.dumps(score(settings),indent=2));return
    if args.mode=='watch':
        stop=time.monotonic()+args.hours*3600
        while True:
            print('Settled outcome inserts:',settle(settings),flush=True);score(settings)
            if args.once or time.monotonic()>=stop:return
            for _ in range(60):
                time.sleep(60)
                if time.monotonic()>=stop:return
    from app.forecast import Forecaster
    from app.main import refresh_live
    forecaster=Forecaster();stop=time.monotonic()+args.hours*3600
    while True:
        forecaster.run(settings,refresh_live(),force=True)
        print('Recorded immutable issue; settled outcomes:',settle(settings),flush=True);score(settings)
        if args.once or time.monotonic()>=stop:break
        # Short waits allow clean interruption and resumable first-write storage.
        for _ in range(60):
            time.sleep(60)
            if time.monotonic()>=stop:return

if __name__=='__main__':main()

"""Bounded chronological forecast experiments, shared by load and panel models.

Scores use issued weather only. Historical weather remains an explicit ablation.
Every fitting/stopping target finishes before the corresponding scored fold.
"""
import hashlib,os,time
import joblib,lightgbm as lgb,numpy as np,pandas as pd
from .config import DATA,read_json,write_json, solar_banks
from .forecast_policy import (FEATURE_SCHEMA,INTERVAL_POLICY,archived_run,ordered,diagnostics,fit_calibration,
                             recent_correction,LEAD_GROUPS,SETTLEMENT_HOURS)
from .datasets import FEATURES,SOLAR_FEATURES,LEGACY_SOLAR_FEATURES,features,bank_features
from .quality import QUALITY_POLICY,LOAD_PIPELINE,dark_hours
from .load_splits import training_mask,split_coverage

LOAD_SCHEMA=None
BASE=dict(num_leaves=31,max_depth=5,min_child_samples=100,learning_rate=.05,reg_lambda=2.,reg_alpha=.1,subsample=.9,colsample_bytree=.9)
# Staged comparisons: never a Cartesian product.
SHAPES=[dict(num_leaves=n,max_depth=d) for n,d in ((15,4),(63,6),(127,7),(31,7))]+[dict(min_child_samples=n) for n in (30,300)]+[dict(learning_rate=r) for r in (.02,.1)]+[dict(reg_lambda=5.,reg_alpha=.5,subsample=.8,colsample_bytree=.8)]
CORRECTIONS=[None,{'days':3,'bound':.25,'taper':24},{'days':7,'bound':.5,'taper':48}]

def usable_columns(kind,frame=None):
    extra=[c for c in frame.columns if c.startswith('bank_irradiance_')] if frame is not None else []
    return FEATURES if kind=='load' else LEGACY_SOLAR_FEATURES+extra

def schema(kind,frame=None):
    from .load_training import LOAD_FEATURES
    return LOAD_FEATURES if kind=='load' else SOLAR_FEATURES+([c for c in frame.columns if c.startswith('bank_irradiance_')] if frame is not None else [])

def _finite(x,kind):
    return x[usable_columns(kind,x)].notna().all(axis=1)

def origins(start,end,settings):
    """Exact local clocks, including half-hour zones and DST boundaries."""
    first=pd.Timestamp(start).tz_convert(settings.timezone).normalize()
    result=set()
    for day in pd.date_range(first,pd.Timestamp(end).tz_convert(settings.timezone).normalize(),freq='D'):
        for clock in {'00:00','06:00','12:00','18:00',*[w.start for w in settings.export_windows]}:
            t=pd.Timestamp(f'{day.date()} {clock}').tz_localize(settings.timezone,ambiguous=False,nonexistent='shift_forward').tz_convert('UTC')
            if start<=t<end:result.add(t)
    return sorted(result)


def make_case(frame,origin,end,settings,kind,bank=None,basis='issued',calendar=None):
    from .sources import consumption_weather,solar_weather,Calendar
    from .load_training import feature_rows,next_charge_start
    index=pd.date_range(pd.Timestamp(origin).floor('h'),min(pd.Timestamp(origin).floor('h')+pd.Timedelta(hours=73),pd.Timestamp(end).ceil('h')),freq='h',inclusive='left')
    if len(index)<24:return None
    weather=None
    if basis=='issued':
        run=archived_run(origin)
        weather=(consumption_weather(settings,run,run+pd.Timedelta(days=7),issued_at=origin) if kind=='load' else solar_weather(settings,bank,run,run+pd.Timedelta(days=7),issued_at=origin))
        weather=weather.reindex(index)
    if kind=='load':
        context=frame.loc[(frame.index<origin-pd.Timedelta(hours=SETTLEMENT_HOURS))&(frame.index>=origin-pd.Timedelta(hours=169))]
        if context.get('context_target',context.target).count()<126:return None
        future=frame.reindex(index).copy();attrs=frame.attrs.copy()
        if basis=='issued':future[FEATURES]=features(weather,settings,calendar or Calendar(settings).load())[FEATURES]
        future['target']=np.nan;future['context_target']=np.nan
        safe=frame.loc[(frame.index<origin.floor('h'))&(frame.index>=origin-pd.Timedelta(days=58))].copy()
        # One full additional hour for recorder settlement, in training/replay/live.
        safe.loc[safe.index+pd.Timedelta(hours=1)>origin-pd.Timedelta(hours=SETTLEMENT_HOURS),'context_target']=np.nan
        safe.attrs={};future.attrs={};safe=pd.concat([safe,future]);safe.attrs=attrs
        x=feature_rows(safe,origin.floor('h'),index,settings.timezone)
        run=(weather.attrs.get('run') if weather is not None else None)
        x['weather_run_age']=(pd.Timestamp(origin)-pd.Timestamp(run)).total_seconds()/3600 if run else np.nan
        dark=np.zeros(len(x),bool)
    else:
        x=bank_features(weather,bank,settings,origin) if basis=='issued' else frame.reindex(index)[schema(kind,frame)].copy()
        x['lead_hours']=np.asarray((index-pd.Timestamp(origin)).total_seconds()/3600)
        dark=dark_hours(settings,index);x['full_dark']=dark.astype(float)
    actual=frame.target.reindex(index).to_numpy(float)
    measured=frame.get('measurement_valid',frame.target.notna()).reindex(index).fillna(False).to_numpy(bool)
    eligible=frame.validation_eligible.reindex(index).fillna(False).to_numpy(bool)
    derived=frame.get('derived_method',pd.Series('',index=frame.index)).reindex(index).eq('dark_zero').to_numpy()
    valid=(eligible&measured&np.isfinite(actual)) | (dark&derived) if kind=='solar' else eligible&np.isfinite(actual)
    actual=np.where(dark&derived,0,actual)
    if not _finite(x,kind).all():return None
    if kind=='load' and x.clock_profile.isna().any():return None
    local=origin.tz_convert(settings.timezone);daily_end=(local.normalize()+pd.DateOffset(days=1)).tz_convert('UTC')
    charge_start=next_charge_start(origin,settings) if kind=='load' else None
    return {'origin':origin,'hours':index,'x':x.reset_index(drop=True),'actual':actual,'valid':valid,'daylight':~dark,
            'daily_end':daily_end,'end':end,'charge_start':charge_start,'weather_basis':basis,'archive':basis=='issued',
            'weather_metadata':weather.attrs if weather is not None else {'basis':'historical proxy'},'quiet':frame.get('quiet_regime',pd.Series(False,index=frame.index)).reindex(index).fillna(False).to_numpy(bool)}

def cases(frame,start,end,settings,kind,bank=None,basis='issued',progress=lambda s:None):
    from .sources import Calendar
    result=[];skipped=[];calendar=Calendar(settings).load() if kind=='load' else None
    for origin in origins(start,end,settings):
        try:
            c=make_case(frame,origin,end,settings,kind,bank,basis,calendar)
            if c is not None and any(window(c,h) is not None for h in (24,48,72)):result.append(c)
            else:skipped.append({'origin':origin.isoformat(),'reason':'Incomplete context, weather or targets'})
        except Exception as exc:skipped.append({'origin':origin.isoformat(),'reason':str(exc)})
    progress(f'{kind}: {len(result)} issued cases; {len(skipped)} unavailable')
    return result,skipped

def window(c,h):
    until=c['origin']+pd.Timedelta(hours=h) if isinstance(h,(int,float)) else pd.Timestamp(h)
    if until>c['end'] or until<=c['origin']:return None
    t=c['hours'];weights=np.maximum(0,(np.minimum((t+pd.Timedelta(hours=1)).as_unit('ns').asi8,until.value)-np.maximum(t.as_unit('ns').asi8,c['origin'].value))/3.6e12)
    m=weights>0
    if not m.any() or not c['valid'][m].all():return None
    return weights

def fitting_rows(frame,start,end,settings,kind,bank=None,basis='issued',progress=lambda s:None):
    """Deterministic origins with all 0–72 leads; never repeat reanalysis as issued input."""
    from .sources import Calendar
    from .load_training import training_rows
    calendar=Calendar(settings).load() if kind=='load' else None
    # Archive-supported comparison period is identical for both weather bases.
    start=pd.Timestamp(start) if basis=='historical_full' else max(pd.Timestamp(start),pd.Timestamp('2024-04-01',tz='UTC'))
    limit=int(os.getenv('ENERGY_FIT_ORIGINS','240'))
    # Fixed anchors reuse the same issued runs across expanding folds. Daily tail
    # origins retain recent sensor/layout changes rather than resampling all history.
    anchors=pd.date_range(start.ceil('D'),end-pd.Timedelta(hours=73),freq='7D' if basis=='historical_full' else '4D')
    if len(anchors)>max(1,limit-14):anchors=anchors[-max(1,limit-14):]
    tail=pd.date_range(max(start.ceil('D'),end.floor('D')-pd.Timedelta(days=17)),end-pd.Timedelta(hours=73),freq='D')
    available=anchors.union(tail).sort_values()
    chunks=[];ys=[];times=[];methods=[];failed=[]
    for i,day in enumerate(available):
        # Cycle issue clocks while retaining lead representation.
        origin=(day+pd.Timedelta(hours=(day.dayofyear%4)*6)).floor('h')
        if origin+pd.Timedelta(hours=73)>end:continue
        try:
            c=make_case(frame,origin,end,settings,kind,bank,'historical' if basis=='historical_full' else 'issued',calendar)
            if c is None:continue
            index=c['hours'];target=frame.reindex(index)
            eligible=target.train_eligible.fillna(False).to_numpy(bool)&target.target.notna().to_numpy()&(index+pd.Timedelta(hours=1)<=end)
            if kind=='solar':
                # Derived interpolation may not borrow a right endpoint across the fitting boundary.
                right=pd.to_datetime(target.derived_right.replace('',None),utc=True)
                eligible &= target.derived_method.ne('interpolation').to_numpy()|((right+pd.Timedelta(hours=1))<=end).to_numpy()
            if not eligible.any():continue
            x=c['x'].copy()
            if basis in ('historical','historical_full'):
                historic=frame.reindex(index)
                x[usable_columns(kind,frame)]=historic[usable_columns(kind,frame)].to_numpy()
                x['weather_run_age']=np.nan
            eligible &= _finite(x,kind).to_numpy()
            chunks.append(x.loc[eligible]);ys.append(target.target.to_numpy(float)[eligible]);times.extend(index[eligible]);methods.extend(target.get('derived_method',pd.Series('',index=index)).fillna('').to_numpy()[eligible])
        except Exception as exc:failed.append({'origin':str(origin),'reason':str(exc)})
        if i%40==0:progress(f'{kind}: preparing {basis} fitting origins {i+1}/{len(available)}')
    if not chunks:raise ValueError(f'No archive-supported fitting origins for {kind}')
    x=pd.concat(chunks,ignore_index=True)[schema(kind,frame)].astype(np.float32);y=np.concatenate(ys);times=pd.DatetimeIndex(times)
    # Stitched operational forecasts do not disclose one initialization. Train that missing-age case.
    x.loc[np.arange(len(x))%7==0,'weather_run_age']=np.nan
    x.attrs['preparation']={'origin_limit':limit,'origins':len(chunks),'rows':len(x),'unique_target_hours':len(times.unique()),'unavailable':failed,'weather_basis':basis,'start':str(start),'end':str(end)}
    return x,y,times,np.asarray(methods)

def row_filter(frame,x,y,times,methods,config,kind):
    mask=np.ones(len(y),bool)
    if kind=='solar':
        recipe=config.get('recipe','measured_only')
        mask &= (methods=='') if recipe=='measured_only' else (methods!='interpolation') if recipe=='dark_zero' else True
        if recipe=='measured_only':mask &= x.full_dark.to_numpy()==0
    if config.get('exclude_quiet'):mask &= ~frame.quiet_regime.reindex(times).fillna(False).to_numpy(bool)
    if config.get('recent_only'):
        sources=frame.source_entity.reindex(times);latest=sources.loc[(sources!='')&sources.notna()].iloc[-1]
        mask &= sources.eq(latest).to_numpy()
    if mask.sum()<100:raise ValueError('Insufficient fitting rows for recipe')
    return x.loc[mask].reset_index(drop=True),y[mask],times[mask],methods[mask]

def fit(frame,data,cutoff,config,kind,heads=('middle',),progress=lambda s:None):
    x,y,times,methods=row_filter(frame,*data,config,kind)
    issued_x=x.copy()
    if config.get('weather_basis')=='historical':
        x=x.copy();x[usable_columns(kind,frame)]=frame.reindex(times)[usable_columns(kind,frame)].to_numpy();x['weather_run_age']=np.nan
    unique=times.unique().sort_values();split=unique[int(len(unique)*.8)]
    early=times+pd.Timedelta(hours=1)<=split;late=times>=split
    # Stopping truth: measured daylight, never interpolation/dark-derived targets.
    if kind=='solar':late &= (methods=='')&(x.full_dark.to_numpy()==0)
    if early.sum()<100 or late.sum()<24:raise ValueError('Insufficient chronological stopping data')
    weight=np.ones(len(y));half=config.get('half_life_days')
    if half:weight*=2**(-np.asarray((cutoff-times).total_seconds())/86400/half)
    if kind=='solar':
        counts=pd.Series(times).value_counts();weight/=np.array([counts[t] for t in times]);weight/=weight.mean()
    weight[methods=='interpolation']*=.25
    params={**BASE,**{k:v for k,v in config.items() if k in BASE},'subsample_freq':1,'random_state':42,'n_jobs':int(os.getenv('ENERGY_TRAIN_THREADS','2')),'verbosity':-1,'deterministic':True,'force_col_wise':True}
    models={};iterations={};curves={}
    ceiling=int(os.getenv('ENERGY_TRAIN_TREES','1000'))
    for key in heads:
        if kind=='solar' and key=='mean' and config.get('point_family') in ('physical','residual'):
            from .solar_estimators import SolarEnergyModel
            model=SolarEnergyModel(config['point_family']).fit(x,y,weight);models[key]=model;iterations[key]=model.iterations_;curves[key]={'basis':'Shared physical scale; bounded residual fit' if config['point_family']=='residual' else 'Shared learned irradiance scale'}
            progress(f'Solar {config["point_family"]}: {len(times.unique()):,} distinct target hours');continue
        alpha={'low':.1,'middle':.5,'high':.9}.get(key)
        objective='regression' if key=='mean' else 'quantile';metric='l2' if key=='mean' else 'quantile'
        cfg={**params,'objective':objective,'metric':metric,'n_estimators':ceiling}
        if alpha is not None:cfg['alpha']=alpha
        probe=lgb.LGBMRegressor(**cfg);probe.fit(x.loc[early],y[early],sample_weight=weight[early],eval_set=[(issued_x.loc[late],y[late])],eval_sample_weight=[weight[late]] if kind=='solar' else None,callbacks=[lgb.early_stopping(60,verbose=False)])
        if (probe.best_iteration_ or ceiling)>=ceiling*.95:
            cfg['n_estimators']=min(3000,ceiling*3)
            probe=lgb.LGBMRegressor(**cfg);probe.fit(x.loc[early],y[early],sample_weight=weight[early],eval_set=[(issued_x.loc[late],y[late])],eval_sample_weight=[weight[late]] if kind=='solar' else None,callbacks=[lgb.early_stopping(100,verbose=False)])
        stop=probe.best_iteration_ or cfg['n_estimators'];curves[key]={'ceiling':cfg['n_estimators'],'best_iteration':stop,'near_ceiling':stop>=cfg['n_estimators']*.95,'stopping_rows':int(late.sum()),'fitting_rows':int(early.sum()),'sampled_loss':probe.evals_result_['valid_0'][metric][::50]}
        model=lgb.LGBMRegressor(**{**cfg,'n_estimators':stop});model.fit(x,y,sample_weight=weight);models[key]=model;iterations[key]=stop
        progress(f'{kind} {key}: {len(x):,} rows; {stop} trees')
    return models,iterations,curves

def predict(models,c,kind,point_method=None,calibration=None,cutoff=0,correction=None,records=()):
    from .forecast_policy import model_columns
    x=c['x'];raw={k:m.predict(x.reindex(columns=model_columns(m,schema(kind)))) for k,m in models.items()}
    if 'low' not in raw:raw.update(low=raw['middle'],high=raw['middle'])
    dark=(~c['daylight']) if kind=='solar' else None
    if kind=='solar' and cutoff:dark=dark|(x.tilted_irradiance.to_numpy()<cutoff)
    q=ordered(raw,x.lead_hours,calibration,dark)
    method=point_method or {'head':'middle','baseline_weight':0}
    base=x.clock_profile.to_numpy() if kind=='load' else q['middle']
    head=raw.get('mean') if method['head']=='mean' and 'mean' in raw else ordered(raw,dark=dark)['middle']
    point=method.get('baseline_weight',0)*base+(1-method.get('baseline_weight',0))*head
    delta=recent_correction(records,c['origin'],x.lead_hours,correction,kind=='solar')
    point=point*delta if kind=='solar' else point+delta
    point=np.maximum(0,point)
    if dark is not None:point[dark]=0
    q['point']=point
    return q

def assessment_chart(records,limit=480):
    # Pick the latest issue for each target (shortest lead), then sample the full span.
    chosen={}
    for r in records:
        if r['time'] not in chosen or r['origin']>chosen[r['time']]['origin']:chosen[r['time']]=r
    rows=sorted(chosen.values(),key=lambda r:r['time'])
    indices=np.unique(np.linspace(0,len(rows)-1,min(limit,len(rows)),dtype=int)) if rows else []
    return [{'time':rows[i]['time'],'actual':rows[i]['actual'],'predicted':rows[i]['point']} for i in indices]


def summarize(cases,predictions,settings,kind):
    rows=[];energy={str(h):[] for h in (24,48,72)};energy['daily']=[];energy['overnight']=[];hourly=[];ys=[];qs={k:[] for k in ('low','middle','high')};points=[];records=[]
    for c,q in zip(cases,predictions):
        y=c['actual'];p=q['point'];m=c['valid'];leads=c['x'].lead_hours.to_numpy();local=c['hours'].tz_convert(settings.timezone)
        for j in np.flatnonzero(m):
            records.append({'origin':c['origin'].isoformat(),'time':c['hours'][j].isoformat(),'lead':float(leads[j]),'actual':float(y[j]),**{k:float(q[k][j]) for k in q},'scored':bool(c['daylight'][j]) if kind=='solar' else True,'season':('summer','autumn','winter','spring')[(local[j].month%12)//3], 'daylight':bool(c['daylight'][j]) if kind=='solar' else 7<=local[j].hour<19,'bright':bool(c['x'].get('tilted_irradiance',pd.Series(0,index=c['x'].index)).iloc[j]>250),'quiet':bool(c.get('quiet',np.zeros(len(y),bool))[j])})
        def add(label,until):
            w=window(c,until)
            if w is not None:
                active=w>0;error=float((p[active]-y[active])@w[active]);energy[label].append({'origin':c['origin'].isoformat(),'actual_kwh':float(y[active]@w[active]),'predicted_kwh':float(p[active]@w[active]),'error_kwh':error})
        for h in (24,48,72):add(str(h),h)
        # A calendar day is a full local-midnight issue, with fractional endpoint weights.
        if c['origin'].tz_convert(settings.timezone).hour==0:add('daily',c['daily_end'])
        if kind=='load' and c['origin'].tz_convert(settings.timezone).strftime('%H:%M') in {w.start for w in settings.export_windows} and c['charge_start'] is not None:add('overnight',c['charge_start'])
    quiet_days={pd.Timestamp(r['time']).tz_convert(settings.timezone).date() for r in records if r['quiet']}
    for r in records:r['return_day']=not r['quiet'] and (pd.Timestamp(r['time']).tz_convert(settings.timezone).date()-pd.Timedelta(days=1)) in quiet_days
    def em(items):
        e=np.array([r['error_kwh'] for r in items]);return {'samples':len(e),'windows':len(e),'energy_mae_kwh':float(np.mean(abs(e))),'energy_bias_kwh':float(e.mean()),'distinct_issue_days':len({r['origin'][:10] for r in items}),'energy_bias_pct':float(100*e.sum()/sum(r['actual_kwh'] for r in items)) if sum(r['actual_kwh'] for r in items)>0 else None} if len(e) else {'samples':0,'windows':0}
    selected=[r for r in records if r['scored']];actual=np.array([r['actual'] for r in records]);point=np.array([r['point'] for r in records])
    if not selected:raise ValueError('No measured scoring targets')
    diag=lambda rr:diagnostics([r['actual'] for r in rr],{k:[r[k] for r in rr] for k in ('low','middle','high')},[r['point'] for r in rr])
    groups={f'{left}–{right}h':diag([r for r in selected if left<=r['lead']<right]) for left,right in LEAD_GROUPS}
    hourly={'mae':float(np.mean(abs(point-actual))),'rmse':float(np.sqrt(np.mean((point-actual)**2))),'bias':float(np.mean(point-actual)),'samples':len(actual)}
    summary={'hourly':hourly,'horizons':{h:em(energy[h]) for h in ('24','48','72')},'daily':em(energy['daily']),'overnight':em(energy['overnight']),
        'regimes':{name:{'samples':len(rr),'mae':float(np.mean([abs(r['point']-r['actual']) for r in rr])) if rr else None,'bias_kw':float(np.mean([r['point']-r['actual'] for r in rr])) if rr else None} for name,rr in [('overall_valid',records),('normal_use',[r for r in records if not r['quiet']]),('quiet',[r for r in records if r['quiet']]),('return_day',[r for r in records if r['return_day']])]},
        'origins':len(cases),'distinct_issue_days':len({c['origin'].tz_convert(settings.timezone).date() for c in cases}),'unique_target_hours':len({r['time'] for r in records}),
        'quantile':diag(selected),'lead_groups':groups,'seasonal':{s:diag([r for r in selected if r['season']==s]) for s in sorted({r['season'] for r in selected})},
        'day_night':{name:diag([r for r in records if r['daylight']==day]) for name,day in [('day',True),('night',False)]},
        'brightness':diag([r for r in selected if r['bright']]),'weather_basis':'issued only','issued_weather_windows':len(cases),'historical_weather_windows':0,
        'energy_interval_basis':'Hourly marginal bounds are not cumulative percentiles; use complete sampled trajectories',
        'bias_direction':'predicted minus actual; positive means overprediction','metric_schema':'energy-diagnostics-v3',
        'note':'Retrospective development/diagnostic evidence. Overlapping horizons are not independent; prospective assessment pending.'}
    if kind=='solar':
        from .solar_training import power_metrics
        day=[r for r in records if r['scored']];day_metrics=power_metrics([r['actual'] for r in day],[r['point'] for r in day],np.ones(len(day)))
        summary['daytime']=day_metrics;summary['all_hours']=hourly
        summary['hourly']={**day_metrics,'bias':day_metrics['bias_kw'],'basis':'Measured daylight forecast/target pairs; repeated targets across issues'}
        summary['unique_daylight_hours']=len({r['time'] for r in day})
        summary['chart_basis']='Latest available issue per target; shortest lead; evenly sampled across the full validation period'
    return summary,energy,records

def normalized(summary,reference,kind):
    components=[];missing=[];bias=[]
    for label in (('overnight','24','48','72') if kind=='load' else ('daily','24','48','72')):
        a=summary.get(label,summary.get('horizons',{}).get(label,{}));b=reference.get(label,reference.get('horizons',{}).get(label,{}))
        if not a.get('samples') or not b.get('samples'):missing.append(label);continue
        # A fixed 0.05 kWh floor prevents a zero reference error from exploding a ratio.
        components.append(a['energy_mae_kwh']/max(.05,b['energy_mae_kwh']));bias.append(abs(a['energy_bias_kwh'])/max(.05,b['energy_mae_kwh']))
    return {'score':float(np.mean(components)) if components else float('inf'),'absolute_bias_score':float(np.mean(bias)) if bias else float('inf'),'supported_components':len(components),'missing_components':missing,'denominator_floor_kwh':.05}

def paired_bootstrap(new,old):
    common={r['origin']:r for r in old};pairs=[(r['origin'][:10],abs(r['error_kwh'])-abs(common[r['origin']]['error_kwh'])) for r in new if r['origin'] in common]
    days=sorted({d for d,e in pairs})
    if not days:return {'status':'No common windows'}
    delta=np.array([np.mean([e for d,e in pairs if d==day]) for day in days]);block=7 if len(days)>=21 else 3
    rng=np.random.default_rng(42);draws=[]
    for _ in range(500):
        indices=[]
        while len(indices)<len(days):
            at=int(rng.integers(max(1,len(days)-block+1)));indices.extend(range(at,min(at+block,len(days))))
        draws.append(float(delta[indices[:len(days)]].mean()))
    return {'paired_issue_days':len(days),'block_days':block,'delta_mae_kwh':float(delta.mean()),'interval_95_kwh':np.quantile(draws,[.025,.975]).tolist(),'interpretation':'Large uncertainty with few blocks; overlapping pairs are not independent','nonoverlapping_delta_kwh':float(delta[::3].mean())}

def _score(models,cc,settings,kind,**kw):
    predictions=[predict(models,c,kind,**kw) for c in cc]
    return summarize(cc,predictions,settings,kind)

def run(settings,summary,d,progress):
    from .models import signature
    from .load_training import selection_folds,fold_frame,LoadPredictor
    from .uncertainty import fit_dependence
    started=time.perf_counter();kind=summary['kind'];name=summary['id'];frame=d['frame'];start=d['start'];end=d['train_end'];bank=next((b for b in solar_banks(settings) if b.id==summary.get('bank_id')),None)
    if kind=='solar':
        if bank.history_start:
            start=max(start,pd.Timestamp(bank.history_start,tz=settings.timezone).tz_convert('UTC'))
            if start>=end:raise ValueError('Training dates precede the current panel layout. Choose dates after the production-history start.')
        frame=frame.copy()
        if bank.id=='__combined' and not any(c.startswith('bank_irradiance_') for c in frame.columns):
            from .sources import solar_weather
            progress('Updating separate bank sunlight inputs for the prepared history')
            weather=solar_weather(settings,bank,start,d['end'],historical=True)
            for c in weather.columns:
                if c.startswith('bank_irradiance_'):frame[c]=weather[c].reindex(frame.index)
        # Rebuild shared geometry/darkness features; newly added issue fields stay unknown here.
        frame['full_dark']=dark_hours(settings,frame.index).astype(float);frame['lead_hours']=np.nan;frame['weather_run_age']=np.nan
    folds=selection_folds(frame,start,end,settings)
    if kind=='solar' and len(folds)==1:
        boundary=folds[0]['start'];left=max(start+pd.Timedelta(days=7),boundary-pd.Timedelta(days=4)).ceil('h')
        if boundary-left>=pd.Timedelta(days=3):folds.insert(0,{'start':left,'end':boundary,'season':folds[0]['season'],'reason':'Earlier disjoint solar block for uncertainty calibration; expanding prefix'})
    # Include a latest regime-specific expanding fold when hardware changed recently.
    recent=end-pd.Timedelta(days=7)
    if recent>start+pd.Timedelta(days=14) and all(f['end']<=recent for f in folds):folds.append({'start':recent.ceil('h'),'end':end,'season':'recent','reason':'Recent hardware/regime fold; earlier prefix only; short-history evidence'})
    folds=sorted(folds,key=lambda f:f['start']);root=DATA/(name+'-study');root.mkdir(exist_ok=True)
    identity=hashlib.sha256((DATA/(name+'.pkl')).read_bytes()).hexdigest();code_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(__import__('pathlib').Path(__file__).parent.glob('*.py')))).hexdigest()
    manifest={'schema':FEATURE_SCHEMA,'dataset_hash':identity,'code_hash':code_hash,'frozen_at':pd.Timestamp.now(tz='UTC').isoformat(),'training_start':str(start),'training_end':str(end),'assessment_start':str(d['val_start']),'assessment_end':str(d['end']),
        'status':'Known retrospective diagnostic; no fresh accuracy claim','folds':[{k:str(v) for k,v in f.items()} for f in folds],
        'threshold_gain_pct':2,'material_regression_pct':10,'material_regression_kwh':.25,'normalization_floor_kwh':.05,'settlement_hours':SETTLEMENT_HOURS,'interval_policy':INTERVAL_POLICY,'archive_publication_assumption_hours':12,
        'lead_endpoint_convention':'73 output rows; lead groups [0,24), [24,48), [48,73); energy windows integrate [origin,origin+h)',
        'prospective':{'status':'pending','minimum_days':28,'maximum_days_if_gaps':56},'installation_start':bank.history_start if bank else None,'coverage':{label:split_coverage(frame,left,right,settings,True) for label,left,right in [('training',start,end),('assessment',d['val_start'],d['end'])]}}
    write_json(name+'-comparison-manifest',manifest)
    prepared=[]
    for f in folds:
        left=f['start'];scoped=fold_frame(frame,left,settings) if kind=='load' else frame
        key=hashlib.sha256((identity+code_hash+str(left)+str(f['end'])+os.getenv('ENERGY_FIT_ORIGINS','240')).encode()).hexdigest()[:20];path=root/(key+'.joblib')
        if path.exists():entry=joblib.load(path)
        else:
            cc,skipped=cases(scoped,left,f['end'],settings,kind,bank,progress=progress)
            if not cc:continue
            try:data=fitting_rows(scoped,start,left,settings,kind,bank,progress=progress)
            except ValueError:continue
            entry={'fold':f,'data':data,'cases':cc,'skipped':skipped};joblib.dump(entry,path)
        prepared.append((path,scoped))
    if not prepared:raise ValueError('No issued-weather development folds; collect valid archive-supported issues first')
    manifest['issued_folds']=[{'fold':{k:str(v) for k,v in joblib.load(p)['fold'].items()},'origins':[c['origin'].isoformat() for c in joblib.load(p)['cases']],'preparation':joblib.load(p)['data'][0].attrs.get('preparation',{}),'skipped':joblib.load(p)['skipped']} for p,_ in prepared]
    write_json(name+'-comparison-manifest',manifest)
    comparisons=[];cache={}
    def compare(config,heads=('middle','mean')):
        fold_results=[];all_records=[]
        for path,scoped in prepared:
            entry=joblib.load(path);f=entry['fold'];cc=entry['cases'];key=(str(path),str(sorted(config.items())),heads)
            if key not in cache:cache[key]=fit(scoped,entry['data'],f['start'],config,kind,heads,progress)
            models,iterations,curves=cache[key]
            if kind=='load':ref=[{'low':c['x'].clock_profile.to_numpy(),'middle':c['x'].clock_profile.to_numpy(),'high':c['x'].clock_profile.to_numpy(),'point':c['x'].clock_profile.to_numpy()} for c in cc]
            else:
                # Fixed historical-weather reference fitted on the same archive-supported target pairs.
                refkey=(str(path),'reference')
                if refkey not in cache:
                    cache[refkey]=fit(scoped,entry['data'],f['start'],{**BASE,'recipe':'measured_only','weather_basis':'historical'},kind,('middle',),progress)[0]
                ref=[predict(cache[refkey],c,kind) for c in cc]
            refsummary,_,_=summarize(cc,ref,settings,kind)
            choices=[]
            methods=[{'head':h,'baseline_weight':w} for h in ('middle','mean') for w in (0,.5,1)] if kind=='load' else [{'head':h,'baseline_weight':0} for h in ('middle','mean')]
            for method in methods:
                result,energy,records=_score(models,cc,settings,kind,point_method=method)
                choices.append({'method':method,**normalized(result,refsummary,kind),'summary':result})
            chosen=min(choices,key=lambda r:(round(r['score']/.005),r['absolute_bias_score'], -r['method']['baseline_weight']))
            result,energy,records=_score(models,cc,settings,kind,point_method=chosen['method']);all_records.extend(records)
            fold_results.append({'fold':{k:str(v) for k,v in f.items()},'score':chosen['score'],'absolute_bias_score':chosen['absolute_bias_score'],'summary':result,'reference':refsummary,'point_choices':choices,'iterations':iterations,'curves':curves})
        # One fixed point method must win across folds; a fold-specific oracle is optimistic.
        method_scores={}
        for f in fold_results:
            for r in f['point_choices']:method_scores.setdefault(str(r['method']),{'method':r['method'],'scores':[],'bias':[]})['scores'].append(r['score']);method_scores[str(r['method'])]['bias'].append(r['absolute_bias_score'])
        winning=min(method_scores.values(),key=lambda r:(round(float(np.mean(r['scores']))/.005),float(np.mean(r['bias'])),-r['method']['baseline_weight']))
        for f in fold_results:
            chosen=next(r for r in f['point_choices'] if r['method']==winning['method'])
            f.update(score=chosen['score'],summary=chosen['summary'],absolute_bias_score=chosen['absolute_bias_score'])
        result={'configuration':config,'score':float(np.mean(winning['scores'])),'point_method':winning['method'],'folds':fold_results}
        comparisons.append(result);progress(f'{name}: internal normalized energy score {result["score"]:.4f}; {config}; point {result["point_method"]}');return result
    initial=compare({**BASE,'half_life_days':None,'recipe':'measured_only','weather_basis':'issued'})
    historical_initial=compare({**BASE,'half_life_days':None,'recipe':'measured_only','weather_basis':'historical'})
    weather_base=min((initial,historical_initial),key=lambda r:r['score'])['configuration']
    # Isolate recency at fixed tree capacity; 30d is labelled a short-history experiment.
    if kind=='load':
        for half in (365,180,90,30):compare({**weather_base,'half_life_days':half})
    chosen=min(comparisons,key=lambda r:r['score']);base=chosen['configuration']
    shapes=SHAPES if kind=='load' or end-start>=pd.Timedelta(days=90) else [dict(num_leaves=7,max_depth=3,min_child_samples=30),dict(num_leaves=15,max_depth=4,min_child_samples=50)]
    for shape in shapes:compare({**base,**shape})
    if kind=='solar':
        for family in ('physical','residual'):compare({**base,'point_family':family})
    # Recipes are tested at a fixed chosen capacity/decay.
    selected=min(comparisons,key=lambda r:r['score']);base=selected['configuration']
    for recipe in (('dark_zero','weighted_interpolation') if kind=='solar' else ('exclude_quiet','recent_only')):
        try:compare({**base,**({'recipe':recipe} if kind=='solar' else {recipe:True})})
        except ValueError as exc:comparisons.append({'configuration':{**base,recipe:True},'unavailable':str(exc)})
    supported=[r for r in comparisons if 'score' in r];finalists=sorted(supported,key=lambda r:r['score'])[:2]
    # Fit all heads for finalists; include tail evidence in final selection.
    finalists=[compare(r['configuration'],('low','middle','high','mean')) for r in finalists]
    best_energy=min(r['score'] for r in finalists);viable=[r for r in finalists if r['score']<=best_energy*1.02]
    selected=min(viable,key=lambda r:(float(np.mean([f['summary']['quantile']['mean_pinball'] for f in r['folds']])),r['configuration']['num_leaves']))
    config=selected['configuration'];all_choices={}
    for f in selected['folds']:
        for r in f['point_choices']:all_choices.setdefault(str(r['method']),{'method':r['method'],'scores':[]})['scores'].append(r['score'])
    point_method=min(all_choices.values(),key=lambda r:(round(np.mean(r['scores'])/.005),-r['method']['baseline_weight']))['method']
    # Earlier OOF records fit adjustments; later folds select them. No final outcomes enter this step.
    earlier=[];post_scores=[];calibration_options=('identity','pooled','lead_group')
    variants=[(cut,cor,cal) for cut in ((0,5) if kind=='solar' else (0,)) for cor in CORRECTIONS for cal in calibration_options]
    for path,scoped in prepared:
        entry=joblib.load(path);cc=entry['cases'];f=entry['fold'];heads=('low','middle','high','mean');models=cache[(str(path),str(sorted(config.items())),heads)][0]
        _,_,raw_records=_score(models,cc,settings,kind,point_method=point_method)
        if earlier:
            for cut,cor,cal in variants:
                offsets={} if cal=='identity' else fit_calibration(earlier,cal=='lead_group',solar=kind=='solar')
                # Sequential replay gives corrections only already settled OOF outcomes.
                history=[];pp=[]
                for c in cc:
                    q=predict(models,c,kind,point_method,offsets,cut,cor,history);pp.append(q)
                    _,_,fresh=summarize([c],[predict(models,c,kind,point_method)],settings,kind);history.extend(fresh)
                result,_,_=summarize(cc,pp,settings,kind)
                ref=next(v['reference'] for v in selected['folds'] if v['fold']['start']==str(f['start']))
                post_scores.append({'cutoff':cut,'correction':cor,'calibration':cal,'fold_start':str(f['start']),**normalized(result,ref,kind),'pinball':result['quantile']['mean_pinball'],'width':result['quantile']['width_kw'],'lower_tail':result['quantile']['lower_tail'],'upper_tail':result['quantile']['upper_tail'],'tail_error':abs(result['quantile']['lower_tail']-.1)+abs(result['quantile']['upper_tail']-.1)})
        earlier.extend(raw_records)
    post=[]
    for cut,cor,cal in variants:
        part=[r for r in post_scores if r['cutoff']==cut and r['correction']==cor and r['calibration']==cal]
        if part:post.append({'cutoff':cut,'correction':cor,'calibration':cal,'score':float(np.mean([r['score'] for r in part])),'pinball':float(np.mean([r['pinball'] for r in part])),'width':float(np.mean([r['width'] for r in part])),'tail_error':float(np.mean([r['tail_error'] for r in part]))})
    policy=min(post,key=lambda r:(round(r['score']/.005),round(r.get('tail_error',0)/.02) if kind=='solar' else 0,r['pinball'],r['correction'] is not None,r['calibration']!='identity')) if post else {'cutoff':0,'correction':None,'calibration':'identity'}
    if kind=='solar' and not post and fit_calibration(earlier,solar=True):
        policy={**policy,'calibration':'pooled'}
    calibration={} if policy['calibration']=='identity' else fit_calibration(earlier,policy['calibration']=='lead_group',solar=kind=='solar')
    # Weather basis is selected on identical issued-origin/target pairs. Older
    # history is a separate recent-fold diagnostic, with its limits recorded.
    last_path,last_scoped=prepared[-1];entry=joblib.load(last_path)
    alternative_basis='historical' if config.get('weather_basis')=='issued' else 'issued'
    alternative,_,_=fit(last_scoped,entry['data'],entry['fold']['start'],{**config,'weather_basis':alternative_basis},kind,('middle','mean'),progress)
    weather_ablation={'alternative_basis':alternative_basis,'summary':_score(alternative,entry['cases'],settings,kind,point_method=point_method)[0],
                      'common_period_initial_scores':{'issued':initial['score'],'historical':historical_initial['score']}}
    older_history={}
    try:
        old_data=fitting_rows(last_scoped,start,entry['fold']['start'],settings,kind,bank,basis='historical_full',progress=progress)
        old_models,_,_=fit(last_scoped,old_data,entry['fold']['start'],{**config,'weather_basis':'historical'},kind,('middle','mean'),progress)
        older_history={'summary':_score(old_models,entry['cases'],settings,kind,point_method=point_method)[0],
                       'preparation':old_data[0].attrs.get('preparation',{}),'status':'Separate older-history recent-fold diagnostic; not promoted from one-fold evidence'}
    except ValueError as exc:older_history={'unavailable':str(exc)}
    progress(f'{name}: fitting frozen final recipe')
    final_data=fitting_rows(frame,start,end,settings,kind,bank,progress=progress)
    models,iterations,curves=fit(frame,final_data,end,config,kind,('low','middle','high','mean'),progress)
    dependence_records=[]
    for origin in sorted({r['origin'] for r in earlier}):
        rr=sorted([r for r in earlier if r['origin']==origin],key=lambda r:r['time'])
        q=ordered({k:[r[k] for r in rr] for k in ('low','middle','high')},[r['lead'] for r in rr],calibration)
        dependence_records.extend({**r,**{k:float(q[k][j]) for k in q}} for j,r in enumerate(rr))
    dependence=fit_dependence(dependence_records,solar=kind=='solar')
    artifact={'schema':FEATURE_SCHEMA,'pipeline':LOAD_PIPELINE,'quality_policy':QUALITY_POLICY,'models':{k:v for k,v in models.items() if k!='mean'},'mean_model':models.get('mean'),'timezone':settings.timezone,'features':schema(kind,frame),'point_method':point_method,'calibration':calibration,'correction':policy['correction'],'irradiance_cutoff':policy['cutoff'],'sampling_rho':dependence['rho'],'sampling_dependence':dependence}
    # This assessment is inspected once, after recipe/postprocessing are frozen.
    final_cases,skipped=cases(frame,d['val_start'],d['end'],settings,kind,bank,progress=progress)
    pp=[];history=[]
    for c in final_cases:
        pp.append(predict(models,c,kind,point_method,calibration,policy['cutoff'],policy['correction'],history))
        _,_,fresh=_score(models,[c],settings,kind,point_method=point_method);history.extend(fresh)
    result,energy,records=summarize(final_cases,pp,settings,kind)
    legacy_path=DATA/(name+'.joblib');comparison=None;old_energy=None
    comparison_result=result
    comparison_note=''
    if legacy_path.exists():
        old=joblib.load(legacy_path);old_predictions=[]
        old_meta=read_json('model-'+name,{})
        old_cutoff=old_meta.get('training_end')
        paired=[(c,q) for c,q in zip(final_cases,pp) if kind!='solar' or old_cutoff and c['origin']>=pd.Timestamp(old_cutoff)]
        if not paired:comparison_note='No paired comparison: the active model has no known earlier fitting cutoff for these validation issues.'
        for c,_ in paired:
            if kind=='load':
                from .load_training import legacy_feature_rows
                old_x=legacy_feature_rows(frame,c,settings) if old.get('pipeline')=='lightgbm-quantile-direct-v2' else c['x']
                pred=LoadPredictor(old).predict(old_x);pred['point']=pred.get('point',pred['middle'])
            else:
                from .solar_training import solar_predict
                pred=solar_predict(old,c['x']);pred['point']=pred.get('point',pred['middle'])
            old_predictions.append(pred)
        if paired:
            paired_cases=[c for c,_ in paired]
            old_summary,old_energy,old_records=summarize(paired_cases,old_predictions,settings,kind)
            comparison_result,_,_=summarize(paired_cases,[q for _,q in paired],settings,kind)
            comparison={'metrics':old_summary['hourly'],'chart':assessment_chart(old_records),('load_validation' if kind=='load' else 'forecast_validation'):old_summary,'candidate_metrics':comparison_result['hourly'],'candidate_validation':comparison_result,'comparison_basis':'Identical issues after both models’ fitting cutoffs; known retrospective benchmark','issue_start':paired_cases[0]['origin'].isoformat()}
    baseline=None
    if kind=='load':
        baseline=summarize(final_cases,[{'low':c['x'].clock_profile.to_numpy(),'middle':c['x'].clock_profile.to_numpy(),'high':c['x'].clock_profile.to_numpy(),'point':c['x'].clock_profile.to_numpy()} for c in final_cases],settings,kind)[0]
    diagnostic=result['quantile'];diagnostic.update(pinball_low=diagnostic['low']['pinball'],pinball_mid=diagnostic['middle']['pinball'],pinball_high=diagnostic['high']['pinball'],width=diagnostic['width_kw'],coverage_daylight=diagnostic['coverage'],width_daylight_kw=diagnostic['width_kw'])
    # Existing UI/clients receive versioned metrics with an explicit bias direction.
    result['weather_model']=settings.weather_model;result['publication_buffer_hours']=12;result['runs']=[];result['skipped']=skipped;result['daily_chart']=energy['daily']
    chart=assessment_chart(records)
    metadata={'trained_at':pd.Timestamp.now(tz='UTC').isoformat(),'dataset_id':name,'kind':kind,'bank_id':summary.get('bank_id'),'model':'Issued-weather LightGBM energy / quantiles','pipeline':LOAD_PIPELINE,'solar_pipeline':__import__('app.datasets',fromlist=['SOLAR_PIPELINE']).SOLAR_PIPELINE if kind=='solar' else None,'schema':FEATURE_SCHEMA,'quality_policy':QUALITY_POLICY,'config_signature':signature(settings,kind,summary.get('bank_id')),'dataset_created_at':summary['created_at'],'training_rows':len(final_data[1]),'unique_target_hours':len(final_data[2].unique()),'validation_rows':result['hourly']['samples'],'metrics':result['hourly'],'chart':chart,('load_validation' if kind=='load' else 'forecast_validation'):result,'comparison':comparison,'history_baseline':baseline,'training_start':str(start),'training_end':str(end),'validation_start':str(d['val_start']),'validation_end':str(d['end']),'point_method':point_method,'calibration':calibration,'correction':policy['correction'],'irradiance_cutoff':policy['cutoff'],'sampling_rho':artifact['sampling_rho'],'sampling_dependence':dependence,'weather_basis':('Historical relationship on archive-supported target pairs; issued-weather selection' if config.get('weather_basis')=='historical' else 'Issued archived runs; 12h conservative availability; live stitched run age unknown'),'parameters':config,'iterations':iterations,'learning_curves':curves,'configuration_search':comparisons,'postprocessing_search':post,'postprocessing_folds':post_scores,'weather_ablation':weather_ablation,'older_history_ablation':older_history,'split_manifest':manifest,'selection_windows':manifest['folds'],'feature_importance':dict(zip(schema(kind,frame),models['middle'].feature_importances_.tolist())),'timings_seconds':{'total':time.perf_counter()-started},'assessment_classification':'promising but insufficient evidence' if comparison and result['hourly']['mae']<comparison['metrics']['mae']*.98 else 'no material change' if not comparison or result['hourly']['mae']<=comparison['metrics']['mae']*1.02 else 'regression','paired_bootstrap':{h:paired_bootstrap(energy[h],old_energy[h]) for h in ('daily','overnight','24','48','72')} if old_energy else {},'fitting_preparation':final_data[0].attrs.get('preparation',{})}
    metadata['comparison_note']=comparison_note
    if kind=='load':
        eligible=frame.loc[training_mask(frame.index,start,end)&frame.train_eligible];profile=eligible.target.groupby(eligible.index.tz_convert(settings.timezone).hour).mean();metadata['hour_profile']={str(k):float(v) for k,v in profile.items()}
    if kind=='solar':
        metadata['uncertainty_calibration']={'status':'residual-adjusted' if calibration else 'identity-selected' if post and fit_calibration(earlier,solar=True) else 'uncalibrated','basis':'Earlier chronological development predictions only; final assessment excluded','independently_selected':bool(post and fit_calibration(earlier,solar=True)),'development_folds':len(prepared),'unique_calibration_hours':len({r['time'] for r in earlier if r['scored']}),'note':'Short-history calibration is experimental; inspect final daylight coverage and P90 exceedances by lead.' if end-start<pd.Timedelta(days=90) else 'Inspect validation coverage; future coverage is not established.'}
        metadata['history_warning']='Short training history: use more measured history from the same panel layout when available.' if end-start<pd.Timedelta(days=90) else ''
    metadata['target_counts']={m:int(np.sum(final_data[3]==m)) for m in ('','dark_zero','interpolation')}
    metadata['stopping_truth']='Measured targets only; measured daylight for solar; chronological target-interval purge'
    metadata['scoring_truth']='Measured daylight plus deterministic full-dark energy accounting; interpolation never scored'
    metadata['selection_score']=selected['score']
    metadata['coverage_goals']={'minimum_measured_pct':90,'minimum_72h_issue_days':10,'minimum_local_solar_days':21,'status':'See frozen comparison manifest; retrospective assessment shorter than requested fresh evidence'}
    if comparison:
        old_summary=comparison['load_validation' if kind=='load' else 'forecast_validation']
        change=normalized(comparison_result,old_summary,kind)
        material=[]
        for h in ('24','48','72'):
            a=comparison_result['horizons'][h];b=old_summary['horizons'][h]
            if a.get('samples') and b.get('samples') and a['energy_mae_kwh']>b['energy_mae_kwh']*1.1+.25:material.append(h)
        metadata['assessment_primary_ratio']=change
        metadata['material_regressions']=material
        metadata['assessment_classification']='regression' if change['score']>1.02 or material else 'promising but insufficient evidence' if change['score']<.98 else 'no material change'
    audit_target_counts(metadata,frame,final_data,config,kind,records)
    artifact['metadata']=metadata
    joblib.dump(artifact,DATA/(name+'-candidate.joblib'));write_json(name+'-assessment-details',{'energy':energy,'records':records,'old_records':old_records if comparison else [],'old_energy':old_energy,'manifest':manifest,'model_identities':{'candidate':metadata['trained_at'],'existing':read_json('model-'+name,{}).get('trained_at')},'sampling_metadata':{'candidate':{k:metadata.get(k) for k in ('sampling_rho','sampling_dependence')},'existing':{k:read_json('model-'+name,{}).get(k) for k in ('sampling_rho','sampling_dependence')}}})
    return metadata


def audit_target_counts(metadata,frame,data,config,kind,records=()):
    """Separate prepared pairs from the rows actually fitted/stopped/scored."""
    x,y,times,methods=row_filter(frame,*data,config,kind)
    def count(mm,xx):
        result={'rows':len(mm),'measured':int(np.sum(mm=='')),'dark_derived':int(np.sum(mm=='dark_zero')),'interpolated':int(np.sum(mm=='interpolation'))}
        if kind=='solar':result['measured_daylight']=int(np.sum((mm=='')&(xx.full_dark.to_numpy()==0)))
        return result
    unique=times.unique().sort_values();split=unique[int(len(unique)*.8)];stop=times>=split
    if kind=='solar':stop &= (methods=='')&(x.full_dark.to_numpy()==0)
    metadata['prepared_training_pairs']=len(data[1]);metadata['training_rows']=len(y);metadata['unique_target_hours']=len(times.unique())
    metadata['effective_training_target_start']=times.min().isoformat();metadata['effective_training_target_end']=(times.max()+pd.Timedelta(hours=1)).isoformat()
    metadata['target_counts_by_stage']={'fitting':count(methods,x),'stopping':count(methods[stop],x.loc[stop])}
    metadata['target_counts']={m:int(np.sum(methods==m)) for m in ('','dark_zero','interpolation')}
    if records:
        at=pd.DatetimeIndex([r['time'] for r in records]);mm=frame.derived_method.reindex(at).fillna('').to_numpy()
        metadata['target_counts_by_stage']['assessment']={'rows':len(records),'measured':int(np.sum(mm=='')),'measured_daylight':sum(r['scored'] for r in records) if kind=='solar' else None,'dark_derived':int(np.sum(mm=='dark_zero')),'interpolated':int(np.sum(mm=='interpolation'))}
    finalists=[r for r in metadata['configuration_search'] if r.get('configuration')==config and any('low' in f.get('iterations',{}) for f in r.get('folds',[]))]
    if finalists:
        folds=finalists[-1]['folds'];metadata['target_counts_by_stage']['calibration']={'measured_scoring_pairs':sum(f['summary']['quantile']['samples'] for f in folds),'unique_target_hours_sum_of_disjoint_folds':sum(f['summary']['unique_target_hours'] for f in folds),'basis':'Earlier chronological OOF only; measured daylight for solar'}
    return metadata

"""Bounded local training and shared issued-weather validation."""
import os,time
import joblib,lightgbm as lgb,numpy as np,pandas as pd
from .config import DATA,read_json,write_json
from .forecast_policy import (FEATURE_SCHEMA,archived_run,ordered,diagnostics,fit_calibration,
                             recent_correction,LEAD_GROUPS,SETTLEMENT_HOURS)
from .datasets import FEATURES,SOLAR_FEATURES,LEGACY_SOLAR_FEATURES,features,bank_features
from .quality import QUALITY_POLICY,LOAD_PIPELINE,dark_hours
from .load_splits import training_mask

BASE=dict(num_leaves=31,max_depth=5,min_child_samples=100,learning_rate=.05,reg_lambda=2.,reg_alpha=.1,subsample=.9,colsample_bytree=.9)

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


def make_case(frame,origin,end,settings,kind,bank=None,basis='issued',calendar=None,cached_only=False):
    from .sources import consumption_weather,solar_weather,Calendar
    from .load_training import feature_rows,next_charge_start
    index=pd.date_range(pd.Timestamp(origin).floor('h'),min(pd.Timestamp(origin).floor('h')+pd.Timedelta(hours=73),pd.Timestamp(end).ceil('h')),freq='h',inclusive='left')
    if len(index)<23:return None
    # Reject incomplete measurements before spending time fetching a weather run.
    actual=frame.target.reindex(index).to_numpy(float)
    dark=frame.full_dark.reindex(index).fillna(False).to_numpy(bool) if kind=='solar' and 'full_dark' in frame else dark_hours(settings,index) if kind=='solar' else np.zeros(len(index),bool)
    eligible=frame.validation_eligible.reindex(index).fillna(False).to_numpy(bool)
    derived=frame.get('derived_method',pd.Series('',index=frame.index)).reindex(index).eq('dark_zero').to_numpy()
    valid=eligible&np.isfinite(actual)
    if kind=='solar':valid |= dark&derived
    local=origin.tz_convert(settings.timezone);daily_end=(local.normalize()+pd.DateOffset(days=1)).tz_convert('UTC')
    preview={'origin':origin,'hours':index,'end':end,'valid':valid}
    midnight=local.strftime('%H:%M')=='00:00'
    if window(preview,24) is None and not (kind=='solar' and midnight and window(preview,daily_end) is not None):return None
    if kind=='load':
        context=frame.loc[(frame.index<origin-pd.Timedelta(hours=SETTLEMENT_HOURS))&(frame.index>=origin-pd.Timedelta(hours=169))]
        if context.get('context_target',context.target).count()<126:return None
    weather=None
    if basis=='issued':
        run=archived_run(origin)
        weather=(consumption_weather(settings,run,run+pd.Timedelta(days=7),issued_at=origin,cached_only=cached_only) if kind=='load' else solar_weather(settings,bank,run,run+pd.Timedelta(days=7),issued_at=origin,cached_only=cached_only))
        weather=weather.reindex(index)
    if kind=='load':
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
        x['full_dark']=dark.astype(float)
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

def cases(frame,start,end,settings,kind,bank=None,basis='issued',progress=lambda s:None,cached_only=False,issue_times=None):
    from types import SimpleNamespace
    calendar=SimpleNamespace(features=lambda index:frame.reindex(index)[['public_holiday','school_holiday','school_calendar_known']]) if kind=='load' else None
    result=[];skipped=[]
    issue_times=origins(start,end,settings) if issue_times is None else issue_times
    weather_failures=0
    for i,origin in enumerate(issue_times):
        progress(f'{kind}: checking validation issue {i+1}/{len(issue_times)}')
        try:
            c=make_case(frame,origin,end,settings,kind,bank,basis,calendar,cached_only)
            if c is not None:result.append(c)
            else:skipped.append({'origin':origin.isoformat(),'reason':'Incomplete context, weather or targets'})
        except Exception as exc:
            skipped.append({'origin':origin.isoformat(),'reason':str(exc)})
            # A service outage should not turn preparation into hours of retries.
            import httpx
            if isinstance(exc,(httpx.HTTPError,TimeoutError)):
                weather_failures+=1
                if weather_failures>=3:
                    raise ValueError('Archived weather is unavailable. Try preparing data again later.') from exc
    progress(f'{kind}: {len(result)} {basis} cases; {len(skipped)} unavailable')
    return result,skipped

def window(c,h):
    until=c['origin']+pd.Timedelta(hours=h) if isinstance(h,(int,float)) else pd.Timestamp(h)
    if until>c['end'] or until<=c['origin']:return None
    t=c['hours'];weights=np.maximum(0,(np.minimum((t+pd.Timedelta(hours=1)).as_unit('ns').asi8,until.value)-np.maximum(t.as_unit('ns').asi8,c['origin'].value))/3.6e12)
    m=weights>0
    if not m.any() or not c['valid'][m].all():return None
    return weights

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
    weight=method.get('baseline_weight',0)
    if weight:base=np.where(np.isfinite(base),base,head)
    point=head.copy() if weight==0 else weight*base+(1-weight)*head
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


def fitting_rows(frame,start,end,settings,kind,progress):
    """Use prepared history only. Bound load origin expansion before allocating it."""
    from .load_training import feature_rows
    if kind=='solar':
        times=frame.index[training_mask(frame.index,start,end)&frame.train_eligible&frame.target.notna()&~frame.full_dark.astype(bool)]
        x=frame.reindex(times)[schema(kind,frame)].reset_index(drop=True).copy()
        # Historical weather has no forecast initialization or lead time.
        x[['lead_hours','weather_run_age']]=np.nan
    else:
        limit=max(24,int(os.getenv('ENERGY_FIT_ORIGINS','240')))
        anchors=pd.date_range(pd.Timestamp(start).ceil('D'),end-pd.Timedelta(hours=73),freq='4D')[-(limit-14):]
        recent=pd.date_range(max(pd.Timestamp(start).ceil('D'),end.floor('D')-pd.Timedelta(days=17)),end-pd.Timedelta(hours=73),freq='D')
        days=anchors.union(recent).sort_values()
        origins=pd.DatetimeIndex([(day+pd.Timedelta(hours=(day.dayofyear%4)*6)).floor('h') for day in days])
        origins=origins[origins+pd.Timedelta(hours=73)<=end]
        issues=origins.repeat(73);times=issues+pd.to_timedelta(np.tile(np.arange(73),len(origins)),unit='h')
        target=frame.reindex(times)
        eligible=training_mask(times,start,end)&target.train_eligible.fillna(False).to_numpy(bool)&target.target.notna().to_numpy()
        issues=issues[eligible];times=times[eligible]
        if not len(times):raise ValueError('No eligible consumption training targets')
        progress(f'Building consumption features for {len(origins)} sampled issues')
        x=feature_rows(frame,issues,times,settings.timezone)
    good=_finite(x,kind).to_numpy(copy=True)
    x=x.loc[good].reset_index(drop=True).astype(np.float32);times=times[good]
    target=frame.reindex(times)
    return x,target.target.to_numpy(float),times,target.derived_method.fillna('').to_numpy()


def fit_heads(x,y,times,methods,kind,development,settings,progress):
    """At most two chronological probes per head, then one complete refit."""
    from .fitting import EnergyValidation
    weight=np.ones(len(y));weight[methods=='interpolation']*=.25;weight/=weight.mean()
    ceiling=min(4000,max(1,int(os.getenv('ENERGY_TRAIN_TREES','2000'))))
    params={**BASE,'n_estimators':ceiling,'n_jobs':max(1,int(os.getenv('ENERGY_TRAIN_THREADS','2'))),
            'subsample_freq':1,'random_state':42,'verbosity':-1,'deterministic':True,'force_col_wise':True}
    heads=[('middle',.5),('low',.1),('high',.9),('mean',None)]
    probe_sets=[];validations=[];losses={k:[] for k,_ in heads};fold_curves=[]
    folds=development.get('folds',[development])
    for i,fold in enumerate(folds):
        validation=EnergyValidation(fold['cases'],kind,settings);validations.append(validation)
        early=times+pd.Timedelta(hours=1)<=fold['start']
        if early.sum()<48:raise ValueError('Not enough fitting rows before the stopping period. Prepare more training history.')
        vx=validation.x.reindex(columns=x.columns).astype(np.float32);probes={};details={}
        child=min(BASE['min_child_samples'],max(10,int(early.sum())//20))
        training=lgb.Dataset(x.loc[early],label=y[early],weight=weight[early])
        stopping=lgb.Dataset(vx,label=validation.y,reference=training)
        for key,alpha in heads:
            point=key in ('middle','mean')
            progress(f'{kind} development {i+1}/{len(folds)}: fitting {key} (maximum {ceiling} trees)')
            cfg={**params,'min_child_samples':child,'objective':'regression' if alpha is None else 'quantile','metric':'None' if point else 'quantile'}
            if alpha is not None:cfg['alpha']=alpha
            native={k:v for k,v in cfg.items() if k not in ('n_estimators','n_jobs','random_state')}
            native.update(num_threads=params['n_jobs'],seed=42);history={}
            probe=lgb.train(native,training,num_boost_round=ceiling,valid_sets=[stopping],valid_names=['valid_0'],
                            feval=(lambda prediction,data:validation.metric(data.get_label(),prediction)) if point else None,
                            keep_training_booster=True,
                            callbacks=[lgb.record_evaluation(history),lgb.early_stopping(150,first_metric_only=True,verbose=False)])
            metric='energy_and_hourly' if point else 'quantile'
            loss=np.array(history['valid_0'][metric])
            losses[key].append(loss/max(.1,float(np.mean(validation.y))))
            probes[key]=probe
            details[key]={'best_iteration':probe.best_iteration or ceiling,'stopping_rows':len(vx),'probe_fitting_rows':int(early.sum()),
                          'weather_basis':fold['weather_basis'],'sampled_loss':loss[::50].tolist()}
        probe_sets.append(probes);fold_curves.append(details)
    models={};iterations={};curves={}
    for key,alpha in heads:
        # Compare the same iteration across both periods, rather than one noisy winner.
        common=min(map(len,losses[key]));average=np.mean([loss[:common] for loss in losses[key]],axis=0)
        iteration=int(np.argmin(average))+1
        for probes in probe_sets:probes[key].best_iteration=iteration
        cfg={**params,'min_child_samples':min(BASE['min_child_samples'],max(10,len(y)//20)),'n_estimators':iteration,'objective':'regression' if alpha is None else 'quantile','metric':'l2' if alpha is None else 'quantile'}
        if alpha is not None:cfg['alpha']=alpha
        model=lgb.LGBMRegressor(**cfg);model.fit(x,y,sample_weight=weight)
        models[key]=model;iterations[key]=iteration
        curves[key]={'ceiling':ceiling,'best_iteration':iteration,'near_ceiling':iteration>=ceiling*.95,'fitting_rows':len(y),
                     'metric':'energy_and_hourly' if key in ('middle','mean') else 'quantile','folds':[fold[key] for fold in fold_curves]}
        progress(f'{kind} {key}: refitted all training rows with {iteration} trees')
    return models,probe_sets,iterations,curves,validations


def run(settings,summary,d,progress):
    from .models import signature,validate_artifact
    from .quality import effective_policy
    from .load_training import LoadPredictor,legacy_feature_rows
    from .solar_training import solar_predict
    from .uncertainty import fit_dependence
    started=time.perf_counter();kind=summary['kind'];name=summary['id'];frame=d['frame']
    final_cases=d.get('validation_cases')
    if not final_cases:raise ValueError('Prepare data again to save issued-weather validation before training.')
    x,y,times,methods=fitting_rows(frame,d['start'],d['train_end'],settings,kind,progress)
    from .fitting import development_cases
    from .config import solar_banks
    bank=next((b for b in solar_banks(settings) if b.id==summary.get('bank_id')),None) if kind=='solar' else None
    development=d.get('development') or development_cases(frame,d['start'],d['train_end'],settings,kind,bank,progress,cached_only=True)
    folds=[]
    for fold in development.get('folds',[development]):
        cc=fold['cases']
        if len(cc)>16:cc=[cc[i] for i in np.linspace(0,len(cc)-1,16,dtype=int)]
        folds.append({**fold,'cases':cc})
    development={**folds[-1],'folds':folds}
    models,probe_sets,iterations,curves,validations=fit_heads(x,y,times,methods,kind,development,settings,progress)
    choices=[];all_predictions=[]
    folds=development.get('folds',[development])
    for fold,probes,validation in zip(folds,probe_sets,validations):
        # Predict once per head; all point formulas share those predictions.
        pp=[predict(probes,c,kind,{'head':'mean','baseline_weight':0}) for c in fold['cases']]
        all_predictions.append(pp)
        heads={'mean':validation.pack(pp),'middle':np.concatenate([q['middle'][mask] for q,mask in zip(pp,validation.masks)])}
        for head in heads:
            for weight in ((0,.5) if kind=='load' else (0,)):
                point=heads[head] if weight==0 else weight*validation.x.clock_profile.to_numpy()+(1-weight)*heads[head]
                score,energy_mae,hourly_mae=validation.scores(point)
                choices.append({'method':{'head':head,'baseline_weight':weight},'score':score/max(.1,float(np.mean(validation.y))),
                                'energy_mae_per_day':energy_mae,'hourly_mae':hourly_mae})
    point_methods=[{'head':head,'baseline_weight':w} for head in heads for w in ((0,.5) if kind=='load' else (0,))]
    point_method=min(point_methods,key=lambda method:np.mean([c['score'] for c in choices if c['method']==method]))
    chosen_predictions=[]
    for c,q in zip(development['cases'],all_predictions[-1]):
        learned=q['point'] if point_method['head']=='mean' else q[point_method['head']]
        weight=point_method['baseline_weight']
        point=learned if weight==0 else weight*c['x'].clock_profile.to_numpy()+(1-weight)*learned
        chosen_predictions.append({**q,'point':point})
    _,_,calibration_records=summarize(development['cases'],chosen_predictions,settings,kind)
    # Deduplicate target hours for both model kinds when assessing calibration evidence.
    calibration=fit_calibration(calibration_records,solar=kind=='solar')
    adjusted=[]
    if calibration_records:
        q=ordered({k:[r[k] for r in calibration_records] for k in ('low','middle','high')},
                  [r['lead'] for r in calibration_records],calibration,
                  [not r['scored'] for r in calibration_records] if kind=='solar' else None)
        adjusted=[{**r,**{k:float(v[j]) for k,v in q.items()}} for j,r in enumerate(calibration_records)]
    dependence=fit_dependence(adjusted,solar=kind=='solar')
    artifact={'schema':FEATURE_SCHEMA,'pipeline':LOAD_PIPELINE,'quality_policy':QUALITY_POLICY,
        'models':{k:models[k] for k in ('low','middle','high')},'mean_model':models['mean'],
        'timezone':settings.timezone,'features':list(x),'point_method':point_method,'calibration':calibration,
        'correction':None,'irradiance_cutoff':0,'sampling_rho':dependence['rho'],'sampling_dependence':dependence,
        'validation_probe':final_cases[0]['x'].iloc[:24].copy()}
    validate_artifact(artifact,kind)
    progress('Scoring saved issued-weather validation')
    pp=[predict(models,c,kind,point_method,calibration) for c in final_cases]
    result,energy,records=summarize(final_cases,pp,settings,kind)
    old_records=[];old_energy=None;comparison=None;comparison_result=result;comparison_note=''
    old_path=DATA/(name+'.joblib');old_meta=read_json('model-'+name,{})
    if old_path.exists():
        old_cutoff=old_meta.get('training_end')
        paired=[(c,q) for c,q in zip(final_cases,pp) if old_cutoff and c['origin']>=pd.Timestamp(old_cutoff)]
        if not paired:comparison_note='No paired comparison: the active model has no known earlier fitting cutoff for these issues.'
        else:
            try:
                old=joblib.load(old_path);old_predictions=[]
                for c,_ in paired:
                    if kind=='load':
                        old_x=legacy_feature_rows(frame,c,settings) if old.get('pipeline')=='lightgbm-quantile-direct-v2' else c['x']
                        old_predictions.append(LoadPredictor(old).predict(old_x))
                    else:old_predictions.append(solar_predict(old,c['x']))
                paired_cases=[c for c,_ in paired]
                old_summary,old_energy,old_records=summarize(paired_cases,old_predictions,settings,kind)
                comparison_result,paired_energy,paired_records=summarize(paired_cases,[q for _,q in paired],settings,kind)
                comparison={'metrics':old_summary['hourly'],'chart':assessment_chart(old_records),
                    ('load_validation' if kind=='load' else 'forecast_validation'):old_summary,
                    'candidate_metrics':comparison_result['hourly'],'candidate_validation':comparison_result,
                    'candidate_chart':assessment_chart(paired_records),'comparison_basis':'Identical issues after both fitting cutoffs'}
            except Exception as exc:
                comparison_note='Active model comparison unavailable: '+str(exc)
                old_records=[];old_energy=None
    baseline=None
    if kind=='load':
        baseline=summarize(final_cases,[{k:c['x'].clock_profile.to_numpy() for k in ('low','middle','high','point')} for c in final_cases],settings,kind)[0]
    diag=result['quantile'];diag.update(pinball_low=diag['low']['pinball'],pinball_mid=diag['middle']['pinball'],
        pinball_high=diag['high']['pinball'],width=diag['width_kw'],coverage_daylight=diag['coverage'],width_daylight_kw=diag['width_kw'])
    result.update(weather_model=settings.weather_model,publication_buffer_hours=12,skipped=d.get('validation_skipped',[]),daily_chart=energy['daily'])
    manifest={'fitting_start':str(times.min()),'stopping_start':str(development['start']),'stopping_end':str(development['end']),
        'training_end':str(d['train_end']),'assessment_start':str(d['val_start']),'assessment_end':str(d['end']),
        'calibration_basis':'Earlier-model chronological development predictions; '+development['weather_basis']+' weather; final refit includes all training rows; final assessment excluded'}
    metadata={'trained_at':pd.Timestamp.now(tz='UTC').isoformat(),'dataset_id':name,'kind':kind,'bank_id':summary.get('bank_id'),
        'model':'LightGBM energy / quantiles','pipeline':LOAD_PIPELINE,'solar_pipeline':__import__('app.datasets',fromlist=['SOLAR_PIPELINE']).SOLAR_PIPELINE if kind=='solar' else None,
        'schema':FEATURE_SCHEMA,'quality_policy':QUALITY_POLICY,'config_signature':signature(settings,kind,summary.get('bank_id')),
        'training_policy':effective_policy(settings,kind,summary['request']),'dataset_created_at':summary['created_at'],
        'training_rows':len(y),'unique_target_hours':len(times.unique()),'validation_rows':result['hourly']['samples'],
        'metrics':result['hourly'],'chart':assessment_chart(records),('load_validation' if kind=='load' else 'forecast_validation'):result,
        'comparison':comparison,'comparison_note':comparison_note,'history_baseline':baseline,
        'training_start':str(d['start']),'training_end':str(d['train_end']),
        'validation_start':str(d['val_start']),'validation_end':str(d['end']),'point_method':point_method,
        'calibration':calibration,'correction':None,'irradiance_cutoff':0,'sampling_rho':dependence['rho'],'sampling_dependence':dependence,
        'weather_basis':'Prepared historical weather for fitting; saved issued-weather replay for assessment',
        'parameters':{**BASE,'min_child_samples':models['middle'].get_params()['min_child_samples']},'iterations':iterations,'learning_curves':curves,'split_manifest':manifest,'point_selection':choices,'stopping_weather':development['weather_basis'],
        'effective_training_target_start':times.min().isoformat(),
        'effective_training_target_end':(times.max()+pd.Timedelta(hours=1)).isoformat(),
        'feature_importance':dict(zip(x.columns,models['middle'].feature_importances_.tolist())),
        'timings_seconds':{'total':time.perf_counter()-started},'assessment_classification':'No paired comparison',
        'paired_bootstrap':{h:paired_bootstrap(paired_energy[h],old_energy[h]) for h in ('daily','overnight','24','48','72')} if comparison else {}}
    def counts(mask):
        mm=methods[mask]
        return {'rows':len(mm),'measured':int(np.sum(mm=='')),'dark_derived':int(np.sum(mm=='dark_zero')),'interpolated':int(np.sum(mm=='interpolation'))}
    metadata['target_counts_by_stage']={'fitting':counts(np.ones(len(y),bool)),'calibration':{'rows':len(calibration_records),'unique_target_hours':len({r['time'] for r in calibration_records})}}
    metadata['uncertainty_calibration']={'status':'residual-adjusted' if calibration else 'uncalibrated',
        'basis':manifest['calibration_basis'],'independently_selected':False,
        'unique_calibration_hours':len({r['time'] for r in calibration_records if r['scored']}),
        'note':'Stopping and residual adjustment share the earlier development period. Inspect the separate final validation coverage.'}
    if kind=='load':
        part=frame.loc[training_mask(frame.index,d['start'],d['train_end'])&frame.train_eligible]
        metadata['hour_profile']={str(k):float(v) for k,v in part.target.groupby(part.index.tz_convert(settings.timezone).hour).mean().items()}
        if d['train_end']-times.max()>pd.Timedelta(days=7):metadata['history_warning']='The most recent usable fitting targets end '+str(times.max().date())+'. Check training dates and selected history sources.'
    else:metadata['history_warning']='Short training history: use more history from the same panel layout when available.' if d['train_end']-d['start']<pd.Timedelta(days=90) else ''
    if comparison:
        change=normalized(comparison_result,old_summary,kind);metadata['assessment_primary_ratio']=change
        metadata['assessment_classification']='regression' if change['score']>1.02 else 'promising but insufficient evidence' if change['score']<.98 else 'no material change'
    artifact['metadata']=metadata
    path=DATA/(name+'-candidate.joblib');tmp=path.with_suffix('.tmp');joblib.dump(artifact,tmp);tmp.replace(path)
    write_json(name+'-assessment-details',{'energy':energy,'records':records,'old_records':old_records,'old_energy':old_energy,
        'manifest':manifest,'model_identities':{'candidate':metadata['trained_at'],'existing':old_meta.get('trained_at')},
        'sampling_metadata':{'candidate':{k:metadata.get(k) for k in ('sampling_rho','sampling_dependence')},
                             'existing':{k:old_meta.get(k) for k in ('sampling_rho','sampling_dependence')}}})
    progress('Candidate saved for review')
    return metadata

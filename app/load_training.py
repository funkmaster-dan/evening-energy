"""One direct LightGBM predictor shared by calibration, replay and live issues."""
import os
import hashlib,inspect,json
from pathlib import Path
from time import perf_counter
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from .config import DATA,read_json
from .datasets import FEATURES
from .forecast_policy import FEATURE_SCHEMA,ordered,model_columns,SETTLEMENT_HOURS
from .load_splits import training_mask
from .quality import LOAD_PIPELINE,LEGACY_LOAD_PIPELINE,QUALITY_POLICY,flat_run_known_at

LOAD_FEATURES=FEATURES+['month','lead_hours','heating','cooling']+[f'lag{h}' for h in (1,2,3,24,48,168)]+[f'mean{h}' for h in (3,6,24,168)]+['std24']+[f'clock{d}' for d in (1,2,7,14)]+['clock_profile','weather_run_age']
PARAMETERS=dict(boosting_type='gbdt',objective='quantile',n_estimators=max(1,int(os.getenv('ENERGY_TRAIN_TREES','1000'))),learning_rate=.03,max_bin=255,reg_lambda=2.,subsample=.9,subsample_freq=1,colsample_bytree=.9,random_state=42,n_jobs=max(1,int(os.getenv('ENERGY_TRAIN_THREADS','2'))),deterministic=True,force_col_wise=True,verbosity=-1)
SELECTION_ROWS=max(1,int(os.getenv('ENERGY_SELECTION_ROWS','150000')))
QUANTILES=(0.1,0.5,0.9)
QKEYS=('low','middle','high')
QMAP=dict(zip(QUANTILES,QKEYS))
CONFIGURATIONS=[dict(num_leaves=leaves,max_depth=depth,min_child_samples=child,half_life_days=half) for leaves,depth,child,half in [(31,5,100,365),(31,5,300,None),(63,6,100,None),(63,6,300,365)]]

def feature_rows(frame,origin,hours,timezone,profile_days=28,omit=(),_quality_applied=False,legacy=False):
    """Shared vector builder for one issue or expanded origin/target pairs."""
    target_index=pd.DatetimeIndex(hours)
    origin_index=pd.DatetimeIndex([pd.Timestamp(origin)]*len(hours)) if isinstance(origin,(str,pd.Timestamp)) else pd.DatetimeIndex(origin)
    if len(origin_index)!=len(target_index):raise ValueError('Each target requires its forecast origin')
    lead=np.asarray((target_index-origin_index).total_seconds()/3600)
    if np.any(lead<0) or np.any(lead>72):raise ValueError('Direct load forecasts support leads 0–72 only')
    origins=origin_index.unique().sort_values();oi=origins.get_indexer(origin_index)
    grid=pd.date_range(frame.index.min().floor('h'),max(frame.index.max(),target_index.max()),freq='h');f=frame.reindex(grid)
    f.attrs={}
    y=pd.to_numeric(f.get('context_target',f.target),errors='coerce')
    lag_offset=0 if legacy else SETTLEMENT_HOURS
    if legacy and len(origins)==1:y=y.mask(grid+pd.Timedelta(hours=1)>origins[0]-pd.Timedelta(hours=SETTLEMENT_HOURS))
    settings=type('Clock',(),{'timezone':timezone})()
    context=pd.DataFrame(index=grid)
    for h in (1,2,3,24,48,168):context[f'lag{h}']=y.shift(h+lag_offset)
    for h in (3,6,24,168):context[f'mean{h}']=y.shift(1+lag_offset).rolling(h,min_periods=int(np.ceil(.75*h))).mean()
    context['std24']=y.shift(1+lag_offset).rolling(24,min_periods=18).std()
    x=f.reindex(target_index)[FEATURES].reset_index(drop=True).astype(np.float32);local=target_index.tz_convert(settings.timezone)
    x['month']=local.month;x['lead_hours']=lead;x['heating']=(18-x.temperature_2m).clip(lower=0);x['cooling']=(x.temperature_2m-24).clip(lower=0)
    for col in context:x[col]=context[col].reindex(origin_index).to_numpy(dtype=np.float32)
    clock=y.index.tz_convert(settings.timezone).tz_localize(None)
    # Two observations in an autumn duplicated hour may only be averaged once both completed.
    for d in (1,2,7,14):
        lookup=local.tz_localize(None)-pd.Timedelta(days=d);total=np.zeros(len(x));count=np.zeros(len(x))
        # Vectorized first/second occurrence lookup avoids merging future autumn observations.
        first=pd.DataFrame({'clock':clock,'utc':grid,'value':y.to_numpy()}).drop_duplicates('clock').set_index('clock').reindex(lookup)
        valid=first.value.notna().to_numpy()&(pd.DatetimeIndex(first.utc).asi8<(origin_index-pd.Timedelta(hours=SETTLEMENT_HOURS)).asi8)
        total+=np.where(valid,first.value.to_numpy(),0);count+=valid
        dup=pd.DataFrame({'clock':clock,'utc':grid,'value':y.to_numpy()});dup=dup.loc[dup.clock.duplicated(keep='first')].set_index('clock').reindex(lookup)
        valid=dup.value.notna().to_numpy()&(pd.DatetimeIndex(dup.utc).asi8<(origin_index-pd.Timedelta(hours=SETTLEMENT_HOURS)).asi8)
        total+=np.where(valid,dup.value.to_numpy(),0);count+=valid
        x[f'clock{d}']=np.divide(total,count,out=np.full(len(x),np.nan),where=count>0).astype(np.float32)
    # Profiles are small (48 groups per origin) and exclude only completed away days.
    profiles=np.full((len(origins),48),np.nan)
    local_history=grid.tz_convert(settings.timezone)
    eligible_profile=f.get('train_eligible',f.target.notna()).fillna(False).to_numpy(dtype=bool)&y.notna().to_numpy()
    keys=local_history.hour+24*(local_history.dayofweek>=5)
    for k in range(48):
        selected=(keys==k)&eligible_profile;values=y.loc[selected].to_numpy();times=grid[selected].asi8
        left=np.searchsorted(times,(origins-pd.Timedelta(days=profile_days)).asi8,side='left');right=np.searchsorted(times,(origins-pd.Timedelta(hours=SETTLEMENT_HOURS)).asi8,side='left')
        sums=np.r_[0,np.cumsum(values)];counts=right-left
        profiles[:,k]=np.divide(sums[right]-sums[left],counts,out=np.full(len(origins),np.nan),where=counts>=2)
    # Background flags for the issue's incomplete local day were not yet known.
    if 'quality_flags' in f:
        away=f.quality_flags.eq('background_only').to_numpy()&y.notna().to_numpy()
        local_dates=local_history.date;away_dates=set(local_dates[away])
        for j,origin in enumerate(origins):
            if origin.tz_convert(settings.timezone).date() not in away_dates:continue
            same=away&(grid<origin-pd.Timedelta(hours=SETTLEMENT_HOURS))&(local_dates==origin.tz_convert(settings.timezone).date())
            for k in np.unique(keys[same]):
                recent=eligible_profile&(keys==k)&(grid<origin-pd.Timedelta(hours=SETTLEMENT_HOURS))&(grid>=origin-pd.Timedelta(days=profile_days))
                values=y.loc[recent| (same&(keys==k))]
                if values.count()>=2:profiles[j,int(k)]=values.mean()
    key=local.hour+24*(local.dayofweek>=5);x['clock_profile']=profiles[oi,key].astype(np.float32)
    for group in omit:
        columns=[c for c in x if c.startswith('lag')] if group=='lags' else [c for c in x if c.startswith('mean') or c=='std24'];x[columns]=np.nan
    if not _quality_applied:
        events=frame.attrs.get('quality_runs',[])
        affected=set()
        for event in events:
            if event.get('reason') not in ('implausible_flat_load','frozen_power_run'):continue
            left=pd.Timestamp(event['start']);right=pd.Timestamp(event['end'])
            affected.update(np.flatnonzero((origins>=left)&(origins<=right+pd.Timedelta(days=max(profile_days,14)+3))).tolist())
        # Many origins have the same known corrections. Rebuild those rows in
        # one vectorized pass instead of recursively rebuilding every origin.
        runs=[event for event in events if event.get('reason') in ('implausible_flat_load','frozen_power_run')]
        known=[flat_run_known_at(event) for event in runs]
        groups={}
        for j in sorted(affected):groups.setdefault(tuple(origins[j]>=at for at in known),[]).append(j)
        for state,indices in groups.items():
            positions=np.flatnonzero(np.isin(oi,indices))
            if not len(positions):continue
            prefix=f.copy()
            raw_sources=frame.attrs.get('context_sources')
            sources=raw_sources.reindex(prefix.index).copy() if raw_sources is not None else None
            context=prefix.get('context_target',prefix.target).copy();combined_mask=pd.Series(False,index=prefix.index)
            for event,is_known in zip(runs,state):
                left=pd.Timestamp(event['start']);right=pd.Timestamp(event['end'])
                mask=(prefix.index>=left)&(prefix.index<right)
                if 'quality_flags' in prefix:mask &= prefix.quality_flags.ne('reviewed_restore')
                entity=event.get('source_entity')
                if is_known:
                    if sources is not None and entity in sources:sources.loc[mask,entity]=np.nan
                    else:combined_mask.loc[mask]=True;context.loc[mask]=np.nan
                else:
                    # Historical exclusions were not yet known to this issue.
                    restore=mask&prefix.get('quality_flags',pd.Series('',index=prefix.index)).isin(['implausible_flat_load','frozen_power_run','sensor_fault'])
                    prefix.loc[restore,'train_eligible']=True
            if sources is not None:
                from .datasets import splice
                from .quality import physical
                context=splice(sources,list(sources),frame.attrs.get('history_combine','splice'))
                context,_=physical(context,'load',frame.attrs.get('context_limit',40))
            context=context.mask(combined_mask)
            if 'quality_flags' in prefix:context=context.mask(prefix.quality_flags.eq('reviewed_exclusion'))
            prefix['context_target']=context
            prefix.attrs={}
            x.iloc[positions]=feature_rows(prefix,origin_index[positions],target_index[positions],timezone,profile_days,omit,True,legacy).to_numpy()
    x['weather_run_age']=np.nan
    return x[LOAD_FEATURES].astype(np.float32)

class LoadPredictor:
    def __init__(self,artifact):
        if artifact.get('pipeline') not in (LOAD_PIPELINE,LEGACY_LOAD_PIPELINE) or artifact.get('quality_policy')!=QUALITY_POLICY:raise ValueError('Rebuild and calibrate the direct LightGBM load model')
        self.artifact=artifact;self.models=artifact['models']
    def forecast(self,frame,hours,origin=None):
        origin=pd.Timestamp(origin) if origin is not None else hours[0]
        x=feature_rows(frame,origin,hours,self.artifact['timezone'],self.artifact.get('profile_days',28),self.artifact.get('omit',()),legacy=self.artifact.get('pipeline')==LEGACY_LOAD_PIPELINE)
        if x[FEATURES].isna().any().any():raise ValueError('Incomplete target weather/calendar coverage')
        return self.predict(x)
    def predict(self,x):
        raw={k:m.predict(x.reindex(columns=model_columns(m,LOAD_FEATURES))) for k,m in self.models.items()}
        if set(raw)=={'middle'}:return {'middle':np.maximum(0,raw['middle'])}
        q=ordered(raw,x.get('lead_hours'),self.artifact.get('calibration'))
        method=self.artifact.get('point_method',{'head':'middle','baseline_weight':0})
        mean=self.artifact.get('mean_model')
        learned=np.maximum(0,mean.predict(x.reindex(columns=model_columns(mean,LOAD_FEATURES)))) if method['head']=='mean' and mean is not None else np.maximum(0,np.sort(np.column_stack([raw[k] for k in QKEYS]),axis=1)[:,1])
        weight=method.get('baseline_weight',0)
        q['point']=np.maximum(0,weight*x.clock_profile.to_numpy()+(1-weight)*learned)
        return q

def pinball(y,q,alpha):
    y=np.asarray(y,float); q=np.asarray(q,float)
    return float(np.mean(np.maximum(alpha*(y-q),(alpha-1)*(y-q))))

def season(t,timezone):
    month=t.tz_convert(timezone).month
    return ('summer' if month in (12,1,2) else 'autumn' if month in (3,4,5) else 'winter' if month in (6,7,8) else 'spring')

def next_charge_start(origin,settings):
    local=origin.tz_convert(settings.timezone);starts=[]
    for days in range(5):
        day=local.date()+pd.Timedelta(days=days)
        for w in settings.charging_windows:
            t=pd.Timestamp(f'{day} {w.start}').tz_localize(settings.timezone,ambiguous=False,nonexistent='shift_forward').tz_convert('UTC')
            if t>origin:starts.append(t)
    return min(starts) if starts else None

def issue_origins(start,end,settings):
    start=pd.Timestamp(start);end=pd.Timestamp(end);origins=set()
    first=start.tz_convert(settings.timezone).normalize()-pd.DateOffset(days=1)
    for day in pd.date_range(first,end.tz_convert(settings.timezone).normalize(),freq='D'):
        for clock in {'00:00','06:00','12:00','18:00',*[w.start for w in settings.export_windows]}:
            t=pd.Timestamp(f'{day.date()} {clock}').tz_localize(settings.timezone,ambiguous=False,nonexistent='shift_forward').tz_convert('UTC').floor('h')
            if start<=t<end:origins.add(t)
    return sorted(origins)

def evaluation_cases(frame,start,end,settings,progress=lambda s:None,issued_weather=True):
    from .sources import consumption_weather,Calendar
    from .datasets import features
    cases=[];skipped={'targets':0,'context':0,'weather':0};weather_cache={};calendar=None
    for origin in issue_origins(start,end,settings):
        hours=pd.date_range(origin,min(origin+pd.Timedelta(hours=71),pd.Timestamp(end)-pd.Timedelta(hours=1)),freq='h')
        if len(hours)<24:continue
        f=frame.reindex(hours);eligible=f.get('validation_eligible',f.target.notna()).fillna(False)
        # Shorter complete windows remain usable; never fill a scored measurement.
        bad=np.flatnonzero(~eligible.to_numpy()|f.target.isna().to_numpy());length=int(bad[0]) if len(bad) else len(hours)
        if length<24:skipped['targets']+=1;continue
        hours=hours[:length]
        context_frame=frame.loc[(frame.index<origin)&(frame.index>=origin-pd.Timedelta(hours=168))]
        context=context_frame.get('context_target',context_frame.target)
        if context.count()<126:skipped['context']+=1;continue
        safe=frame.loc[(frame.index<origin)&(frame.index>=origin-pd.Timedelta(days=58))].copy()
        future=frame.reindex(hours).copy();source='historical-weather proxy';archive=False
        if issued_weather:
            run=(origin-pd.Timedelta(hours=12)).floor('6h')
            if run not in weather_cache:
                progress('Checking issued weather for '+str(origin.date()))
                try:
                    weather_cache[run]=consumption_weather(settings,run,run+pd.Timedelta(days=7),issued_at=origin)
                except Exception:weather_cache[run]=None
            weather=weather_cache[run]
            if weather is not None:
                if calendar is None:calendar=Calendar(settings).load()
                weather=features(weather.reindex(hours),settings,calendar)
                if not weather[FEATURES].isna().any().any():future[FEATURES]=weather[FEATURES];source='archived issued weather (12h publication allowance)';archive=True
            if not archive:skipped['weather']+=1
        future['target']=np.nan
        metadata=frame.attrs.copy()
        if metadata.get('context_sources') is not None:metadata['context_sources']=metadata['context_sources'].reindex(safe.index).copy()
        safe.attrs={};future.attrs={}
        safe['context_target']=safe.get('context_target',safe.target)
        future['context_target']=np.nan
        safe=pd.concat([safe,future]);safe.attrs=metadata;x=feature_rows(safe,origin,hours,settings.timezone)
        if x[FEATURES].isna().any().any():skipped['weather']+=1;continue
        if x.clock_profile.isna().any():skipped['context']+=1;continue
        cases.append({'origin':origin,'hours':hours,'frame':safe,'x':x,'actual':f.target.iloc[:length].to_numpy(float),'weather_source':source,'archive':archive,'skipped':skipped,'context_coverage_pct':100*context.count()/168})
    return cases

def energy_metrics(errors):
    return {'energy_mae_kwh':round(float(np.mean(np.abs(errors))),4),'energy_bias_kwh':round(float(np.mean(errors)),4),'underprediction_kwh':round(float(np.mean(np.maximum(errors,0))),4),'samples':len(errors)} if errors else {'samples':0}

def evaluate(artifact,cases,settings,baseline=False,point_only=False):
    from .models import metrics,validation_chart
    if not cases:raise ValueError('No complete normal-use validation windows with sufficient prior context')
    predictor=None if baseline else LoadPredictor(artifact); ys=[];ps=[];errors={h:[] for h in (24,48,72)};nights=[];seasonal={};cumulative=[];chart=[];worst=[]
    intervals=not baseline and not point_only
    pins={a:[] for a in QUANTILES};hits=[];widths=[];lead_hits={};lead_widths={};season_hits={};day_hits=[];night_hits=[];day_widths=[];night_widths=[];season_widths={};cov24=[];cov48=[];cov72=[];zero_low=[]
    regimes={k:{'actual':[],'predicted':[],'hits':[]} for k in ('overall_valid','normal_use','quiet','return_day')}
    quiet_dates=set(frame_date for c in cases for frame_date in c['frame'].index[c['frame'].get('quiet_regime',pd.Series(False,index=c['frame'].index)).fillna(False)].tz_convert(settings.timezone).date)
    for c in cases:
        if baseline:
            p=c['x'].clock_profile.to_numpy(float)
            # Baseline is unavailable where no prior same-clock reference exists.
            if not np.isfinite(p).all():continue
            q={'low':p,'middle':p,'high':p}
        elif artifact.get('profile_days',28)==28 and not artifact.get('omit'):
            q=predictor.predict(c['x'])
        else:q=predictor.forecast(c['frame'],c['hours'],c['origin'])
        p=q['middle']
        y=c['actual']; e=y-p;ys.extend(y);ps.extend(p)
        quiet=c['frame'].get('quiet_regime',pd.Series(False,index=c['frame'].index)).reindex(c['hours']).fillna(False).to_numpy(dtype=bool)
        returning=np.array([(pd.Timestamp(t.date())-pd.Timedelta(days=1)).date() in quiet_dates for t in c['hours'].tz_convert(settings.timezone)])&~quiet
        for label,mask in [('overall_valid',np.ones(len(y),dtype=bool)),('normal_use',~quiet),('quiet',quiet),('return_day',returning)]:
            regimes[label]['actual'].extend(y[mask]);regimes[label]['predicted'].extend(p[mask])
            if intervals:regimes[label]['hits'].extend(((y>=q['low'])&(y<=q['high']))[mask])
        if intervals:
            in_band=(y>=q['low'])&(y<=q['high']);width=q['high']-q['low']
            hits.extend(in_band.tolist());widths.extend(width.tolist());zero_low.append(float(np.mean(q['low']==0)))
            local_hours=c['hours'].tz_convert(settings.timezone);day_mask=(local_hours.hour>=7)&(local_hours.hour<19)
            day_hits.extend(in_band[day_mask].tolist());night_hits.extend(in_band[~day_mask].tolist())
            day_widths.extend(width[day_mask].tolist());night_widths.extend(width[~day_mask].tolist())
            s0=season(c['origin'],settings.timezone);season_hits.setdefault(s0,[]).extend(in_band.tolist());season_widths.setdefault(s0,[]).extend(width.tolist())
            for lead,hit in enumerate(in_band):
                h=int((c['hours'][lead]-c['origin']).total_seconds()//3600)
                lead_hits.setdefault(h,[]).append(bool(hit));lead_widths.setdefault(h,[]).append(float(width[lead]))
            for h,store in ((24,cov24),(48,cov48),(72,cov72)):
                if len(y)>=h:store.append(bool(q['low'][:h].sum()<=y[:h].sum()<=q['high'][:h].sum()))
            for a in QUANTILES: pins[a].extend(np.maximum(a*(y-q[QMAP[a]]),(a-1)*(y-q[QMAP[a]])).tolist())
        for h in errors:
            if len(y)>=h:errors[h].append(float(e[:h].sum()))
        cumulative.append({'origin':c['origin'].isoformat(),'errors_kwh':np.cumsum(e).tolist()})
        worst.append({'origin':c['origin'].isoformat(),'hours':len(y),'error_kwh':float(e.sum()),'weather_source':c['weather_source']})
        s=season(c['origin'],settings.timezone);seasonal.setdefault(s,[]).extend(e.tolist())
        local=c['origin'].tz_convert(settings.timezone)
        for window in settings.export_windows:
            export=pd.Timestamp(f'{local.date()} {window.start}').tz_localize(settings.timezone,ambiguous=False,nonexistent='shift_forward').tz_convert('UTC')
            if export.floor('h')==c['origin']:
                stop=next_charge_start(export,settings)
                if stop is not None and export>=c['origin'] and stop<=c['hours'][-1]+pd.Timedelta(hours=1):
                    weights=np.array([max(0,(min(t+pd.Timedelta(hours=1),stop)-max(t,export)).total_seconds()/3600) for t in c['hours']]);nights.append(float(e@weights))
        if not chart:chart=validation_chart(c['hours'],y,p)
    if not ys:raise ValueError('History baseline has insufficient clock coverage')
    hourly=metrics(ys,ps);hourly['bias']=round(float(np.mean(np.array(ys)-np.array(ps))),4)
    quantile=None
    if intervals:
        quantile={'pinball_low':round(float(np.mean(pins[0.1])),4),'pinball_mid':round(float(np.mean(pins[0.5])),4),'pinball_high':round(float(np.mean(pins[0.9])),4),
                  'coverage':round(float(np.mean(hits)),4),'width_kw':round(float(np.mean(widths)),4),
                  'coverage_day':round(float(np.mean(day_hits)),4) if day_hits else None,'coverage_night':round(float(np.mean(night_hits)),4) if night_hits else None,
                  'width_day_kw':round(float(np.mean(day_widths)),4) if day_widths else None,'width_night_kw':round(float(np.mean(night_widths)),4) if night_widths else None,
                  'by_lead':{str(k):{'coverage':round(float(np.mean(lead_hits[k])),4),'width_kw':round(float(np.mean(lead_widths[k])),4),'samples':len(lead_hits[k])} for k in sorted(lead_hits)},
                  'by_season':{s:{'coverage':round(float(np.mean(v)),4),'width_kw':round(float(np.mean(season_widths[s])),4),'samples':len(v)} for s,v in season_hits.items()},
                  'energy_coverage':{h:round(float(np.mean(s)),4) if s else None for h,s in (('24',cov24),('48',cov48),('72',cov72))},
                  'energy_interval_basis':'Sum of hourly marginal P10/P90 bounds; not MC energy percentiles',
                  'zero_truncated_share':round(float(np.mean(zero_low)),4)}
    summary={'label':'Overall-valid validation','horizons':{str(h):{**energy_metrics(v),'energy_coverage':round(float(np.mean(s)),4) if s else None} for (h,v),s in zip(errors.items(),(cov24,cov48,cov72))},'overnight':energy_metrics(nights),'hourly':hourly,'origins':len(cases),'skipped_windows':cases[-1]['skipped'],'excluded_background_hours':sum(int(c['frame'].get('quality_flags',pd.Series(dtype=str)).eq('background_only').sum()) for c in cases),'issued_weather_windows':sum(c['archive'] for c in cases),'historical_weather_windows':sum(not c['archive'] for c in cases),'context_coverage_pct':float(np.mean([c['context_coverage_pct'] for c in cases])),'seasonal':{s:{'mae_kw':float(np.mean(np.abs(v))),'bias_kw':float(np.mean(v)),'samples':len(v)} for s,v in seasonal.items()},'quantile':quantile,'note':'Positive bias means underprediction. Overlapping windows are not independent. Historical-weather proxy results do not estimate live accuracy.'}
    summary['regimes']={k:{**metrics(v['actual'],v['predicted']),'bias_kw':float(np.mean(np.array(v['actual'])-np.array(v['predicted']))),'coverage':float(np.mean(v['hits'])) if v['hits'] else None} if v['actual'] else {'samples':0} for k,v in regimes.items()}
    from .uncertainty import residual_rho
    summary['rho']=residual_rho({'cumulative':cumulative})
    return {'summary':summary,'chart':chart,'cumulative':cumulative,'worst_windows':sorted(worst,key=lambda r:abs(r['error_kwh']),reverse=True)[:10]}

def training_rows(frame,start,end,settings,profile_days=28,omit=(),progress=lambda s:None,recipe=None):
    origins=pd.date_range(pd.Timestamp(start).ceil('h'),pd.Timestamp(end).ceil('h'),freq='h',inclusive='left')
    origin_index=origins[np.repeat(np.arange(len(origins)),73)];lead=np.tile(np.arange(73),len(origins));target_index=origin_index+pd.to_timedelta(lead,unit='h')
    target=frame[FEATURES+['target','train_eligible']].reindex(target_index)
    eligible=target.train_eligible.fillna(False).to_numpy(dtype=bool)&target.target.notna().to_numpy()&(target_index+pd.Timedelta(hours=1)<=end)&target[FEATURES].notna().all(axis=1).to_numpy()
    if not eligible.any():raise ValueError('No eligible measured load targets')
    recipe=recipe or {}
    if recipe.get('exclude_quiet'):eligible &= ~frame.quiet_regime.reindex(target_index).fillna(False).to_numpy(dtype=bool)
    if recipe.get('recent_only'):
        sources=frame.get('source_entity',pd.Series('',index=frame.index));available=sources.loc[training_mask(frame.index,start,end)&frame.train_eligible]
        recent=available.iloc[-1] if len(available) else ''
        eligible &= sources.reindex(target_index).eq(recent).to_numpy()
    origin_index=origin_index[eligible];target_index=target_index[eligible]
    return feature_rows(frame,origin_index,target_index,settings.timezone,profile_days,omit),target.target.loc[eligible].to_numpy(float),target_index

def fold_frame(frame,cutoff,settings):
    """Freeze learned target-selection thresholds at each fitting prefix."""
    from .quality import quality_frame
    f=frame.copy()
    if 'raw_target' not in frame:return f
    raw=frame.raw_target.copy();raw.attrs={}
    source=frame.get('source_entity')
    if source is not None:source=source.copy();source.attrs={}
    bounds=frame[['sensor_min','sensor_max']].rename(columns={'sensor_min':'min','sensor_max':'max'}) if 'sensor_min' in frame else None
    if bounds is not None:bounds.attrs={}
    q,_=quality_frame(raw,settings,'load',frame.attrs.get('context_limit',settings.load_max_power_kw),source=source,cutoff=pd.Timestamp(cutoff).floor('h'),corrections=read_json('dataset-load',{}).get('request',{}).get('corrections',[]),minmax=bounds)
    for col in ('target','quality_flags','train_eligible','validation_eligible','quiet_regime','normal_use_eligible'):f[col]=q[col]
    if 'quality_flags' in frame:
        rejected=frame.quality_flags.isin(['physical_limit','reviewed_exclusion','mean_outside_minmax','negative_energy_delta','recorder_interval_gap','incomplete_sum'])
        f.loc[rejected,'target']=np.nan;f.loc[rejected,['train_eligible','validation_eligible']]=False
    f['train_eligible'] &= f[FEATURES].notna().all(axis=1)
    f['validation_eligible'] &= f[FEATURES].notna().all(axis=1)
    return f

def selection_folds(frame,start,end,settings):
    folds=[]
    if end-start>=pd.Timedelta(days=180):
        candidates=[]
        for left in pd.date_range(max(start+pd.Timedelta(days=28),end-pd.Timedelta(days=365)),end-pd.Timedelta(days=28),freq='7D'):
            right=left+pd.Timedelta(days=28)
            if season(left,settings.timezone)!=season(right-pd.Timedelta(hours=1),settings.timezone):continue
            f=frame.loc[(frame.index>=left)&(frame.index<right)];eligible=f.train_eligible.fillna(False)
            score=int(eligible.rolling(72).sum().eq(72).sum());candidates.append((season(left,settings.timezone),score,left,right))
        used=[]
        for s in ('summer','autumn','winter','spring'):
            viable=[c for c in candidates if c[0]==s and c[1]>0 and all(c[3]<=a or c[2]>=b for a,b in used)]
            if viable:
                _,score,left,right=max(viable,key=lambda c:(c[1],c[2]));folds.append({'start':left,'end':right,'season':s,'reason':'Complete eligible 72-hour windows, then recency; expanding prefix'});used.append((left,right))
    if not folds:
        left=max(start+pd.Timedelta(days=2),end-max(pd.Timedelta(hours=48),(end-start)*.2))
        folds=[{'start':left.ceil('h'),'end':end,'season':season(left,settings.timezone),'reason':'Short experiment: chronological final 20% of training'}]
    return sorted(folds,key=lambda r:r['start'])

def weights(times,end,half):
    if not half:return None
    w=2**(-np.asarray((end-times).total_seconds())/86400/half);return w/w.mean()

def rank(summary):
    night=summary['overnight'];daily=summary['horizons'].get('24',{})
    base=(night.get('energy_mae_kwh',daily.get('energy_mae_kwh',float('inf'))),daily.get('energy_mae_kwh',float('inf')),abs(night.get('energy_bias_kwh',daily.get('energy_bias_kwh',float('inf')))))
    cov=(summary.get('quantile') or {}).get('coverage')
    penalty=max(0.,0.8-cov)*10. if cov is not None else 0.
    return (base[0]+penalty,base[1],base[2])

def fit_configuration(config,fold_data,settings,progress):
    results=[];iterations=[]
    for entry in fold_data:
        fold,x,y,times,vx,vy,cases=joblib.load(entry) if isinstance(entry,(str,os.PathLike)) else entry
        x,y,times=recipe_filter(x,y,times,config,fold['start'],settings)
        if len(times.unique())<168:raise ValueError('Recipe lacks seven days of internal training measurements')
        ix=tuning_indices(len(x));vi=tuning_indices(len(vx),50000)
        params={**PARAMETERS,'metric':'quantile','alpha':0.5,**{k:v for k,v in config.items() if k in ('num_leaves','max_depth','min_child_samples')}}
        model=lgb.LGBMRegressor(**params)
        progress(f'Scoring {fold["season"]} P50 configuration on {len(ix):,} tuning rows')
        model.fit(x.iloc[ix],y[ix],sample_weight=weights(times[ix],fold['start'],config.get('half_life_days')),eval_set=[(vx.iloc[vi],vy[vi])],eval_metric='quantile',callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False)])
        iterations.append(model.best_iteration_ or PARAMETERS['n_estimators'])
        artifact={'pipeline':LOAD_PIPELINE,'quality_policy':QUALITY_POLICY,'models':{'middle':model},'timezone':settings.timezone,'profile_days':config.get('profile_days',28),'omit':config.get('omit',())}
        result=evaluate(artifact,cases,settings,point_only=True);results.append({'fold':{k:str(v) for k,v in fold.items()},'iteration':iterations[-1],'summary':result['summary']})
        progress(f"LightGBM {config}: {fold['season']} energy MAE {rank(result['summary'])[0]:.3f} kWh")
    score=tuple(float(np.mean([rank(r['summary'])[i] for r in results])) for i in range(3))
    return {'configuration':config,'score':score,'iteration':int(np.median(iterations)),'folds':results}

def tuning_indices(size,limit=None):
    """Fixed uniform tuning sample; full refits and scored windows stay intact."""
    limit=SELECTION_ROWS if limit is None else limit
    return np.arange(size) if size<=limit else np.sort(np.random.default_rng(42).choice(size,limit,replace=False))

def selection_cache_key(settings):
    from .models import signature
    functions=(feature_rows,training_rows,fold_frame,evaluation_cases,issue_origins)
    code=''.join(inspect.getsource(f) for f in functions)+repr(LOAD_FEATURES)
    for name in ('quality.py','datasets.py','sources.py'):
        code+=(Path(__file__).parent/name).read_text()
    return {'dataset':hashlib.sha256((DATA/'load.pkl').read_bytes()).hexdigest(),
            'features':hashlib.sha256(code.encode()).hexdigest(),'signature':signature(settings,'load')}

def search_configurations(configurations,fold_data,settings,progress):
    """Read each large fold once, then score the small P50 grid on it."""
    comparisons=[{'configuration':c,'folds':[]} for c in configurations]
    for entry in fold_data:
        data=joblib.load(entry) if isinstance(entry,(str,os.PathLike)) else entry
        for result in comparisons:
            result['folds'].extend(fit_configuration(result['configuration'],[data],settings,progress)['folds'])
    for result in comparisons:
        result['score']=tuple(float(np.mean([rank(f['summary'])[i] for f in result['folds']])) for i in range(3))
        result['iteration']=int(np.median([f['iteration'] for f in result['folds']]))
    return comparisons

def save_artifact(artifact,name='load.joblib'):
    path=DATA/name;tmp=path.with_suffix('.tmp');joblib.dump(artifact,tmp);os.replace(tmp,path)

def fit_quantile_heads(x,y,times,end,config,iteration,progress):
    x,y,times=recipe_filter(x,y,times,config,end,None)
    sw=weights(times,end,config.get('half_life_days'))
    # Tails converge differently: chronological slice gives each head its own iteration.
    cut=int(len(times.unique())*0.8);split=sorted(times.unique())[cut] if cut<len(times.unique()) else end
    early_mask=times+pd.Timedelta(hours=1)<=split;late_mask=times>=split
    base={k:v for k,v in config.items() if k in ('num_leaves','max_depth','min_child_samples')}
    iterations={'middle':iteration};models={}
    for alpha,key in ((0.5,'middle'),(0.1,'low'),(0.9,'high')):
        if alpha==0.5:stop=iteration
        else:
            probe=lgb.LGBMRegressor(**{**PARAMETERS,**base,'alpha':alpha,'metric':'quantile'})
            early=np.flatnonzero(early_mask);late=np.flatnonzero(late_mask)
            early=early[tuning_indices(len(early))];late=late[tuning_indices(len(late),50000)]
            w=weights(times[early],split,config.get('half_life_days'))
            probe.fit(x.iloc[early],y[early],sample_weight=w,eval_set=[(x.iloc[late],y[late])],eval_metric='quantile',callbacks=[lgb.early_stopping(50,first_metric_only=True,verbose=False)])
            stop=probe.best_iteration_ or PARAMETERS['n_estimators']
        model=lgb.LGBMRegressor(**{**PARAMETERS,**base,'alpha':alpha,'metric':'quantile','n_estimators':stop})
        model.fit(x,y,sample_weight=sw);models[key]=model;iterations[key]=stop
        progress(f'Load {key} head: {stop} trees')
    params={**PARAMETERS,**base,'n_estimators':iterations['middle']}
    return models,iterations,params

def train_load(settings,summary,d,progress):
    from .study import run
    return run(settings,summary,d,progress)


def recipe_filter(x,y,times,config,cutoff,settings):
    if not config.get('recent_only') and not config.get('exclude_quiet'):return x,y,times
    data=pd.read_pickle(DATA/'load.pkl');frame=data['frame']
    if settings is not None:frame=fold_frame(frame,cutoff,settings)
    mask=np.ones(len(y),dtype=bool)
    if config.get('exclude_quiet'):mask &= ~frame.quiet_regime.reindex(times).fillna(False).to_numpy(dtype=bool)
    if config.get('recent_only'):
        available=frame.loc[(frame.index+pd.Timedelta(hours=1)<=cutoff)&frame.train_eligible,'source_entity']
        latest=available.iloc[-1] if len(available) else ''
        mask &= frame.source_entity.reindex(times).eq(latest).to_numpy()
    return x.loc[mask].reset_index(drop=True),y[mask],times[mask]


def legacy_feature_rows(frame,case,settings):
    """Original lag semantics, with the same settled information as new models."""
    origin=case['origin'];hours=case['hours'];attrs=frame.attrs.copy()
    past=frame.loc[(frame.index<origin.floor('h'))&(frame.index>=origin-pd.Timedelta(days=58))].copy()
    past['context_target']=past.get('context_target',past.target)
    past.loc[past.index+pd.Timedelta(hours=1)>origin-pd.Timedelta(hours=SETTLEMENT_HOURS),'context_target']=np.nan
    future=frame.reindex(hours).copy();future[FEATURES]=case['x'][FEATURES].to_numpy();future['target']=np.nan;future['context_target']=np.nan
    past.attrs={};future.attrs={};safe=pd.concat([past,future]);safe.attrs=attrs
    return feature_rows(safe,origin.floor('h'),hours,settings.timezone,legacy=True)

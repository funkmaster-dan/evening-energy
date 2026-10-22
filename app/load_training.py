"""One direct LightGBM predictor shared by calibration, replay and live issues."""
import numpy as np
import pandas as pd
from .datasets import FEATURES
from .forecast_policy import ordered,model_columns,SETTLEMENT_HOURS
from .quality import LOAD_PIPELINE,LEGACY_LOAD_PIPELINE,QUALITY_POLICY,flat_run_known_at

LOAD_FEATURES=FEATURES+['month','lead_hours','heating','cooling']+[f'lag{h}' for h in (1,2,3,24,48,168)]+[f'mean{h}' for h in (3,6,24,168)]+['std24']+[f'clock{d}' for d in (1,2,7,14)]+['clock_profile','weather_run_age']
QKEYS=('low','middle','high')

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
    x['weather_run_age']=np.nan
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
        baseline=np.where(np.isfinite(x.clock_profile.to_numpy()),x.clock_profile.to_numpy(),learned)
        q['point']=np.maximum(0,learned if weight==0 else weight*baseline+(1-weight)*learned)
        return q

def next_charge_start(origin,settings):
    local=origin.tz_convert(settings.timezone);starts=[]
    for days in range(5):
        day=local.date()+pd.Timedelta(days=days)
        for w in settings.charging_windows:
            t=pd.Timestamp(f'{day} {w.start}').tz_localize(settings.timezone,ambiguous=False,nonexistent='shift_forward').tz_convert('UTC')
            if t>origin:starts.append(t)
    return min(starts) if starts else None

def train_load(settings,summary,d,progress):
    from .study import run
    return run(settings,summary,d,progress)


def legacy_feature_rows(frame,case,settings):
    """Original lag semantics, with the same settled information as new models."""
    origin=case['origin'];hours=case['hours'];attrs=frame.attrs.copy()
    past=frame.loc[(frame.index<origin.floor('h'))&(frame.index>=origin-pd.Timedelta(days=58))].copy()
    past['context_target']=past.get('context_target',past.target)
    past.loc[past.index+pd.Timedelta(hours=1)>origin-pd.Timedelta(hours=SETTLEMENT_HOURS),'context_target']=np.nan
    future=frame.reindex(hours).copy();future[FEATURES]=case['x'][FEATURES].to_numpy();future['target']=np.nan;future['context_target']=np.nan
    past.attrs={};future.attrs={};safe=pd.concat([past,future]);safe.attrs=attrs
    return feature_rows(safe,origin.floor('h'),hours,settings.timezone,legacy=True)

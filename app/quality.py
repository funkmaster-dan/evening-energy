"""Reproducible sensor cleaning; measured values are never interpolated here."""
import numpy as np
import pandas as pd
import pvlib
from .config import read_json

QUALITY_POLICY='sensor-quality-v2'
LOAD_PIPELINE='lightgbm-issued-energy-v3'
LEGACY_LOAD_PIPELINE='lightgbm-quantile-direct-v2'

def dark_hours(settings,index):
    index=pd.DatetimeIndex(index)
    return np.max([pvlib.solarposition.get_solarposition(index+pd.Timedelta(minutes=m),settings.latitude,settings.longitude).apparent_elevation.to_numpy() for m in (0,15,30,45,60)],axis=0)<=0

def physical(values,kind='load',limit=40,noise=.005):
    raw=pd.to_numeric(values,errors='coerce').astype(float)
    flags=pd.Series('',index=raw.index)
    flags.loc[~np.isfinite(raw)]='missing'
    invalid=(raw.abs()>limit) if kind=='battery_power' else ((raw < -noise)|(raw>limit))
    if kind=='soc': invalid=(raw<0)|(raw>100)
    flags.loc[invalid]='physical_limit'
    small=(raw<0)&(raw>=-noise) if kind in ('load','solar') else pd.Series(False,index=raw.index)
    flags.loc[small]='noise_normalized'
    return raw.mask(invalid|~np.isfinite(raw)).mask(small,0),flags

def resolution(values):
    differences=np.abs(np.diff(values.dropna().to_numpy()))
    differences=differences[differences>1e-9]
    return float(np.quantile(differences,.05)) if len(differences) else 1e-6

def constant_reading_runs(values,minmax=None,cutoff=None):
    """Shared unchanged-reading segments; real variations break the run.

    Only floating-point precision is ignored. Retained min/max variation also
    disproves a stuck hourly mean.
    """
    epsilon=1e-9
    cadence=values.index.to_series().diff().dropna().median() if len(values)>1 else pd.Timedelta(hours=1)
    bounds=minmax.reindex(values.index) if minmax is not None else None
    left=None;low=high=0.
    for i,(time,value) in enumerate(values.items()):
        varying=bounds is not None and pd.notna(bounds['min'].iloc[i]) and pd.notna(bounds['max'].iloc[i]) and bounds['max'].iloc[i]-bounds['min'].iloc[i]>epsilon
        broken=not np.isfinite(value) or varying
        boundary=left is not None and cutoff is not None and values.index[i-1]+cadence<=pd.Timestamp(cutoff)<time+cadence
        outside=left is not None and (boundary or time-values.index[i-1]>cadence*1.5 or max(high,value)-min(low,value)>epsilon)
        if left is not None and (broken or outside):
            part=values.iloc[left:i];yield part,(part.index[-1]-part.index[0]+cadence).total_seconds()/3600
            left=None
        if not broken:
            if left is None:left=i;low=high=value
            else:low=min(low,value);high=max(high,value)
    if left is not None:
        part=values.iloc[left:];yield part,(part.index[-1]-part.index[0]+cadence).total_seconds()/3600


def clean_sensor(values,kind='load',limit=40,settings=None,minmax=None,cutoff=None):
    target,flags=physical(values,kind,limit)
    runs=[];retained=[];previous=[]
    dark_all=pd.Series(dark_hours(settings,target.index),index=target.index) if kind=='solar' and settings is not None else None
    for part,dt in constant_reading_runs(target,minmax,cutoff):
        start=part.index[0];level=float(part.median())
        if dt<=1:
            previous.append((start,dt));continue
        reference=target.loc[(target.index<start)&(target.index>=start-pd.Timedelta(days=56))].dropna()
        if cutoff is not None:reference=reference.loc[reference.index+pd.Timedelta(hours=1)<=pd.Timestamp(cutoff)]
        res=resolution(reference)
        normal=[duration for time,duration in previous if time>=start-pd.Timedelta(days=56) and (cutoff is None or time+pd.Timedelta(hours=duration)<=pd.Timestamp(cutoff))]
        threshold=max(12,3*float(np.quantile(normal,.99))) if len(normal)>=24 else float('inf')
        fault=None;evidence='';known=part.index[-1]+pd.Timedelta(hours=1)
        if kind=='load':
            if dt>=24:
                fault='frozen_power_run';known=start+pd.Timedelta(hours=24);evidence='Power is unchanged for at least 24 hours; no retained within-interval variation disproves the plateau'
            elif dt>threshold and len(reference)>=24*7:
                comparable=reference.loc[(reference-level).abs()<=max(res*10,float(reference.quantile(.25)))]
                if len(comparable)>=24 and comparable.quantile(.95)-comparable.quantile(.05)>3*res:
                    fault='implausible_flat_load';known=start+pd.Timedelta(hours=int(np.floor(threshold))+1);evidence='Unchanged power exceeds the prior normal repeat duration'
        if kind=='solar' and settings is not None:
            dark=dark_all.reindex(part.index).to_numpy()
            if dark.any() and level>.005:
                fault='frozen_solar_run';known=part.index[dark][0]+pd.Timedelta(hours=1)
                evidence='Unchanged positive solar power continues into full darkness; min/max are not required when no variation contradicts the freeze'
            elif dt>=24 and level<=.005 and (~dark).any():
                fault='frozen_solar_run';known=start+pd.Timedelta(hours=24);evidence='Unchanged near-zero production persists for at least 24 hours including daylight; excluded as a stuck reading or production outage'
            elif dt>=6:
                retained.append({'start':start.isoformat(),'hours':dt,'reason':'Normal night zeros or daytime clipping/curtailment remain valid'})
        if fault:
            target.loc[part.index]=np.nan;flags.loc[part.index]=fault
            runs.append({'start':start.isoformat(),'end':(part.index[-1]+pd.Timedelta(hours=1)).isoformat(),'hours':dt,'reason':fault,'observable_at':known.isoformat(),'suspect_recorded_kwh':float(part.sum()),'evidence':evidence})
        else:previous.append((start,dt))
    if dark_all is not None:
        night_bad=dark_all&(target>.005)
        target.loc[night_bad]=np.nan;flags.loc[night_bad]='positive_solar_in_darkness'
    if kind=='load':
        for hour in range(24):
            part=target.loc[target.index.tz_convert(settings.timezone if settings else 'UTC').hour==hour]
            past=part.shift().rolling('56D',min_periods=14)
            median=past.median();mad=past.apply(lambda x:np.median(np.abs(x-np.median(x))),raw=True)
            candidate=part>median+np.maximum(8*1.4826*mad,1)
            ix=candidate.index[candidate];flags.loc[ix]=flags.loc[ix].where(flags.loc[ix]!='','spike_review')
    return target,flags,{'flat_runs':runs,'valid_plateaus':retained}


def background_reference(history,timezone):
    history=history.dropna(); blocks=[]
    local=history.index.tz_convert(timezone)
    for day,night in history.loc[(local.hour<6)].groupby(local[(local.hour<6)].date):
        duration=(pd.Timestamp(f'{day} 06:00',tz=timezone)-pd.Timestamp(day,tz=timezone)).total_seconds()/3600
        if len(night)<int(np.ceil(.75*duration)):continue
        choices=[]
        for i in range(max(0,len(night)-2)):
            part=night.iloc[i:i+3]
            if len(part)==3 and part.index[-1]-part.index[0]==pd.Timedelta(hours=2):
                end=(part.index[-1]+pd.Timedelta(hours=1)).tz_convert(timezone)
                if end.hour<6 or (end.hour==6 and end.minute==0):choices.append(part)
        if choices: blocks.append(min(choices,key=lambda p:p.mean()))
    if len(blocks)<14: return None
    blocks=sorted(blocks,key=lambda p:p.mean()); chosen=blocks[:max(1,int(np.ceil(len(blocks)/4)))]; values=np.concatenate([p.to_numpy() for p in chosen])
    b=float(np.median([p.mean() for p in chosen])); spread=1.4826*float(np.median(np.abs(values-np.median(values))))
    if b<=0 or np.quantile(values,.95)>max(b*3,b+spread*6): return None
    return {'B':b,'U':float(np.quantile(values,.95)),'S':max(spread,resolution(history),.10*b),'nights':len(blocks),'start':history.index.min().isoformat(),'end':history.index.max().isoformat()}

def classify_background(target,settings,cutoff=None,source=None):
    mask=pd.Series(False,index=target.index); days=[]; references=[]
    local=target.index.tz_convert(settings.timezone)
    for date,part in target.groupby(local.date):
        start=pd.Timestamp(date,tz=settings.timezone).tz_convert('UTC'); end=(pd.Timestamp(date,tz=settings.timezone)+pd.DateOffset(days=1)).tz_convert('UTC')
        boundary=min(start,pd.Timestamp(cutoff)) if cutoff is not None else start
        history=target.loc[(target.index+pd.Timedelta(hours=1)<=boundary)&(target.index>=boundary-pd.Timedelta(days=56))]
        if isinstance(source,pd.Series):
            current=source.reindex(part.index).replace('',np.nan).dropna().unique()
            if len(current)!=1:continue
            history=history.loc[source.reindex(history.index)==current[0]]
        ref=background_reference(history,settings.timezone)
        if ref is None: continue
        if not references or references[-1]['reference']!=ref: references.append({'date':str(date),'reference':ref})
        hours=pd.date_range(start.floor('h'),end.ceil('h'),freq='h',inclusive='left')
        full=target.reindex(hours)
        if full.isna().any():continue
        weights=np.array([max(0,(min(t+pd.Timedelta(hours=1),end)-max(t,start)).total_seconds()/3600) for t in hours])
        duration=float(weights.sum());b,u,s=ref['B'],ref['U'],ref['S'];excess=float((full-u).clip(lower=0).to_numpy()@weights);high=full>u+2*s
        qualifies=float(full.to_numpy()@weights)/duration<=b+2*s and full.quantile(.95)<=u+2*s and full.max()<=u+4*s and excess<=2*s*duration and not (high&high.shift(fill_value=False)).any()
        if qualifies:
            # Keep boundary hours on active departure/return days; only whole target intervals inside this day are omitted.
            inside=(part.index>=start)&(part.index+pd.Timedelta(hours=1)<=end)
            mask.loc[part.index[inside]]=True;days.append({'date':str(date),'start':start.isoformat(),'end':end.isoformat(),'hours':duration,'annotated_target_hours':int(inside.sum()),'energy_kwh':float(full.to_numpy()@weights),'variation_kw':float(full.std()),'excess_energy_kwh':excess,'reference':ref,'coverage_pct':100})
    return mask,days,references

def history_availability(values):
    """No prepared values before all required inputs have a recorded reading."""
    observed=values.notna()
    if isinstance(observed,pd.DataFrame):observed=observed.all(axis=1)
    first=observed.index[observed][0] if observed.any() else None
    return pd.Series(values.index>=first,index=values.index) if first is not None else pd.Series(False,index=values.index)


def apply_corrections(raw,target,flags,settings,kind,limit,corrections,original_flags=None):
    """One exclusion/restore policy for sensor history and prepared targets."""
    target=target.copy();flags=flags.copy()
    original_flags=flags.copy() if original_flags is None else original_flags
    for correction in corrections or []:
        left=pd.Timestamp(correction['start']);right=pd.Timestamp(correction['end'])
        if left.tzinfo is None:left=left.tz_localize(settings.timezone)
        if right.tzinfo is None:right=right.tz_localize(settings.timezone)
        selected=(target.index>=left)&(target.index<right)
        if correction.get('action')=='restore':
            valid,_=physical(raw,kind,limit)
            selected &= valid.notna()&original_flags.isin(['','spike_review','noise_normalized','implausible_flat_load','frozen_power_run'])
            target.loc[selected]=valid.loc[selected];flags.loc[selected]='reviewed_restore'
        else:target.loc[selected]=np.nan;flags.loc[selected]='reviewed_exclusion'
    return target,flags


def exclusion_report(frame):
    """Shared counts and representative examples, after all preparation rules."""
    reasons={}
    for reason,part in frame.groupby('quality_flags'):
        if reason and reason not in ('spike_review','noise_normalized','reviewed_restore','valid_replacement_fallback'):
            reasons[reason]={'hours':len(part),'measured_energy_kwh':float(part.raw_target.sum()) if part.raw_target.notna().any() and np.isfinite(part.raw_target.dropna()).all() else None}
    samples=[]
    for reason in reasons:
        for time in frame.index[frame.quality_flags==reason][:3]:
            for at,row in frame.loc[time-pd.Timedelta(hours=1):time+pd.Timedelta(hours=1)].iterrows():
                samples.append({'time':at.isoformat(),'raw':None if not np.isfinite(row.raw_target) else float(row.raw_target),'cleaned':None if not np.isfinite(row.target) else float(row.target),'reason':row.quality_flags or 'valid'})
    return {'excluded_by_reason':reasons,'samples':samples,'usable_hours':int(frame.train_eligible.sum())}


def quality_frame(raw,settings,kind='load',limit=40,source=None,cutoff=None,corrections=None,background=True,minmax=None,precleaned=None):
    raw=raw.copy(deep=False);raw.attrs={}
    if isinstance(source,pd.Series):source=source.copy(deep=False);source.attrs={}
    if precleaned is not None:
        precleaned=tuple(value.copy(deep=False) for value in precleaned)
        for value in precleaned:value.attrs={}
    if precleaned is None:
        clean,flags,details=clean_sensor(raw,kind,limit,settings,cutoff=cutoff,minmax=minmax)
        if isinstance(source,pd.Series) and kind=='load':
            clean,flags=physical(raw,kind,limit);details={'flat_runs':[],'valid_plateaus':[]}
            for entity in source.replace('',np.nan).dropna().unique():
                selected=source==entity
                values=raw.where(selected)
                part,labels,report=clean_sensor(values,kind,limit,settings,cutoff=cutoff,minmax=minmax)
                clean.loc[selected]=part.loc[selected];flags.loc[selected]=labels.loc[selected]
                for key in details:details[key].extend([{**r,'source_entity':entity} for r in report[key]])
    else:
        clean,flags=precleaned
        clean,aggregate_flags=physical(clean,kind,limit)
        flags=flags.copy();flags.loc[aggregate_flags.eq('physical_limit')]='physical_limit'
        details={'flat_runs':[],'valid_plateaus':[]}
    f=pd.DataFrame({'raw_target':raw,'target':clean,'observed':raw.notna(),'quality_flags':flags,'source_entity':source if source is not None else '', 'coverage':clean.notna().astype(float)})
    f['history_available']=history_availability(raw)
    f.loc[~f.history_available,'quality_flags']='before_sensor_history'
    away,days,refs=classify_background(clean,settings,cutoff,source) if kind=='load' and background else (pd.Series(False,index=raw.index),[],[])
    f['original_reason']=f.quality_flags
    f['quiet_regime']=away
    f['regime_observable_at']=[(pd.Timestamp(t.tz_convert(settings.timezone).date(),tz=settings.timezone)+pd.DateOffset(days=1)).tz_convert('UTC') if a else pd.NaT for t,a in zip(f.index,away)]
    f['target'],f['quality_flags']=apply_corrections(f.raw_target,f.target,f.quality_flags,settings,kind,limit,corrections,f.original_reason)
    f['measurement_valid']=f.target.notna()&f.observed;f['normal_use_eligible']=f.measurement_valid&~away
    f['train_eligible']=f.measurement_valid;f['validation_eligible']=f.measurement_valid
    f['derived_method']='';f['sample_weight']=1.;f['derived_left']='';f['derived_right']=''
    details.update(policy=QUALITY_POLICY,corrections=corrections or [],raw_hours=int(f.observed.sum()),expected_hours=len(f),spike_review_hours=int((f.quality_flags=='spike_review').sum()),noise_normalized_hours=int((f.quality_flags=='noise_normalized').sum()),background_days=days,background_references=refs,background_status='Learned from prior reliable nights' if refs else 'Unavailable: insufficient coherent prior nights',missing_energy='Unknown; never zero',evidence_limit='Retained hourly statistics; raw/five-minute history may have expired')
    details.update(exclusion_report(f))
    return f,details

def battery_quality(frame,settings,cutoff=None):
    f=frame.copy();power=settings.battery_power_entity;soc=settings.battery_soc_entity
    f[power],_=physical(f[power],'battery_power',max(settings.battery_charge_limit_kw,settings.battery_discharge_limit_kw)*1.2)
    f[soc],_=physical(f[soc],'soc',100)
    from .config import read_json
    calibration=read_json('model-battery',{})
    sign=calibration.get('power_sign',1)
    if 'power_sign' not in calibration and f[power].std()>0 and f[soc].diff().std()>0 and f[power].corr(f[soc].diff())>0:sign=-1
    for readings,_ in constant_reading_runs(f[soc],cutoff=cutoff):
        part=f.loc[readings.index]
        hours=(part.index[-1]-part.index[0]).total_seconds()/3600
        if hours<4 or part[soc].isna().any() or part[power].isna().any():continue
        dt=part.index.to_series().diff().dt.total_seconds().fillna(0)/3600; net=float((part[power]*sign*dt).sum());level=float(part[soc].iloc[0]); step=max(1.,resolution(f.loc[f.index<part.index[0],soc]))
        if level>=100-step:continue  # A full battery may legitimately remain at 100%.
        enough=abs(net)>max(.05*settings.battery_capacity_kwh,3*step/100*settings.battery_capacity_kwh)
        incompatible=(settings.minimum_soc+step<level<100-step) or (level>=100-step and net>0) or (level<=settings.minimum_soc+step and net<0)
        if enough and incompatible:f.loc[part.index,soc]=np.nan
    # Constant controller power remains valid unless SoC moves against its flow.
    for readings,_ in constant_reading_runs(f[power],cutoff=cutoff):
        part=f.loc[readings.index]
        if len(part)<2 or part[power].isna().any() or part[soc].isna().any():continue
        hours=(part.index[-1]-part.index[0]).total_seconds()/3600
        if hours<4:continue
        energy=float(part[power].iloc[0]*sign*hours);delta=float((part[soc].iloc[-1]-part[soc].iloc[0])*settings.battery_capacity_kwh/100)
        if abs(energy)>.05*settings.battery_capacity_kwh and abs(delta)>.05*settings.battery_capacity_kwh and energy*delta>0:f.loc[part.index,power]=np.nan
    return f


def target_bounds(frame,entities,combine='splice'):
    """Normalize retained sensor min/max; missing evidence stays missing."""
    bounds={label:pd.DataFrame(index=frame.index) for label in ('min','max')}
    for entity in entities:
        info=frame.attrs.get('sensors',{}).get(entity,{})
        stats=info.get('statistics');unit=info.get('unit')
        for label in bounds:
            bounds[label][entity]=pd.to_numeric(stats[label],errors='coerce').reindex(frame.index)/(1000 if unit=='W' else 1) if stats is not None and unit in ('W','kW') else np.nan
    if combine=='sum':return pd.DataFrame({'min':np.nan,'max':np.nan},index=frame.index)
    chosen=frame.reindex(columns=entities).notna().iloc[:,::-1].idxmax(axis=1)
    result=pd.DataFrame(np.nan,index=frame.index,columns=['min','max'])
    for entity in entities:
        mask=chosen==entity
        for label in result:result.loc[mask,label]=bounds[label].loc[mask,entity]
    return result


def load_context_metadata(frame,entities,settings,quality=None,combine='splice',limit=None):
    """Raw physical observations plus when learned faults became knowable."""
    raw=frame.attrs.get('raw',frame).reindex(columns=entities)
    sources=pd.DataFrame(index=frame.index)
    events=[]
    for entity in entities:
        sources[entity],_=physical(raw[entity],'load',limit if limit is not None else settings.load_max_power_kw)
        info=frame.attrs.get('sensors',{}).get(entity,{})
        flags=info.get('quality_flags')
        if flags is not None:sources[entity]=sources[entity].mask(flags.reindex(sources.index).isin(['physical_limit','reviewed_exclusion','mean_outside_minmax']))
        events.extend([{**r,'source_entity':entity} for r in info.get('details',{}).get('flat_runs',[]) if r.get('reason') in ('implausible_flat_load','frozen_power_run')])
    events.extend(r for r in (quality or {}).get('flat_runs',[]) if r.get('reason') in ('implausible_flat_load','frozen_power_run'))
    unique={ (r['start'],r['end'],r.get('source_entity')):r for r in events}
    return {'context_sources':sources,'quality_runs':list(unique.values()),'history_combine':combine,'context_limit':limit if limit is not None else settings.load_max_power_kw}

def flat_run_known_at(event):
    return pd.Timestamp(event['observable_at']) if event.get('observable_at') else pd.Timestamp(event['start'])+pd.Timedelta(hours=int(np.floor(max(12,3*event.get('normal_repeat_p99_hours',0))))+1)


def attach_load_context(frame,metadata):
    """Store a causal default; the shared builder revises known runs per issue."""
    from .datasets import splice
    sources=metadata['context_sources'].reindex(frame.index).copy();combined_mask=pd.Series(False,index=frame.index)
    for event in metadata['quality_runs']:
        entity=event.get('source_entity')
        known=flat_run_known_at(event)
        # At this observation's completion the run became known, not earlier.
        mask=(sources.index+pd.Timedelta(hours=1)>=known)&(sources.index<pd.Timestamp(event['end']))&(sources.index>=pd.Timestamp(event['start']))
        if 'quality_flags' in frame:mask &= frame.quality_flags.ne('reviewed_restore')
        if entity in sources:sources.loc[mask,entity]=np.nan
        else:combined_mask.loc[mask]=True
    context=splice(sources,list(sources),metadata['history_combine'])
    context,_=physical(context,'load',metadata['context_limit']);context=context.mask(combined_mask)
    if 'quality_flags' in frame:context=context.mask(frame.quality_flags.eq('reviewed_exclusion'))
    frame['context_target']=context
    frame.attrs=metadata
    return frame


def effective_policy(settings,kind,request=None):
    request=request or {}
    return {'max_power_kw':float(request.get('max_power_kw',settings.load_max_power_kw if kind=='load' else settings.solar_max_power_kw)), 'corrections':request.get('corrections',[]),'units':request.get('units') or ({request['entities'][0]:request['unit']} if request.get('unit') and len(request.get('entities',[]))==1 else {}),'quality_policy':QUALITY_POLICY,'combine':request.get('combine','splice'),'entities':request.get('entities',[])}


def active_policy(settings,kind,bank_id=None):
    """Preparing another dataset must not change the active model's inputs."""
    name='solar-'+bank_id if kind=='solar' and bank_id else kind
    metadata=read_json('model-'+name,{})
    request=metadata.get('training_policy')
    if request is None:
        request=metadata.get('config_signature',{}).copy()
        if kind=='load':request['corrections']=request.get('quality_corrections',[])
    return effective_policy(settings,kind,request)

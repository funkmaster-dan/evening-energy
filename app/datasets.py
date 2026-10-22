import json
from datetime import datetime
import numpy as np, pandas as pd, pvlib
from .config import DATA, write_json, read_json, solar_banks, solar_configuration
from .sources import HomeAssistant, Weather, Calendar, solar_weather
from .load_splits import inventory, recommendations, training_mask, split_coverage
from .quality import QUALITY_POLICY, LOAD_PIPELINE, quality_frame, dark_hours, physical, battery_quality, target_bounds, load_context_metadata, attach_load_context, effective_policy, history_availability, exclusion_report

FEATURES=['temperature_2m','shortwave_radiation','hour_sin','hour_cos','year_sin','year_cos','weekday','weekend','public_holiday','school_holiday','school_calendar_known']
LEGACY_SOLAR_FEATURES=['temperature_2m','shortwave_radiation','tilted_irradiance','hour_sin','hour_cos','year_sin','year_cos','solar_elevation','solar_azimuth']
SOLAR_FEATURES=LEGACY_SOLAR_FEATURES+['lead_hours','weather_run_age','full_dark']
SOLAR_PIPELINE='open-meteo-issued-energy-v3'
PREVIEW_VERSION=2

def dataset_preview(series):
    # Average every hour in each bucket: a weekly midnight sample hides all solar.
    span=(series.index[-1]-series.index[0]).total_seconds()/3600+1 if len(series) else 0
    hours=max(1,int(np.ceil(span/240)))
    if hours>24: hours=int(np.ceil(hours/24))*24
    groups=series.resample(f'{hours}h',origin=series.index[0]) if len(series) else []
    points=[]
    for time,part in groups:
        valid=part.dropna()
        points.append({'time':str(time),'value':round(float(valid.mean()),3) if len(valid) else None,
                       'available_readings':len(valid),'expected_readings':len(part)})
    return {'preview':points,'preview_hours':hours,'preview_version':PREVIEW_VERSION}

def time_features(frame,settings):
    """Use the same local hour/year encoding for consumption and solar."""
    f=frame.copy();local=f.index.tz_convert(settings.timezone);hour=local.hour+local.minute/60
    f['hour_sin']=np.sin(hour*2*np.pi/24);f['hour_cos']=np.cos(hour*2*np.pi/24)
    f['year_sin']=np.sin(local.dayofyear*2*np.pi/365.25);f['year_cos']=np.cos(local.dayofyear*2*np.pi/365.25)
    return f


def features(weather,settings,calendar):
    f=time_features(weather,settings);local=f.index.tz_convert(settings.timezone)
    f['weekday']=local.dayofweek;f['weekend']=(local.dayofweek>=5).astype(int)
    return f.join(calendar.features(f.index))

def bank_features(f, bank, settings, origin=None):
    # Open-Meteo supplies tilted irradiance. pvlib is used only for sun position.
    position=pvlib.solarposition.get_solarposition(f.index+pd.Timedelta(minutes=30),settings.latitude,settings.longitude)
    x=time_features(f[['temperature_2m','shortwave_radiation','tilted_irradiance']],settings)
    x['solar_elevation']=position.apparent_elevation.to_numpy(); x['solar_azimuth']=position.azimuth.to_numpy()
    x['full_dark']=dark_hours(settings,f.index).astype(float)
    x['lead_hours']=np.asarray((f.index-pd.Timestamp(origin)).total_seconds()/3600) if origin is not None else np.nan
    run=f.attrs.get('run');x['weather_run_age']=(pd.Timestamp(origin)-pd.Timestamp(run)).total_seconds()/3600 if origin is not None and run else np.nan
    for column in f.columns:
        if column.startswith('bank_irradiance_'):x[column]=f[column]
    return x[SOLAR_FEATURES+[c for c in x.columns if c.startswith('bank_irradiance_')]]

def splice(frame, entities, mode='splice'):
    f=frame.reindex(columns=entities);f.attrs={}
    if mode=='sum': return f.sum(axis=1,min_count=len(entities))
    # Last-listed sensor takes priority when replacing a prior sensor.
    return f.iloc[:,::-1].bfill(axis=1).iloc[:,0]

def build_dataset(settings, req, progress):
    kind=req['kind']
    if kind not in ('load','solar','battery'):raise ValueError('Choose consumption, solar production or battery efficiency')
    dataset_id=kind if kind in ('load','battery') else 'solar-'+req['bank_id']
    bank=next((b for b in solar_banks(settings) if b.id==req.get('bank_id')),None) if kind=='solar' else None
    if kind=='solar' and bank is None: raise ValueError('Choose a solar model for the current sensor mode')
    if bank and bank.history_start:
        req={**req,'train_start':max(req['train_start'],bank.history_start)}
    start=pd.Timestamp(req['train_start'],tz=settings.timezone).tz_convert('UTC')
    end=(pd.Timestamp(req['validation_end'],tz=settings.timezone)+pd.DateOffset(days=1)).tz_convert('UTC')
    train_end=(pd.Timestamp(req['train_end'],tz=settings.timezone)+pd.DateOffset(days=1)).tz_convert('UTC')
    val_start=pd.Timestamp(req['validation_start'],tz=settings.timezone).tz_convert('UTC')
    if start>=train_end or val_start>=end or train_end>val_start: raise ValueError('Choose non-overlapping training and validation periods, with training first.')
    if end>pd.Timestamp.now(tz='UTC')+pd.Timedelta(days=1): raise ValueError('Training data must be historical.')
    entities=req.get('entities',[])
    if not entities: raise ValueError('Select at least one relevant Home Assistant sensor.')
    if kind=='battery': entities=list(dict.fromkeys(entities+[settings.battery_power_entity,settings.battery_soc_entity]))
    ha=HomeAssistant(settings)
    context_start=start-pd.Timedelta(days=58) if kind=='load' else start
    if req.get('unit') and len(entities)>1:raise ValueError('Use per-entity units for multiple sources')
    policy=effective_policy(settings,kind,req)
    units=policy['units']
    raw=ha.frame(entities,context_start,end,period='5minute' if kind=='battery' else 'hour',progress=progress,units=units,corrections=req.get('corrections',[]),quality_cutoff=train_end,measurement_kind=kind if kind in ('load','solar') else None,max_power_kw=policy['max_power_kw'])
    if raw.empty: raise ValueError('Home Assistant has no recorded data in that period.')
    step='5min' if kind=='battery' else '1h'
    expected=pd.date_range(context_start.ceil(step),end,freq=step,inclusive='left')
    raw=raw.reindex(expected)
    if kind in ('load','solar'):
        progress('Fetching temperature and irradiance from Open-Meteo')
        if kind=='solar':
            weather=solar_weather(settings,bank,start,end,historical=True)
            f=bank_features(weather,bank,settings).reindex(expected)
        else:
            weather=Weather(settings).fetch(context_start,end,historical=True)
            calendar=Calendar(settings).load()
            f=features(weather,settings,calendar).reindex(expected)
        target=splice(raw,entities,req.get('combine','splice'))
        source=raw.notna().iloc[:,::-1].idxmax(axis=1).where(raw.notna().any(axis=1),'') if req.get('combine','splice')=='splice' else pd.Series('+'.join(entities),index=expected)
        source.attrs={}
        original=raw.attrs.get('raw',raw).reindex(expected)
        raw_target=splice(original,entities,req.get('combine','splice'))
        if req.get('combine','splice')=='splice':
            for entity in entities:
                selected=source==entity;raw_target.loc[selected]=original.loc[selected,entity]
        limit=policy['max_power_kw']
        source_flags=pd.Series('missing',index=expected)
        for entity in entities:
            selected=source.eq(entity)
            if not selected.any():continue
            source_flags.loc[selected]=raw.attrs['sensors'][entity]['quality_flags'].reindex(expected).loc[selected]
        # Where all usable sources fail, retain the preferred raw source and its actual reason.
        if req.get('combine','splice')=='splice':
            preferred=original.notna().iloc[:,::-1].idxmax(axis=1).where(original.notna().any(axis=1),'')
            for entity in entities:
                selected=source.eq('')&preferred.eq(entity)
                source.loc[selected]=entity
                source_flags.loc[selected]=raw.attrs['sensors'][entity]['quality_flags'].reindex(expected).loc[selected]
        else:
            incomplete=target.isna();source_flags.loc[incomplete]='incomplete_sum'
        if req.get('combine','splice')=='splice':
            # An explicit outage/manual exclusion has provenance even with no numeric value.
            for entity in reversed(entities):
                flags=raw.attrs['sensors'][entity]['quality_flags'].reindex(expected)
                selected=source.eq('')&source_flags.eq('missing')&~flags.isin(['','missing'])&flags.notna()
                source_flags.loc[selected]=flags.loc[selected]
        bounds=target_bounds(raw,entities,req.get('combine','splice'))
        quality,details=quality_frame(raw_target,settings,kind,limit,source,train_end,req.get('corrections') if kind=='solar' else None,minmax=bounds,precleaned=(target,source_flags))
        for entity in entities:
            selected=source.eq(entity)
            original_flags=raw.attrs.get('sensors',{}).get(entity,{}).get('original_quality_flags')
            if original_flags is not None:quality.loc[selected,'original_reason']=original_flags.reindex(expected).loc[selected]
        if req.get('combine','splice')=='splice':
            quality['preferred_source_entity']=preferred
            quality['preferred_rejection_reason']=''
            for entity in entities:
                selected=preferred.eq(entity)
                quality.loc[selected,'preferred_rejection_reason']=raw.attrs['sensors'][entity]['quality_flags'].reindex(expected).loc[selected]
            fallback=quality.target.notna()&source.ne(preferred)&preferred.ne('')
            quality.loc[fallback&quality.quality_flags.eq(''),'quality_flags']='valid_replacement_fallback'
            details['replacement_fallbacks']={'hours':int(fallback.sum()),'preferred_rejection_reasons':quality.loc[fallback,'preferred_rejection_reason'].value_counts().to_dict()}
        details['corrections']=req.get('corrections',[])
        details['effective_policy']=policy
        if kind=='solar':
            quality=derive_solar(quality,settings,start,train_end,int(req.get('interpolate_hours',1)),limit)
        # Preserve a full expected grid, including missing context and targets.
        cleaned=f.join(quality)
        cleaned['sensor_min']=bounds['min'];cleaned['sensor_max']=bounds['max']
        cleaned['train_eligible'] &= cleaned[FEATURES if kind=='load' else LEGACY_SOLAR_FEATURES].notna().all(axis=1)
        cleaned['coverage']=(cleaned.observed&cleaned.target.notna()&~cleaned.quality_flags.str.startswith('derived')).astype(float)
        cleaned['weather_available']=cleaned[FEATURES if kind=='load' else LEGACY_SOLAR_FEATURES].notna().all(axis=1)
        cleaned['validation_eligible'] &= cleaned.weather_available
        details['source_history']=source_history(raw,entities,source,start,train_end,val_start,end)
        details['split_coverage']={name:split_coverage(cleaned,left,right,settings,complete=name=='training') for name,left,right in [('training',start,train_end),('validation',val_start,end)]}
        for name,left,right in [('training',start,train_end),('validation',val_start,end)]:
            part=cleaned.loc[training_mask(cleaned.index,left,right) if name=='training' else (cleaned.index+pd.Timedelta(hours=1)>left)&(cleaned.index<right)]
            details['split_coverage'][name]['missing_weather_hours']=int((~part.weather_available).sum())
        if kind=='load':
            context_metadata=load_context_metadata(raw,entities,settings,details,req.get('combine','splice'),limit);attach_load_context(cleaned,context_metadata)
            details['flat_runs']=context_metadata['quality_runs']
            details['valid_plateaus'].extend({**plateau,'source_entity':entity} for entity in entities for plateau in raw.attrs.get('sensors',{}).get(entity,{}).get('details',{}).get('valid_plateaus',[]))
        details.update(exclusion_report(cleaned))
        extra={'filtered_outliers':sum(v['hours'] for k,v in details['excluded_by_reason'].items() if k in ('physical_limit','implausible_flat_load','frozen_power_run','frozen_solar_run')),'filled_hours':int(cleaned.quality_flags.str.startswith('derived').sum()),'weather_source':'Open-Meteo historical weather','quality':details,'quality_policy':QUALITY_POLICY}
        if kind=='solar':
            for entity in entities:
                for key in ('flat_runs','valid_plateaus'):details[key].extend({**r,'source_entity':entity} for r in raw.attrs.get('sensors',{}).get(entity,{}).get('details',{}).get(key,[]))
        extra['quality']['sensor_checks']={e:raw.attrs.get('sensors',{}).get(e,{}).get('details',{}) for e in entities}
        if kind=='solar':extra.update(solar_pipeline=SOLAR_PIPELINE,irradiance_source='Open-Meteo global_tilted_irradiance',features=SOLAR_FEATURES)
        else:extra['calendar']=calendar.summary(expected)
    else:
        cleaned=battery_quality(raw,settings,train_end);cleaned['history_available']=history_availability(raw);extra={'filtered_outliers':0,'filled_hours':0}
        removed=raw.notna()&cleaned.isna()
        extra['quality']={'policy':QUALITY_POLICY,'expected_hours':len(raw),'raw_hours':int(raw.notna().all(axis=1).sum()),'usable_hours':int(cleaned.notna().all(axis=1).sum()),'excluded_by_reason':{'sensor_fault':{'hours':int(removed.any(axis=1).sum()),'measured_energy_kwh':None}},'flat_runs':[],'valid_plateaus':[],'background_days':[],'background_references':[],'background_status':'Background exclusion applies only to consumption','samples':[],'evidence_limit':'Battery power and SoC are never interpolated; full/idle plateaus retained'}
    available=cleaned.history_available
    first_recorded=available.index[available][0] if available.any() else None
    extra['availability']={'first_recorded_at':first_recorded.isoformat() if first_recorded is not None else None,'excluded_leading_hours':float(((cleaned.index>=start)&~available).sum())*(5/60 if kind=='battery' else 1),'requested_start':start.isoformat()}
    train=cleaned[training_mask(cleaned.index,start,train_end,'5min' if kind=='battery' else '1h')]
    valid=cleaned[(cleaned.index+pd.Timedelta(minutes=5 if kind=='battery' else 60)>val_start)&(cleaned.index<end)]
    if 'train_eligible' in cleaned:
        train=train.loc[train.train_eligible];valid=valid.loc[valid.validation_eligible]
    else:
        train=train.dropna();valid=valid.dropna()
    minimum=24*7 if kind in ('load','solar') else 48
    sufficient=len(train)>=minimum and len(valid)>=24
    if kind=='solar':sufficient=sufficient and int(train.measurement_valid.sum())>=100
    manifest={'quality_policy':QUALITY_POLICY,'training_start':start.isoformat(),'training_end':train_end.isoformat(),'final_periods':[{'start':val_start.isoformat(),'end':end.isoformat(),'reason':'Explicit user dates; untouched final test'}]}
    if kind=='load':
        fvalid=cleaned.loc[(cleaned.index>=val_start)&(cleaned.index<end)]
        raw_counts={str(h):int(fvalid.validation_eligible.astype(int).rolling(h).sum().eq(h).sum()) for h in (24,48,72)}
        from .load_training import evaluation_cases
        cases=evaluation_cases(cleaned,val_start,end,settings,progress)
        counts={str(h):sum(len(c['hours'])>=h for c in cases) for h in (24,48,72)}
        manifest['raw_contiguous_target_windows']=raw_counts
        manifest['usable_issue_origins']=[c['origin'].isoformat() for c in cases]
        manifest['distinct_issue_days']={str(h):len({str(c['origin'].tz_convert(settings.timezone).date()) for c in cases if len(c['hours'])>=h}) for h in (24,48,72)}
        manifest['weather_sources']={label:sum(c['archive']==archive for c in cases) for label,archive in [('archived',True),('historical_proxy',False)]}
        manifest['discarded_windows']=cases[-1]['skipped'] if cases else {'reason':'No origin has complete targets, context/clock profile and weather'}
        sufficient=sufficient and (end-val_start)>=pd.Timedelta(days=4) and counts['72']>0 and len(valid)>=96
        manifest['complete_windows']=counts;manifest['eligible_hours']=len(valid);manifest['excluded_hours']=int((~fvalid.validation_eligible|fvalid.target.isna()).sum())
        manifest['recommendation']='Reserve 28–56 recent days, at least ten distinct issue days with complete 72h/overnight windows. Four days is an experiment minimum.'
        manifest['eligible_target_hours']=[t.isoformat() for t in valid.index]
        manifest['coverage_only_inventory']=inventory(cleaned,val_start,end,settings)
        manifest['holdout_recommendation']=recommendations(cleaned,start,end,settings)
        extra['split_manifest']=manifest
    if kind=='solar':
        from .solar_training import solar_cases
        provisional={'frame':cleaned,'start':start,'train_end':train_end,'val_start':val_start,'end':end}
        cases,skipped=solar_cases(settings,bank,provisional,progress)
        counts={str(h):sum(c['valid_window'](c['origin']+pd.Timedelta(hours=h)) is not None for c in cases if c['origin']+pd.Timedelta(hours=h)<=end) for h in (24,48,72)}
        manifest.update(complete_windows=counts,usable_issue_origins=[c['origin'].isoformat() for c in cases],discarded_windows=skipped)
        sufficient=sufficient and any(c['valid_window'](c['daily_end']) is not None for c in cases)
        extra['split_manifest']=manifest
    preview=cleaned.loc[(cleaned.index>=start)&available];series=preview['target'] if 'target' in preview else preview.iloc[:,0]
    summary={'solar_configuration':solar_configuration(settings,bank.id) if bank else None,'id':dataset_id,'preparation_version':2,'kind':kind,'bank_id':req.get('bank_id'),'request':req,'created_at':pd.Timestamp.now(tz='UTC').isoformat(),'rows':len(cleaned),'training_rows':len(train),'validation_rows':len(valid),'coverage_pct':round(100*(len(train)+len(valid))/max(1,len(expected[expected>=start])),1),'sufficient':sufficient,'readiness':'Ready to calibrate' if sufficient else ('LightGBM needs 7 measured training days, 4 validation days and a complete 72-hour window with prior context' if kind=='load' else 'Need measured solar training targets, a complete measured daylight validation day and archived weather' if kind=='solar' else f'Need at least {minimum} training rows and 24 validation rows'),'recommendation':'Use one or several comparable years for seasonal learning and 28–56 final validation days; shorter periods are experiments.','sources':entities,**extra,**dataset_preview(series)}
    pd.to_pickle({'frame':cleaned,'start':start,'train_end':train_end,'val_start':val_start,'end':end,'entities':entities,'split_manifest':manifest,'source_metadata':raw.attrs}, DATA/(dataset_id+'.pkl'))
    write_json('dataset-'+dataset_id,summary)
    progress(f'Dataset built: {len(train)} training rows, {len(valid)} validation rows')
    return summary

def list_datasets(settings=None):
    summaries=[]
    for path in sorted(DATA.glob('dataset-*.json')):
        summary=json.loads(path.read_text())
        if summary.get('kind')=='inverter':continue
        frame_path=DATA/(summary['id']+'.pkl')
        if summary.get('preview_version')!=PREVIEW_VERSION and frame_path.exists():
            saved=pd.read_pickle(frame_path)
            frame=saved['frame']
            # Older datasets may include preceding context for consumption.
            start=saved.get('start')
            if start is not None: frame=frame.loc[frame.index>=start]
            if 'history_available' in frame:frame=frame.loc[frame.history_available]
            series=frame['target'] if 'target' in frame else frame.iloc[:,0]
            summary.update(dataset_preview(series))
            write_json(path.stem,summary)
        if settings is not None and summary.get('kind')=='solar':
            try:matches=summary.get('solar_configuration')==solar_configuration(settings,summary.get('bank_id'))
            except ValueError:matches=False
            if not matches:summary={**summary,'sufficient':False,'readiness':'Solar settings have changed or this dataset predates configuration checks. Prepare data again.'}
        summaries.append(summary)
    return summaries


def derive_solar(quality,settings,start,train_end,fill,limit):
    q=quality.copy();dark=pd.Series(dark_hours(settings,q.index),index=q.index)
    ordinary=q.raw_target.isna()&q.original_reason.isin(['','missing'])
    night=q.raw_target.isna()&dark&q.history_available&q.original_reason.isin(['','missing','raw_history_unavailable'])
    q.loc[night,'target']=0.;q.loc[night,'quality_flags']='derived_dark_zero';q.loc[night,'derived_method']='dark_zero'
    q.loc[night,'train_eligible']=True;q.loc[night,'validation_eligible']=False
    # Only one isolated original gap; endpoints must both be measured and same-source.
    if fill:
        inside=pd.Series(training_mask(q.index,start,train_end),index=q.index)
        measured=q.measurement_valid&inside
        mask=ordinary&~dark&inside&measured.shift(fill_value=False)&measured.shift(-1,fill_value=False)
        mask &= q.source_entity.shift().eq(q.source_entity.shift(-1))&q.source_entity.shift().ne('')
        candidate=(q.target.shift()+q.target.shift(-1))/2
        checked,flags=physical(candidate,'solar',limit)
        mask &= checked.notna()&~dark
        q.loc[mask,'target']=checked.loc[mask];q.loc[mask,'quality_flags']='derived_interpolation';q.loc[mask,'derived_method']='interpolation';q.loc[mask,'sample_weight']=.25
        q.loc[mask,'source_entity']=q.source_entity.shift().loc[mask]
        q.loc[mask,'derived_left']=[(t-pd.Timedelta(hours=1)).isoformat() for t in q.index[mask]]
        q.loc[mask,'derived_right']=[(t+pd.Timedelta(hours=1)).isoformat() for t in q.index[mask]]
        q.loc[mask,'train_eligible']=True;q.loc[mask,'validation_eligible']=False
    q['observed']=q.raw_target.notna()
    return q


def source_history(raw,entities,source,start,train_end,val_start,end):
    metadata=raw.attrs
    raw=raw.copy(deep=False);raw.attrs={}
    source=source.copy(deep=False);source.attrs={}
    rows=[];transitions=[];previous=None
    for entity in entities:
        info=metadata.get('sensors',{}).get(entity,{})
        entry={'entity':entity,'unit':info.get('normalized_unit',info.get('unit')),'stored_unit':info.get('stored_unit'),'basis':info.get('unit_basis'),'source_unit':info.get('source_unit'),'measurement_basis':'Interval average power (kW); cumulative energy uses endpoint delta','splits':{}}
        for name,left,right in [('training',start,train_end),('validation',val_start,end)]:
            part=raw[entity].loc[training_mask(raw.index,left,right) if name=='training' else (raw.index+pd.Timedelta(hours=1)>left)&(raw.index<right)].dropna();chosen=part.loc[source.reindex(part.index).eq(entity)]
            gaps=[];expected=pd.date_range(pd.Timestamp(left).ceil('h'),right,freq='h',inclusive='left');missing=raw[entity].reindex(expected).isna()
            for _,group in missing.groupby(missing.ne(missing.shift()).cumsum()):
                if group.iloc[0]:gaps.append({'start':group.index[0].isoformat(),'end':(group.index[-1]+pd.Timedelta(hours=1)).isoformat(),'hours':len(group)})
            entry['splits'][name]={'first_valid':part.index[0].isoformat() if len(part) else None,'last_valid':part.index[-1].isoformat() if len(part) else None,'valid_hours':len(part),'selected_hours':len(chosen),'gaps':gaps}
        rows.append(entry)
    selected=source.loc[(source.index>=start)&source.ne('')]
    for time,entity in selected.items():
        if previous and previous[1]!=entity:transitions.append({'from':previous[1],'to':entity,'last_previous':previous[0].isoformat(),'first_replacement':time.isoformat(),'gap_hours':max(0,(time-previous[0]).total_seconds()/3600-1),'annotation':'Ordered hardware replacement; no transfer multiplier inferred'})
        previous=(time,entity)
    return {'sources':rows,'transitions':transitions,'order':entities,'combination':'Last usable replacement takes priority; gaps are never bridged'}

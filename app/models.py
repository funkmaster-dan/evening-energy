"""Direct load/solar models and measured battery/inverter calibration."""
import warnings
warnings.filterwarnings('ignore',message='X does not have valid feature names')
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import DATA, read_json, write_json
from .datasets import SOLAR_PIPELINE
from .load_splits import training_mask
from .quality import LOAD_PIPELINE, QUALITY_POLICY, effective_policy

def metrics(y,p):
    return {'mae':round(float(mean_absolute_error(y,p)),4),'rmse':round(float(np.sqrt(mean_squared_error(y,p))),4),'r2':round(float(r2_score(y,p)),4) if len(y)>1 else 0,'samples':len(y)}

def validation_chart(index,y,p):
    step=max(1,len(y)//240)
    return [{'time':str(index[i]),'actual':round(float(y[i]),3),'predicted':round(float(p[i]),3)} for i in range(0,len(y),step)]

def train(settings,dataset_id,progress):
    summary=read_json('dataset-'+dataset_id)
    if not summary or not summary['sufficient']: raise ValueError('Build a sufficient dataset first.')
    if summary.get('quality',{}).get('policy')!=QUALITY_POLICY:raise ValueError('Prepare this dataset again for the current cleaning policy.')
    if read_json('candidate-'+dataset_id): raise ValueError('Promote or delete the existing candidate before calibrating again.')
    d=pd.read_pickle(DATA/(dataset_id+'.pkl')); f=d['frame']; kind=summary['kind']
    metadata={'trained_at':pd.Timestamp.now(tz='UTC').isoformat(),'dataset_id':dataset_id,'training_rows':summary['training_rows'],'validation_rows':summary['validation_rows'],'kind':kind,'quality_policy':QUALITY_POLICY,'bank_id':summary.get('bank_id'),'dataset_created_at':summary['created_at'],'config_signature':signature(settings,kind,summary.get('bank_id'))}
    if kind=='load':
        from .load_training import train_load
        metadata=train_load(settings,summary,d,progress)
        write_json('candidate-'+dataset_id,metadata)
        return metadata
    elif kind=='solar':
        from .solar_training import train_solar
        metadata=train_solar(settings,summary,d,metadata,progress)
        write_json('candidate-'+dataset_id,metadata)
        return metadata
    elif kind=='battery':
        power=f[settings.battery_power_entity]; soc=f[settings.battery_soc_entity]
        # Positive delta SoC means charging. Align preceding-interval mean power with SoC delta.
        dt=f.index.to_series().diff().dt.total_seconds()/3600
        de=soc.diff()*settings.battery_capacity_kwh/100
        energy=(power+power.shift(1))/2*dt
        usable=(dt>0)&(dt<=0.12)&(soc>5)&(soc<97)&(soc.shift(1)>5)&(soc.shift(1)<97)&(energy.abs()>.025)&(de.abs()>.01)
        tr=training_mask(f.index,d['start'],d['train_end'],'5min')&usable; va=(f.index>=d['val_start'])&usable
        if tr.sum()<24: raise ValueError('Not enough changing SoC and power samples to calibrate battery efficiency.')
        correlation=float(energy.loc[tr].corr(de.loc[tr])); sign=1 if correlation<0 else -1
        # Normalize positive power to discharge.
        e=energy*sign
        charge=tr&(e<0)&(de>0); discharge=tr&(e>0)&(de<0)
        if charge.sum()<10 or discharge.sum()<10: raise ValueError('Need both charge and discharge history to calibrate efficiency.')
        charge_eta=float(de[charge].sum()/(-e[charge]).sum()); discharge_eta=float(e[discharge].sum()/(-de[discharge]).sum())
        if not (.5<=charge_eta<=1.15 and .5<=discharge_eta<=1.15): raise ValueError(f'Efficiency does not match measured SoC: charge {charge_eta:.2f}, discharge {discharge_eta:.2f}. Check the power and SoC sensors.')
        charge_eta=min(.995,charge_eta); discharge_eta=min(.995,discharge_eta)
        predicted=np.where(e<0,-e*charge_eta,-e/discharge_eta)
        actual=de.loc[va].to_numpy(); pred=predicted[va]
        calibration={'charge_efficiency':charge_eta,'discharge_efficiency':discharge_eta,'power_sign':sign,'observed_charge_kw':round(float((-power.loc[tr]*sign).clip(lower=0).quantile(.995)),2),'observed_discharge_kw':round(float((power.loc[tr]*sign).clip(lower=0).quantile(.995)),2),'charge_samples':int(charge.sum()),'discharge_samples':int(discharge.sum()),'correlation':round(correlation,3)}
        metadata.update(model='Measured power / SoC calibration',**calibration,metrics=metrics(actual,pred) if len(actual) else {},chart=validation_chart(f.index[va],actual,pred),metric_unit='kWh change')
        baseline=None
        if model_ready(settings,dataset_id,kind):
            current=read_json('model-'+dataset_id); be=energy*current['power_sign']
            baseline=np.where(be<0,-be*current['charge_efficiency'],-be/current['discharge_efficiency'])[va]
        if not len(actual): raise ValueError('Need usable battery validation samples')
        y=de
    else:
        total=f[[b.entity for b in settings.banks]].sum(axis=1,min_count=len(settings.banks)); ct=f[settings.solar_ct_entity]
        usable=(total>.5)&(ct>=0)&(ct/total>.3)&(ct/total<1.2)
        tr=training_mask(f.index,d['start'],d['train_end'])&usable; va=(f.index>=d['val_start'])&usable
        if tr.sum()<24: raise ValueError('Need at least 24 simultaneous daytime inverter and CT readings.')
        eta=float(np.clip((ct[tr]/total[tr]).median(),.3,1.0)); pred=total[va].to_numpy()*eta
        metadata.update(model='Inverter / CT efficiency calibration',efficiency=eta,metrics=metrics(ct[va],pred) if va.sum() else {},chart=validation_chart(f.index[va],ct[va].to_numpy(),pred))
        baseline=None
        if model_ready(settings,dataset_id,kind): baseline=total[va].to_numpy()*read_json('model-'+dataset_id)['efficiency']
        if not va.sum(): raise ValueError('Need usable inverter validation samples')
        y=ct
    metadata['validation_rows']=int(va.sum())
    metadata['validation_start']=d['val_start'].isoformat(); metadata['validation_end']=d['end'].isoformat()
    metadata['comparison']={'metrics':metrics(y.loc[va],baseline),'chart':validation_chart(f.index[va],y.loc[va].to_numpy(),baseline)} if baseline is not None else None
    write_json('candidate-'+dataset_id,metadata)
    return metadata

def signature(settings,kind,bank_id=None):
    policy=effective_policy(settings,kind) if kind in ('load','solar') else {'quality_policy':QUALITY_POLICY}
    if kind=='solar':policy=effective_policy(settings,kind,read_json('dataset-solar-'+bank_id,{}).get('request',{}))
    if kind=='load': return {**policy,'load_pipeline':LOAD_PIPELINE,'quality_policy':QUALITY_POLICY,'weather_model':settings.weather_model,'load_max_power_kw':settings.load_max_power_kw,'quality_corrections':read_json('dataset-load',{}).get('request',{}).get('corrections',[]),'location':[settings.latitude,settings.longitude],'timezone':settings.timezone,'state':settings.state,'school_calendar_url':settings.school_calendar_url,'entities':settings.historical_consumption_entities}
    if kind=='solar':
        b=next(b for b in settings.banks if b.id==bank_id)
        return {**policy,'location':[settings.latitude,settings.longitude],'timezone':settings.timezone,'bank':b.model_dump(),'solar_pipeline':SOLAR_PIPELINE,'weather_model':settings.weather_model}
    if kind=='battery': return {**policy,'capacity':settings.battery_capacity_kwh,'power':settings.battery_power_entity,'soc':settings.battery_soc_entity}
    return {**policy,'banks':[b.entity for b in settings.banks],'ct':settings.solar_ct_entity}

def model_ready(settings,name,kind,bank_id=None):
    meta=read_json('model-'+name)
    legacy=meta and ((kind=='solar' and meta.get('solar_pipeline')=='open-meteo-gti-angles-quantile-v2') or (kind=='load' and meta.get('pipeline')=='lightgbm-quantile-direct-v2'))
    if legacy:
        old=meta.get('config_signature',{}).copy();current=signature(settings,kind,bank_id).copy()
        old.pop('solar_pipeline' if kind=='solar' else 'load_pipeline',None);current.pop('solar_pipeline' if kind=='solar' else 'load_pipeline',None)
        return old==current and all(p.exists() for p in artifact_paths(name))
    return bool(meta and (kind!='load' or meta.get('pipeline')==LOAD_PIPELINE) and meta.get('config_signature')==signature(settings,kind,bank_id) and all(p.exists() for p in artifact_paths(name)))

def list_models(): return [read_json(p.stem) for p in sorted(DATA.glob('model-*.json'))]

def list_candidates(): return [read_json(p.stem) for p in sorted(DATA.glob('candidate-*.json'))]

def artifact_paths(dataset_id,candidate=False):
    name=dataset_id+('-candidate' if candidate else '')
    if dataset_id=='load': return [DATA/(name+'.joblib')]
    if dataset_id.startswith('solar-'): return [DATA/(name+'.joblib')]
    return []


def promote_candidate(settings,dataset_id):
    metadata=read_json('candidate-'+dataset_id)
    if not metadata: raise ValueError('Candidate not found')
    if dataset_id=='load' and metadata.get('pipeline')!=LOAD_PIPELINE: raise ValueError('Rebuild and calibrate the load dataset first')
    if metadata['config_signature']!=signature(settings,metadata['kind'],metadata.get('bank_id')):
        raise ValueError('Configuration has changed. Delete this candidate and calibrate again.')
    sources=artifact_paths(dataset_id,True); targets=artifact_paths(dataset_id)
    if any(not p.exists() for p in sources): raise ValueError('Candidate artifact is missing. Delete it and calibrate again.')
    if sources:
        import joblib
        artifact=joblib.load(sources[0]);models=artifact.get('models',artifact)
        if set(models)!= {'low','middle','high'}:raise ValueError('Candidate lacks quantile heads')
        d=pd.read_pickle(DATA/(dataset_id+'.pkl'));frame=d['frame']
        if dataset_id=='load':
            from .load_training import evaluation_cases,LoadPredictor
            cases=evaluation_cases(frame,d['val_start'],d['end'],settings,issued_weather=False)
            if not cases:raise ValueError('No operational load validation case')
            predicted=LoadPredictor(artifact).predict(cases[0]['x'])
        else:
            from .solar_training import solar_predict
            from .datasets import SOLAR_FEATURES
            predicted=solar_predict(artifact,frame.dropna(subset=[c for c in SOLAR_FEATURES if c not in ('weather_run_age','lead_hours')]).iloc[-24:])
        if any(not np.isfinite(values).all() for values in predicted.values()):raise ValueError('Candidate predictions must be finite')
    for source,target in zip(sources,targets): source.replace(target)
    # Keep paired validation evidence with the promoted model.
    write_json('model-'+dataset_id,metadata)
    (DATA/('candidate-'+dataset_id+'.json')).unlink()
    return metadata


def delete_candidate(dataset_id):
    if not read_json('candidate-'+dataset_id): raise ValueError('Candidate not found')
    for path in [*artifact_paths(dataset_id,True),DATA/('candidate-'+dataset_id+'.json')]: path.unlink(missing_ok=True)


def migrate_uncertainty(settings):
    """Preserve incompatible artifacts and requests before a policy upgrade.

    Readiness uses signatures; old models remain available for rollback and paired
    private evaluation until a reviewed candidate replaces them.
    """
    import tarfile,sqlite3,tempfile
    marker='preparation-migration-'+QUALITY_POLICY
    if read_json(marker):return
    paths=[p for pattern in ('model-*.json','candidate-*.json','*.joblib','dataset-*.json','config.json') for p in DATA.glob(pattern)]
    if paths:
        backup_dir=DATA/'backups';backup_dir.mkdir(exist_ok=True)
        backup=backup_dir/(marker+'-'+pd.Timestamp.now(tz='UTC').strftime('%Y%m%d-%H%M%S-%f')+'.tar.gz')
        with tarfile.open(backup,'w:gz') as bundle:
            for path in sorted(set(paths)):bundle.add(path,arcname=path.name)
            if (DATA/'history.sqlite').exists():
                with tempfile.TemporaryDirectory() as temp:
                    copy=__import__('pathlib').Path(temp)/'history.sqlite'
                    with sqlite3.connect(DATA/'history.sqlite') as source,sqlite3.connect(copy) as destination:source.backup(destination)
                    bundle.add(copy,arcname='history.sqlite')
        backup.chmod(0o600)
    write_json(marker,{'completed_at':pd.Timestamp.now(tz='UTC').isoformat(),'policy':QUALITY_POLICY})


def clean_model_storage(settings):
    """Keep active models and a reviewable candidate for configured models."""
    names={'load','battery','inverter',*('solar-'+b.id for b in settings.banks)}
    keep=set()
    for name in names:
        for prefix,candidate in [('model-',False),('candidate-',True)]:
            metadata=DATA/(prefix+name+'.json')
            if metadata.exists():
                keep.add(metadata); keep.update(artifact_paths(name,candidate))
    for pattern in ('model-*.json','candidate-*.json','*.pt','*.joblib'):
        for path in DATA.glob(pattern):
            if path not in keep: path.unlink()
    for name in ('battery-calibration','inverter-calibration','load-uncertainty','fine-tuning-status','fine-tune-schedule','direct-trial','weather-comparison','forecast-measurement'):
        (DATA/(name+'.json')).unlink(missing_ok=True)
    (DATA/'load-history.pkl').unlink(missing_ok=True)

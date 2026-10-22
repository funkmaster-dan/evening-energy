"""Direct load/solar models and measured battery calibration."""
import warnings
warnings.filterwarnings('ignore',message='X does not have valid feature names')
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import DATA, read_json, write_json, solar_banks, solar_configuration
from .load_splits import training_mask
from .quality import QUALITY_POLICY

def metrics(y,p):
    return {'mae':round(float(mean_absolute_error(y,p)),4),'rmse':round(float(np.sqrt(mean_squared_error(y,p))),4),'r2':round(float(r2_score(y,p)),4) if len(y)>1 else 0,'samples':len(y)}

def validation_chart(index,y,p):
    step=max(1,len(y)//240)
    return [{'time':str(index[i]),'actual':round(float(y[i]),3),'predicted':round(float(p[i]),3)} for i in range(0,len(y),step)]

def train(settings,dataset_id,progress):
    if dataset_id=='inverter':raise ValueError('Inverter calibration has been removed')
    summary=read_json('dataset-'+dataset_id)
    if not summary or not summary['sufficient']: raise ValueError('Build a sufficient dataset first.')
    if summary.get('quality',{}).get('policy')!=QUALITY_POLICY:raise ValueError('Prepare this dataset again for the current cleaning policy.')
    if summary.get('model_settings') is not None and summary['model_settings']!=signature(settings,summary['kind'],summary.get('bank_id')):
        raise ValueError('Model inputs have changed. Prepare data again.')
    if read_json('candidate-'+dataset_id): raise ValueError('Promote or delete the existing candidate before calibrating again.')
    if summary['kind']=='solar' and summary.get('bank_id') not in {b.id for b in solar_banks(settings)}: raise ValueError('Prepare a solar dataset for the current sensor mode')
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
    else:raise ValueError('Unsupported model type')
    metadata['validation_rows']=int(va.sum())
    metadata['validation_start']=d['val_start'].isoformat(); metadata['validation_end']=d['end'].isoformat()
    metadata['comparison']={'metrics':metrics(y.loc[va],baseline),'chart':validation_chart(f.index[va],y.loc[va].to_numpy(),baseline)} if baseline is not None else None
    write_json('candidate-'+dataset_id,metadata)
    return metadata

def combined_configuration(settings):
    return solar_configuration(settings,'__combined')


def signature(settings,kind,bank_id=None):
    context={'location':[settings.latitude,settings.longitude],'timezone':settings.timezone,'weather_model':settings.weather_model}
    if kind=='load': return {**context,'load_max_power_kw':settings.load_max_power_kw,'state':settings.state,'school_calendar_url':settings.school_calendar_url,'entities':settings.historical_consumption_entities}
    if kind=='solar':
        b=next(b for b in solar_banks(settings) if b.id==bank_id)
        geometry={'combined':combined_configuration(settings),'mode':'combined-ac-v1'} if bank_id=='__combined' else {'bank':b.model_dump(exclude={'panel_count','name'})}
        return comparable_settings({**context,**geometry},kind)
    if kind=='battery': return {'capacity':settings.battery_capacity_kwh,'power':settings.battery_power_entity,'soc':settings.battery_soc_entity}
    raise ValueError('Unsupported model type')

def comparable_settings(saved,kind):
    # Older models mixed dataset options and software labels into this snapshot.
    saved={k:v for k,v in saved.items() if k not in {'quality_policy','load_pipeline','solar_pipeline','max_power_kw','corrections','quality_corrections','units','combine'}}
    if kind=='solar':
        saved.pop('entities',None)
        if 'bank' in saved:saved['bank']={k:v for k,v in saved['bank'].items() if k not in {'name','panel_count'}}
        if 'combined' in saved:saved['combined']={k:v for k,v in saved['combined'].items() if k!='version'}
    return saved

def settings_match(settings,metadata):
    try:return comparable_settings(metadata.get('config_signature',{}),metadata['kind'])==signature(settings,metadata['kind'],metadata.get('bank_id'))
    except (ValueError,StopIteration):return False

def model_ready(settings,name,kind,bank_id=None):
    if kind not in ('load','solar','battery'):return False
    if kind=='solar' and bank_id not in {b.id for b in solar_banks(settings)}: return False
    meta=read_json('model-'+name)
    return bool(meta and settings_match(settings,meta) and all(p.exists() for p in artifact_paths(name)))

def list_models(): return [read_json(p.stem) for p in sorted(DATA.glob('model-*.json')) if p.stem!='model-inverter']

def list_candidates(): return [read_json(p.stem) for p in sorted(DATA.glob('candidate-*.json')) if p.stem!='candidate-inverter']

def artifact_paths(dataset_id,candidate=False):
    name=dataset_id+('-candidate' if candidate else '')
    if dataset_id=='load': return [DATA/(name+'.joblib')]
    if dataset_id.startswith('solar-'): return [DATA/(name+'.joblib')]
    return []


def promote_candidate(settings,dataset_id):
    metadata=read_json('candidate-'+dataset_id)
    if not metadata: raise ValueError('Candidate not found')
    if metadata.get('kind') not in ('load','solar','battery'):raise ValueError('Unsupported model type')
    if not settings_match(settings,metadata):
        raise ValueError('Configuration has changed. Delete this candidate and calibrate again.')
    sources=artifact_paths(dataset_id,True); targets=artifact_paths(dataset_id)
    if any(not p.exists() for p in sources): raise ValueError('Candidate artifact is missing. Delete it and calibrate again.')
    if sources:
        import joblib
        try:validate_artifact(joblib.load(sources[0]),metadata['kind'])
        except Exception as exc:raise ValueError('Cannot use this candidate: '+str(exc)) from exc
    for source,target in zip(sources,targets): source.replace(target)
    # Keep paired validation evidence with the promoted model.
    write_json('model-'+dataset_id,metadata)
    (DATA/('candidate-'+dataset_id+'.json')).unlink()
    return metadata


def delete_candidate(dataset_id):
    if not read_json('candidate-'+dataset_id): raise ValueError('Candidate not found')
    for path in [*artifact_paths(dataset_id,True),DATA/('candidate-'+dataset_id+'.json')]: path.unlink(missing_ok=True)


def validate_artifact(artifact,kind):
    """A small local prediction check; activation never needs history or weather."""
    from .forecast_policy import model_columns
    models=artifact.get('models',artifact)
    if set(models)!={'low','middle','high'}:raise ValueError('Candidate lacks quantile heads')
    if kind=='load':
        from .load_training import LoadPredictor,LOAD_FEATURES
        columns=LOAD_FEATURES
        predict=LoadPredictor(artifact).predict
    else:
        from .solar_training import solar_predict
        from .datasets import SOLAR_FEATURES
        columns=SOLAR_FEATURES
        predict=lambda x:solar_predict(artifact,x)
    probe=artifact.get('validation_probe')
    if probe is None:
        # Older candidates have no saved sample. Check their own feature names.
        columns=list(dict.fromkeys([*columns,*(c for model in models.values() for c in model_columns(model,columns))]))
        probe=pd.DataFrame(0.,index=[0],columns=columns)
        probe['weather_run_age']=np.nan
    if probe.empty:raise ValueError('Candidate prediction sample is empty')
    predicted=predict(probe)
    if any(len(values)!=len(probe) or not np.isfinite(values).all() for values in predicted.values()):
        raise ValueError('Candidate predictions must be finite')


def clean_model_storage(settings):
    """Keep active models and a reviewable candidate for configured models."""
    names={'load','battery','solar-__combined',*('solar-'+b.id for b in settings.banks)}
    for name in ('dataset-inverter.json','inverter.pkl','inverter.joblib','inverter-candidate.joblib'):(DATA/name).unlink(missing_ok=True)
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

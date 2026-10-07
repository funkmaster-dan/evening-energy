"""Small CNN, per-bank LightGBM, and measured battery/inverter calibration."""
import warnings
warnings.filterwarnings('ignore',message='X does not have valid feature names')
import joblib, numpy as np, pandas as pd, torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from .config import DATA, read_json, write_json
from .datasets import FEATURES

torch.set_num_threads(2)
SEQ=24
LEGACY_LOAD_FEATURES=FEATURES+['lag1','lag24','rolling24']
CONTEXT_FEATURES=['last_week','weekday_average','weekend_average','heating','cooling']
LOAD_FEATURES=LEGACY_LOAD_FEATURES+CONTEXT_FEATURES

class LoadCNN(nn.Module):
    def __init__(self,feature_count=len(LOAD_FEATURES)):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(feature_count,24,3,padding=1),nn.ReLU(),nn.Conv1d(24,24,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(48,24),nn.ReLU(),nn.Linear(24,1),nn.Softplus())
    def forward(self,x):
        h=self.conv(x.transpose(1,2)); return self.head(torch.cat([h[:,:,-1],h.mean(dim=2)],dim=1)).squeeze(-1)

def load_features(f, timezone='Australia/Adelaide'):
    x=f[FEATURES].copy(); y=pd.to_numeric(f['target'],errors='coerce').astype(float)
    x['lag1']=y.shift(1); x['lag24']=y.shift(24); x['rolling24']=y.shift(1).rolling(24,min_periods=20).mean()
    # Duplicate autumn clock hours are averaged; missing spring hours use the recent mean.
    clock=f.index.tz_convert(timezone).tz_localize(None)
    by_clock=pd.Series(y.to_numpy(),index=clock).groupby(level=0).mean()
    x['last_week']=by_clock.reindex(clock-pd.Timedelta(days=7)).to_numpy()
    for label,weekend in [('weekday_average',False),('weekend_average',True)]:
        total=np.zeros(len(f)); count=np.zeros(len(f))
        for days in range(1,29):
            previous=clock-pd.Timedelta(days=days)
            values=by_clock.reindex(previous).to_numpy()
            valid=np.isfinite(values)&((previous.dayofweek>=5)==weekend)
            total+=np.where(valid,values,0);count+=valid
        x[label]=np.divide(total,count,out=np.full(len(f),np.nan),where=count>0)
    for key in CONTEXT_FEATURES[:3]: x[key]=x[key].fillna(x['rolling24'])
    x['heating']=(18-x.temperature_2m).clip(lower=0)
    x['cooling']=(x.temperature_2m-24).clip(lower=0)
    return x

def sequences(f,timezone='Australia/Adelaide'):
    # Reindex to expose gaps instead of treating separated months as consecutive hours.
    f=f.reindex(pd.date_range(f.index.min(),f.index.max(),freq='1h'))
    if len(f)<SEQ+24: raise ValueError('Need at least 48 contiguous hourly slots to build CNN inputs')
    x=load_features(f,timezone).to_numpy(dtype=np.float32); y=f.target.to_numpy(dtype=np.float32)
    seq=np.lib.stride_tricks.sliding_window_view(x,SEQ,axis=0).transpose(0,2,1).copy()
    targets=y[SEQ-1:]; times=f.index[SEQ-1:]
    mask=np.isfinite(seq).all(axis=(1,2))&np.isfinite(targets)
    return seq[mask],targets[mask],times[mask]

def metrics(y,p):
    return {'mae':round(float(mean_absolute_error(y,p)),4),'rmse':round(float(np.sqrt(mean_squared_error(y,p))),4),'r2':round(float(r2_score(y,p)),4) if len(y)>1 else 0,'samples':len(y)}

def validation_chart(index,y,p):
    step=max(1,len(y)//240)
    return [{'time':str(index[i]),'actual':round(float(y[i]),3),'predicted':round(float(p[i]),3)} for i in range(0,len(y),step)]

def save_residuals(name,residuals): np.save(DATA/(name+'-residuals.npy'),np.asarray(residuals))

def train(settings,dataset_id,progress,epochs=30,fine_tune=False):
    summary=read_json('dataset-'+dataset_id)
    if not summary or not summary['sufficient']: raise ValueError('Build a sufficient dataset first.')
    d=pd.read_pickle(DATA/(dataset_id+'.pkl')); f=d['frame']; kind=summary['kind']
    metadata={'trained_at':pd.Timestamp.now(tz='UTC').isoformat(),'dataset_id':dataset_id,'training_rows':summary['training_rows'],'validation_rows':summary['validation_rows'],'fine_tuned':fine_tune,'dataset_created_at':summary['created_at'],'config_signature':signature(settings,kind,summary.get('bank_id'))}
    if kind=='load':
        from .load_training import train_load
        return train_load(settings,summary,d,progress,epochs,fine_tune)
    elif kind=='solar':
        cols=FEATURES+['tilted_irradiance']; tr=f.index<d['train_end']; va=f.index>=d['val_start']
        x=f[cols]; y=f.target
        model=LGBMRegressor(n_estimators=240,num_leaves=20,learning_rate=0.04,min_child_samples=20,n_jobs=2,verbosity=-1)
        progress('Fitting LightGBM to irradiance and panel output')
        model.fit(x.loc[tr],y.loc[tr]); pred=np.maximum(0,model.predict(x.loc[va])); pred[x.loc[va,'tilted_irradiance'].to_numpy()<5]=0
        joblib.dump(model,DATA/(dataset_id+'.joblib'))
        daylight=f[(f.tilted_irradiance>250)&(f.target>0)]
        capacity=float((daylight.target/(daylight.tilted_irradiance/1000)).quantile(.9)) if len(daylight) else 0
        metadata.update(model='LightGBM',metrics=metrics(y.loc[va],pred),chart=validation_chart(f.index[va],y.loc[va].to_numpy(),pred),inferred_capacity_kw=round(capacity,2),feature_importance=dict(zip(cols,map(int,model.feature_importances_))))
        save_residuals(dataset_id,y.loc[va].to_numpy()-pred)
    elif kind=='battery':
        power=f[settings.battery_power_entity]; soc=f[settings.battery_soc_entity]
        # Positive delta SoC means charging. Align preceding-interval mean power with SoC delta.
        dt=f.index.to_series().diff().dt.total_seconds()/3600
        de=soc.diff()*settings.battery_capacity_kwh/100
        energy=(power+power.shift(1))/2*dt
        usable=(dt>0)&(dt<=0.12)&(soc>5)&(soc<97)&(soc.shift(1)>5)&(soc.shift(1)<97)&(energy.abs()>.025)&(de.abs()>.01)
        tr=(f.index<d['train_end'])&usable; va=(f.index>=d['val_start'])&usable
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
        calibration={'charge_efficiency':charge_eta,'discharge_efficiency':discharge_eta,'power_sign':sign,'observed_charge_kw':round(float((-power*sign).clip(lower=0).quantile(.995)),2),'observed_discharge_kw':round(float((power*sign).clip(lower=0).quantile(.995)),2),'charge_samples':int(charge.sum()),'discharge_samples':int(discharge.sum()),'correlation':round(correlation,3)}
        metadata.update(model='Measured power / SoC calibration',**calibration,metrics=metrics(actual,pred) if len(actual) else {},chart=validation_chart(f.index[va],actual,pred),metric_unit='kWh change')
        write_json('battery-calibration',metadata)
    else:
        total=f[[b.entity for b in settings.banks]].sum(axis=1,min_count=len(settings.banks)); ct=f[settings.solar_ct_entity]
        usable=(total>.5)&(ct>=0)&(ct/total>.3)&(ct/total<1.2)
        tr=(f.index<d['train_end'])&usable; va=(f.index>=d['val_start'])&usable
        if tr.sum()<24: raise ValueError('Need at least 24 simultaneous daytime inverter and CT readings.')
        eta=float(np.clip((ct[tr]/total[tr]).median(),.3,1.0)); pred=total[va].to_numpy()*eta
        metadata.update(model='Inverter / CT efficiency calibration',efficiency=eta,metrics=metrics(ct[va],pred) if va.sum() else {},chart=validation_chart(f.index[va],ct[va].to_numpy(),pred))
        write_json('inverter-calibration',metadata)
    write_json('model-'+dataset_id,metadata)
    return metadata

def signature(settings,kind,bank_id=None):
    if kind=='load': return {'location':[settings.latitude,settings.longitude],'timezone':settings.timezone,'state':settings.state,'school_calendar_url':settings.school_calendar_url,'entities':settings.historical_consumption_entities}
    if kind=='solar':
        b=next(b for b in settings.banks if b.id==bank_id)
        return {'location':[settings.latitude,settings.longitude],'bank':b.model_dump()}
    if kind=='battery': return {'capacity':settings.battery_capacity_kwh,'power':settings.battery_power_entity,'soc':settings.battery_soc_entity}
    return {'banks':[b.entity for b in settings.banks],'ct':settings.solar_ct_entity}

def model_ready(settings,name,kind,bank_id=None):
    meta=read_json('model-'+name)
    return bool(meta and meta.get('config_signature')==signature(settings,kind,bank_id))

def list_models(): return [read_json(p.stem) for p in sorted(DATA.glob('model-*.json'))]

def interval(name,confidence):
    path=DATA/(name+'-residuals.npy')
    if not path.exists(): return (-1,1)
    residual=np.load(path); alpha=(1-confidence)/2
    lo,hi=np.quantile(residual,[alpha,1-alpha])
    return (min(0,float(lo)),max(0,float(hi)))

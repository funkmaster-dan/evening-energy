import json
from datetime import datetime
import numpy as np, pandas as pd, pvlib
from .config import DATA, write_json, read_json
from .sources import HomeAssistant, Weather, Calendar

FEATURES=['temperature_2m','shortwave_radiation','hour_sin','hour_cos','year_sin','year_cos','weekday','weekend','public_holiday','school_holiday','school_calendar_known']

def features(weather, settings, calendar):
    f=weather.copy(); local=f.index.tz_convert(settings.timezone)
    hour=local.hour+local.minute/60
    f['hour_sin']=np.sin(hour*2*np.pi/24); f['hour_cos']=np.cos(hour*2*np.pi/24)
    f['year_sin']=np.sin(local.dayofyear*2*np.pi/365.25); f['year_cos']=np.cos(local.dayofyear*2*np.pi/365.25)
    f['weekday']=local.dayofweek; f['weekend']=(local.dayofweek>=5).astype(int)
    return f.join(calendar.features(f.index))

def bank_features(f, bank, settings):
    # Open-Meteo radiation is the preceding-hour mean: use the hour midpoint.
    position=pvlib.solarposition.get_solarposition(f.index+pd.Timedelta(minutes=30),settings.latitude,settings.longitude)
    irradiance=pvlib.irradiance.get_total_irradiance(bank.tilt,bank.azimuth,position.apparent_zenith.to_numpy(),position.azimuth.to_numpy(),f.direct_normal_irradiance.to_numpy(),f.shortwave_radiation.to_numpy(),f.diffuse_radiation.to_numpy())
    x=f[FEATURES].copy(); x['tilted_irradiance']=np.maximum(0, np.nan_to_num(irradiance['poa_global']))
    return x

def splice(frame, entities, mode='splice'):
    f=frame.reindex(columns=entities)
    if mode=='sum': return f.sum(axis=1,min_count=len(entities))
    # Last-listed sensor takes priority when replacing a prior sensor.
    return f.iloc[:,::-1].bfill(axis=1).iloc[:,0]

def build_dataset(settings, req, progress):
    kind=req['kind']; dataset_id=kind if kind in ('load','battery','inverter') else 'solar-'+req['bank_id']
    start=pd.Timestamp(req['train_start'],tz=settings.timezone).tz_convert('UTC')
    end=(pd.Timestamp(req['validation_end'],tz=settings.timezone)+pd.DateOffset(days=1)).tz_convert('UTC')
    train_end=(pd.Timestamp(req['train_end'],tz=settings.timezone)+pd.DateOffset(days=1)).tz_convert('UTC')
    val_start=pd.Timestamp(req['validation_start'],tz=settings.timezone).tz_convert('UTC')
    if start>=train_end or val_start>=end or train_end>val_start: raise ValueError('Choose non-overlapping training and validation periods, with training first.')
    if end>pd.Timestamp.now(tz='UTC')+pd.Timedelta(days=1): raise ValueError('Training data must be historical.')
    entities=req.get('entities',[])
    if not entities: raise ValueError('Select at least one relevant Home Assistant sensor.')
    if kind=='battery': entities=list(dict.fromkeys(entities+[settings.battery_power_entity,settings.battery_soc_entity]))
    if kind=='inverter': entities=list(dict.fromkeys(entities+[b.entity for b in settings.banks]+[settings.solar_ct_entity]))
    ha=HomeAssistant(settings)
    raw=ha.frame(entities,start,end,period='5minute' if kind=='battery' else 'hour',progress=progress)
    if raw.empty: raise ValueError('Home Assistant has no recorded data in that period.')
    step='5min' if kind=='battery' else '1h'
    expected=pd.date_range(start.ceil(step),end,freq=step,inclusive='left')
    raw=raw.reindex(expected)
    if kind in ('load','solar'):
        progress('Fetching temperature and irradiance from Open-Meteo')
        weather=Weather(settings).fetch(start,end,historical=True)
        calendar=Calendar(settings).load()
        f=features(weather,settings,calendar).reindex(expected)
        target=splice(raw,entities,req.get('combine','splice'))
        limit=float(req.get('max_power_kw',40))
        invalid=(target<0)|(target>limit)
        target=target.mask(invalid)
        if kind=='solar': target=target.mask(target.isna() & (f['shortwave_radiation']<5),0)
        # Only interpolate short gaps; never bridge sensor replacement outages.
        fill=int(req.get('interpolate_hours',2))
        before=int(target.isna().sum())
        if fill: target=target.interpolate(limit=fill,limit_area='inside')
        if kind=='solar':
            bank=next(b for b in settings.banks if b.id==req['bank_id'])
            f=bank_features(f,bank,settings)
        f['target']=target
        cleaned=f.dropna()
        extra={'filtered_outliers':int(invalid.sum()),'filled_hours':before-int(target.isna().sum()),'calendar':calendar.summary(expected),'weather_source':'Open-Meteo reanalysis + recent forecast archive'}
    else:
        cleaned=raw.dropna(); extra={'filtered_outliers':0,'filled_hours':0}
        if kind=='battery':
            soc=cleaned[settings.battery_soc_entity]; cleaned=cleaned[(soc>=0)&(soc<=100)]
    train=cleaned[(cleaned.index>=start)&(cleaned.index<train_end)]
    valid=cleaned[(cleaned.index>=val_start)&(cleaned.index<end)]
    minimum=24*7 if kind in ('load','solar') else 48
    sufficient=len(train)>=minimum and len(valid)>=24
    if kind=='load': sufficient=sufficient and (end.tz_convert(settings.timezone).date()-val_start.tz_convert(settings.timezone).date()).days>=10 and len(valid)>=168
    summary={'id':dataset_id,'kind':kind,'bank_id':req.get('bank_id'),'request':req,'created_at':pd.Timestamp.now(tz='UTC').isoformat(),'rows':len(cleaned),'training_rows':len(train),'validation_rows':len(valid),'coverage_pct':round(100*len(cleaned)/len(expected),1),'sufficient':sufficient,'readiness':'Ready to calibrate' if sufficient else ('CNN needs 10 validation days (selection plus independent holdout) and at least 7 training days' if kind=='load' else f'Need at least {minimum} training rows and 24 validation rows'),'recommendation':'A full year improves seasonal coverage; 7 days is the minimum for an experiment.','sources':entities,**extra,'preview':[{'time':str(t), 'value':round(float(v),3)} for t,v in (cleaned['target'] if 'target' in cleaned else cleaned.iloc[:,0]).iloc[::max(1,len(cleaned)//240)].items()]}
    pd.to_pickle({'frame':cleaned,'train_end':train_end,'val_start':val_start,'end':end,'entities':entities}, DATA/(dataset_id+'.pkl'))
    if kind=='load':
        from .load_training import preserve_history
        preserve_history(settings, cleaned, entities)
    write_json('dataset-'+dataset_id,summary)
    progress(f'Dataset built: {len(train)} training rows, {len(valid)} validation rows')
    return summary

def list_datasets(): return [json.loads(p.read_text()) for p in sorted(DATA.glob('dataset-*.json'))]

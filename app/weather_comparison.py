"""Compare archived weather runs against independent panel validation days."""
import joblib, numpy as np, pandas as pd
from .config import DATA,read_json,write_json
from .sources import forecast_weather,Calendar
from .datasets import features,bank_features


def assess(settings,progress):
    frames={b.id:pd.read_pickle(DATA/('solar-'+b.id+'.pkl')) for b in settings.banks}
    start=max(d['val_start'] for d in frames.values()).tz_convert(settings.timezone).ceil('D')
    end=min(d['end'] for d in frames.values()).tz_convert(settings.timezone).normalize()
    models={b.id:joblib.load(DATA/('solar-'+b.id+'.joblib')) for b in settings.banks}
    calendar=Calendar(settings).load();errors={};daily={};days=[];skipped=[]
    for day in pd.date_range(start,end,freq='D',inclusive='left')[-21:]:
        stop=day+pd.DateOffset(days=1);grid=pd.date_range(day.tz_convert('UTC').floor('h'),stop.tz_convert('UTC').ceil('h'),freq='1h',inclusive='left')
        weights=np.array([max(0,(min(stop.tz_convert('UTC'),t+pd.Timedelta(hours=1))-max(day.tz_convert('UTC'),t)).total_seconds()/3600) for t in grid])
        targets=[frames[b.id]['frame'].target.reindex(grid) for b in settings.banks]
        if any(y.isna().any() for y in targets):continue
        actual=sum(y.to_numpy() for y in targets);predictions={}
        # Forecast before the target local day begins; publication time buffer matches load holdouts.
        issued=day.tz_convert('UTC')-pd.Timedelta(hours=1)
        for name in ('ecmwf_ifs','gfs_global','icon_global'):
            progress(f'Comparing {name} against measured panels on {day.date()}')
            try:
                f=features(forecast_weather(settings,grid[0],grid[-1],name,issued),settings,calendar).reindex(grid)
                output=np.zeros(len(grid))
                for b in settings.banks:
                    x=bank_features(f,b,settings)
                    if x.isna().any().any():raise ValueError('Incomplete forecast irradiance')
                    pred=np.maximum(0,models[b.id].predict(x));pred[x.tilted_irradiance.to_numpy()<5]=0;output+=pred
                predictions[name]=output
            except Exception as exc:skipped.append({'day':str(day.date()),'model':name,'reason':str(exc)})
        # Common days keep each comparison fair. No substitute observed weather.
        if len(predictions)!=3:continue
        predictions['equal_blend']=np.mean(list(predictions.values()),axis=0);days.append(str(day.date()))
        for name,pred in predictions.items():
            errors.setdefault(name,[]).extend((actual-pred).tolist());daily.setdefault(name,[]).append(float((actual-pred)@weights))
    result={'completed_at':pd.Timestamp.now(tz='UTC').isoformat(),'days':days,'skipped':skipped,'models':{name:{'samples':len(daily[name]),'energy_mae_kwh':round(float(np.mean(np.abs(daily[name]))),3),'energy_bias_kwh':round(float(np.mean(daily[name])),3),'hourly_mae_kw':round(float(np.mean(np.abs(err))),3)} for name,err in errors.items()},'note':'Same complete held-out days and measured sum of panel-bank output. Runs predate the local day with a 12-hour publication buffer. Not an operational issued-forecast score or a calibrated ensemble.'}
    result['recommended']=min(result['models'],key=lambda name:result['models'][name]['energy_mae_kwh']) if len(days)>=7 else None
    write_json('weather-comparison',result);return result

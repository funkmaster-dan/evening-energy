"""Panel quantile models and validation against weather available at forecast issue time."""
import os, joblib, numpy as np, pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from .config import DATA
from .datasets import SOLAR_FEATURES, SOLAR_PIPELINE, bank_features
from .sources import solar_weather
from .quality import dark_hours
from .load_splits import training_mask

QUANTILES=(0.1,0.5,0.9)
QKEYS=('low','middle','high')
QMAP=dict(zip(QUANTILES,QKEYS))
SOLAR_PARAMETERS=dict(boosting_type='gbdt',objective='quantile',n_estimators=1000,learning_rate=.05,max_bin=255,reg_lambda=2.,reg_alpha=.1,subsample=.9,subsample_freq=1,colsample_bytree=.9,random_state=42,n_jobs=max(1,int(os.getenv('ENERGY_TRAIN_THREADS','2'))),deterministic=True,force_col_wise=True,verbosity=-1)
# Small grid; larger min_child_samples candidates stabilize the tails.
SOLAR_CONFIGURATIONS=[dict(num_leaves=leaves,max_depth=depth,min_child_samples=child) for leaves,depth,child in [(15,4,100),(31,5,100),(31,5,300)]]


def pinball(y,q,alpha):
    y=np.asarray(y,float); q=np.asarray(q,float)
    return float(np.mean(np.maximum(alpha*(y-q),(alpha-1)*(y-q))))


def solar_predict(artifact,x,records=(),origin=None):
    """Shared inference for legacy quantile maps and issued-weather artifacts."""
    from .forecast_policy import ordered,model_columns,recent_correction
    from .datasets import LEGACY_SOLAR_FEATURES
    current=artifact.get('models',artifact);modern='models' in artifact
    raw={key:current[key].predict(x.reindex(columns=model_columns(current[key],LEGACY_SOLAR_FEATURES))) for key in QKEYS}
    # full_dark comes from the same interval predicate used in target preparation.
    dark=x.full_dark.to_numpy(dtype=bool) if modern and 'full_dark' in x else x.solar_elevation.to_numpy()<=0
    cutoff=artifact.get('irradiance_cutoff',0) if modern else 5
    if cutoff:dark=dark|(x.tilted_irradiance.to_numpy()<cutoff)
    q=ordered(raw,x.get('lead_hours'),artifact.get('calibration') if modern else None,dark)
    point_model=artifact.get('mean_model') if modern and artifact.get('point_method',{}).get('head')=='mean' else None
    q['point']=np.maximum(0,point_model.predict(x.reindex(columns=model_columns(point_model,LEGACY_SOLAR_FEATURES)))) if point_model is not None else ordered(raw,dark=dark)['middle']
    q['base_point']=q['point'].copy()
    if modern and origin is not None:q['point']*=recent_correction(records,origin,x.lead_hours,artifact.get('correction'),True)
    q['point'][dark]=0; q['base_point'][dark]=0
    return q


def power_metrics(actual,predicted,weights):
    actual=np.asarray(actual); predicted=np.asarray(predicted); weights=np.asarray(weights)
    error=predicted-actual; mean=np.average(actual,weights=weights)
    variance=np.sum(weights*(actual-mean)**2)
    return {'mae':round(float(np.average(abs(error),weights=weights)),4),
            'rmse':round(float(np.sqrt(np.average(error**2,weights=weights))),4),
            'bias_kw':round(float(np.average(error,weights=weights)),4),
            'r2':round(float(1-np.sum(weights*error**2)/variance),4) if variance>0 else 0.,
            'samples':len(actual)}


def energy_metrics(rows):
    if not rows: return {'windows':0}
    actual=np.array([r['actual_kwh'] for r in rows]); predicted=np.array([r['predicted_kwh'] for r in rows]); error=predicted-actual
    return {'windows':len(rows),'energy_mae_kwh':round(float(abs(error).mean()),4),
            'energy_rmse_kwh':round(float(np.sqrt(np.mean(error**2))),4),
            'energy_bias_kwh':round(float(error.mean()),4),
            'energy_bias_pct':round(float(100*error.sum()/actual.sum()),2) if actual.sum()>0 else 0.,
            'actual_mean_kwh':round(float(actual.mean()),4),'predicted_mean_kwh':round(float(predicted.mean()),4),
            'energy_coverage':round(float(np.mean([(r.get('low_kwh',-np.inf)<=r['actual_kwh']<=r.get('high_kwh',np.inf)) for r in rows])),4) if any('low_kwh' in r for r in rows) else None}


def evaluate_solar(settings,bank,d,models,baseline,progress):
    """Daily local-midnight issues; fixed 12h publication buffer for archived runs.

    Daily totals respect DST. Longer windows use elapsed 24/48/72 hours.
    Missing/interpolated daylight targets are excluded, never scored as truth.
    `models`/`baseline` map low/middle/high to fitted regressors; point metrics
    use the middle (p50) head so candidates stay comparable to prior models.
    """
    cases,skipped=solar_cases(settings,bank,d,progress)
    predictions={key:{'actual':[],'middle':[],'low':[],'high':[],'weights':[],'daylight':[],'bright':[],'times':[],'daily':[],'horizons':{h:[] for h in (24,48,72)}} for key in (('candidate','existing') if baseline is not None else ('candidate',))}
    runs=[]
    for case in cases:
        origin=case['origin'];daily_end=case['daily_end'];end=d['end'];x=case['x'];index=x.index;actual=case['actual'];daylight=case['daylight'];bright=case['bright'];weather=case['weather']
        runs.append({'issue_time':origin.isoformat(),'run':weather.attrs['run']})
        for key,current in [('candidate',models),*([('existing',baseline)] if baseline is not None else [])]:
            predicted=solar_predict(current,x); out=predictions[key]
            def window(until):
                result=case['valid_window'](until)
                if result is None:return None
                mask,weights=result
                return mask,weights,{'issue_time':origin.isoformat(),'end':until.isoformat(),
                                     'actual_kwh':float(np.sum(actual*weights)),
                                     'predicted_kwh':float(np.sum(predicted['middle']*weights)),
                                     'low_kwh':float(np.sum(predicted['low']*weights)),
                                     'high_kwh':float(np.sum(predicted['high']*weights))}
            daily=window(daily_end)
            if daily:
                mask,weights,row=daily; out['daily'].append(row)
                for label,values in [('actual',actual),('middle',predicted['middle']),('low',predicted['low']),('high',predicted['high']),('weights',weights),('daylight',daylight),('bright',bright)]: out[label].extend(values[mask].tolist())
                out['times'].extend(index[mask].tolist())
            elif key=='candidate': skipped.append({'issue_time':origin.isoformat(),'reason':'Missing measured daylight targets for daily total'})
            for horizon in (24,48,72):
                until=origin+pd.Timedelta(hours=horizon)
                if until<=end:
                    result=window(until)
                    if result: out['horizons'][horizon].append(result[2])
    results={}
    for key,out in predictions.items():
        if not out['actual']: raise ValueError('Solar validation needs a complete day with measured daylight targets and an available archived weather run. Choose recent validation dates.')
        actual=np.asarray(out['actual']); middle=np.asarray(out['middle']); low=np.asarray(out['low']); high=np.asarray(out['high']); weights=np.asarray(out['weights']); day=np.asarray(out['daylight'],dtype=bool)
        if not day.any(): raise ValueError('Solar validation has no daytime samples')
        br=np.asarray(out['bright'],dtype=bool)
        in_band=(actual>=low)&(actual<=high)
        quantile={'pinball_low':round(pinball(actual[day],low[day],0.1),4),'pinball_mid':round(pinball(actual[day],middle[day],0.5),4),'pinball_high':round(pinball(actual[day],high[day],0.9),4),
                  'coverage_daylight':round(float(in_band[day].mean()),4),'coverage_bright':round(float(in_band[br].mean()),4) if br.any() else None,
                  'width_daylight_kw':round(float((high[day]-low[day]).mean()),4),
                  'width_bright_kw':round(float((high[br]-low[br]).mean()),4) if br.any() else None,
                  'p90_bright_max_kw':round(float(high[br].max()),4) if br.any() else None,
                  'zero_truncated_share':round(float((low[day]==0).mean()),4)}
        summary={'method':'Archived forecast runs available before local-midnight issue time',
                 'weather_model':settings.weather_model,'publication_buffer_hours':12,
                 'daytime':power_metrics(actual[day],middle[day],weights[day]),
                 'daily':energy_metrics(out['daily']),
                 'horizons':{str(h):energy_metrics(rows) for h,rows in out['horizons'].items()},
                 'quantile':quantile,
                 'energy_interval_basis':'Sum of hourly marginal P10/P90 bounds; not MC energy percentiles',
                 'daily_chart':out['daily'],'runs':runs,'skipped':skipped,
                 'target_basis':'Measured daylight power; missing dark-hour power is zero',
                 'daytime_definition':'Solar elevation above zero'}
        results[key]={'metrics':power_metrics(actual,middle,weights),
                      'chart':[{'time':str(t),'actual':round(float(a),3),'predicted':round(float(p),3)} for t,a,p in zip(out['times'],actual,middle)],
                      'forecast_validation':summary}
    return results


def train_solar(settings,summary,d,metadata,progress):
    from .study import run
    return {**metadata,**run(settings,summary,d,progress)}


def solar_cases(settings,bank,d,progress=lambda s:None):
    frame=d['frame'];start=d['val_start'];end=d['end'];cases=[];skipped=[]
    first=start.tz_convert(settings.timezone).normalize()
    if first<start:first+=pd.DateOffset(days=1)
    for local in pd.date_range(first,end.tz_convert(settings.timezone),freq='D',inclusive='left'):
        origin=local.tz_convert('UTC');daily_end=(local+pd.DateOffset(days=1)).tz_convert('UTC')
        if daily_end>end:continue
        finish=min(origin+pd.Timedelta(hours=72),end)
        index=pd.date_range(origin.floor('h'),finish.ceil('h'),freq='h',inclusive='left')
        progress(f'Checking solar issue weather: {bank.name} {local.date()}')
        try:
            weather=solar_weather(settings,bank,index[0],finish.ceil('h'),issued_at=origin)
            x=bank_features(weather.reindex(index),bank,settings,origin)
            if x.isna().any().any():raise ValueError('Incomplete archived forecast weather')
        except Exception as exc:
            skipped.append({'issue_time':origin.isoformat(),'reason':'Archived weather unavailable: '+str(exc)});continue
        y=frame.target.reindex(index);dark=dark_hours(settings,index)
        eligible=frame.validation_eligible.reindex(index).fillna(False).to_numpy(dtype=bool)
        derived=frame.derived_method.reindex(index).eq('dark_zero').to_numpy()
        valid=(eligible&y.notna().to_numpy()) | (dark&derived)
        actual=y.fillna(0).to_numpy()
        def window(until,index=index,origin=origin,valid=valid):
            if until>end:return None
            left=np.maximum(index.as_unit('ns').asi8,origin.value);right=np.minimum((index+pd.Timedelta(hours=1)).as_unit('ns').asi8,until.value)
            weights=np.maximum(0,(right-left)/3.6e12);mask=weights>0
            return (mask,weights) if mask.any() and valid[mask].all() else None
        cases.append({'origin':origin,'daily_end':daily_end,'x':x,'weather':weather,'actual':actual,'daylight':x.solar_elevation.to_numpy()>0,'bright':(x.tilted_irradiance.to_numpy()>250)&(actual>0),'valid_window':window})
        if window(daily_end) is None:skipped.append({'issue_time':origin.isoformat(),'reason':'Missing valid measured daylight or rejected dark target'})
    return cases,skipped

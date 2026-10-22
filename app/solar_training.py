"""Solar inference and training entry point; replay is shared with consumption."""
import numpy as np
QKEYS=('low','middle','high')

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


def train_solar(settings,summary,d,metadata,progress):
    from .study import run
    return {**metadata,**run(settings,summary,d,progress)}

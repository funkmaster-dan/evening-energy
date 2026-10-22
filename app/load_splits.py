"""Coverage-only recommendations; model errors never select a holdout."""
import numpy as np
import pandas as pd

def training_mask(index,start,end,step='1h'):
    """A fitting target's complete interval must be within its split."""
    return (index>=pd.Timestamp(start))&(index+pd.Timedelta(step)<=pd.Timestamp(end))


def split_coverage(frame,start,end,settings,complete=False):
    from .quality import dark_hours
    part=frame.loc[training_mask(frame.index,start,end) if complete else (frame.index+pd.Timedelta(hours=1)>start)&(frame.index<end)]
    dark=pd.Series(dark_hours(settings,part.index),index=part.index)
    measured=part.get('measurement_valid',part.target.notna())
    day=~dark
    weights=np.array([max(0,(min(t+pd.Timedelta(hours=1),end)-max(t,start)).total_seconds()/3600) for t in part.index])
    return {'interval_weighted_hours':float(weights.sum()),'measured_weighted_hours':float(weights@measured.to_numpy(dtype=float)),'expected_hours':len(part),'measured_hours':int(measured.sum()),'measured_pct':round(100*measured.sum()/max(1,len(part)),2),'daylight_hours':int(day.sum()),'measured_daylight_hours':int((measured&day).sum()),'measured_daylight_pct':round(100*(measured&day).sum()/max(1,day.sum()),2),'derived_dark_zero_hours':int(part.derived_method.eq('dark_zero').sum()),'interpolated_hours':int(part.derived_method.eq('interpolation').sum()),'quiet_hours':int(part.quiet_regime.sum()),'reasons':part.quality_flags.value_counts().to_dict()}

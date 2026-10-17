"""HA availability and successful snapshot time; state-change time is diagnostic only."""
import math
import pandas as pd


def state_available(state):
    return state.get('state') not in (None,'unknown','unavailable','none','')


def numeric_value(state,power=False):
    if not state_available(state):return None
    try:value=float(state['state'])
    except (KeyError,TypeError,ValueError):return None
    if not math.isfinite(value):return None
    if power:
        unit=state.get('attributes',{}).get('unit_of_measurement','')
        if unit=='W':value/=1000
        elif unit!='kW':return None
    return value


def snapshot_age(live,now=None):
    updated=live.get('updated_at')
    if not updated:return 1e9
    try:return max(0,((now if now is not None else pd.Timestamp.now(tz='UTC'))-pd.Timestamp(updated)).total_seconds())
    except (TypeError,ValueError):return 1e9


def history_values(pieces,start,end,step):
    """Compatibility wrapper; duration evidence travels with the interval means."""
    intervals=history_intervals(pieces,start,end,step)
    result=intervals['mean'].copy()
    result.attrs['durations']=intervals.drop(columns='mean')
    return result


def normalized_value(value, unit, kind):
    """Normalize one representation, rejecting incompatible dimensions."""
    if kind=='soc':
        if unit!='%':raise ValueError('State of charge requires percent (%)')
        return value
    if unit not in ('W','kW','Wh','kWh'):raise ValueError('Power requires W/kW or an energy-delta source in Wh/kWh')
    return value/(1000 if unit in ('W','Wh') else 1) if value is not None else None


def history_intervals(pieces,start,end,step,cumulative=False):
    """Integrate held states at actual change times; no minute sampling."""
    import numpy as np
    start,end=pd.Timestamp(start),pd.Timestamp(end);step=pd.Timedelta(step)
    events=pd.DataFrame(pieces,columns=['time','value'])
    events['time']=pd.to_datetime(events.time,utc=True)
    events=events.drop_duplicates('time',keep='last').sort_values('time').set_index('time').value
    events=pd.to_numeric(events,errors='coerce').astype(float).replace([np.inf,-np.inf],np.nan)
    grid=pd.date_range(start.floor(step),end.ceil(step),freq=step,inclusive='left')
    rows=[]
    for bin_start in grid:
        left=max(start,bin_start);right=min(end,bin_start+step)
        knots=events.index[(events.index>left)&(events.index<right)].union(pd.DatetimeIndex([left,right])).sort_values()
        held=events.reindex(knots,method='ffill')
        durations=np.diff(knots.as_unit('ns').asi8)/1e9
        known=np.isfinite(held.iloc[:-1].to_numpy());observed=float(durations[known].sum());total=(right-left).total_seconds()
        complete=abs(observed-total)<1e-6
        if cumulative:
            # Include the right endpoint and reject resets anywhere in this bin.
            delta=held.diff().iloc[1:];valid=complete and held.notna().all() and (delta>=0).all()
            mean=float((held.iloc[-1]-held.iloc[0])/(total/3600)) if valid else np.nan
        else:mean=float(np.sum(held.iloc[:-1].fillna(0).to_numpy()*durations)/total) if complete else np.nan
        rows.append((mean,observed,total-observed,total))
    return pd.DataFrame(rows,index=grid,columns=['mean','observed_seconds','unknown_seconds','interval_seconds'])

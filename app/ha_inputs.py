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
        elif unit not in ('kW',''):return None
    return value


def snapshot_age(live,now=None):
    updated=live.get('updated_at')
    if not updated:return 1e9
    try:return max(0,((now if now is not None else pd.Timestamp.now(tz='UTC'))-pd.Timestamp(updated)).total_seconds())
    except (TypeError,ValueError):return 1e9


def history_values(pieces,start,end,step):
    """HA states hold until changed/unavailable; silence alone is not a history gap."""
    f=pd.DataFrame(pieces,columns=['time','value']).drop_duplicates('time',keep='last').set_index('time').sort_index()['value']
    f=pd.to_numeric(f,errors='coerce').astype(float)
    grid=pd.date_range(pd.Timestamp(start).floor('min'),pd.Timestamp(end).ceil('min'),freq='1min',inclusive='left')
    held=f.reindex(grid,method='ffill')  # Explicit NaN unavailable transitions remain NaN.
    result=held.resample(step).mean()
    return result.mask(held.resample(step).count()!=held.resample(step).size())

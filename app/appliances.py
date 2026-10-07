"""Explicit HA appliance assumptions; no control or schedule mutation."""
import pandas as pd
from .forecast import inside
from .ha_inputs import snapshot_age


def known_load(settings,live,t,now):
    local=t.tz_convert(settings.timezone);total=0;additional=0;names=[]
    states=live.get('appliance_states',{})
    live_fresh=snapshot_age(live,now)<900
    for a in settings.appliances:
        scheduled=a.start!=a.end and inside(local,a.start,a.end)
        day=local-pd.DateOffset(days=1) if a.start>a.end and local.strftime('%H:%M')<a.end else local
        scheduled=scheduled and day.dayofweek in a.weekdays
        state=states.get(a.entity or a.power_entity or a.name,{})
        fresh=live_fresh and state.get('available',False)
        active=fresh and (state.get('hvac_action') or state.get('state')) in a.active_states and (t-now).total_seconds()/3600<a.live_hours
        measured=state.get('power_kw')
        if a.power_entity and not a.entity:active=live_fresh and measured is not None and measured>.1 and (t-now).total_seconds()/3600<a.live_hours
        event=state.get('next_event')
        if event and fresh and a.entity.startswith('schedule.'):
            event=pd.Timestamp(event)
            if state.get('state') in a.active_states:active=live_fresh and t<event
            else:active=live_fresh and event<=t<event+pd.Timedelta(hours=a.live_hours)
        if scheduled or active:
            power=measured if active and live_fresh and measured is not None and measured>.1 else a.power_kw
            names.append(a.name)
            if a.mode=='additional':additional+=power
            else:total+=power
    return total,additional,names

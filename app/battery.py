"""Shared battery equations for point forecasts and Monte Carlo paths."""
from datetime import datetime,timedelta
import numpy as np, pandas as pd


def inside(t,start,end):
    a=datetime.strptime(start,'%H:%M').time();b=datetime.strptime(end,'%H:%M').time();v=t.time().replace(tzinfo=None)
    return a<=v<b if a<b else v>=a or v<b


def export_active(t,settings):
    return any(inside(t,w.start,w.end) for w in settings.export_windows)


def export_rate(t,settings):
    window=next((w for w in settings.export_windows if inside(t,w.start,w.end)),None)
    if window is None:return 0.
    return min(settings.battery_discharge_limit_kw,window.discharge_kw if window.discharge_kw is not None else settings.battery_discharge_limit_kw)


def window_day(t,w):
    if w.start>w.end and t.strftime('%H:%M')<w.end:return (t-timedelta(days=1)).date().isoformat()
    return t.date().isoformat()


# Shared acceptance tolerances, also exposed with the advisory schedule.
IMPORT_TOLERANCE_KWH=.01
RESERVE_TOLERANCE_SOC=.001
DELIVERY_TOLERANCE_KWH=.001
POLICY='half-day-energy-fixed-discharge-v1'


def export_mask(settings,rows,cutoff=None):
    """Only the first current/upcoming export window is eligible."""
    eligible=np.array([export_active(pd.Timestamp(r['time']).tz_convert(settings.timezone),settings)
                       and (cutoff is None or pd.Timestamp(r['time'])<cutoff) for r in rows])
    active=np.flatnonzero(eligible)
    if len(active):
        last=active[0]
        while last+1<len(rows) and eligible[last+1]:last+=1
        eligible[last+1:]=False
    return eligible


def acceptable(settings,rows,candidate,outside):
    weights=np.asarray([r['hours'] for r in rows])
    unallowed=(outside*weights).sum(axis=1)
    reserve_soc=min(100,settings.minimum_soc+100*settings.reserve_kwh/settings.battery_capacity_kwh)
    reserve_ok=(candidate+RESERVE_TOLERANCE_SOC>=reserve_soc).all(axis=1)
    return (unallowed<=IMPORT_TOLERANCE_KWH)&reserve_ok


def simulate_paths(settings,rows,net,initial_soc,eta_c,eta_d,used_allowance=None,
                   export_kwh=0.,export_cutoff=None,record_flows=False):
    """Advance one or many paths; optionally retain detailed display flows."""
    net=np.asarray(net,float);n,steps=net.shape
    capacity=settings.battery_capacity_kwh;minimum=capacity*settings.minimum_soc/100
    energy=np.broadcast_to(np.asarray(initial_soc,float),(n,)).copy()*capacity/100
    reserve=minimum+settings.reserve_kwh
    eligible=export_mask(settings,rows,export_cutoff)
    requested=np.broadcast_to(np.asarray(export_kwh,float),(n,)).copy();remaining=requested.copy()
    usage={k:np.full(n,v,float) for k,v in (used_allowance or {}).items()}
    soc=np.empty((n,steps));outside=np.zeros((n,steps))
    flows={key:np.empty((n,steps)) for key in ('grid_import_kw','battery_export_kw','power_limited_import_kw','energy_shortfall_import_kw','charge_kw','discharge_kw','household_import_kw','grid_charge_kw','requested_discharge_kw','battery_discharge_kw')} if record_flows else None
    if record_flows:flows['outside_window_import_kw']=outside
    for i,row in enumerate(rows):
        t=pd.Timestamp(row['time']).tz_convert(settings.timezone);dt=row['hours']
        window=next(((j,w) for j,w in enumerate(settings.charging_windows) if inside(t,w.start,w.end)),None)
        surplus=np.maximum(0,-net[:,i]);demand=np.maximum(0,net[:,i])
        charge=np.where(net[:,i]<=0,np.minimum(np.minimum(surplus,settings.battery_charge_limit_kw),(capacity-energy)/(eta_c*dt)),0.)
        energy+=charge*eta_c*dt
        imported=np.zeros(n)
        if window:
            j,w=window;key=f'{j}:{window_day(t,w)}'
            used=usage.setdefault(key,np.zeros(n))
            available=np.maximum(0,w.limit_kwh-used)/dt
            imported=np.minimum(demand,available);used+=imported*dt;demand-=imported
        discharge=np.minimum(np.minimum(demand,settings.battery_discharge_limit_kw),np.maximum(0,energy-minimum)*eta_d/dt)
        energy-=discharge*dt/eta_d;outside[:,i]=np.maximum(0,demand-discharge)
        imported+=outside[:,i]
        if record_flows:
            power_import=np.maximum(0,demand-settings.battery_discharge_limit_kw)
            flows['power_limited_import_kw'][:,i]=power_import
            flows['energy_shortfall_import_kw'][:,i]=np.maximum(0,outside[:,i]-power_import)
        household_import=imported.copy();grid_charge=np.zeros(n)
        if window:
            available=np.maximum(0,w.limit_kwh-used)/dt
            grid_charge=np.minimum(np.minimum(settings.battery_charge_limit_kw if w.charge_kw is None else w.charge_kw,np.maximum(0,settings.battery_charge_limit_kw-charge)),np.minimum(available,np.maximum(0,capacity-energy)/(eta_c*dt)))
            energy+=grid_charge*eta_c*dt;used+=grid_charge*dt
            charge+=grid_charge;imported+=grid_charge
        # Fixed total battery AC discharge while exporting. Home demand takes
        # priority; the remainder reaches the grid. Stop at the energy target.
        rate=export_rate(t,settings)
        active=eligible[i]&(remaining>1e-9)
        amount=np.where(active,np.maximum(0,remaining)/dt,0.)
        extra=np.minimum(np.maximum(0,amount),np.minimum(np.maximum(0,rate-discharge),np.maximum(0,energy-reserve)*eta_d/dt))
        energy-=extra*dt/eta_d;remaining-=extra*dt
        soc[:,i]=100*np.maximum(0,energy)/capacity
        if record_flows:
            for key,value in [('grid_import_kw',imported),('battery_export_kw',extra),('charge_kw',charge),('discharge_kw',discharge),('household_import_kw',household_import),('grid_charge_kw',grid_charge),('requested_discharge_kw',np.where(active,rate,0.)),('battery_discharge_kw',discharge+extra)]:flows[key][:,i]=value
    result=(soc,outside,requested-remaining)
    return (*result,flows) if record_flows else result


def allowances(settings,rows,net,soc,eta_c,eta_d,usage,cutoff,include_stored=True):
    """Shared counterfactual searches for stored excess and achievable export."""
    reserve_soc=min(100,settings.minimum_soc+100*settings.reserve_kwh/settings.battery_capacity_kwh)
    n=len(net)
    def valid(candidate,outside):
        return acceptable(settings,rows,candidate,outside)
    stored=np.zeros(n)
    if include_stored:
        lower=np.zeros(n);upper=np.full(n,max(0,(soc-reserve_soc)*settings.battery_capacity_kwh/100))
        for _ in range(22):
            amount=(lower+upper)/2
            candidate,outside,_=simulate_paths(settings,rows,net,soc-100*amount/settings.battery_capacity_kwh,eta_c,eta_d,usage)
            ok=valid(candidate,outside);lower=np.where(ok,amount,lower);upper=np.where(ok,upper,amount)
        stored=np.maximum(0,np.floor(lower*100)/100)
    # Integrated requested power, not one stored capacity: replenishment can occur.
    maximum=sum(r['hours']*export_rate(pd.Timestamp(r['time']).tz_convert(settings.timezone),settings) for r,on in zip(rows,export_mask(settings,rows,cutoff)) if on)
    lower=np.zeros(n);upper=np.full(n,maximum)
    if cutoff is not None:
        candidate,outside,delivered=simulate_paths(settings,rows,net,soc,eta_c,eta_d,usage,upper,cutoff)
        upper_ok=valid(candidate,outside)&(delivered>=upper-DELIVERY_TOLERANCE_KWH)
        lower=np.where(upper_ok,upper,lower)
        for _ in range(22):
            amount=(lower+upper)/2
            candidate,outside,delivered=simulate_paths(settings,rows,net,soc,eta_c,eta_d,usage,amount,cutoff)
            ok=valid(candidate,outside)&(delivered>=amount-DELIVERY_TOLERANCE_KWH)
            lower=np.where(ok,amount,lower);upper=np.where(ok,upper,amount)
    return stored,np.maximum(0,np.floor(lower*100)/100)

import joblib, numpy as np, pandas as pd
from .config import DATA, read_json, write_json
from .sources import HomeAssistant, Calendar, consumption_weather, solar_weather
from .datasets import features, bank_features, splice
from .models import model_ready
from .ha_inputs import snapshot_age
from .forecast_policy import record_issue,settled_records,recent_correction,SETTLEMENT_HOURS
from .load_training import LoadPredictor
from .quality import QUALITY_POLICY, effective_policy, LOAD_PIPELINE, quality_frame, target_bounds, load_context_metadata, attach_load_context
from .uncertainty import energy_scenarios
from .battery import (inside, export_active, export_rate, window_day, simulate_paths, allowances, acceptable, export_periods, plan_exports,
                      POLICY, IMPORT_TOLERANCE_KWH, RESERVE_TOLERANCE_SOC, DELIVERY_TOLERANCE_KWH)

def simulate(settings, rows, initial_soc, eta_c, eta_d, export_kwh=0, export_cutoff=None, used_allowance=None, export_schedule=None):
    """Format a single path from the shared battery engine for existing callers."""
    net=np.asarray([r['net_demand_kw'] for r in rows],float).reshape(1,-1)
    soc,outside,delivered,flows=simulate_paths(settings,rows,net,initial_soc,eta_c,eta_d,used_allowance,
                                             export_kwh,export_cutoff,record_flows=True,export_schedule=export_schedule)
    rounded={'grid_import_kw','battery_export_kw','charge_kw','discharge_kw'}
    result=[]
    for i,row in enumerate(rows):
        values={key:round(float(array[0,i]),3) if key in rounded else float(array[0,i]) for key,array in flows.items()}
        result.append({**row,'soc':float(soc[0,i]),**values})
    outside_import=sum(float(outside[0,i])*row['hours'] for i,row in enumerate(rows))
    return result,outside_import,float(delivered[0])

def stored_excess(settings,rows,soc,eta_c,eta_d,used_allowance):
    """Stored energy from the shared counterfactual search."""
    net=np.asarray([r['net_demand_kw'] for r in rows],float).reshape(1,-1)
    stored,_=allowances(settings,rows,net,soc,eta_c,eta_d,used_allowance,None)
    return float(stored[0])

def excess(settings,rows,soc,eta_c,eta_d,used_allowance):
    # Export only in the first current/upcoming ToU window.
    peak=[i for i,r in enumerate(rows) if export_active(pd.Timestamp(r['time']).tz_convert(settings.timezone),settings)]
    if not peak:return 0.,simulate(settings,rows,soc,eta_c,eta_d,used_allowance=used_allowance)[0]
    last=peak[0]
    while last+1 in peak:last+=1
    cutoff=pd.Timestamp(rows[last]['time'])+pd.Timedelta(hours=rows[last]['hours'])
    net=np.asarray([r['net_demand_kw'] for r in rows],float).reshape(1,-1)
    _,exported=allowances(settings,rows,net,soc,eta_c,eta_d,used_allowance,cutoff,include_stored=False)
    amount=float(exported[0])
    path,_,_=simulate(settings,rows,soc,eta_c,eta_d,amount,cutoff,used_allowance)
    return amount,path

def allowance_used(settings, now):
    """Estimate actual charging-window import from measured load, solar and battery I/O."""
    calibration=read_json('model-battery',{})
    sign=calibration.get('power_sign',1); local=now.tz_convert(settings.timezone)
    start=local.normalize()-pd.Timedelta(days=1)
    try:
        frame=HomeAssistant(settings).frame([settings.consumption_entity,settings.solar_ct_entity,settings.battery_power_entity],start.tz_convert('UTC'),now,'5minute')
        use={}
        for t,row in frame.dropna().iterrows():
            tl=t.tz_convert(settings.timezone)
            imported=max(0,row[settings.consumption_entity]-row[settings.solar_ct_entity]-row[settings.battery_power_entity]*sign)/12
            for i,w in enumerate(settings.charging_windows):
                if inside(tl,w.start,w.end):
                    key=f'{i}:{window_day(tl,w)}'; use[key]=use.get(key,0)+imported; break
        # No past records during an already started window: assume its allowance used.
        for i,w in enumerate(settings.charging_windows):
            if inside(local,w.start,w.end):
                key=f'{i}:{window_day(local,w)}'
                if key not in use: use[key]=w.limit_kwh
        return use
    except Exception:
        return {f'{i}:{window_day(local,w)}':w.limit_kwh for i,w in enumerate(settings.charging_windows) if inside(local,w.start,w.end)}

SCENARIOS={
    'low':{'name':'Conservative','solar_percentile':10,'load_percentile':90,'energy_basis':'12-hour','solar_key':'low','load_key':'high'},
    'middle':{'name':'Middle','solar_percentile':50,'load_percentile':50,'energy_basis':'12-hour','solar_key':'middle','load_key':'middle'},
    'high':{'name':'Optimistic','solar_percentile':90,'load_percentile':10,'energy_basis':'12-hour','solar_key':'high','load_key':'low'},
}


def safety_end(settings,now):
    """Protect the full configured forecast period."""
    return now+pd.Timedelta(hours=settings.horizon_hours)


def forecast_end(settings,now):
    return safety_end(settings,now)


def project(settings,rows,soc,eta_c,eta_d,usage,trusted,periods):
    """One hourly battery trajectory per 12-hour energy risk profile."""
    initial=soc if soc is not None else settings.minimum_soc;keys=tuple(SCENARIOS)
    positions={'low':0,'middle':1,'high':2};scenario_rows={}
    for key,scenario in SCENARIOS.items():
        load=periods['load'][positions[scenario['load_key']]];solar=periods['solar'][positions[scenario['solar_key']]]
        scenario_rows[key]=[{**r,'load_scenario_kw':float(load[i]),'solar_scenario_kw':float(solar[i]),'net_demand_kw':float(load[i]-solar[i])} for i,r in enumerate(rows)]
    scenario_net=np.asarray([[r['net_demand_kw'] for r in scenario_rows[k]] for k in keys])
    available={k:0. for k in keys};values=dict(available)
    periods_export=export_periods(settings,rows)
    peak=periods_export[0] if periods_export else [];cutoff=None
    if peak:
        last=peak[-1]
        cutoff=pd.Timestamp(rows[last]['time'])+pd.Timedelta(hours=rows[last]['hours'])
    schedule={}
    if trusted:
        stored,_=allowances(settings,rows,scenario_net,initial,eta_c,eta_d,usage,None)
        schedule=plan_exports(settings,rows,scenario_net,initial,eta_c,eta_d,usage)
        available=dict(zip(keys,stored.tolist()))
        if peak:values=dict(zip(keys,schedule[peak[0]].tolist()))
    baseline_states,baseline_imports,_=simulate_paths(settings,rows,scenario_net,initial,eta_c,eta_d,usage)
    baseline_safe=acceptable(settings,rows,baseline_states,baseline_imports)
    imports=sum(float(baseline_imports[1,i])*r['hours'] for i,r in enumerate(rows));projections={}
    for plan in ('none',*keys):
        key='middle' if plan=='none' else plan;amount=0. if plan=='none' else values[plan]
        selected_schedule={start:float(target[keys.index(key)]) if plan!='none' else 0. for start,target in schedule.items()}
        centre,unallowed,delivered=simulate(settings,scenario_rows[key],initial,eta_c,eta_d,used_allowance=usage,export_schedule=selected_schedule)
        centre_safe=bool(acceptable(settings,rows,np.asarray([[r['soc'] for r in centre]]),np.asarray([[r['outside_window_import_kw'] for r in centre]]))[0])
        states=simulate_paths(settings,rows,scenario_net,initial,eta_c,eta_d,usage,export_schedule=selected_schedule)[0]
        lower=states.min(axis=0);upper=states.max(axis=0);projection=[]
        for i,r in enumerate(rows):
            projection.append({'time':r['time'],'hours':r['hours'],'peak_window':r['peak_window'],
                               'charging_window':any(inside(pd.Timestamp(r['time']).tz_convert(settings.timezone),w.start,w.end) for w in settings.charging_windows),
                               'soc_point':centre[i]['soc'] if soc is not None else None,
                               'soc_scenario_low':float(lower[i]) if soc is not None else None,
                               'soc_scenario_high':float(upper[i]) if soc is not None else None,
                               'load_scenario_kw':scenario_rows[key][i]['load_scenario_kw'],
                               'solar_scenario_kw':scenario_rows[key][i]['solar_scenario_kw'],
                               **{f:centre[i][f] for f in ('requested_discharge_kw','battery_discharge_kw','grid_import_kw','battery_export_kw')}})
        projections[plan]={'requested_export_kwh':amount,'modeled_export_kwh':round(sum(centre[i]['battery_export_kw']*rows[i]['hours'] for i in peak),3),
                           'total_modeled_export_kwh':round(delivered,3),
                           'total_requested_export_kwh':round(sum(selected_schedule.values()),3),
                           'export_windows':[{'start':rows[period[0]]['time'],
                                              'end':(pd.Timestamp(rows[period[-1]]['time'])+pd.Timedelta(hours=rows[period[-1]]['hours'])).isoformat(),
                                              'requested_export_kwh':selected_schedule.get(period[0],0.),
                                              'discharge_rate_kw':export_rate(pd.Timestamp(rows[period[0]]['time']).tz_convert(settings.timezone),settings)} for period in periods_export],
                           'discharge_rate_kw':export_rate(pd.Timestamp(rows[peak[0]]['time']).tz_convert(settings.timezone),settings) if peak else 0.,'scenario':SCENARIOS[key],
                           'baseline_feasible':bool(baseline_safe[keys.index(key)]) if trusted else None,
                           'scenario_feasible':centre_safe if trusted else None,'unallowed_import_kwh':round(unallowed,3),
                           'flow_basis':'Selected 12-hour energy scenario with a separate allowance for each export window at its configured discharge rate',
                           'policy':POLICY,'forecast':projection}
    return available,values,projections,imports,peak,cutoff


def chart_history(settings,now):
    """Twenty-four hours of recorded five-minute averages from the live sensors."""
    start=now-pd.Timedelta(hours=24)
    sensors={'load_actual':settings.consumption_entity,'solar_actual':settings.solar_ct_entity,'soc_actual':settings.battery_soc_entity}
    frame=HomeAssistant(settings).frame(list(dict.fromkeys(sensors.values())),start,now,period='5minute')
    rows=[]
    for t,values in frame.iterrows():
        time=max(start,t+pd.Timedelta(minutes=2.5))
        if time>=now:continue
        rows.append({'time':time.isoformat(),**{key:float(values[entity]) if pd.notna(values[entity]) else None for key,entity in sensors.items()}})
    # Open-Meteo's recent-past coverage is model-based, not an on-site observation.
    try:
        weather=consumption_weather(settings,start.floor('h'),now.ceil('h'))['temperature_2m']
        times=pd.DatetimeIndex([pd.Timestamp(r['time']) for r in rows])
        temperatures=weather.reindex(weather.index.union(times)).sort_index().interpolate(method='time',limit_area='inside').reindex(times)
    except Exception:
        temperatures=pd.Series(float('nan'),index=range(len(rows)))
    for row,value in zip(rows,temperatures):row['temperature_actual']=float(value) if pd.notna(value) else None

    return rows


class Forecaster:
    def __init__(self):
        self.load_cache=None; self.solar_cache=None; self.load_at=0; self.solar_at=0
        self.history_cache=[];self.history_at=0

    def reset(self): self.load_at=0; self.solar_at=0; self.load_cache=None; self.solar_cache=None

    def run(self,settings,live,force=False):
        now=pd.Timestamp.now(tz='UTC'); end=forecast_end(settings,now)
        hours=pd.date_range(now.floor('h'),end.ceil('h'),freq='1h')
        warnings=[]; ready_load=model_ready(settings,'load','load'); ready_solar=all(model_ready(settings,'solar-'+b.id,'solar',b.id) for b in settings.banks)
        eta_meta=read_json('model-battery',{}); inv=read_json('model-inverter',{})
        battery_ready=model_ready(settings,'battery','battery'); inverter_ready=model_ready(settings,'inverter','inverter')
        if not ready_load: warnings.append('Consumption LightGBM needs calibration for the current configuration.')
        if not ready_solar: warnings.append('One or more panel-bank models need calibration.')
        if not battery_ready: warnings.append('Battery efficiencies need measured calibration.')
        if not inverter_ready: warnings.append('CT / inverter efficiency needs calibration.')
        if settings.solar_ct_proxy: warnings.append('Solar monitoring uses a PV proxy; inverter efficiency and achievable export are approximate until an independent AC/CT sensor is configured.')
        elapsed=now.timestamp()
        need_load=force or self.load_cache is None or elapsed-self.load_at>=settings.load_interval_minutes*60 or self.load_cache.index.max()<hours[-1]
        need_solar=force or self.solar_cache is None or elapsed-self.solar_at>=settings.solar_interval_minutes*60 or self.solar_cache.index.max()<hours[-1]
        if need_load or need_solar:
            # Fetch weather only when an actual model refresh is due.
            cache_hours=pd.date_range(hours[0],periods=73,freq='h')
            weather_available=True
            try:
                weather=consumption_weather(settings,cache_hours[0],cache_hours[-1]+pd.Timedelta(hours=1))
            except Exception as exc:
                weather_available=False
                warnings.append(f'Chosen weather model unavailable: {exc}')
                if ready_load:raise ValueError('Selected weather unavailable for consumption forecast') from exc
                weather=pd.DataFrame({'temperature_2m':20.,'shortwave_radiation':0.},index=cache_hours)
            calendar=Calendar(settings).load();f=features(weather,settings,calendar).reindex(cache_hours)
            if calendar.summary(hours)['missing_school_hours']:warnings.append('School calendar does not cover the forecast period.')
            if need_load:
                history_start=hours[0]-pd.Timedelta(days=58)
                policy=effective_policy(settings,'load')
                recent=HomeAssistant(settings).frame(settings.historical_consumption_entities,history_start,hours[0],measurement_kind='load',max_power_kw=policy['max_power_kw'],units=policy['units'],corrections=policy['corrections'])
                observed=splice(recent,settings.historical_consumption_entities,read_json('dataset-load',{}).get('request',{}).get('combine','splice')) if not recent.empty else pd.Series(dtype=float,index=pd.DatetimeIndex([],tz='UTC'))
                context,quality=quality_frame(observed,settings,'load',policy['max_power_kw'],corrections=read_json('dataset-load',{}).get('request',{}).get('corrections',[]),minmax=target_bounds(recent,settings.historical_consumption_entities))
                metadata=load_context_metadata(recent,settings.historical_consumption_entities,settings,quality,read_json('dataset-load',{}).get('request',{}).get('combine','splice'),policy['max_power_kw'])
                context.attrs={};f.attrs={}
                f=pd.concat([context,f]).sort_index();attach_load_context(f,metadata)
                profile=read_json('model-load',{}).get('hour_profile',{})
                if ready_load:
                    artifact=joblib.load(DATA/'load.joblib');predictor=LoadPredictor(artifact)
                    from .load_training import feature_rows
                    x=feature_rows(f,hours[0],cache_hours,settings.timezone,legacy=artifact.get('pipeline')=='lightgbm-quantile-direct-v2')
                    run=weather.attrs.get('run');x['weather_run_age']=(hours[0]-pd.Timestamp(run)).total_seconds()/3600 if run else np.nan
                    q=predictor.predict(x)
                    model_id=read_json('model-load',{}).get('trained_at','legacy')
                    q['base_point']=q['point'].copy()
                    q['point']=np.maximum(0,q['point']+recent_correction(settled_records('load',hours[0],model_identity=model_id),hours[0],x.lead_hours,artifact.get('correction')))
                    record_issue('load',hours[0],cache_hours,x,q,weather,model_id,issued_at=now.isoformat())
                    values={f'load_{k}':q[k] for k in ('low','middle','high','point')}
                else:
                    base=np.array([float(profile.get(str(t.tz_convert(settings.timezone).hour),1.)) for t in cache_hours])
                    values={f'load_{k}':base for k in ('low','middle','high','point')}
                    warnings.append('Generic hourly consumption profile is display-only; allowances require calibration.')
                values['temperature_c']=weather['temperature_2m'].reindex(cache_hours).astype(object).where(lambda x:x.notna(),None).to_numpy() if weather_available else [None]*len(cache_hours)
                self.load_cache=pd.DataFrame(values,index=cache_hours);self.load_at=elapsed
            if need_solar:
                total=pd.DataFrame(0.,index=cache_hours,columns=['solar_low','solar_middle','solar_high','solar_point',*[f'solar_{k}_{b.id}' for b in settings.banks for k in ('low','middle','high','point')]])
                for bank in settings.banks:
                    name='solar-'+bank.id
                    if model_ready(settings,name,'solar',bank.id):
                        from .solar_training import solar_predict as bank_quantiles
                        solar=solar_weather(settings,bank,cache_hours[0],cache_hours[-1]+pd.Timedelta(hours=1))
                        x=bank_features(solar.reindex(cache_hours),bank,settings,hours[0])
                        if x.drop(columns=['weather_run_age','lead_hours']).isna().any().any(): raise ValueError(f'Incomplete Open-Meteo solar forecast for {bank.name}')
                        model_id=read_json('model-'+name,{}).get('trained_at','legacy')
                        pred=bank_quantiles(joblib.load(DATA/(name+'.joblib')),x,settled_records(name,hours[0],model_identity=model_id),hours[0])
                        record_issue(name,hours[0],cache_hours,x,pred,solar,model_id,issued_at=now.isoformat())
                    else: pred={k:np.zeros(len(cache_hours)) for k in ('low','middle','high','point')}
                    for k in ('low','middle','high','point'):
                        total[f'solar_{k}_{bank.id}']=pred[k];total['solar_'+k]+=pred[k]
                eff=inv.get('efficiency',.9) if inverter_ready else .9
                total[['solar_low','solar_middle','solar_high','solar_point']]*=eff
                self.solar_cache=total; self.solar_at=elapsed
        # Fifteen minute dispatch plus a partial first/final step. Preserve exact horizon duration.
        boundaries=[now]+list(pd.date_range(now.ceil('15min'),end,freq='15min'))+[end]
        first_day=now.tz_convert(settings.timezone).normalize()-pd.DateOffset(days=1)
        protected_until=end
        clocks={'00:00','06:00','18:00',*[clock for w in settings.export_windows for clock in (w.start,w.end)],*[clock for w in settings.charging_windows for clock in (w.start,w.end)]}
        for day in pd.date_range(first_day,end.tz_convert(settings.timezone).normalize()+pd.DateOffset(days=1),freq='D'):
            for clock in clocks:
                boundary=pd.Timestamp(f'{day.date()} {clock}').tz_localize(settings.timezone,ambiguous=False,nonexistent='shift_forward').tz_convert('UTC')
                if now<boundary<end:boundaries.append(boundary)
        boundaries=sorted(set(boundaries)); rows=[]
        for a,b in zip(boundaries,boundaries[1:]):
            t=a.floor('h')
            row={'time':a.isoformat(),'hours':(b-a).total_seconds()/3600,'peak_window':export_active(a.tz_convert(settings.timezone),settings),**self.load_cache.loc[t].to_dict(),**self.solar_cache.loc[t].to_dict()}
            row['charging_window']=any(inside(a.tz_convert(settings.timezone),w.start,w.end) for w in settings.charging_windows)
            row['net_demand_kw']=row['load_point']-row['solar_point'];row['net_demand_low']=row['load_low']-row['solar_high'];row['net_demand_high']=row['load_high']-row['solar_low'];rows.append(row)
        soc=live.get('soc'); age=snapshot_age(live,now)
        trusted=bool(ready_load and ready_solar and battery_ready and inverter_ready and soc is not None and age<900)
        if soc is not None and soc<settings.minimum_soc:warnings.append('Reported battery SoC is below the operating minimum; the projection retains the reported state.')
        if settings.mc_samples<500:warnings.append('Small Monte Carlo sample count: percentile and risk estimates may be unstable.')
        if soc is None: warnings.append('Battery SoC is unavailable or invalid in Home Assistant; export allowance is zero.')
        elif age>=900: warnings.append('No successful Home Assistant snapshot within 15 minutes; export allowance is zero.')
        missing=[name for name in ('load_kw','solar_kw','battery_kw') if live.get(name) is None]
        if missing:warnings.append('Live HA readings unavailable: '+', '.join(missing)+'. Historical forecasts remain in use.')
        eta_c=eta_meta.get('charge_efficiency',.9) if battery_ready else .9; eta_d=eta_meta.get('discharge_efficiency',.9) if battery_ready else .9
        usage=allowance_used(settings,now)
        periods,history=energy_scenarios(settings,rows,now,inv.get('efficiency',.9) if inverter_ready else .9)
        trusted=trusted and bool(periods['periods'])
        history['status']='Forecast available; coverage not established' if trusted else 'Awaiting calibrated models and current SoC'
        mapping={'low':10,'middle':50,'high':90}
        available,values,projections,imports,peak_indices,cutoff=project(settings,rows,soc,eta_c,eta_d,usage,trusted,periods)
        if force or elapsed-self.history_at>=300:
            try:
                self.history_cache=chart_history(settings,now);self.history_at=elapsed
            except Exception:
                warnings.append('Recorded chart history is temporarily unavailable.')
        actual_history=[r for r in self.history_cache if now-pd.Timedelta(hours=24)<=pd.Timestamp(r['time'])<now]
        chart=[{**row,**projections['low']['forecast'][i]} for i,row in enumerate(rows)]
        result={'updated_at':now.isoformat(),'valid_until':(now+pd.Timedelta(minutes=max(15,settings.battery_interval_minutes*3))).isoformat(),
                'ready':trusted,'peak_active':export_active(now.tz_convert(settings.timezone),settings),
                'percentiles':mapping,'percentile_basis':'Scenario labels only; see scenarios for separate solar/load input quantiles','horizon_hours':round((end-now).total_seconds()/3600,2),'scenarios':SCENARIOS,'energy_period_forecast':periods['periods'],'energy_period_basis':periods['basis'],'excess_kwh':available,'achievable_export_kwh':values,
                'excess_percent':{k:round(v/settings.battery_capacity_kwh*100,2) for k,v in available.items()},
                'achievable_export_percent':{k:round(v/eta_d/settings.battery_capacity_kwh*100,2) for k,v in values.items()},
                'forecast':chart,'soc_projections':projections,'actual_history':actual_history,
                'export_policy':{'name':POLICY,'schedule':'Every forecast export window uses its configured discharge rate and its own energy allowance, allocated in time order','discharge_rate_kw':projections['middle']['discharge_rate_kw'],
                    'safety_horizon_hours':round((end-now).total_seconds()/3600,2),'safety_end':end.isoformat(),'horizon_basis':'Safety checks use the complete configured forecast period',
                    'reserve_rule':'Retain configured minimum plus reserve at every forecast step',
                    'import_rule':'No household imports beyond configured charging-window allowances, including baseline imports',
                    'import_tolerance_kwh':IMPORT_TOLERANCE_KWH,'reserve_tolerance_soc':RESERVE_TOLERANCE_SOC,'delivery_tolerance_kwh':DELIVERY_TOLERANCE_KWH,
                    'execution':'Advisory only; external controllers must execute this schedule to share its assumptions'},
                'planned_export_window':{'start':rows[peak_indices[0]]['time'],'end':cutoff.isoformat(),'discharge_rate_kw':projections['middle']['discharge_rate_kw']} if peak_indices else None,
                'warnings':warnings,'live':live,'outside_window_import_kwh':round(imports,3),
                'model_updated_at':{'load':pd.Timestamp(self.load_at,unit='s',tz='UTC').isoformat(),'solar':pd.Timestamp(self.solar_at,unit='s',tz='UTC').isoformat()},
                'uncertainty':history,'load_point_method':read_json('model-load',{}).get('point_method',{'head':'middle','baseline_weight':0}),'load_forecast_basis':LOAD_PIPELINE,'soc_band_method':POLICY,'weather_model':settings.weather_model,'quality_policy':QUALITY_POLICY,'source_policy':effective_policy(settings,'load'),
                'calibration':{'charge_efficiency':eta_c,'discharge_efficiency':eta_d,'inverter_efficiency':inv.get('efficiency')},
                'used_window_allowance_kwh':usage,
                'energy_metric_basis':{'excess_kwh':'Stored battery energy above forecast household needs and configured minimum/reserve; not capped by export power or peak time',
                                       'achievable_export_kwh':'AC energy deliverable to the grid in the current/upcoming peak; limited by excess, discharge losses, inverter power and remaining time'},
                'note':'Conservative uses P10 solar/P90 consumption per 12-hour block, middle P50/P50, optimistic P90/P10. Blocks are 06:00–18:00 and 18:00–06:00 local time. Each block follows the forecast hourly shape. A single trajectory per scenario determines its export allowance over the full forecast at the same fixed discharge rate. Every charging and export window is simulated; export allowances are allocated in time order. Shading spans scenarios under the same per-window export targets; it is not a probability interval.'}
        write_json('forecast',result); return result

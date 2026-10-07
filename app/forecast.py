import json, math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import joblib, numpy as np, pandas as pd
from .config import DATA, read_json, write_json
from .sources import HomeAssistant, Weather, Calendar, forecast_weather
from .datasets import features, bank_features, splice
from .models import interval, model_ready
from .ha_inputs import snapshot_age
from .load_training import LoadPredictor, cumulative_bands, consumption_paths
import torch

def inside(t,start,end):
    a=datetime.strptime(start,'%H:%M').time(); b=datetime.strptime(end,'%H:%M').time(); v=t.time().replace(tzinfo=None)
    return a<=v<b if a<b else v>=a or v<b

def window_day(t,w):
    if w.start>w.end and t.strftime('%H:%M')<w.end: return (t-timedelta(days=1)).date().isoformat()
    return t.date().isoformat()

def simulate(settings, rows, initial_soc, eta_c, eta_d, export_kwh=0, export_cutoff=None, scenario='middle', used_allowance=None, export_schedule=None):
    capacity=settings.battery_capacity_kwh; minimum=capacity*settings.minimum_soc/100
    energy=capacity*initial_soc/100; reserve=minimum+settings.reserve_kwh
    usage=dict(used_allowance or {}); remaining=export_kwh; result=[]; outside_import=0
    for row_index,row in enumerate(rows):
        t=pd.Timestamp(row['time']).tz_convert(settings.timezone); dt=row['hours']
        load=row['load_'+('high' if scenario=='low' else 'low' if scenario=='high' else 'middle')]
        solar=row['solar_'+scenario]
        charge=0.; discharge=0.; imported=0.; exported=0.; outside_kw=0.; power_import=0.; energy_import=0.
        net=solar-load
        windows=[(i,w) for i,w in enumerate(settings.charging_windows) if inside(t,w.start,w.end)]
        window=windows[0] if windows else None
        if net>=0:
            charge=min(net,settings.battery_charge_limit_kw,(capacity-energy)/(eta_c*dt))
            energy+=charge*eta_c*dt
        else:
            demand=-net
            # Planned grid windows supply household load first, then the battery.
            if window:
                i,w=window; key=f'{i}:{window_day(t,w)}'; available=max(0,w.limit_kwh-usage.get(key,0))/dt
                imported=min(demand,available); usage[key]=usage.get(key,0)+imported*dt; demand-=imported
            discharge=min(demand,settings.battery_discharge_limit_kw,max(0,energy-minimum)*eta_d/dt)
            energy-=discharge*dt/eta_d; imported+=max(0,demand-discharge)
            outside_kw=max(0,demand-discharge)
            power_import=max(0,demand-settings.battery_discharge_limit_kw)
            energy_import=max(0,outside_kw-power_import)
            outside_import+=outside_kw*dt
        if window:
            i,w=window; key=f'{i}:{window_day(t,w)}'; available=max(0,w.limit_kwh-usage.get(key,0))/dt
            grid_charge=min(w.charge_kw,max(0,settings.battery_charge_limit_kw-charge),available,max(0,capacity-energy)/(eta_c*dt))
            charge+=grid_charge; energy+=grid_charge*eta_c*dt; imported+=grid_charge; usage[key]=usage.get(key,0)+grid_charge*dt
        eligible=inside(t,settings.export_start,settings.export_end) and (export_cutoff is None or pd.Timestamp(row['time'])<export_cutoff)
        requested_export=export_schedule[row_index] if export_schedule is not None else (remaining/dt if eligible else 0)
        if requested_export>0:
            extra=min(requested_export,max(0,settings.battery_discharge_limit_kw-discharge),max(0,energy-reserve)*eta_d/dt)
            energy-=extra*dt/eta_d; remaining-=extra*dt; exported=extra
        result.append({**row,'soc':round(100*max(minimum,energy)/capacity,3),'grid_import_kw':round(imported,3),'battery_export_kw':round(exported,3),'outside_window_import_kw':outside_kw,'power_limited_import_kw':power_import,'energy_shortfall_import_kw':energy_import,'charge_kw':round(charge,3),'discharge_kw':round(discharge,3)})
    return result,outside_import,export_kwh-remaining

def stored_excess(settings,rows,soc,eta_c,eta_d,scenario,used_allowance):
    """Stored kWh removable now without additional imports or using forecast reserves.

    The counterfactual removes energy from initial storage, so neither the peak
    window nor export inverter power caps it. Normal household/charging physics
    still apply. Later battery saturation naturally removes today's influence.
    """
    baseline=simulate(settings,rows,soc,eta_c,eta_d,scenario=scenario,used_allowance=used_allowance)[0]
    reserve_soc=min(100,settings.minimum_soc+100*settings.reserve_kwh/settings.battery_capacity_kwh)
    lower=0.;upper=max(0,(soc-reserve_soc)*settings.battery_capacity_kwh/100)
    for _ in range(22):
        amount=(lower+upper)/2
        candidate=simulate(settings,rows,soc-100*amount/settings.battery_capacity_kwh,eta_c,eta_d,scenario=scenario,used_allowance=used_allowance)[0]
        extra_import=sum(max(0,c['outside_window_import_kw']-b['outside_window_import_kw'])*c['hours'] for c,b in zip(candidate,baseline))
        reserves=all(c['soc']+.001>=min(reserve_soc,b['soc']) for c,b in zip(candidate,baseline))
        if extra_import<=.01 and reserves:lower=amount
        else:upper=amount
    return max(0,math.floor(lower*100)/100)

def excess(settings,rows,soc,eta_c,eta_d,scenario,used_allowance):
    # Export only in the first current/upcoming peak window. Preserve reserves through the entire horizon.
    peak=[i for i,r in enumerate(rows) if inside(pd.Timestamp(r['time']).tz_convert(settings.timezone),settings.export_start,settings.export_end)]
    if not peak: return 0.,simulate(settings,rows,soc,eta_c,eta_d,scenario=scenario,used_allowance=used_allowance)[0]
    first=peak[0]; last=first
    while last+1<len(rows) and last+1 in peak: last+=1
    cutoff=pd.Timestamp(rows[last]['time'])+pd.Timedelta(hours=rows[last]['hours'])
    base,imports,_=simulate(settings,rows,soc,eta_c,eta_d,scenario=scenario,used_allowance=used_allowance)
    reserve_soc=min(100,settings.minimum_soc+100*settings.reserve_kwh/settings.battery_capacity_kwh)
    lower=0.; upper=settings.battery_capacity_kwh*eta_d
    for _ in range(22):
        mid=(lower+upper)/2
        candidate,imp,delivered=simulate(settings,rows,soc,eta_c,eta_d,mid,cutoff,scenario,used_allowance)
        extra_import=sum(max(0,c['outside_window_import_kw']-b['outside_window_import_kw'])*c['hours'] for c,b in zip(candidate,base))
        reserves=all(c['soc']+.001>=min(reserve_soc,b['soc']) for c,b in zip(candidate,base))
        if extra_import<=.01 and delivered>=mid-.001 and reserves: lower=mid
        else: upper=mid
    amount=max(0,math.floor(lower*100)/100)
    path,_,_=simulate(settings,rows,soc,eta_c,eta_d,amount,cutoff,scenario,used_allowance)
    return amount,path

def allowance_used(settings, now):
    """Estimate today's actual free-window import from measured load, solar and battery I/O."""
    calibration=read_json('battery-calibration',{})
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

class Forecaster:
    def __init__(self): self.load_cache=None; self.solar_cache=None; self.load_at=0; self.solar_at=0; self.load_range_info={}; self.load_path_cache=None; self.load_chart_range_info={}; self.weather_inputs={}; self.load_weather_inputs={}; self.solar_weather_inputs={}; self.solar_alternatives={}

    def reset(self): self.load_at=0; self.solar_at=0

    def run(self,settings,live,force=False):
        now=pd.Timestamp.now(tz='UTC'); end=now+pd.Timedelta(hours=settings.horizon_hours)
        hours=pd.date_range(now.floor('h'),end.ceil('h'),freq='1h')
        warnings=[]; ready_load=model_ready(settings,'load','load'); ready_solar=all(model_ready(settings,'solar-'+b.id,'solar',b.id) for b in settings.banks)
        eta_meta=read_json('battery-calibration',{}); inv=read_json('inverter-calibration',{})
        battery_ready=model_ready(settings,'battery','battery'); inverter_ready=model_ready(settings,'inverter','inverter')
        if not ready_load: warnings.append('Consumption CNN needs calibration for the current configuration.')
        if not ready_solar: warnings.append('One or more panel-bank models need calibration.')
        if not battery_ready: warnings.append('Battery efficiencies need measured calibration.')
        if not inverter_ready: warnings.append('CT / inverter efficiency needs calibration.')
        elapsed=now.timestamp()
        need_load=force or self.load_cache is None or elapsed-self.load_at>=settings.load_interval_minutes*60 or self.load_cache.index.max()<hours[-1]
        need_solar=force or self.solar_cache is None or elapsed-self.solar_at>=settings.solar_interval_minutes*60 or self.solar_cache.index.max()<hours[-1]
        if need_load or need_solar:
            # Fetch weather only when an actual model refresh is due.
            cache_hours=pd.date_range(hours[0],hours[-1]+pd.Timedelta(hours=max(24,min(72,max(settings.load_interval_minutes,settings.solar_interval_minutes)/60))),freq='1h')
            weather=Weather(settings).fetch(hours[0]-pd.Timedelta(hours=72),cache_hours[-1],historical=False)
            try:
                chosen=forecast_weather(settings,hours[0]-pd.Timedelta(hours=1),cache_hours[-1],settings.weather_model)
                weather.loc[weather.index>=hours[0],chosen.columns]=chosen.reindex(weather.index[weather.index>=hours[0]])
                self.weather_inputs={'source':chosen.attrs,'hourly':{str(t):{k:float(v) for k,v in row.items()} for t,row in chosen.loc[(chosen.index>=hours[0])&(chosen.index<=cache_hours[-1])].dropna().iterrows()}}
            except Exception as exc:
                warnings.append(f'Chosen weather model unavailable; using best match: {exc}')
                self.weather_inputs={'source':'Open-Meteo best match','retrieved_at':now.isoformat()}
            calendar=Calendar(settings).load(); f=features(weather,settings,calendar)
            if calendar.summary(hours)['missing_school_hours']: warnings.append('School calendar does not cover the forecast period.')
            if need_load:
                self.load_weather_inputs=self.weather_inputs
                history_start=hours[0]-pd.Timedelta(days=31)
                historical=features(Weather(settings).fetch(history_start,hours[0],historical=True),settings,calendar)
                f=pd.concat([historical.loc[historical.index<hours[0]-pd.Timedelta(hours=72)],f]).sort_index();f=f[~f.index.duplicated(keep='last')]
                recent=HomeAssistant(settings).frame(settings.historical_consumption_entities,history_start,hours[0])
                observed=splice(recent,settings.historical_consumption_entities) if not recent.empty else pd.Series(dtype=float)
                meta=read_json('model-load',{})
                profile=meta.get('hour_profile',{})
                f['target']=observed.reindex(f.index)
                for t in f.index[f.index<hours[0]]:
                    if pd.isna(f.at[t,'target']): f.at[t,'target']=profile.get(str(t.tz_convert(settings.timezone).hour),1.)
                if ready_load:
                    artifact=torch.load(DATA/'load.pt',map_location='cpu',weights_only=False)
                    predictor=LoadPredictor(artifact)
                    values=predictor.forecast(f,cache_hours)
                    calibration=artifact.get('uncertainty')
                    if calibration and calibration.get('paths'):
                        lows,highs,self.load_range_info=cumulative_bands(values,calibration,settings.confidence,cache_hours[0].tz_convert(settings.timezone).hour)
                        trajectories,display_low,display_high,self.load_chart_range_info=consumption_paths(values,calibration,settings.confidence,cache_hours[0].tz_convert(settings.timezone).hour)
                    else:
                        lo,hi=interval('load',settings.confidence)
                        lows=np.maximum(0,values+lo);highs=np.maximum(0,values+hi)
                        display_low=lows;display_high=highs;trajectories=np.stack([lows,values,highs]);self.load_chart_range_info={'method':'legacy hourly errors'}
                        self.load_range_info={'method':'legacy hourly errors','note':'Run fine tuning to calibrate cumulative recursive ranges'}
                else:
                    values=np.array([float(profile.get(str(t.tz_convert(settings.timezone).hour),1.)) for t in cache_hours])
                    lows=np.zeros(len(values));highs=values+1
                    self.load_range_info={'method':'uncalibrated'}
                    display_low=lows;display_high=highs;trajectories=np.stack([lows,values,highs]);self.load_chart_range_info={'method':'uncalibrated'}
                self.load_cache=pd.DataFrame({'load_low':lows,'load_middle':values,'load_high':highs,'load_band_low':display_low,'load_band_high':display_high},index=cache_hours)
                self.load_path_cache=pd.DataFrame(trajectories.T,index=cache_hours);self.load_at=elapsed
            if need_solar:
                self.solar_weather_inputs={'primary':self.weather_inputs,'comparisons':{}}
                total=pd.DataFrame(0.,index=cache_hours,columns=['solar_low','solar_middle','solar_high'])
                for bank in settings.banks:
                    name='solar-'+bank.id; x=bank_features(f.reindex(cache_hours),bank,settings)
                    if model_ready(settings,name,'solar',bank.id):
                        model=joblib.load(DATA/(name+'.joblib')); pred=np.maximum(0,model.predict(x)); lo,hi=interval(name,settings.confidence)
                    else: pred=np.zeros(len(x)); lo=0.; hi=0.
                    night=x.tilted_irradiance.to_numpy()<5
                    for label,offset in [('low',lo),('middle',0),('high',hi)]:
                        p=np.maximum(0,pred+offset); p[night]=0; total['solar_'+label]+=p
                total*=inv.get('efficiency',.9) if inverter_ready else .9
                solar_source=settings.weather_model if isinstance(self.weather_inputs.get('source'),dict) else 'best_match'
                self.solar_alternatives={solar_source:{str(t):float(v) for t,v in total.solar_middle.items()}}
                if settings.compare_weather_models and ready_solar:
                    predictions=[]
                    for weather_model in ('ecmwf_ifs','gfs_global','icon_global'):
                        if weather_model==solar_source: predictions.append(total.solar_middle.to_numpy());continue
                        try:
                            alt_weather=forecast_weather(settings,cache_hours[0],cache_hours[-1],weather_model)
                            self.solar_weather_inputs['comparisons'][weather_model]={'source':alt_weather.attrs,'hourly':{str(t):{k:float(v) for k,v in row.items()} for t,row in alt_weather.reindex(cache_hours).dropna().iterrows()}}
                            alt=features(alt_weather,settings,calendar)
                            output=np.zeros(len(cache_hours))
                            for bank in settings.banks:
                                x=bank_features(alt.reindex(cache_hours),bank,settings)
                                if x.isna().any().any(): raise ValueError('Incomplete irradiance coverage')
                                pred=np.maximum(0,joblib.load(DATA/('solar-'+bank.id+'.joblib')).predict(x));pred[x.tilted_irradiance.to_numpy()<5]=0
                                output+=pred
                            output*=inv.get('efficiency',.9)
                            self.solar_alternatives[weather_model]={str(t):float(v) for t,v in zip(cache_hours,output)};predictions.append(output)
                        except Exception as exc: warnings.append(f'{weather_model} comparison unavailable: {exc}')
                    if len(predictions)>1:
                        # Model disagreement supplements held-out residuals; it is not a calibrated ensemble probability.
                        spread=np.std(predictions,axis=0)
                        total['solar_low']=np.maximum(0,total.solar_low-spread)
                        total['solar_high']+=spread
                        blend=np.mean(predictions,axis=0)
                        if settings.solar_blend:
                            total['solar_middle']=blend
                            total['solar_low']=np.minimum(total.solar_low,blend)
                            total['solar_high']=np.maximum(total.solar_high,blend)
                        self.solar_alternatives['equal_blend']={str(t):float(v) for t,v in zip(cache_hours,blend)}
                self.solar_cache=total; self.solar_at=elapsed
        from .forecast_measurement import solar_correction, archive_issue
        corrected_solar,correction_info=solar_correction(settings,now,self.solar_cache,live)
        from .appliances import known_load
        # Fifteen minute dispatch plus a partial first/final step. Preserve exact horizon duration.
        boundaries=[now]+list(pd.date_range(now.ceil('15min'),end,freq='15min'))+[end]
        boundaries=sorted(set(boundaries)); rows=[]
        for a,b in zip(boundaries,boundaries[1:]):
            t=a.floor('h')
            row={'time':a.isoformat(),'hours':(b-a).total_seconds()/3600,'peak_window':inside(a.tz_convert(settings.timezone),settings.export_start,settings.export_end),**self.load_cache.loc[t].to_dict(),**corrected_solar.loc[t].to_dict()}
            floor,extra,names=known_load(settings,live,a,now)
            for key in ('load_low','load_middle','load_high','load_band_low','load_band_high'):row[key]=max(row[key],floor)+extra
            row['known_load_kw']=floor+extra;row['known_loads']=names;rows.append(row)
        soc=live.get('soc'); age=snapshot_age(live,now)
        trusted=bool(ready_load and ready_solar and battery_ready and inverter_ready and soc is not None and age<900)
        if soc is None: warnings.append('Battery SoC is unavailable or invalid in Home Assistant; export allowance is zero.')
        elif age>=900: warnings.append('No successful Home Assistant snapshot within 15 minutes; export allowance is zero.')
        missing=[name for name in ('load_kw','solar_kw','battery_kw') if live.get(name) is None]
        if missing:warnings.append('Live HA readings unavailable: '+', '.join(missing)+'. Historical forecasts remain in use.')
        eta_c=eta_meta.get('charge_efficiency',.9) if battery_ready else .9; eta_d=eta_meta.get('discharge_efficiency',.9) if battery_ready else .9
        usage=allowance_used(settings,now); values={}; energy_excess={}; paths={}
        for scenario in ('low','middle','high'):
            energy_excess[scenario]=stored_excess(settings,rows,soc,eta_c,eta_d,scenario,usage) if trusted else 0.
            amount,path=excess(settings,rows,soc if soc is not None else settings.minimum_soc,eta_c,eta_d,scenario,usage)
            values[scenario]=amount if trusted else 0.; paths[scenario]=path
        active=inside(now.tz_convert(settings.timezone),settings.export_start,settings.export_end)
        available=dict(energy_excess)
        times=pd.DatetimeIndex([pd.Timestamp(r['time']).floor('h') for r in rows])
        power_paths=self.load_path_cache.reindex(times).to_numpy().T.copy()
        for i,row in enumerate(rows):
            floor,extra,_=known_load(settings,live,pd.Timestamp(row['time']),now)
            power_paths[:,i]=np.maximum(power_paths[:,i],floor)+extra
        # Check the conservative allowance against the same correlated error trajectories.
        # This preserves uncertainty after a full-battery reset as well as before it.
        trajectory_imports=0.
        if trusted:
            for powers in power_paths:
                sample_rows=[{**r,**{'load_'+level:float(p) for level in ('low','middle','high')}} for r,p in zip(rows,powers)]
                _,sample_imports,_=simulate(settings,sample_rows,soc,eta_c,eta_d,scenario='low',used_allowance=usage)
                trajectory_imports=max(trajectory_imports,sample_imports)
                raw=stored_excess(settings,sample_rows,soc,eta_c,eta_d,'low',usage)
                energy_excess['low']=min(energy_excess['low'],raw)
                allowance,_=excess(settings,sample_rows,soc,eta_c,eta_d,'low',usage)
                values['low']=min(values['low'],allowance)
            energy_excess['low']=min(energy_excess['low'],energy_excess['middle'])
            values['low']=min(values['low'],values['middle'])
            available['low']=energy_excess['low']
        # Compare all chart ranges under the selected plan, never different plans inside one band.
        peak_indices=[i for i,r in enumerate(rows) if r['peak_window']]
        cutoff=None
        if peak_indices:
            last=peak_indices[0]
            while last+1 in peak_indices: last+=1
            cutoff=pd.Timestamp(rows[last]['time'])+pd.Timedelta(hours=rows[last]['hours'])
        projections={}
        for plan in ('none','low','middle','high'):
            amount=0. if plan=='none' else values[plan]
            planned=simulate(settings,rows,soc if soc is not None else settings.minimum_soc,eta_c,eta_d,amount,cutoff,'middle' if plan=='none' else plan,usage)[0]
            schedule=[r['battery_export_kw'] for r in planned] if trusted else [0.]*len(rows)
            centre=simulate(settings,rows,soc if soc is not None else settings.minimum_soc,eta_c,eta_d,used_allowance=usage,export_schedule=schedule)[0]
            sample_soc=[]; sample_added=[]
            for powers in power_paths:
                for solar_case in ('low','middle','high'):
                    sample_rows=[{**r,'load_middle':float(p),'solar_middle':r['solar_'+solar_case]} for r,p in zip(rows,powers)]
                    path=simulate(settings,sample_rows,soc if soc is not None else settings.minimum_soc,eta_c,eta_d,used_allowance=usage,export_schedule=schedule)[0]
                    no_export=simulate(settings,sample_rows,soc if soc is not None else settings.minimum_soc,eta_c,eta_d,used_allowance=usage)[0]
                    sample_added.append(sum(max(0,c['outside_window_import_kw']-b['outside_window_import_kw'])*c['hours'] for c,b in zip(path,no_export)))
                    sample_soc.append([r['soc'] for r in path])
            sample_soc=np.asarray(sample_soc)
            soc_low=np.minimum(sample_soc.min(axis=0),[r['soc'] for r in centre])
            soc_high=np.maximum(sample_soc.max(axis=0),[r['soc'] for r in centre])
            projection=[]
            for i,r in enumerate(rows):
                projection.append({'time':r['time'],'hours':r['hours'],'peak_window':r['peak_window'],'soc_low':float(soc_low[i]),'soc_middle':centre[i]['soc'],'soc_high':float(soc_high[i]),'grid_import_kw':centre[i]['grid_import_kw'],'battery_export_kw':centre[i]['battery_export_kw']})
            delivered=sum(r['battery_export_kw']*r['hours'] for r in centre)
            projections[plan]={'requested_export_kwh':amount,'modeled_export_kwh':round(delivered,3),'max_added_import_kwh':round(max(sample_added),3),'forecast':projection}
        chart=[{**row,**projections['low']['forecast'][i]} for i,row in enumerate(rows)]
        baseline,imports,_=simulate(settings,rows,soc if soc is not None else settings.minimum_soc,eta_c,eta_d,used_allowance=usage,scenario='low')
        imports=max(imports,trajectory_imports)
        if imports>.01:
            warnings.append(f'Up to {imports:.2f} kWh of outside-window import is already forecast without exporting. Allowances prevent additional imports caused by export; unavoidable inverter-limit imports do not zero the energy surplus.')
        result={'updated_at':now.isoformat(),'valid_until':(now+pd.Timedelta(minutes=max(15,settings.battery_interval_minutes*3))).isoformat(),'ready':trusted,'peak_active':active,'confidence':settings.confidence,'horizon_hours':settings.horizon_hours,'excess_kwh':available,'achievable_export_kwh':{k:round(v,2) for k,v in values.items()},'excess_percent':{k:round(v/settings.battery_capacity_kwh*100,2) for k,v in available.items()},'achievable_export_percent':{k:round(v/eta_d/settings.battery_capacity_kwh*100,2) for k,v in values.items()},'forecast':chart,'soc_projections':projections,'planned_export_window':{'start':rows[peak_indices[0]]['time'],'end':cutoff.isoformat()} if peak_indices else None,'warnings':warnings,'live':live,'outside_window_import_kwh':round(imports,3),'model_updated_at':{'load':pd.Timestamp(self.load_at,unit='s',tz='UTC').isoformat(),'solar':pd.Timestamp(self.solar_at,unit='s',tz='UTC').isoformat()},'load_range_calibration':self.load_range_info,'consumption_band_calibration':self.load_chart_range_info,'soc_band_method':'Scenario envelope of correlated consumption-error paths and solar scenarios under the selected common export plan','calibration':{'charge_efficiency':eta_c,'discharge_efficiency':eta_d,'inverter_efficiency':inv.get('efficiency')},'used_window_allowance_kwh':usage,'energy_metric_basis':{'excess_kwh':'Stored battery energy above forecast household needs and configured minimum/reserve; not capped by export power or peak time','achievable_export_kwh':'AC energy deliverable to the grid in the current/upcoming peak; limited by excess, discharge losses, inverter power and remaining time'},'note':'Excess kWh is stored battery energy; excess percent is its share of battery capacity. Achievable export kWh is AC grid energy; its percent is the battery SoC consumed including discharge losses. Chart plans export the achievable amount, leaving any energy excess that cannot be exported in the window. Low uses high load and low solar. Surplus is reported continuously; peak_active independently indicates tariff timing.'}
        result['solar_correction']=correction_info;result['weather_model']=settings.weather_model
        archive_issue(settings,result,{'load':self.load_weather_inputs,'solar':self.solar_weather_inputs},self.solar_alternatives)
        write_json('forecast',result); return result

"""Paired common-window aggregate and trajectory diagnostics; no model tuning."""
import numpy as np,pandas as pd
from .config import DATA,read_json,write_json
from .study import paired_bootstrap
from .uncertainty import sample,solar_dependence,METHOD,evidence_signature

def aggregate(settings):
    names=['load',*['solar-'+b.id for b in settings.banks]]
    details={name:read_json(name+'-assessment-details') for name in names}
    if not all(details.values()):raise ValueError('Calibrate load and every bank before aggregate comparison')
    results={};efficiency=read_json('model-inverter',{}).get('efficiency',1.)
    for variant,key in [('candidate','records'),('existing','old_records')]:
        maps={name:{(r['origin'],r['time']):r for r in d.get(key,[])} for name,d in details.items()}
        common=set.intersection(*(set(m) for m in maps.values()));rows=[]
        for origin,time in sorted(common):
            load=maps['load'][(origin,time)];banks=[maps[name][(origin,time)] for name in names[1:]]
            actual=sum(r['actual'] for r in banks);point=sum(r['point'] for r in banks)
            rows.append({'origin':origin,'time':time,'solar_actual':actual,'solar_point':point,'net_actual':load['actual']-actual*efficiency,'net_point':load['point']-point*efficiency,'load':load,'banks':banks})
        by_origin={origin:[r for r in rows if r['origin']==origin] for origin in sorted({r['origin'] for r in rows})}
        outcomes={kind:{str(h):[] for h in (24,48,72)} for kind in ('solar','net')};trajectory={kind:{str(h):[] for h in (24,48,72)} for kind in ('load','solar','net')}
        metadata={name:read_json('candidate-'+name,read_json('model-'+name,{})) if variant=='candidate' else read_json('model-'+name,{}) for name in names}
        metadata={name:{**metadata[name],**details[name].get('sampling_metadata',{}).get(variant,{})} for name in names}
        rho=metadata['load'].get('sampling_rho') or 0.
        solar_rho=solar_dependence([metadata[name] for name in names[1:]])['rho']
        for origin,rr in by_origin.items():
            rr=sorted(rr,key=lambda r:r['time']);index=pd.DatetimeIndex([r['time'] for r in rr]);origin=pd.Timestamp(origin)
            load_q={k:np.array([r['load'][k] for r in rr]) for k in ('low','middle','high')};bank_q=[{k:np.array([r['banks'][j][k] for r in rr]) for k in ('low','middle','high')} for j in range(len(names)-1)]
            zero={k:np.zeros(len(rr)) for k in load_q}
            net_paths,assumptions=sample(settings,load_q,bank_q,efficiency,seed=42,rho=rho,solar_rho=solar_rho,times=index)
            load_paths,_=sample(settings,load_q,[],1,seed=42,rho=rho,solar_rho=solar_rho,times=index)
            solar_paths,_=sample(settings,zero,bank_q,1,seed=42,rho=rho,solar_rho=solar_rho,times=index);solar_paths=-solar_paths
            for h in (24,48,72):
                stop=origin+pd.Timedelta(hours=h);expected=pd.date_range(origin.floor('h'),stop.ceil('h'),freq='h',inclusive='left')
                if not expected.isin(index).all():continue
                weights=np.maximum(0,(np.minimum((index+pd.Timedelta(hours=1)).as_unit('ns').asi8,stop.value)-np.maximum(index.as_unit('ns').asi8,origin.value))/3.6e12)
                for kind in ('solar','net'):
                    actual=float(np.array([r[kind+'_actual'] for r in rr])@weights);point=float(np.array([r[kind+'_point'] for r in rr])@weights)
                    outcomes[kind][str(h)].append({'origin':origin.isoformat(),'actual_kwh':actual,'predicted_kwh':point,'error_kwh':point-actual})
                for kind,paths in [('load',load_paths),('solar',solar_paths),('net',net_paths)]:
                    actual=float(np.array([r['load']['actual'] if kind=='load' else r[kind+'_actual'] for r in rr])@weights)
                    low,mid,high=np.percentile(paths@weights,[10,50,90]);trajectory[kind][str(h)].append({'covered':bool(low<=actual<=high),'width_kwh':float(high-low),'actual_kwh':actual,'p10':float(low),'p50':float(mid),'p90':float(high)})
        def summary(items):
            e=np.array([r['error_kwh'] for r in items]);return {'windows':len(e),'mae_kwh':float(np.mean(abs(e))),'bias_kwh':float(np.mean(e))} if len(e) else {'windows':0}
        results[variant]={'common_origin_target_pairs':len(rows),'unique_target_hours':len({r['time'] for r in rows}),'solar':{h:summary(v) for h,v in outcomes['solar'].items()},'net':{h:summary(v) for h,v in outcomes['net'].items()},'windows':outcomes}
        results[variant]['sampling_metadata']={name:{k:metadata[name].get(k) for k in ('sampling_rho','sampling_dependence')} for name in names}
        results[variant]['sampled_energy']={kind:{h:{'windows':len(v),'coverage':float(np.mean([r['covered'] for r in v])) if v else None,'width_kwh':float(np.mean([r['width_kwh'] for r in v])) if v else None} for h,v in horizons.items()} for kind,horizons in trajectory.items()}
    results['sampled_energy']=results['candidate']['sampled_energy']  # Existing consumers
    identities={name:read_json('model-'+name,{}).get('trained_at') for name in names}
    for variant in ('existing','candidate'):
        if all(identities[name] and identities[name]==details[name].get('model_identities',{}).get(variant)
               and results[variant]['sampling_metadata'][name]=={k:read_json('model-'+name,{}).get(k) for k in ('sampling_rho','sampling_dependence')} for name in names):
            results['active_signature']=evidence_signature(settings);results['active_sampled_energy']=results[variant]['sampled_energy']
    results['paired_bootstrap']={kind:{h:paired_bootstrap(results['candidate']['windows'][kind][h],results['existing']['windows'][kind][h]) for h in ('24','48','72')} for kind in ('solar','net')}
    results['uncertainty_assumptions']={'method':METHOD,'percentiles':[10,90],'rho_basis':'Earlier internal OOF residuals only','solar_latent_cross_bank':.64,'solar_persistence':'Gap-aware earlier daylight residuals; independent fallback if missing','battery_parameters_fixed':True,'status':'Retrospective cumulative trajectory diagnostic; no claim of calibrated battery outcomes'}
    write_json('aggregate-assessment',results);return results

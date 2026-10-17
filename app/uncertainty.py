"""Asymmetric marginal draws, persistent hourly errors, fixed battery inputs."""
import numpy as np, pandas as pd

METHOD='monte-carlo-v2'
Z90=1.2815515655446004


def residual_rho(summary):
    """AR(1) from internal calibration P50 residuals via lag-1 autocorrelation."""
    try:
        if 'rho' in summary:return float(np.clip(summary['rho'],0,0.95))
        left=[];right=[]
        for entry in summary.get('cumulative',[]):
            e=np.asarray(entry.get('errors_kwh',[]),float)
            residuals=np.diff(e,prepend=0.)
            valid=np.isfinite(residuals[:-1])&np.isfinite(residuals[1:])
            left.extend(residuals[:-1][valid]);right.extend(residuals[1:][valid])
        if len(left)<48 or np.std(left)==0 or np.std(right)==0: return 0.0
        c=np.corrcoef(left,right)[0,1]
        return round(float(np.clip(c,0,0.95)),3) if np.isfinite(c) else 0.0
    except Exception:
        return 0.0


def fit_dependence(records,solar=False):
    """Gap-aware latent residual lag-1 on earlier records; no issue demeaning.

    Invert the asymmetric map away from zero/degenerate widths. Solar uses
    measured daylight only. Count issue dates separately from overlapping pairs.
    """
    groups={}
    for r in records:
        if solar and not r.get('scored',r.get('daylight',False)):continue
        y,lo,mid,hi=(float(r[k]) for k in ('actual','low','middle','high'))
        width=(mid-lo) if y<mid else (hi-mid)
        if not np.isfinite([y,lo,mid,hi]).all() or y<=0 or width<.05:continue
        z=Z90*(y-mid)/width
        groups.setdefault(r['origin'],{})[pd.Timestamp(r['time'])]=float(np.clip(z,-4,4))
    pairs=[];days=set()
    for origin,rr in groups.items():
        for t,z in rr.items():
            previous=rr.get(t-pd.Timedelta(hours=1))
            if previous is not None:
                pairs.append((previous,z));days.add(pd.Timestamp(origin).date().isoformat())
    info={'pairs':len(pairs),'issue_days':len(days),'basis':'Earlier asymmetric-standardized residuals; exact adjacent UTC hours within issue; width >= 0.05 kW; positive measured daylight for solar',
          'shared_and_bank_persistence':'Same pooled coefficient assumed for cloud and bank noise'}
    if len(pairs)<48 or len(days)<3:
        return {**info,'rho':0.,'status':'Insufficient earlier pairs; independent fallback'}
    left,right=np.asarray(pairs).T
    rho=float(np.corrcoef(left,right)[0,1]) if np.std(left)>0 and np.std(right)>0 else float('nan')
    return {**info,'rho':round(float(np.clip(rho,0,.95)),3) if np.isfinite(rho) else 0.,'raw_rho':rho if np.isfinite(rho) else None,'status':'Estimated; constrained to [0, 0.95]'}


def quantile_draws(q,z):
    """Piecewise Gaussian map reproducing P10/P50/P90, including tied heads."""
    lo,mid,hi=(np.asarray(q[k],float) for k in ('low','middle','high'))
    return np.maximum(0,mid+z*np.where(z<0,mid-lo,hi-mid)/Z90)


def _ar(rng,samples,n,rho,active=None,times=None):
    z=rng.standard_normal((samples,n));rho=float(np.clip(rho,0,.95))
    for t in range(1,n):
        adjacent=times is None or pd.Timestamp(times[t])-pd.Timestamp(times[t-1])==pd.Timedelta(hours=1)
        if adjacent and (active is None or (active[t] and active[t-1])):
            z[:,t]=rho*z[:,t-1]+np.sqrt(1-rho*rho)*z[:,t]
    return z


def solar_dependence(metadata):
    fitted=[m.get('sampling_dependence') or {} for m in metadata]
    usable=[m for m in fitted if m.get('pairs',0)>=48 and m.get('issue_days',0)>=3]
    if not usable:return {'rho':0.,'status':'No gap-aware daylight estimate; independent fallback','pairs':0,'issue_days':0}
    return {'rho':float(np.average([m['rho'] for m in usable],weights=[m['pairs'] for m in usable])),
            'status':'Pooled daylight persistence; common and bank noise share coefficient',
            'pairs':sum(m['pairs'] for m in usable),'issue_days':min(m['issue_days'] for m in usable)}


def sample_components(settings,load_q,solar_qs,efficiency,seed=None,rho=None,solar_rho=None,times=None):
    """Draw complete hourly paths. Cross-bank loading .8 remains an assumption."""
    from .config import read_json
    rng=np.random.default_rng(seed)
    n=len(load_q['middle']);samples=max(10,min(5000,int(getattr(settings,'mc_samples',5000))))
    if rho is None:rho=read_json('model-load',{}).get('sampling_rho',0.)
    rho=float(np.clip(rho,0,.95))
    if solar_rho is None:
        solar_rho=solar_dependence([read_json('model-solar-'+b.id,{}) for b in settings.banks])['rho']
    solar_rho=float(np.clip(solar_rho,0,.95))
    load_paths=quantile_draws(load_q,_ar(rng,samples,n,rho,times=times))
    solar_paths=np.zeros((samples,n))
    active=np.any([np.asarray(b['high'])>0 for b in solar_qs],axis=0) if solar_qs else np.zeros(n,bool)
    cloud=_ar(rng,samples,n,solar_rho,active,times)
    for bank in solar_qs:
        bank_active=np.asarray(bank['high'])>0
        z=.8*cloud+.6*_ar(rng,samples,n,solar_rho,bank_active,times)
        solar_paths+=quantile_draws(bank,z)
    solar_paths*=efficiency
    return load_paths,solar_paths,{'method':METHOD,'samples':samples,'rho':rho,'solar_rho':solar_rho,
        'solar_correlation':{'basis':'Assumed shared cloud loading; same persistence for cloud and bank noise; independent of load','loading':.8,'latent_cross_bank':.64}}


def sample(settings,load_q,solar_qs,efficiency,seed=None,rho=None,solar_rho=None,times=None):
    load,solar,info=sample_components(settings,load_q,solar_qs,efficiency,seed,rho,solar_rho,times)
    return load-solar,info


def energy_period_profiles(settings,rows,load,solar):
    """P10/P50/P90 energy for fixed local 06–18 and 18–06 windows.

    Quantiles follow integration of complete sampled paths. Scale the shared
    hourly shape to each window total. Partial windows use only their forecast
    portion; local clock windows naturally span 11/13 hours at DST transitions.
    """
    weights=np.asarray([r['hours'] for r in rows])
    local=pd.DatetimeIndex([r['time'] for r in rows]).tz_convert(settings.timezone).tz_localize(None)
    anchors=(local-pd.Timedelta(hours=6)).floor('12h')+pd.Timedelta(hours=6)
    profiles={k:np.zeros((3,len(rows))) for k in ('load','solar')};periods=[]
    for anchor in dict.fromkeys(anchors):
        anchor=pd.Timestamp(anchor);indices=np.flatnonzero(anchors==anchor);w=weights[indices]
        start=anchor.tz_localize(settings.timezone)
        end=(anchor+pd.Timedelta(hours=12)).tz_localize(settings.timezone)
        duration=(end-start).total_seconds()/3600
        entry={'start':start.isoformat(),'end':end.isoformat(),'period':'day' if anchor.hour==6 else 'night',
               'forecast_hours':float(w.sum()),'partial_period':bool(w.sum()<duration-1e-8)}
        for name,paths in (('load',load),('solar',solar)):
            totals=np.percentile((paths[:,indices]*w).sum(axis=1),[10,50,90])
            shape=np.maximum(0,np.asarray([rows[i][name+'_middle'] for i in indices],float))
            energy=float(shape@w)
            if energy<=1e-12:
                shape=paths[:,indices].mean(axis=0);energy=float(shape@w)
            if energy>1e-12:profiles[name][:,indices]=totals[:,None]*shape[None,:]/energy
            entry[name+'_kwh']=dict(zip(('low','middle','high'),totals.tolist()))
        periods.append(entry)
    return {**profiles,'periods':periods,'basis':'Sampled energy quantiles for local 06:00–18:00 and 18:00–06:00 windows; shared hourly shapes scaled to each total; partial windows use only their forecast portion'}


def evidence_signature(settings):
    """Bind aggregate evidence to model identities and sampler/settings in use."""
    from .config import read_json
    import hashlib,json
    names=['load',*['solar-'+b.id for b in settings.banks],'battery','inverter']
    models={name:{k:read_json('model-'+name,{}).get(k) for k in ('trained_at','sampling_rho','sampling_dependence','efficiency','charge_efficiency','discharge_efficiency')} for name in names}
    return hashlib.sha256(json.dumps({'method':METHOD,'settings':settings.model_dump(exclude={'ha_token'}),'models':models},sort_keys=True,default=str).encode()).hexdigest()


def energy_scenarios(settings,rows,now,efficiency=None):
    """Sample energy totals for the three fixed-shape battery scenarios."""
    from .config import read_json
    # Each hourly draw is held through all dispatch substeps of that hour.
    codes,unique=pd.factorize(pd.DatetimeIndex([r['time'] for r in rows]).floor('h'))
    hourly=[rows[int(np.flatnonzero(codes==i)[0])] for i in range(len(unique))]
    load_q={k:np.array([r['load_'+k] for r in hourly]) for k in ('low','middle','high')}
    solar_qs=[{k:np.array([r.get(f'solar_{k}_{b.id}',0.) for r in hourly]) for k in ('low','middle','high')} for b in settings.banks]
    inv=read_json('model-inverter',{}).get('efficiency',.9) if efficiency is None else efficiency
    # Stable within an issue hour, so refreshing SoC does not redraw the band.
    load,solar,info=sample_components(settings,load_q,solar_qs,inv,seed=int(pd.Timestamp(now).floor('h').timestamp())%2**32,times=unique)
    load=load[:,codes];solar=solar[:,codes]
    periods=energy_period_profiles(settings,rows,load,solar)
    load_meta=read_json('model-load',{})
    solar_meta=[read_json('model-solar-'+b.id,{}) for b in settings.banks]
    status={'method':METHOD,'status':'Forecast available; coverage not established','samples':info['samples'],'rho':info['rho'],
            'coverage':{'nominal':.8,'load_validation':bool(load_meta.get('load_validation')),'solar_validations':sum(bool(m.get('forecast_validation')) for m in solar_meta)},
            'solar_correlation':info['solar_correlation'],'solar_persistence':solar_dependence(solar_meta),
            'issue_hour_utc':pd.Timestamp(now).hour,
            'cumulative_energy_validation':read_json('aggregate-assessment',{}).get('active_sampled_energy',{}) if read_json('aggregate-assessment',{}).get('active_signature')==evidence_signature(settings) else {},
            'validation_basis':'Retrospective sampled cumulative energy; not validated battery-outcome coverage',
            'note':'Sampled load/solar paths supply 12-hour energy quantiles; battery projections use the three fixed-shape scenarios. Three marginal heads assume interpolated/extrapolated tails. Small samples give unstable percentiles.'}
    return periods,status

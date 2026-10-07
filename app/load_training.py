"""Age-weighted CNN training, history replay, recursive backtests and promotion."""
import copy, os
import numpy as np, pandas as pd, torch
from torch import nn
from .config import DATA, read_json, write_json
from .datasets import FEATURES
from .models import LoadCNN, LOAD_FEATURES, SEQ, sequences, metrics, validation_chart, load_features, signature, LEGACY_LOAD_FEATURES, CONTEXT_FEATURES


def age_weights(index, reference, enabled=True, half_life_days=365):
    age=np.maximum(0,(pd.Timestamp(reference)-index).total_seconds().to_numpy()/86400)
    return np.exp2(-age/half_life_days).astype(np.float32) if enabled else np.ones(len(index),np.float32)


def history_signature(settings, entities):
    return {'features':2,'location':[settings.latitude,settings.longitude],'timezone':settings.timezone,'state':settings.state,'calendar':settings.school_calendar_url,'entities':entities}


def preserve_history(settings,frame,entities):
    path=DATA/'load-history.pkl'; sig=history_signature(settings,entities)
    if path.exists():
        old=pd.read_pickle(path)
        if old['signature']==sig:
            frame=pd.concat([old['frame'],frame]).sort_index()
            frame=frame[~frame.index.duplicated(keep='last')]
    pd.to_pickle({'signature':sig,'frame':frame},path)


def prepare_fine_tune(settings,progress):
    from .datasets import build_dataset
    now=pd.Timestamp.now(tz=settings.timezone).normalize()
    path=DATA/'load-history.pkl'; old=pd.read_pickle(path) if path.exists() else None
    requested=pd.Timestamp(settings.load_history_start,tz=settings.timezone).tz_convert('UTC')
    usable=old is not None and old['signature']==history_signature(settings,settings.historical_consumption_entities) and old['frame'].index.min()<requested+pd.Timedelta(days=2)
    start=now-pd.DateOffset(days=30) if usable else requested.tz_convert(settings.timezone)
    progress('Refreshing recent history' if usable else 'Building the retained long-term history archive')
    req={'kind':'load','entities':settings.historical_consumption_entities,'combine':'splice','train_start':str(start.date()),'train_end':str((now-pd.DateOffset(days=11)).date()),'validation_start':str((now-pd.DateOffset(days=10)).date()),'validation_end':str((now-pd.DateOffset(days=1)).date()),'interpolate_hours':2,'max_power_kw':40}
    build_dataset(settings,req,progress)
    d=pd.read_pickle(DATA/'load.pkl'); history=pd.read_pickle(path)['frame']
    d['frame']=history.loc[(history.index>=requested)&(history.index<d['end'])]
    pd.to_pickle(d,DATA/'load.pkl')
    summary=read_json('dataset-load'); summary.update(rows=len(d['frame']),training_rows=int((d['frame'].index<d['train_end']).sum()),validation_rows=int((d['frame'].index>=d['val_start']).sum()),history_start=str(d['frame'].index.min()),history_end=str(d['frame'].index.max()),history_replay=True)
    expected_rows=len(pd.date_range(requested.ceil('h'),d['end'],freq='1h',inclusive='left'))
    summary['coverage_pct']=round(100*len(d['frame'])/expected_rows,1)
    summary['sufficient']=summary['training_rows']>=168 and summary['validation_rows']>=168
    summary['readiness']='Ready to calibrate' if summary['sufficient'] else 'Need more usable training and validation history'
    summary['request']['train_start']=str(requested.tz_convert(settings.timezone).date())
    summary['preview']=[{'time':str(t),'value':round(float(v),3)} for t,v in d['frame'].target.iloc[::max(1,len(d['frame'])//240)].items()]
    write_json('dataset-load',summary)
    return summary,d


class LoadPredictor:
    """One artifact load and a shared recursive forecast path for live use and validation."""
    def __init__(self,artifact):
        self.artifact=artifact; self.feature_count=len(artifact['mean']); self.direct=artifact.get('architecture')=='direct'
        if self.direct:
            from .direct_load import DirectCNN
            self.model=DirectCNN(self.feature_count)
        else:self.model=LoadCNN(self.feature_count)
        self.model.load_state_dict(artifact['state']); self.model.eval()
        self.mean=artifact['mean']; self.std=artifact['std']

    def forecast_many(self,contexts,future_features,extra=None):
        # Contexts contain exactly 48 observed hours. Future targets are never supplied.
        context=np.asarray(contexts,dtype=np.float32)
        future=np.asarray(future_features,dtype=np.float32)
        n,h,_=future.shape
        target=np.concatenate([context[:,:,-1],np.full((n,h),np.nan,np.float32)],axis=1)
        cov=np.concatenate([context[:,:,:-1],future],axis=1)
        xx=np.full((n,48+h,self.feature_count),np.nan,np.float32)
        xx[:,:,:len(FEATURES)]=cov
        if self.feature_count>len(LEGACY_LOAD_FEATURES):
            if extra is None: raise ValueError('Weekly context is required by this CNN')
            xx[:,:,len(LEGACY_LOAD_FEATURES):]=extra
        for j in range(24,48):
            xx[:,j,len(FEATURES)]=target[:,j-1]; xx[:,j,len(FEATURES)+1]=target[:,j-24]; xx[:,j,len(FEATURES)+2]=target[:,j-24:j].mean(axis=1)
        prediction=np.empty((n,h),np.float32)
        with torch.no_grad():
            for k in range(h):
                j=48+k
                xx[:,j,len(FEATURES)]=target[:,j-1]; xx[:,j,len(FEATURES)+1]=target[:,j-24]; xx[:,j,len(FEATURES)+2]=target[:,j-24:j].mean(axis=1)
                x=(xx[:,j-SEQ+1:j+1]-self.mean)/self.std
                if not np.isfinite(x).all(): raise ValueError('CNN forecast needs contiguous observed context and complete weather features')
                if self.direct:
                    if k%24==0:
                        stop=min(h,k+24)
                        exog=np.concatenate([future[:,k:stop],extra[:,48+k:48+stop]],axis=2)
                        exog=(exog-self.artifact['future_mean'])/self.artifact['future_std']
                        block=self.model(torch.from_numpy(x),torch.from_numpy(exog.astype(np.float32))).numpy()
                    p=block[:,k%24]
                else:p=self.model(torch.from_numpy(x)).numpy()
                prediction[:,k]=p; target[:,j]=p
        return prediction

    def forecast(self,frame,hours):
        origin=hours[0]
        context=frame.reindex(pd.date_range(origin-pd.Timedelta(hours=48),periods=48,freq='1h'))[FEATURES+['target']].to_numpy(np.float32)
        future=frame.reindex(hours)[FEATURES].to_numpy(np.float32)
        safe=frame.copy(); safe.loc[safe.index>=origin,'target']=np.nan
        extra=load_features(safe,self.artifact.get('timezone','Australia/Adelaide')).reindex(pd.date_range(origin-pd.Timedelta(hours=48),periods=48+len(hours),freq='1h'))[CONTEXT_FEATURES]
        # Future recent averages must use observed history, never recursively invented load.
        extra.iloc[48:,:3]=extra.iloc[48:,:3].ffill().fillna(float(safe.target.tail(24+len(hours)).mean()))
        return self.forecast_many(context[None,:,:],future[None,:,:],extra.to_numpy(np.float32)[None,:,:])[0]


def next_free_start(origin,settings):
    local=origin.tz_convert(settings.timezone); starts=[]
    for days in range(4):
        day=local.date()+pd.Timedelta(days=days)
        for w in settings.charging_windows:
            if not w.free: continue
            t=pd.Timestamp(f'{day} {w.start}',tz=settings.timezone).tz_convert('UTC')
            if t>origin: starts.append(t)
    return min(starts) if starts else None


def evaluation_cases(frame,start,end,settings,progress=lambda s:None):
    end=min(pd.Timestamp(end),frame.index.max()+pd.Timedelta(hours=1))
    peak_hour=int(settings.export_start.split(':')[0]); origins=[]
    for t in pd.date_range(pd.Timestamp(start).ceil('h'),end,freq='1h',inclusive='left'):
        if t.tz_convert(settings.timezone).hour not in {0,6,12,18,peak_hour}: continue
        length=min(72,int((end-t).total_seconds()/3600))
        if length<24: continue
        f=frame.reindex(pd.date_range(t-pd.Timedelta(hours=48),periods=48+length,freq='1h'))
        if f[FEATURES+['target']].isna().any().any(): continue
        safe=frame.loc[frame.index<t].copy().reindex(frame.index); safe[FEATURES]=frame[FEATURES]
        extra=load_features(safe,settings.timezone).reindex(f.index)[CONTEXT_FEATURES].ffill()
        source='observed weather'
        if settings.archived_weather_backtest:
            from .sources import forecast_weather, Calendar
            from .datasets import features
            try:
                progress(f'Retrieving weather available before {t.isoformat()}')
                weather=forecast_weather(settings,t,t+pd.Timedelta(hours=length),settings.weather_model,t)
                future=features(weather,settings,Calendar(settings).load()).reindex(f.index[48:])[FEATURES]
                if future.isna().any().any(): continue
                f.loc[f.index[48:],FEATURES]=future
                extra.loc[f.index[48:],'heating']=(18-future.temperature_2m).clip(lower=0)
                extra.loc[f.index[48:],'cooling']=(future.temperature_2m-24).clip(lower=0)
                source='archived forecast run (12h publication buffer)'
            except Exception as exc:
                progress(f'Archived weather unavailable for {t.date()}; skipping origin: {exc}')
                continue
        origins.append({'weather_source':source,'extra' :extra.to_numpy(np.float32),'origin':t,'length':length,'context':f.iloc[:48][FEATURES+['target']].to_numpy(np.float32),'future':f.iloc[48:][FEATURES].to_numpy(np.float32),'actual':f.target.iloc[48:].to_numpy(np.float32),'peak':t.tz_convert(settings.timezone).hour==peak_hour})
    return origins


def evaluate(artifact,cases,settings):
    if not cases: raise ValueError('Need contiguous independent holdout data for recursive validation')
    longest=max(c['length'] for c in cases)
    future=np.stack([np.pad(c['future'],((0,longest-c['length']),(0,0)),mode='edge') for c in cases])
    extra=np.stack([np.pad(c['extra'],((0,longest-c['length']),(0,0)),mode='edge') for c in cases])
    pred=LoadPredictor(artifact).forecast_many(np.stack([c['context'] for c in cases]),future,extra)
    summary={'horizons':{},'origins':len(cases)}; cumulative=[]; nights=[]; chart=[]; actual=[]; predicted=[]
    for i,c in enumerate(cases):
        y=c['actual']; p=pred[i,:c['length']]; error=np.cumsum(y-p)
        cumulative.append({'origin':c['origin'].isoformat(),'local_hour':c['origin'].tz_convert(settings.timezone).hour,'errors_kwh':[float(x) for x in error]})
        actual.extend(y[:24]); predicted.extend(p[:24])
        if c['peak']:
            stop=next_free_start(c['origin'],settings)
            hours=(stop-c['origin']).total_seconds()/3600 if stop is not None else None
            if hours is not None and 0<hours<=c['length']:
                weights=np.clip(hours-np.arange(c['length']),0,1)
                a=float(y@weights); b=float(p@weights)
                nights.append({'time':c['origin'].isoformat(),'hours':hours,'actual':a,'predicted':b,'error_kwh':a-b})
        if i==0:
            times=pd.date_range(c['origin'],periods=c['length'],freq='1h')
            chart=validation_chart(times,y,p)
    for h in (24,48,72):
        errs=[c['errors_kwh'][h-1] for c in cumulative if len(c['errors_kwh'])>=h]
        if errs: summary['horizons'][str(h)]={'energy_mae_kwh':round(float(np.mean(np.abs(errs))),4),'energy_bias_kwh':round(float(np.mean(errs)),4),'underprediction_kwh':round(float(np.mean(np.maximum(errs,0))),4),'samples':len(errs)}
    errors=[n['error_kwh'] for n in nights]
    summary['overnight']={'energy_mae_kwh':round(float(np.mean(np.abs(errors))),4),'energy_bias_kwh':round(float(np.mean(errors)),4),'underprediction_kwh':round(float(np.mean(np.maximum(errors,0))),4),'samples':len(errors)} if errors else {'samples':0}
    summary['hourly']=metrics(actual,predicted)
    summary['note']='Forecast lag inputs exclude future actual consumption. Weather: '+', '.join(sorted({c.get('weather_source','observed weather') for c in cases}))+'. Archived reconstruction is approximate; immutable issued forecasts are the operational score.'
    return {'summary':summary,'cumulative':cumulative,'nights':nights,'chart':chart}


def promotion_decision(baseline,candidate,improvement_percent):
    a=baseline['overnight']; b=candidate['overnight']
    if min(a.get('samples',0),b.get('samples',0))<3 or candidate['horizons'].get('72',{}).get('samples',0)<3:
        return False,'Need at least three independent overnight and 72-hour forecast origins'
    improvement=100*(a['energy_mae_kwh']-b['energy_mae_kwh'])/max(a['energy_mae_kwh'],1e-8)
    if b['energy_mae_kwh']>=a['energy_mae_kwh'] or improvement<improvement_percent:
        return False,f'Overnight error did not improve by the required {improvement_percent:g}% ({improvement:.1f}% measured)'
    if b['underprediction_kwh']>a['underprediction_kwh']+.1:
        return False,'Candidate increased mean overnight underprediction by more than 0.1 kWh'
    if any(candidate['horizons'][h]['energy_mae_kwh']>v['energy_mae_kwh']*1.05+.05 for h,v in baseline['horizons'].items()):
        return False,'Candidate worsened a 24/48/72-hour energy error by more than the allowed 5% + 0.05 kWh'
    return True,f'Overnight error improved {improvement:.1f}% without materially increasing underprediction or worsening multi-hour energy forecasts'


def replay_indices(index,frame,tr,cutoff,replay_percent):
    recent=np.flatnonzero(tr&(index>=cutoff-pd.Timedelta(days=30))); older=np.flatnonzero(tr&(index<cutoff-pd.Timedelta(days=30)))
    if not len(recent): raise ValueError('No recent training sequences are available')
    count=min(len(older),int(len(recent)*replay_percent/max(1,100-replay_percent)))
    if not count: return recent,0
    rng=np.random.default_rng(42); selected=[]
    values=frame.reindex(index[older]); local=index[older]
    groups=pd.DataFrame({'quarter':local.quarter,'day_type':values.weekday.to_numpy()>=5,'temperature':np.digitize(values.temperature_2m.to_numpy(),[15,25]),'holiday':((values.public_holiday+values.school_holiday)>0).to_numpy()},index=older)
    buckets=list(groups.groupby(list(groups.columns)).groups.values()); rng.shuffle(buckets)
    for group in buckets[:count]:
        ids=np.asarray(group,dtype=int); selected.append(int(rng.choice(ids)))
    rest=np.setdiff1d(older,selected)
    if len(selected)<count:
        selected.extend(rng.choice(rest,count-len(selected),replace=False).tolist())
    return np.concatenate([recent,np.array(selected)]),len(selected)


def save_artifact(artifact,name='load.pt'):
    path=DATA/name; tmp=path.with_suffix('.tmp')
    torch.save(artifact,tmp); os.replace(tmp,path)


def uncertainty(evaluation,settings):
    return {'method':'recursive cumulative energy errors','created_at':pd.Timestamp.now(tz='UTC').isoformat(),'signature':{'timezone':settings.timezone,'export_start':settings.export_start,'free_starts':[w.start for w in settings.charging_windows if w.free]},'paths':evaluation['cumulative'],'overnight_errors':evaluation['nights'],'max_hours':72,'note':evaluation['summary']['note']}


def cumulative_bands(prediction,calibration,confidence,origin_hour):
    """Turn cumulative energy quantiles into physically ordered hourly scenario profiles."""
    paths=calibration.get('paths',[])
    nearby=[p for p in paths if min(abs(p['local_hour']-origin_hour),24-abs(p['local_hour']-origin_hour))<=3]
    if sum(len(p['errors_kwh'])>=72 for p in nearby)>=3: paths=nearby
    alpha=(1-confidence)/2; low=[0.]; high=[0.]; counts=[]
    for h in range(1,len(prediction)+1):
        step=min(h,72); errors=[p['errors_kwh'][step-1] for p in paths if len(p['errors_kwh'])>=step]
        if not errors: raise ValueError('Cumulative forecast range calibration is unavailable')
        lo,hi=np.quantile(errors,[alpha,1-alpha]); scale=max(1,h/72)
        low.append(min(0,float(lo))*scale); high.append(max(0,float(hi))*scale); counts.append(len(errors))
    low=np.minimum.accumulate(low); high=np.maximum.accumulate(high)
    middle=np.concatenate([[0.],np.cumsum(prediction)])
    lower=np.maximum.accumulate(np.maximum(0,middle+low)); upper=middle+high
    return np.diff(lower),np.diff(upper),{'method':'cumulative recursive energy errors','origins':len(paths),'minimum_samples':min(counts),'confidence':confidence}



def consumption_paths(prediction,calibration,confidence,origin_hour):
    """Pointwise power bands from whole error trajectories, not differences of quantiles."""
    all_paths=calibration.get('paths',[])
    full=[p for p in all_paths if len(p['errors_kwh'])>=72]
    near=[p for p in full if min(abs(p['local_hour']-origin_hour),24-abs(p['local_hour']-origin_hour))<=3]
    selected=near if len(near)>=3 else full
    if not selected: raise ValueError('Need complete recursive forecast-error paths for chart uncertainty')
    trajectories=[]
    for path in selected:
        errors=np.diff(np.concatenate([[0.],path['errors_kwh']]))
        if len(errors)<len(prediction):
            # Internal forecast cache can extend beyond the supported 72h display horizon.
            tail=np.resize(errors[-24:],len(prediction)-len(errors))
            errors=np.concatenate([errors,tail])
        trajectories.append(np.maximum(0,prediction+errors[:len(prediction)]))
    trajectories=np.asarray(trajectories)
    alpha=(1-confidence)/2
    lo,hi=np.quantile(trajectories,[alpha,1-alpha],axis=0)
    return trajectories,np.minimum(lo,prediction),np.maximum(hi,prediction),{'method':'pointwise recursive error trajectories','historical_paths':len(selected),'confidence':confidence,'note':'Power uncertainty may vary by hour. Cumulative energy bounds are calculated separately.'}


def train_load(settings,summary,d,progress,epochs=30,fine_tune=False):
    frame=d['frame'];x,y,index=sequences(frame,settings.timezone);tr=index<d['train_end']
    # First three validation days select the epoch; later days are untouched until promotion evaluation.
    selection_end=(d['val_start'].tz_convert(settings.timezone)+pd.DateOffset(days=3)).tz_convert('UTC')
    selection=(index>=d['val_start'])&(index<selection_end)
    if tr.sum()<48 or selection.sum()<24: raise ValueError('Need training sequences and a separate three-day selection period')
    cases=evaluation_cases(frame,selection_end,d['end'],settings,progress)
    if len([c for c in cases if c['length']>=72])<3: raise ValueError('Need ten validation days including at least three complete 72-hour holdout origins')
    if sum(c['peak'] and c['length']>=24 for c in cases)<3: raise ValueError('Need at least three evening-origin holdout forecasts')
    active_path=DATA/'load.pt'; baseline=torch.load(active_path,weights_only=False,map_location='cpu') if active_path.exists() else None
    if fine_tune and baseline is None: raise ValueError('Calibrate the CNN before fine tuning')
    migrated=None
    if fine_tune and baseline.get('architecture')=='direct':
        progress('Training a recursive challenger to the direct CNN')
        fine_tune=False
    elif fine_tune and len(baseline['mean'])!=len(LOAD_FEATURES):
        # Preserve all learned responses; new context channels initially contribute zero.
        progress('Adding weekly context to learned CNN weights before recent-data fine tuning')
        migrated=copy.deepcopy(baseline);old_count=len(baseline['mean'])
        torch.manual_seed(42);expanded=LoadCNN().state_dict()
        for key,value in baseline['state'].items():
            if key=='conv.0.weight':expanded[key].zero_();expanded[key][:,:old_count,:]=value
            else:expanded[key]=value.clone()
        migrated['state']=expanded
        weighted=age_weights(index,pd.Timestamp.now(tz='UTC'),settings.age_weighting,settings.age_half_life_days)[tr]
        added=x[tr,:,old_count:]
        mean_added=np.average(added.mean(axis=1),axis=0,weights=weighted).astype(np.float32)
        std_added=np.maximum(np.sqrt(np.average(((added-mean_added)**2).mean(axis=1),axis=0,weights=weighted)),.01).astype(np.float32)
        migrated['mean']=np.concatenate([baseline['mean'],mean_added]);migrated['std']=np.concatenate([baseline['std'],std_added])
    weights=age_weights(index,pd.Timestamp.now(tz='UTC'),settings.age_weighting,settings.age_half_life_days)
    if fine_tune:
        ids,replay_count=replay_indices(index,frame,tr,d['train_end'],settings.history_replay_percent)
        artifact=copy.deepcopy(migrated or baseline); mean=artifact['mean'];std=artifact['std']
    else:
        ids=np.flatnonzero(tr);replay_count=0
        w=weights[ids];mean=np.average(x[ids].mean(axis=1),axis=0,weights=w).astype(np.float32)
        variance=np.average(((x[ids]-mean)**2).mean(axis=1),axis=0,weights=w)
        std=np.maximum(np.sqrt(variance),.01).astype(np.float32);artifact={'mean':mean,'std':std}
    trainw=weights[ids].copy()
    xx=torch.from_numpy((x-mean)/std);yy=torch.from_numpy(y);trainx=xx[ids];trainy=yy[ids];ww=torch.from_numpy(trainw)
    torch.manual_seed(42);model=LoadCNN()
    if fine_tune:model.load_state_dict(artifact['state'])
    opt=torch.optim.Adam(model.parameters(),lr=.0005 if fine_tune else .003)
    lossfn=nn.SmoothL1Loss(reduction='none'); losses=[];best=float('inf');best_state=None
    torch.manual_seed(42)
    progress(f'Training {len(ids)} sequences ({replay_count} older replay examples); age half-life {settings.age_half_life_days:g} days')
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(ids));total=0.;mass=0.
        for batch in order.split(256):
            opt.zero_grad();loss=(lossfn(model(trainx[batch]),trainy[batch])*ww[batch]).sum()/ww[batch].sum();loss.backward();opt.step()
            total+=loss.item()*ww[batch].sum().item();mass+=ww[batch].sum().item()
        model.eval()
        with torch.no_grad():val=float(lossfn(model(xx[selection]),yy[selection]).mean())
        losses.append({'epoch':epoch+1,'training':round(total/mass,5),'validation':round(val,5)})
        if val<best:best=val;best_state={k:v.clone() for k,v in model.state_dict().items()}
        progress(f'CNN epoch {epoch+1}/{epochs}; unweighted selection loss {val:.3f}')
    artifact['state']=best_state; artifact['timezone']=settings.timezone; artifact['architecture']='recursive'
    progress('Backtesting the candidate recursively at 24, 48 and 72 hours')
    candidate_eval=evaluate(artifact,cases,settings)
    baseline_eval=evaluate(baseline,cases,settings) if baseline is not None else None
    accepted,reason=promotion_decision(baseline_eval['summary'],candidate_eval['summary'],settings.promotion_improvement_percent) if baseline_eval else (True,'Initial model; no previous CNN to compare')
    metadata={'trained_at':pd.Timestamp.now(tz='UTC').isoformat(),'dataset_id':'load','model':'Temporal CNN','config_signature':signature(settings,'load'),'fine_tuned':fine_tune,'training_rows':len(ids),'history_rows':int(tr.sum()),'replay_rows':replay_count,'validation_rows':sum(c['length'] for c in cases),'dataset_created_at':summary['created_at'],'training_end':d['train_end'].isoformat(),'selection_end':selection_end.isoformat(),'metrics':candidate_eval['summary']['hourly'],'chart':candidate_eval['chart'],'losses':losses,'recursive_validation':candidate_eval['summary'],'weighting':{'enabled':settings.age_weighting,'half_life_days':settings.age_half_life_days,'replay_percent':settings.history_replay_percent,'minimum_age_weight':float(weights[ids].min()),'validation_weighted':False}}
    # Profile uses training targets only; holdout consumption does not feed future lag seeding.
    train_frame=frame.loc[frame.index<d['train_end']];profile=train_frame.target.groupby(train_frame.index.tz_convert(settings.timezone).hour).median()
    metadata['hour_profile']={str(k):float(v) for k,v in profile.items()}
    artifact['metadata']=metadata;artifact['uncertainty']=uncertainty(candidate_eval,settings)
    save_artifact(artifact,'load-candidate.pt')
    report={'completed_at':metadata['trained_at'],'decision':'promoted' if accepted else 'retained','reason':reason,'baseline':baseline_eval['summary'] if baseline_eval else None,'candidate':candidate_eval['summary'],'weighting':metadata['weighting'],'replay_rows':replay_count,'training_rows':len(ids),'history_rows':int(tr.sum()),'holdout_start':selection_end.isoformat(),'holdout_end':d['end'].isoformat(),'selection_end':selection_end.isoformat(),'overnight_chart':candidate_eval['nights'],'cumulative_chart':[{'hours':h,'baseline':baseline_eval['summary']['horizons'][str(h)]['energy_mae_kwh'] if baseline_eval else None,'candidate':candidate_eval['summary']['horizons'][str(h)]['energy_mae_kwh']} for h in (24,48,72)]}
    if accepted:
        if baseline is not None:
            baseline['metadata']=read_json('model-load',{});baseline['uncertainty']=uncertainty(baseline_eval,settings)
            save_artifact(baseline,'load-previous.pt')
        save_artifact(artifact);write_json('model-load',metadata)
        active_eval=candidate_eval
    else:
        # Keep the active weights and scaling; recalibrate its ranges on the same holdout.
        baseline['uncertainty']=uncertainty(baseline_eval,settings);save_artifact(baseline)
        active_metadata=read_json('model-load',{});active_metadata['recursive_validation']=baseline_eval['summary'];active_metadata['range_calibrated_at']=metadata['trained_at'];write_json('model-load',active_metadata)
        active_eval=baseline_eval
    write_json('load-uncertainty',uncertainty(active_eval,settings));write_json('fine-tuning-status',report)
    progress(('Candidate promoted: ' if accepted else 'Current CNN retained: ')+reason)
    return report


def rollback_load(settings):
    previous=DATA/'load-previous.pt'
    if not previous.exists():raise ValueError('No previous promoted CNN is available')
    backup=torch.load(previous,map_location='cpu',weights_only=False)
    if backup.get('metadata',{}).get('config_signature')!=signature(settings,'load'):raise ValueError('Previous CNN is incompatible with the current sensor/location configuration')
    current=torch.load(DATA/'load.pt',map_location='cpu',weights_only=False)
    save_artifact(current,'load-previous.pt');save_artifact(backup);write_json('model-load',backup['metadata'])
    if backup.get('uncertainty'):write_json('load-uncertainty',backup['uncertainty'])
    report={'completed_at':pd.Timestamp.now(tz='UTC').isoformat(),'decision':'rolled_back','reason':'Restored the previous CNN, input scaling and range calibration'}
    write_json('fine-tuning-status',report);return report

"""A small direct 24-hour CNN experiment, using the existing promotion guard."""
import copy
import numpy as np, pandas as pd, torch
from torch import nn
from .config import DATA,read_json,write_json
from .datasets import FEATURES
from .models import LOAD_FEATURES,CONTEXT_FEATURES,load_features,sequences

BLOCK=24
class DirectCNN(nn.Module):
    def __init__(self,feature_count=len(LOAD_FEATURES)):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(feature_count,24,3,padding=1),nn.ReLU(),nn.Conv1d(24,24,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(48+len(FEATURES)+len(CONTEXT_FEATURES),32),nn.ReLU(),nn.Linear(32,1),nn.Softplus())
    def forward(self,x,future):
        h=self.conv(x.transpose(1,2));context=torch.cat([h[:,:,-1],h.mean(dim=2)],dim=1)
        return self.head(torch.cat([context[:,None,:].expand(-1,future.shape[1],-1),future],dim=2)).squeeze(-1)


def trial(settings,progress):
    from .load_training import evaluation_cases,evaluate,promotion_decision,age_weights,save_artifact,uncertainty
    if not (DATA/'load.pt').exists(): raise ValueError('Calibrate the recursive CNN first')
    d=pd.read_pickle(DATA/'load.pkl');frame=d['frame'];x,y,index=sequences(frame,settings.timezone)
    full=frame.reindex(pd.date_range(frame.index.min(),frame.index.max(),freq='1h'))
    exog=load_features(full,settings.timezone)[FEATURES+CONTEXT_FEATURES].to_numpy(np.float32)
    targets=full.target.to_numpy(np.float32);positions=full.index.get_indexer(index)
    valid=positions+BLOCK<=len(full);positions=positions[valid];x=x[valid];index=index[valid]
    future=np.stack([exog[i:i+BLOCK] for i in positions]);y=np.stack([targets[i:i+BLOCK] for i in positions])
    finite=np.isfinite(future).all(axis=(1,2))&np.isfinite(y).all(axis=1)
    x=x[finite];future=future[finite];y=y[finite];index=index[finite]
    selection_end=(d['val_start'].tz_convert(settings.timezone)+pd.DateOffset(days=3)).tz_convert('UTC')
    tr=index+pd.Timedelta(hours=BLOCK)<=d['train_end'];va=(index>=d['val_start'])&(index+pd.Timedelta(hours=BLOCK)<=selection_end)
    if tr.sum()<48 or va.sum()<24:raise ValueError('Need contiguous training sequences and three selection days')
    # Bound experiment cost while preserving evenly spaced historical and recent examples.
    ids=np.flatnonzero(tr);ids=ids[np.linspace(0,len(ids)-1,min(12000,len(ids)),dtype=int)]
    mean=x[ids].mean(axis=(0,1));std=np.maximum(x[ids].std(axis=(0,1)),.01)
    emean=future[ids].mean(axis=(0,1));estd=np.maximum(future[ids].std(axis=(0,1)),.01)
    xx=torch.from_numpy((x-mean)/std);ff=torch.from_numpy((future-emean)/estd);yy=torch.from_numpy(y)
    weights=torch.from_numpy(age_weights(index,pd.Timestamp.now(tz='UTC'),settings.age_weighting,settings.age_half_life_days))
    torch.manual_seed(42);model=DirectCNN();opt=torch.optim.Adam(model.parameters(),lr=.002);lossfn=nn.SmoothL1Loss(reduction='none');best=float('inf');best_state=None
    for epoch in range(12):
        model.train()
        for batch in torch.from_numpy(ids)[torch.randperm(len(ids))].split(256):
            opt.zero_grad();loss=(lossfn(model(xx[batch],ff[batch]),yy[batch]).mean(dim=1)*weights[batch]).sum()/weights[batch].sum();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():score=float(lossfn(model(xx[va],ff[va]),yy[va]).mean())
        if score<best:best=score;best_state=copy.deepcopy(model.state_dict())
        progress(f'Direct 24-hour CNN epoch {epoch+1}/12; selection loss {score:.3f}')
    artifact={'architecture':'direct','state':best_state,'mean':mean,'std':std,'future_mean':emean,'future_std':estd,'timezone':settings.timezone}
    cases=evaluation_cases(frame,selection_end,d['end'],settings,progress)
    baseline=torch.load(DATA/'load.pt',map_location='cpu',weights_only=False)
    old=evaluate(baseline,cases,settings);candidate=evaluate(artifact,cases,settings)
    accepted,reason=promotion_decision(old['summary'],candidate['summary'],settings.promotion_improvement_percent)
    metadata=copy.deepcopy(read_json('model-load'));metadata.update(model='Direct 24-hour CNN',trained_at=pd.Timestamp.now(tz='UTC').isoformat(),metrics=candidate['summary']['hourly'],chart=candidate['chart'],recursive_validation=candidate['summary'],training_rows=len(ids),fine_tuned=False)
    artifact['metadata']=metadata;artifact['uncertainty']=uncertainty(candidate,settings);save_artifact(artifact,'load-direct-candidate.pt')
    if accepted:
        baseline['metadata']=read_json('model-load');save_artifact(baseline,'load-previous.pt');save_artifact(artifact);write_json('model-load',metadata);write_json('load-uncertainty',artifact['uncertainty'])
    result={'completed_at':pd.Timestamp.now(tz='UTC').isoformat(),'decision':'promoted' if accepted else 'retained','reason':reason,'baseline':old['summary'],'candidate':candidate['summary'],'block_hours':BLOCK,'training_rows':len(ids)}
    write_json('direct-trial',result);return result

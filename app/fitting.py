"""Small chronological development holdouts and inexpensive energy-aware scoring."""
import numpy as np
import pandas as pd
from .load_splits import training_mask


def development_cases(frame,start,end,settings,kind,bank=None,progress=lambda s:None,cached_only=False):
    from .study import cases,origins
    valid=frame.validation_eligible&frame.target.notna()&training_mask(frame.index,start,end)
    if kind=='solar':valid |= frame.full_dark.astype(bool)&frame.derived_method.eq('dark_zero')&training_mask(frame.index,start,end)
    complete=valid.rolling(24,min_periods=24).sum().shift(-23).eq(24)
    if kind=='load':
        context=frame.get('context_target',frame.target).notna().rolling(168,min_periods=1).sum().shift(2)
        complete &= context>=126
    possible=frame.index[complete&(frame.index+pd.Timedelta(hours=24)<=end)]
    if not len(possible):raise ValueError('No complete training-history day is available for stopping. Check dates and history coverage.')
    finish=min(end,possible[-1]+pd.Timedelta(hours=24))
    targets=frame.index[frame.train_eligible&frame.target.notna()&training_mask(frame.index,start,finish)]
    if len(targets)<100:raise ValueError('Not enough training history before the development period')
    count=min(672,max(96,int(len(targets)*.2)),len(targets)-48)
    left=targets[-count]
    def period(left,finish):
        covered=set(possible)
        issues=[t for t in origins(left,finish,settings) if t.floor('h') in covered and t+pd.Timedelta(hours=24)<=finish]
        if len(issues)>16:issues=[issues[i] for i in np.linspace(0,len(issues)-1,16,dtype=int)]
        progress('Preparing chronological stopping forecasts')
        try:cc,_=cases(frame,left,finish,settings,kind,bank,progress=progress,cached_only=cached_only,issue_times=issues)
        except ValueError:cc=[]
        basis='issued'
        if len({c['origin'].date() for c in cc})<3:
            progress('Using prepared historical weather for stopping: too few archived issues are available')
            cc,_=cases(frame,left,finish,settings,kind,bank,basis='historical',progress=progress,issue_times=issues)
            basis='historical'
        if not cc:raise ValueError('No complete stopping forecasts. Prepare a longer training period.')
        return {'start':left,'end':finish,'cases':cc,'weather_basis':basis}
    recent=period(left,finish);folds=[]
    before=targets[targets+pd.Timedelta(hours=1)<=left]
    if len(before)>=240:
        earlier_left=before[-min(672,max(96,int(len(before)*.2)))]
        try:
            earlier=period(earlier_left,left)
            if earlier['weather_basis']==recent['weather_basis']:folds.append(earlier)
        except ValueError:pass
    folds.append(recent.copy());recent['folds']=folds
    return recent



class EnergyValidation:
    """Pack replay rows once; each boosting iteration uses a few vector sums."""
    def __init__(self,cc,kind,settings):
        from .study import window
        self.masks=[];chunks=[];actual=[];rows=[];weights=[];groups=[];durations=[];offset=0
        for c in cc:
            mask=c['valid'].copy()
            if kind=='solar':mask &= c['daylight']
            self.masks.append(mask);chunks.append(c['x'].loc[mask]);actual.extend(c['actual'][mask])
            lookup=np.full(len(mask),-1,int);lookup[mask]=offset+np.arange(mask.sum());offset+=int(mask.sum())
            windows=[(str(h),c['origin']+pd.Timedelta(hours=h)) for h in (24,48,72)]
            local=c['origin'].tz_convert(settings.timezone)
            if kind=='solar' and local.strftime('%H:%M')=='00:00':windows.append(('daily',c['daily_end']))
            if kind=='load' and local.strftime('%H:%M') in {w.start for w in settings.export_windows} and c['charge_start'] is not None:
                windows.append(('overnight',c['charge_start']))
            for label,until in windows:
                w=window(c,until)
                if w is None:continue
                selected=mask&(w>0)
                rows.extend(lookup[selected]);weights.extend(w[selected]);groups.extend([len(durations)]*int(selected.sum()))
                durations.append((label,(until-c['origin']).total_seconds()/86400))
        self.x=pd.concat(chunks,ignore_index=True);self.y=np.asarray(actual,float)
        self.rows=np.asarray(rows,int);self.weights=np.asarray(weights,float);self.groups=np.asarray(groups,int)
        self.labels=np.array([label for label,_ in durations]);self.days=np.array([days for _,days in durations])
        if not len(durations):raise ValueError('Stopping requires complete energy windows')
        self.actual_energy=np.bincount(self.groups,weights=self.y[self.rows]*self.weights,minlength=len(durations))

    def scores(self,prediction):
        prediction=np.maximum(0,np.asarray(prediction,float))
        energy=np.bincount(self.groups,weights=prediction[self.rows]*self.weights,minlength=len(self.days))
        errors=np.abs(energy-self.actual_energy)/self.days
        labels=np.unique(self.labels)
        priority={'24':3,'daily':1,'48':1,'72':1,'overnight':1}
        energy_mae=float(np.average([errors[self.labels==label].mean() for label in labels],weights=[priority[label] for label in labels]))
        hourly_mae=float(np.mean(abs(prediction-self.y)))
        # Energy errors are in kWh per day; the smaller hourly term protects shape.
        return energy_mae+3*hourly_mae,energy_mae,hourly_mae

    def metric(self,actual,prediction):
        score,energy,hourly=self.scores(prediction)
        return [('energy_and_hourly',score,False),('energy_mae_per_day',energy,False),('hourly_mae',hourly,False)]

    def pack(self,predictions):
        return np.concatenate([q['point'][mask] for q,mask in zip(predictions,self.masks)])

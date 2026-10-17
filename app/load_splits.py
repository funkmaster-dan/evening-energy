"""Coverage-only recommendations; model errors never select a holdout."""
import numpy as np
import pandas as pd

def inventory(frame,start,end,settings):
    from .load_training import issue_origins,next_charge_start
    frame=frame.copy(deep=False);frame.attrs={}
    origins=issue_origins(start,end,settings);counts={str(h):0 for h in (24,48,72)};days={str(h):set() for h in (24,48,72)};discarded={'targets':0,'context':0};overnight=set();usable=[]
    for origin in origins:
        context=frame.loc[(frame.index<origin)&(frame.index>=origin-pd.Timedelta(hours=168))]
        context=context.get('context_target',context.target)
        # get() fallback must not replace the selected context slice.
        if context.reindex(pd.date_range(origin-pd.Timedelta(hours=168),periods=168,freq='h')).count()<126:discarded['context']+=1;continue
        passed=False
        for h in (24,48,72):
            times=pd.date_range(origin,periods=h,freq='h');f=frame.reindex(times)
            valid=(times[-1]+pd.Timedelta(hours=1)<=end) and f.validation_eligible.fillna(False).all() and f.target.notna().all()
            if valid:counts[str(h)]+=1;days[str(h)].add(str(origin.tz_convert(settings.timezone).date()));passed=True
        if not passed:discarded['targets']+=1
        else:usable.append(origin.isoformat())
        local=origin.tz_convert(settings.timezone)
        for window in settings.export_windows:
            export=pd.Timestamp(f'{local.date()} {window.start}').tz_localize(settings.timezone,ambiguous=False,nonexistent='shift_forward').tz_convert('UTC');stop=next_charge_start(export,settings)
            if export.floor('h')==origin and stop is not None and stop<=end:
                f=frame.reindex(pd.date_range(export.floor('h'),stop.ceil('h'),freq='h',inclusive='left'))
                if f.validation_eligible.fillna(False).all() and f.target.notna().all():overnight.add(str(local.date()))
    f=frame.loc[(frame.index>=start)&(frame.index<end)];eligible=int(f.validation_eligible.sum())
    return {'start':str(start),'end':str(end),'eligible_hours':eligible,'eligible_pct':100*eligible/max(1,len(f)),'complete_windows':counts,'distinct_issue_days':{h:len(v) for h,v in days.items()},'overnight_issue_days':len(overnight),'discarded_windows':discarded,'origins':usable,'archived_weather_coverage':'Checked and frozen by replay before parameter comparison; proxies reported separately','regime':'Selected sensor layout; replacement units verified during preparation'}

def recommendations(frame,start,end,settings):
    # Most recent completed local day; recorder has a full additional hour to settle.
    settled=(pd.Timestamp.now(tz='UTC')-pd.Timedelta(hours=1)).tz_convert(settings.timezone).normalize().tz_convert('UTC');end=min(pd.Timestamp(end),settled)
    best=None
    for days in (28,35,42,49,56):
        left=max(pd.Timestamp(start),end-pd.DateOffset(days=days));row=inventory(frame,left,end,settings)
        good=row['eligible_pct']>=90 and row['distinct_issue_days']['72']>=10 and (not settings.charging_windows or not settings.export_windows or row['overnight_issue_days']>=10)
        if best is None or row['distinct_issue_days']['72']>best['distinct_issue_days']['72']:best=row
        if good:return {'periods':[row],'training_cutoff':str(left),'reason':'Most recent recent interval meeting practical coverage targets; reserved before model selection','strong_coverage':True}
    # Recent separated blocks avoid silently fitting intervening/later outcomes.
    blocks=[]
    for right in pd.date_range(end,max(pd.Timestamp(start)+pd.Timedelta(days=14),end-pd.Timedelta(days=112)),freq='-14D'):
        left=right-pd.Timedelta(days=14);row=inventory(frame,left,right,settings)
        if row['eligible_pct']>=90 and row['distinct_issue_days']['72']>=3:blocks.append(row)
        if len(blocks)>=2:break
    if len(blocks)>=2:return {'periods':sorted(blocks,key=lambda r:r['start']),'training_cutoff':min(r['start'] for r in blocks),'reason':'Gaps prevent a contiguous recent holdout; later blocks and intervening outcomes are excluded from fitting','strong_coverage':False}
    return {'periods':[best] if best else [],'training_cutoff':best['start'] if best else None,'reason':'Insufficient recent complete windows; use experiment minimum and collect prospective issues','strong_coverage':False}


def training_mask(index,start,end,step='1h'):
    """A fitting target's complete interval must be within its split."""
    return (index>=pd.Timestamp(start))&(index+pd.Timedelta(step)<=pd.Timestamp(end))


def split_coverage(frame,start,end,settings,complete=False):
    from .quality import dark_hours
    part=frame.loc[training_mask(frame.index,start,end) if complete else (frame.index+pd.Timedelta(hours=1)>start)&(frame.index<end)]
    dark=pd.Series(dark_hours(settings,part.index),index=part.index)
    measured=part.get('measurement_valid',part.target.notna())
    day=~dark
    weights=np.array([max(0,(min(t+pd.Timedelta(hours=1),end)-max(t,start)).total_seconds()/3600) for t in part.index])
    return {'interval_weighted_hours':float(weights.sum()),'measured_weighted_hours':float(weights@measured.to_numpy(dtype=float)),'expected_hours':len(part),'measured_hours':int(measured.sum()),'measured_pct':round(100*measured.sum()/max(1,len(part)),2),'daylight_hours':int(day.sum()),'measured_daylight_hours':int((measured&day).sum()),'measured_daylight_pct':round(100*(measured&day).sum()/max(1,day.sum()),2),'derived_dark_zero_hours':int(part.derived_method.eq('dark_zero').sum()),'interpolated_hours':int(part.derived_method.eq('interpolation').sum()),'quiet_hours':int(part.quiet_regime.sum()),'reasons':part.quality_flags.value_counts().to_dict()}

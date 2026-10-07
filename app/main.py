import asyncio, logging, time, uuid
from contextlib import asynccontextmanager
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .config import DATA, Settings, load_settings, read_json, write_json
from .sources import HomeAssistant, store_live
from .ha_inputs import numeric_value, state_available, snapshot_age
from .datasets import build_dataset, list_datasets
from .models import train, list_models, model_ready
from .forecast import Forecaster, inside

log=logging.getLogger('energy-tool')
settings=load_settings(); forecaster=Forecaster(); jobs={}; entities=[]; live={}; forecast=read_json('forecast',{}); last_error=None
if forecast and 'achievable_export_kwh' not in forecast:forecast={}
busy=False; forecast_lock=asyncio.Lock()

def refresh_live():
    global entities,live
    states=HomeAssistant(settings).states(); entities=states
    lookup={s['entity_id']:s for s in states}
    store_live([s for s in states if s['entity_id'] in {settings.consumption_entity,settings.solar_ct_entity,settings.battery_soc_entity,settings.battery_power_entity,*[b.entity for b in settings.banks]}])
    now=pd.Timestamp.now(tz='UTC')
    def value(entity,power=False):return numeric_value(lookup.get(entity,{}),power)
    soc=value(settings.battery_soc_entity)
    if soc is not None and not 0<=soc<=100:soc=None
    ids={settings.consumption_entity,settings.solar_ct_entity,settings.battery_soc_entity,settings.battery_power_entity,*[b.entity for b in settings.banks],*[a.entity for a in settings.appliances if a.entity],*[a.power_entity for a in settings.appliances if a.power_entity]}
    inputs={entity:{'available':state_available(lookup.get(entity,{})),'state':lookup.get(entity,{}).get('state'),'last_changed':lookup.get(entity,{}).get('last_changed'),'last_updated':lookup.get(entity,{}).get('last_updated'),'last_reported':lookup.get(entity,{}).get('last_reported')} for entity in ids}
    live={'soc':soc,'load_kw':value(settings.consumption_entity,True),'solar_kw':value(settings.solar_ct_entity,True),'battery_kw':value(settings.battery_power_entity,True),'soc_age_seconds':0,'updated_at':now.isoformat(),'freshness_basis':'Successful Home Assistant snapshot and entity availability; state timestamps show changes, not sensor expiry','inputs':inputs,'appliance_states':{(a.entity or a.power_entity or a.name):{'state':lookup.get(a.entity,{}).get('state'),'available':state_available(lookup.get(a.entity,{})),'power_kw':value(a.power_entity,True) if a.power_entity else None,'next_event':lookup.get(a.entity,{}).get('attributes',{}).get('next_event'),'hvac_action':lookup.get(a.entity,{}).get('attributes',{}).get('hvac_action')} for a in settings.appliances},'banks':[{'name':b.name,'power_kw':value(b.entity,True)} for b in settings.banks]}

    return live

async def update_forecast(force=False):
    global forecast,last_error
    async with forecast_lock:
        await asyncio.to_thread(refresh_live)
        forecast=await asyncio.to_thread(forecaster.run,settings,live,force)
        last_error=None
        return forecast

def start_job(label,fn):
    global busy
    if busy: raise HTTPException(409,'Another dataset or calibration job is running. Wait for it to finish.')
    busy=True; id=uuid.uuid4().hex[:12]
    jobs[id]={'id':id,'label':label,'status':'running','message':'Starting','started_at':pd.Timestamp.now(tz='UTC').isoformat()}
    def progress(msg):
        jobs[id]['message']=msg
        log.info('%s: %s',label,msg)
    async def worker():
        global busy,last_error
        try:
            result=await asyncio.to_thread(fn,progress)
            jobs[id].update(status='complete',message='Complete',result=result)
            forecaster.reset()
        except Exception as exc:
            log.exception('Job failed'); jobs[id].update(status='failed',message=str(exc)); last_error=str(exc)
        finally:
            busy=False; jobs[id]['finished_at']=pd.Timestamp.now(tz='UTC').isoformat(); write_json('jobs',list(jobs.values())[-20:])
    asyncio.create_task(worker()); return jobs[id]

def fine_tune(progress):
    from .load_training import prepare_fine_tune, train_load
    summary,d=prepare_fine_tune(settings,progress)
    return train_load(settings,summary,d,progress,epochs=6,fine_tune=True)

async def scheduler():
    global last_error
    last_live=0; last_forecast=0; last_score=0; last_tune=read_json('fine-tune-schedule',{}).get('last_attempt',time.time())
    if not read_json('fine-tune-schedule'):
        write_json('fine-tune-schedule',{'last_attempt':last_tune})
    while True:
        try:
            if settings.ha_token:
                now=time.time()
                if now-last_live>=settings.live_interval_seconds:
                    await asyncio.to_thread(refresh_live); last_live=now
                if now-last_forecast>=settings.battery_interval_minutes*60 or forecaster.load_at==0 or forecaster.solar_at==0:
                    try: await update_forecast()
                    finally: last_forecast=now
                if not busy and now-last_score>=3600:
                    from .forecast_measurement import score_issues
                    try: await asyncio.to_thread(score_issues,settings)
                    except Exception as exc: log.warning('Issued forecast scoring: %s',exc)
                    last_score=now
                if settings.auto_fine_tune and not busy and model_ready(settings,'load','load') and now-last_tune>=settings.fine_tune_hours*3600:
                    start_job('Automatic CNN fine tuning',fine_tune); last_tune=now; write_json('fine-tune-schedule',{'last_attempt':now})
        except Exception as exc:
            last_error=str(exc); log.warning('Refresh: %s',exc); last_live=time.time()
        await asyncio.sleep(10)

@asynccontextmanager
async def lifespan(app):
    task=asyncio.create_task(scheduler())
    yield
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass

app=FastAPI(title='Evening Energy',lifespan=lifespan)
app.mount('/static',StaticFiles(directory='app/static'),name='static')

@app.get('/')
def index(): return FileResponse('app/static/index.html')

@app.get('/api/health')
def health(): return {'status':'ok','ha_configured':bool(settings.ha_token)}

@app.get('/api/config')
def config(): return settings.public()

@app.put('/api/config')
async def save_config(body:dict):
    global settings
    data={**settings.model_dump(),**{k:v for k,v in body.items() if k not in ('token_configured',)}}
    if not body.get('ha_token'): data['ha_token']=settings.ha_token
    try:
        next_settings=Settings(**data)
        ZoneInfo(next_settings.timezone)
        pd.Timestamp(next_settings.load_history_start,tz=next_settings.timezone)
        if next_settings.state not in ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']: raise ValueError('Choose an Australian state or territory')
        for w in next_settings.charging_windows:
            for t in (w.start,w.end): datetime.strptime(t,'%H:%M')
            if w.start==w.end: raise ValueError('Charging window start and end must differ')
        for a in next_settings.appliances:
            for t in (a.start,a.end): datetime.strptime(t,'%H:%M')
            if a.mode not in ('floor','additional') or any(day not in range(7) for day in a.weekdays): raise ValueError('Invalid appliance mode or weekdays')
        if next_settings.weather_model not in ('ecmwf_ifs','gfs_global','icon_global'): raise ValueError('Choose an available global weather model')
        for t in (next_settings.export_start,next_settings.export_end): datetime.strptime(t,'%H:%M')
        if next_settings.export_start==next_settings.export_end: raise ValueError('Peak start and end must differ')
        if len({b.id for b in next_settings.banks})!=len(next_settings.banks) or not next_settings.banks: raise ValueError('Panel banks need unique identifiers')
        settings=next_settings; settings.save(); forecaster.reset()
    except Exception as exc: raise HTTPException(400,str(exc))
    # Immediately invalidate old forecasts when configuration changes.
    forecast.clear()
    return settings.public()

@app.post('/api/connect')
async def connect_ha():
    try:
        await asyncio.to_thread(refresh_live)
        return {'connected':True,'entities':len(entities),'live':live}
    except Exception as exc: raise HTTPException(400,str(exc))

@app.get('/api/entities')
async def get_entities(context:str='power',search:str=''):
    if not entities and settings.ha_token:
        try: await asyncio.to_thread(refresh_live)
        except Exception as exc: raise HTTPException(400,str(exc))
    result=[]
    for s in entities:
        a=s.get('attributes',{}); dc=a.get('device_class'); unit=a.get('unit_of_measurement','')
        if context=='appliance': matched=s['entity_id'].split('.')[0] in ('climate','switch','binary_sensor','input_boolean','schedule','sensor')
        elif context=='soc': matched=dc=='battery' or unit=='%'
        elif context=='energy': matched=dc in ('power','energy') or unit in ('W','kW','Wh','kWh')
        else: matched=dc=='power' or unit in ('W','kW')
        if matched and (context=='appliance' or s['entity_id'].startswith('sensor.')):
            d={'id':s['entity_id'],'name':a.get('friendly_name',s['entity_id']),'unit':unit,'state':s['state'],'device_class':dc}
            if not search or search.lower() in (d['id']+' '+d['name']).lower(): result.append(d)
    return sorted(result,key=lambda s:s['id'])

@app.get('/api/dashboard')
def dashboard():
    return {'measurement':read_json('forecast-measurement',{}),'weather_comparison':read_json('weather-comparison',{}),'direct_trial':read_json('direct-trial',{}),'fine_tuning':read_json('fine-tuning-status',{}),'rollback_available':(DATA/'load-previous.pt').exists(),'forecast':forecast,'live':{**live,'soc_age_seconds':snapshot_age(live),'ha_snapshot_age_seconds':snapshot_age(live)},'models':list_models(),'datasets':list_datasets(),'jobs':list(jobs.values())[-20:],'error':last_error,'busy':busy}

@app.get('/api/forecast')
def get_forecast(): return forecast

@app.get('/api/excess')
def get_excess():
    if not forecast: raise HTTPException(503,'First forecast is being prepared')
    result={k:v for k,v in forecast.items() if k not in ('forecast','soc_projections','live','calibration','used_window_allowance_kwh')}
    result['peak_active']=inside(pd.Timestamp.now(tz=settings.timezone),settings.export_start,settings.export_end)
    expired=pd.Timestamp.now(tz='UTC')>pd.Timestamp(forecast['valid_until'])
    if expired:
        result['ready']=False; result['excess_kwh']={k:0 for k in ('low','middle','high')}; result['excess_percent']={k:0 for k in ('low','middle','high')}; result['achievable_export_kwh']={k:0 for k in ('low','middle','high')}; result['achievable_export_percent']={k:0 for k in ('low','middle','high')}; result['warnings']=[*result['warnings'],'Forecast expired; allowance is zero.']
    return result

@app.post('/api/refresh')
async def refresh():
    try: return await update_forecast(force=True)
    except Exception as exc: raise HTTPException(400,str(exc))

@app.get('/api/datasets')
def datasets(): return list_datasets()

@app.post('/api/datasets')
async def build(body:dict): return start_job('Build '+body.get('kind','')+' dataset',lambda p:build_dataset(settings,body,p))

@app.get('/api/models')
def models(): return list_models()

@app.post('/api/train/{dataset_id}')
async def calibration(dataset_id:str):
    if dataset_id not in {d['id'] for d in list_datasets()}: raise HTTPException(404,'Dataset not found')
    return start_job('Calibrate '+dataset_id,lambda p:train(settings,dataset_id,p))

@app.post('/api/fine-tune')
async def tune(): return start_job('Fine tune CNN using recent history',fine_tune)

@app.post('/api/load/rollback')
async def rollback():
    from .load_training import rollback_load
    return start_job('Restore previous CNN',lambda p:rollback_load(settings))

@app.get('/api/jobs/{job_id}')
def job(job_id:str):
    if job_id not in jobs: raise HTTPException(404,'Job not found')
    return jobs[job_id]

@app.post('/api/battery/apply-observed-limits')
def apply_limits():
    meta=read_json('battery-calibration',{})
    if not meta: raise HTTPException(400,'Calibrate the battery first')
    settings.battery_charge_limit_kw=max(.1,meta['observed_charge_kw']); settings.battery_discharge_limit_kw=max(.1,meta['observed_discharge_kw']); settings.save(); forecaster.reset()
    return settings.public()

@app.post('/api/forecast/score')
async def score_forecasts():
    from .forecast_measurement import score_issues
    return start_job('Measure issued forecasts',lambda p:score_issues(settings,p))

@app.post('/api/load/direct-trial')
async def direct_trial():
    from .direct_load import trial
    return start_job('Trial direct 24-hour CNN',lambda p:trial(settings,p))

@app.post('/api/weather/assess')
async def assess_weather():
    from .weather_comparison import assess
    return start_job('Compare archived weather models',lambda p:assess(settings,p))

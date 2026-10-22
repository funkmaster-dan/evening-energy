import asyncio, logging, time, uuid, os
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .config import Settings, load_settings, read_json, write_json, solar_banks
from .sources import HomeAssistant, store_live
from .ha_inputs import numeric_value, state_available, snapshot_age
from .quality import physical, LOAD_PIPELINE, QUALITY_POLICY, effective_policy
from .setup import inspect_setup
from .datasets import build_dataset, list_datasets
from .models import train, list_models, list_candidates, promote_candidate, delete_candidate, clean_model_storage, clear_outdated_preparations
from .forecast import Forecaster
from .battery import POLICY, export_active

log=logging.getLogger('energy-tool')
settings=load_settings(); forecaster=Forecaster(); jobs={}; entities=[]; live={}; forecast=read_json('forecast',{}); last_error=None
if forecast and ('inverter_efficiency' in forecast.get('calibration',{}) or forecast.get('load_forecast_basis')!=LOAD_PIPELINE or forecast.get('quality_policy')!=QUALITY_POLICY or forecast.get('soc_band_method')!=POLICY or forecast.get('export_policy',{}).get('name')!=POLICY):forecast={}
busy=False; forecast_lock=asyncio.Lock()

def refresh_live():
    global entities,live
    states=HomeAssistant(settings).states(); entities=states
    lookup={s['entity_id']:s for s in states}
    store_live([s for s in states if s['entity_id'] in {settings.consumption_entity,settings.solar_ct_entity,settings.battery_soc_entity,settings.battery_power_entity,*[b.entity for b in solar_banks(settings)]}])
    now=pd.Timestamp.now(tz='UTC')
    def value(entity,power=False):
        value=numeric_value(lookup.get(entity,{}),power)
        if not power or value is None:return value
        kind='battery_power' if entity==settings.battery_power_entity else 'load' if entity==settings.consumption_entity else 'solar'
        limit=max(settings.battery_charge_limit_kw,settings.battery_discharge_limit_kw)*1.2 if kind=='battery_power' else effective_policy(settings,kind)['max_power_kw']
        bank=next((b for b in solar_banks(settings) if b.entity==entity),None)
        if bank:limit=effective_policy(settings,'solar',read_json('dataset-solar-'+bank.id,{}).get('request',{}))['max_power_kw']
        cleaned,_=physical(pd.Series([value]),kind,limit)
        return None if pd.isna(cleaned.iloc[0]) else float(cleaned.iloc[0])
    soc=value(settings.battery_soc_entity) if lookup.get(settings.battery_soc_entity,{}).get('attributes',{}).get('unit_of_measurement')=='%' else None
    if soc is not None and not 0<=soc<=100:soc=None
    ids={settings.consumption_entity,settings.solar_ct_entity,settings.battery_soc_entity,settings.battery_power_entity,*[b.entity for b in solar_banks(settings)]}
    inputs={entity:{'available':state_available(lookup.get(entity,{})),'state':lookup.get(entity,{}).get('state'),'last_changed':lookup.get(entity,{}).get('last_changed'),'last_updated':lookup.get(entity,{}).get('last_updated'),'last_reported':lookup.get(entity,{}).get('last_reported')} for entity in ids}
    live={'soc':soc,'load_kw':value(settings.consumption_entity,True),'solar_kw':value(settings.solar_ct_entity,True),'battery_kw':value(settings.battery_power_entity,True),'soc_age_seconds':0,'updated_at':now.isoformat(),'freshness_basis':'Successful Home Assistant snapshot and entity availability; state timestamps show changes, not sensor expiry','inputs':inputs,'banks':[{'name':b.name,'power_kw':value(b.entity,True)} for b in solar_banks(settings)]}

    return live

async def update_forecast(force=False):
    global forecast,last_error
    async with forecast_lock:
        if busy:return forecast
        runner=forecaster; current_settings=settings
        await asyncio.to_thread(refresh_live)
        result=await asyncio.to_thread(runner.run,current_settings,live,force)
        if runner is forecaster:
            forecast=result; last_error=None
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
            # Let any existing projection finish before calibration takes CPU.
            async with forecast_lock:pass
            await asyncio.to_thread(fn,progress)
            jobs[id].update(status='complete',message='Complete')
        except Exception as exc:
            log.exception('Job failed'); jobs[id].update(status='failed',message=str(exc)); last_error=str(exc)
        finally:
            busy=False; jobs[id]['finished_at']=pd.Timestamp.now(tz='UTC').isoformat(); write_json('jobs',list(jobs.values())[-20:])
    asyncio.create_task(worker()); return jobs[id]

async def scheduler():
    global last_error
    last_live=0; last_forecast=0
    while True:
        try:
            if settings.ha_token:
                now=time.time()
                if now-last_live>=settings.live_interval_seconds:
                    await asyncio.to_thread(refresh_live); last_live=now
                if not busy and (not forecast or now-last_forecast>=settings.battery_interval_minutes*60 or forecaster.load_at==0 or forecaster.solar_at==0):
                    try: await update_forecast()
                    finally: last_forecast=now
        except Exception as exc:
            last_error=str(exc); log.warning('Refresh: %s',exc); last_live=time.time()
        await asyncio.sleep(10)

@asynccontextmanager
async def lifespan(app):
    global forecast
    settings.save()
    forecast={};write_json('forecast',{})
    clear_outdated_preparations()
    clean_model_storage(settings)
    task=asyncio.create_task(scheduler())
    yield
    task.cancel()
    try: await task
    except asyncio.CancelledError: pass

app=FastAPI(title='ToU Discharge Forecaster',lifespan=lifespan)
app.mount('/static',StaticFiles(directory='app/static'),name='static')

@app.get('/')
def index(): return FileResponse('app/static/index.html')

@app.get('/api/health')
def health(): return {'status':'ok','ha_configured':bool(settings.ha_token),'release_commit':os.getenv('ENERGY_RELEASE_COMMIT','unknown'),'quality_policy':QUALITY_POLICY}

@app.get('/api/config')
def config(): return settings.public()

@app.put('/api/config')
async def save_config(body:dict):
    global settings,forecaster
    if busy: raise HTTPException(409,'Wait for the current job before changing configuration.')
    data={**settings.model_dump(),**{k:v for k,v in body.items() if k not in ('token_configured',)}}
    if not body.get('ha_token'): data['ha_token']=settings.ha_token
    try:
        next_settings=Settings(**data)
        ZoneInfo(next_settings.timezone)
        if next_settings.state not in ['ACT','NSW','NT','QLD','SA','TAS','VIC','WA']: raise ValueError('Choose an Australian state or territory')
        if next_settings.weather_model not in ('ecmwf_ifs','gfs_global','icon_global'): raise ValueError('Choose an available global weather model')
        if len({b.id for b in next_settings.banks})!=len(next_settings.banks) or not next_settings.banks: raise ValueError('Panel banks need unique identifiers')
    except Exception as exc: raise HTTPException(400,str(exc))
    next_forecaster=Forecaster()
    if settings.model_dump(exclude={'reserve_kwh'})==next_settings.model_dump(exclude={'reserve_kwh'}):
        for name in ('load_cache','solar_cache','load_at','solar_at'):
            setattr(next_forecaster,name,getattr(forecaster,name))
    settings=next_settings; settings.save(); clean_model_storage(settings); forecaster=next_forecaster
    # Keep an in-progress calculation's caches intact; discard its old result.
    forecast.clear()
    return settings.public()

@app.post('/api/connect')
async def connect_ha():
    try:
        await asyncio.to_thread(refresh_live)
        return {'connected':True,'entities':len(entities),'live':live}
    except Exception as exc: raise HTTPException(400,str(exc))

@app.get('/api/setup')
async def setup_check():
    try: return await asyncio.to_thread(inspect_setup, settings)
    except Exception as exc: raise HTTPException(400, str(exc))

@app.get('/api/entities')
async def get_entities(context:str='power',search:str=''):
    if context not in ('power','soc','energy'):raise HTTPException(400,'Unsupported entity context; choose power, soc or energy')
    if not entities and settings.ha_token:
        try: await asyncio.to_thread(refresh_live)
        except Exception as exc: raise HTTPException(400,str(exc))
    result=[]
    for s in entities:
        a=s.get('attributes',{}); dc=a.get('device_class'); unit=a.get('unit_of_measurement','')
        if context=='soc': matched=dc=='battery' or unit=='%'
        elif context=='energy': matched=dc in ('power','energy') or unit in ('W','kW','Wh','kWh')
        else: matched=dc=='power' or unit in ('W','kW')
        if matched and s['entity_id'].startswith('sensor.'):
            d={'id':s['entity_id'],'name':a.get('friendly_name',s['entity_id']),'unit':unit,'state':s['state'],'device_class':dc}
            if not search or search.lower() in (d['id']+' '+d['name']).lower(): result.append(d)
    return sorted(result,key=lambda s:s['id'])

def current_forecast():
    if not forecast:return {}
    result={**forecast}
    if pd.Timestamp.now(tz='UTC')>pd.Timestamp(forecast['valid_until']):
        result['ready']=False
        for key in ('excess_kwh','excess_percent','achievable_export_kwh','achievable_export_percent'):
            result[key]={k:0 for k in ('low','middle','high')}
        result['warnings']=[*result.get('warnings',[]),'Forecast expired; allowance is zero.']
        result['uncertainty']={**result.get('uncertainty',{}),'status':'Forecast expired; refresh required'}
    return result

@app.get('/api/dashboard')
def dashboard():
    return {'candidates':list_candidates(),'forecast':current_forecast(),'live':{**live,'soc_age_seconds':snapshot_age(live),'ha_snapshot_age_seconds':snapshot_age(live)},'models':list_models(),'datasets':list_datasets(settings),'jobs':list(jobs.values())[-20:],'error':last_error,'busy':busy}

@app.get('/api/forecast')
def get_forecast(): return current_forecast()

@app.get('/api/excess')
def get_excess():
    if not forecast: raise HTTPException(503,'First forecast is being prepared')
    result={k:v for k,v in current_forecast().items() if k not in ('forecast','soc_projections','live','calibration','used_window_allowance_kwh')}
    result['peak_active']=export_active(pd.Timestamp.now(tz=settings.timezone),settings)
    return result

@app.post('/api/refresh')
async def refresh():
    try: return await update_forecast(force=True)
    except Exception as exc: raise HTTPException(400,str(exc))

@app.get('/api/datasets')
def datasets(): return list_datasets(settings)

@app.post('/api/datasets')
async def build(body:dict):
    if body.get('kind') not in ('load','solar','battery'):raise HTTPException(400,'Choose consumption, solar production or battery efficiency')
    return start_job('Build '+body['kind']+' dataset',lambda p:build_dataset(settings,body,p))

@app.get('/api/models')
def models(): return list_models()

@app.post('/api/train/{dataset_id}')
async def calibration(dataset_id:str):
    if dataset_id not in {d['id'] for d in list_datasets(settings)}: raise HTTPException(404,'Dataset not found')
    if read_json('candidate-'+dataset_id): raise HTTPException(409,'Promote or delete the existing candidate first.')
    return start_job('Calibrate '+dataset_id,lambda p:train(settings,dataset_id,p))

@app.post('/api/models/{dataset_id}/promote')
async def promote(dataset_id:str):
    if busy: raise HTTPException(409,'Wait for the current job to finish.')
    async with forecast_lock:
        try: result=promote_candidate(settings,dataset_id)
        except ValueError as exc: raise HTTPException(400,str(exc))
        forecaster.reset(); forecast.clear()
        write_json('forecast',{})
    return result

@app.delete('/api/models/{dataset_id}/candidate')
def discard(dataset_id:str):
    if busy: raise HTTPException(409,'Wait for the current job to finish.')
    try: delete_candidate(dataset_id)
    except ValueError as exc: raise HTTPException(404,str(exc))
    return {'deleted':dataset_id}

@app.get('/api/jobs/{job_id}')
def job(job_id:str):
    if job_id not in jobs: raise HTTPException(404,'Job not found')
    return jobs[job_id]

@app.post('/api/battery/apply-observed-limits')
def apply_limits():
    meta=read_json('model-battery',{})
    if not meta: raise HTTPException(400,'Calibrate the battery first')
    settings.battery_charge_limit_kw=max(.1,meta['observed_charge_kw']); settings.battery_discharge_limit_kw=max(.1,meta['observed_discharge_kw']); settings.save(); forecaster.reset()
    return settings.public()

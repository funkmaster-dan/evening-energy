import json, os
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, model_validator

DATA = Path(os.getenv('ENERGY_DATA_DIR', 'data'))
DATA.mkdir(parents=True, exist_ok=True)

class Bank(BaseModel):
    id: str
    name: str
    entity: str
    azimuth: float = Field(ge=0, le=360)
    tilt: float = Field(default=30, ge=0, le=90)
    history_start: str = Field(default='', pattern=r'^(\d{4}-\d{2}-\d{2})?$')

class TimeWindow(BaseModel):
    start: str
    end: str

    @model_validator(mode='after')
    def validate_times(self):
        for clock in (self.start,self.end): datetime.strptime(clock,'%H:%M')
        if self.start==self.end: raise ValueError('Window start and end must differ')
        return self

class Window(TimeWindow):
    start: str = '11:00'
    end: str = '14:00'
    limit_kwh: float = Field(default=50, ge=0)
    charge_kw: float | None = Field(default=None, ge=0)

class ExportWindow(TimeWindow):
    discharge_kw: float | None = Field(default=None, ge=0)

class Settings(BaseModel):
    ha_url: str = 'http://homeassistant.local:8123'
    ha_token: str = ''
    latitude: float = -34.9285
    longitude: float = 138.6007
    timezone: str = 'Australia/Adelaide'
    state: str = 'SA'
    school_calendar_url: str = ''
    weather_model: str = 'ecmwf_ifs'
    load_max_power_kw: float = Field(default=40, gt=0)
    solar_max_power_kw: float = Field(default=40, gt=0)
    horizon_hours: int = Field(default=48, ge=24, le=72)
    mc_samples: int = Field(default=5000, ge=10, le=5000)
    load_interval_minutes: int = Field(default=60, ge=5)
    solar_interval_minutes: int = Field(default=60, ge=5)
    battery_interval_minutes: int = Field(default=5, ge=1)
    live_interval_seconds: int = Field(default=60, ge=15)
    consumption_entity: str = 'sensor.home_power'
    historical_consumption_entities: list[str] = ['sensor.home_power']
    battery_power_entity: str = 'sensor.battery_power'
    battery_soc_entity: str = 'sensor.battery_soc'
    solar_ct_entity: str = 'sensor.solar_power'
    solar_ct_proxy: bool = False
    battery_capacity_kwh: float = Field(default=10, gt=0)
    battery_charge_limit_kw: float = Field(default=5, gt=0)
    battery_discharge_limit_kw: float = Field(default=5, gt=0)
    minimum_soc: float = Field(default=10, ge=0, le=95)
    reserve_kwh: float = Field(default=1, ge=0)
    export_windows: list[ExportWindow] = [ExportWindow(start='18:00',end='21:00')]
    charging_windows: list[Window] = [Window()]
    banks: list[Bank] = [Bank(id='north', name='Main array', entity='sensor.solar_inverter_power', azimuth=0)]

    @model_validator(mode='before')
    @classmethod
    def migrate_schedule(cls, values):
        values=dict(values)
        if 'export_windows' not in values and ('export_start' in values or 'export_end' in values):
            values['export_windows']=[{'start':values.get('export_start','18:00'),'end':values.get('export_end','21:00')}]
            maximum=values.get('battery_charge_limit_kw',5)
            if 'charging_windows' in values: values['charging_windows']=[{**w,'charge_kw':None if w.get('charge_kw') is not None and w['charge_kw']>=maximum else w.get('charge_kw')} for w in values.get('charging_windows',[])]
        return values

    def save(self):
        (DATA / 'config.json').write_text(self.model_dump_json(indent=2))
        (DATA / 'config.json').chmod(0o600)

    def public(self):
        d=self.model_dump(); d['ha_token']=''; d['token_configured']=bool(self.ha_token)
        return d

def load_settings():
    p=DATA / 'config.json'
    values=json.loads(p.read_text()) if p.exists() else {}
    return Settings(**values)

def read_json(name, default=None):
    p=DATA / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else default

def write_json(name, value):
    p=DATA / f'{name}.json'; tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, default=str, allow_nan=False)); tmp.replace(p)

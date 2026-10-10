import json, os
from pathlib import Path
from datetime import date, timedelta
from pydantic import BaseModel, Field, field_validator

DATA = Path(os.getenv('ENERGY_DATA_DIR', 'data'))
DATA.mkdir(parents=True, exist_ok=True)

class Bank(BaseModel):
    id: str
    name: str
    entity: str
    azimuth: float = Field(ge=0, le=360)
    tilt: float = Field(default=30, ge=0, le=90)

class Window(BaseModel):
    start: str = '11:00'
    end: str = '14:00'
    limit_kwh: float = Field(default=50, ge=0)
    charge_kw: float = Field(default=10, ge=0)
    free: bool = True

class Appliance(BaseModel):
    name: str = 'Large load'
    entity: str = ''
    power_entity: str = ''
    active_states: list[str] = ['on','heat','cool','heat_cool','heating','cooling','charging']
    power_kw: float = Field(default=1,ge=0,le=30)
    start: str = '00:00'
    end: str = '00:00'
    weekdays: list[int] = [0,1,2,3,4,5,6]
    live_hours: float = Field(default=2,ge=0,le=24)
    mode: str = 'floor'

class Settings(BaseModel):
    ha_url: str = 'http://homeassistant.local:8123'
    ha_token: str = ''
    latitude: float = -34.9285
    longitude: float = 138.6007
    timezone: str = 'Australia/Adelaide'
    state: str = 'SA'
    school_calendar_url: str = ''
    solar_blend: bool = False
    solar_nowcast: bool = True
    compare_weather_models: bool = True
    weather_model: str = 'ecmwf_ifs'
    archived_weather_backtest: bool = True
    appliances: list[Appliance] = []
    horizon_hours: int = Field(default=48, ge=1, le=72)
    confidence: float = Field(default=0.90, ge=0.5, le=0.99)
    load_interval_minutes: int = Field(default=60, ge=5)
    solar_interval_minutes: int = Field(default=60, ge=5)
    battery_interval_minutes: int = Field(default=5, ge=1)
    live_interval_seconds: int = Field(default=60, ge=15)
    fine_tune_hours: int = Field(default=24, ge=1)
    auto_fine_tune: bool = True
    age_weighting: bool = True
    age_half_life_days: float = Field(default=365, ge=30, le=3650)
    history_replay_percent: float = Field(default=20, ge=0, le=80)
    load_history_start: str = Field(default_factory=lambda: (date.today()-timedelta(days=365)).isoformat())
    promotion_improvement_percent: float = Field(default=2, ge=0, le=50)
    consumption_entity: str = 'sensor.home_power'
    historical_consumption_entities: list[str] = ['sensor.home_power']
    battery_power_entity: str = 'sensor.battery_power'
    battery_soc_entity: str = 'sensor.battery_soc'
    solar_ct_entity: str = 'sensor.solar_power'
    battery_capacity_kwh: float = Field(default=10, gt=0)
    battery_charge_limit_kw: float = Field(default=5, gt=0)
    battery_discharge_limit_kw: float = Field(default=5, gt=0)
    minimum_soc: float = Field(default=10, ge=0, le=95)
    reserve_kwh: float = Field(default=1, ge=0)
    export_start: str = '18:00'
    export_end: str = '21:00'
    charging_windows: list[Window] = [Window()]
    banks: list[Bank] = [Bank(id='north', name='Main array', entity='sensor.solar_inverter_power', azimuth=0)]

    @field_validator('ha_token', 'ha_url')
    @classmethod
    def strip_connection_whitespace(cls, value):
        return value.strip()

    def save(self):
        (DATA / 'config.json').write_text(self.model_dump_json(indent=2))
        (DATA / 'config.json').chmod(0o600)

    def public(self):
        d=self.model_dump(); d['ha_token']=''; d['token_configured']=bool(self.ha_token)
        return d

def load_settings():
    p=DATA / 'config.json'
    return Settings(**json.loads(p.read_text())) if p.exists() else Settings()

def read_json(name, default=None):
    p=DATA / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else default

def write_json(name, value):
    p=DATA / f'{name}.json'; tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, default=str, allow_nan=False)); tmp.replace(p)

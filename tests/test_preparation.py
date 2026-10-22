"""Synthetic regressions for duration, provenance and interval preparation."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from app.config import Settings
from app.ha_inputs import history_values,history_intervals,normalized_value
from app.quality import clean_sensor,quality_frame
from app.datasets import derive_solar,splice
from app.load_splits import training_mask
from app.sources import HomeAssistant,weather_intervals

class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.s=Settings(timezone='UTC',latitude=0,longitude=0)
        self.start=pd.Timestamp('2026-01-01',tz='UTC')
        self.index=pd.date_range(self.start,periods=72,freq='h')
    def test_subminute_duration_and_unknown(self):
        t=self.start
        events=[(t,0),(t+pd.Timedelta(seconds=10),6),(t+pd.Timedelta(seconds=50),0)]
        self.assertAlmostEqual(history_values(events,t,t+pd.Timedelta(hours=1),'1h').iloc[0],.0666666667)
        gap=history_intervals([(t,1),(t+pd.Timedelta(seconds=10),None),(t+pd.Timedelta(seconds=50),1)],t,t+pd.Timedelta(hours=1),'1h')
        self.assertTrue(pd.isna(gap['mean'].iloc[0]));self.assertEqual(gap.unknown_seconds.iloc[0],40)
        self.assertTrue(pd.isna(history_values([(t+pd.Timedelta(seconds=10),1)],t,t+pd.Timedelta(hours=1),'1h').iloc[0]))
        five=history_intervals(events,t,t+pd.Timedelta(hours=1),'5min')
        self.assertAlmostEqual(float((five['mean']*five.interval_seconds/3600).sum()),.0666666667)
    def test_boundaries_duplicates_and_energy(self):
        t=self.start;end=t+pd.Timedelta(hours=1)
        events=[(t,2),(t,3),(end,9)]
        result=history_values(events,t,end,'1h');self.assertEqual(result.iloc[0],3)
        partial=history_intervals(events,t+pd.Timedelta(seconds=20),end-pd.Timedelta(seconds=20),'1h')
        self.assertEqual(partial.observed_seconds.iloc[0],3560)
        energy=history_intervals([(t,10),(end,12)],t,end,'1h',True)
        self.assertEqual(energy['mean'].iloc[0],2)
        reset=history_intervals([(t,10),(t+pd.Timedelta(minutes=30),1),(end,12)],t,end,'1h',True)
        self.assertTrue(pd.isna(reset['mean'].iloc[0]))
    def test_frozen_solar_and_review(self):
        raw=pd.Series(2.,index=self.index);bounds=pd.DataFrame({'min':raw,'max':raw})
        dark=np.array([False]*12+[True]*12+[False]*48)
        with patch('app.quality.dark_hours',return_value=dark):
            target,flags,details=clean_sensor(raw,'solar',40,self.s,bounds)
            self.assertTrue(target.isna().all());self.assertTrue(flags.eq('frozen_solar_run').all())
            self.assertEqual(details['flat_runs'][0]['observable_at'],(self.index[12]+pd.Timedelta(hours=1)).isoformat())
            target,flags,details=clean_sensor(raw,'solar',40,self.s)
            self.assertTrue(target.isna().all());self.assertEqual(details['flat_runs'][0]['reason'],'frozen_solar_run')
        with patch('app.quality.dark_hours',return_value=np.ones(72,dtype=bool)):
            target,_,_=clean_sensor(raw*0,'solar',40,self.s,bounds*0);self.assertTrue(target.eq(0).all())
        with patch('app.quality.dark_hours',return_value=np.zeros(72,dtype=bool)):
            target,_,_=clean_sensor(raw,'solar',40,self.s,bounds);self.assertTrue(target.eq(2).all())
    def test_selective_derivation(self):
        raw=pd.Series([1,np.nan,3,4,np.nan,6,7,8],index=self.index[:8]);source=pd.Series('a',index=raw.index)
        with patch('app.quality.dark_hours',return_value=np.zeros(8,dtype=bool)):
            q,_=quality_frame(raw,self.s,'solar',40,source,background=False)
        q.loc[q.index[4],'original_reason']='frozen_solar_run'
        with patch('app.datasets.dark_hours',return_value=np.zeros(8,dtype=bool)):
            result=derive_solar(q,self.s,self.start,self.start+pd.Timedelta(hours=8),1,40)
        self.assertEqual(result.target.iloc[1],2);self.assertEqual(result.sample_weight.iloc[1],.25)
        self.assertEqual(result.original_reason.iloc[1],'missing');self.assertFalse(result.validation_eligible.iloc[1]);self.assertTrue(pd.isna(result.target.iloc[4]))
        q.loc[q.index[2],'source_entity']='b'
        with patch('app.datasets.dark_hours',return_value=np.zeros(8,dtype=bool)):
            self.assertTrue(pd.isna(derive_solar(q,self.s,self.start,self.index[7],1,40).target.iloc[1]))
        q.loc[q.index[1],'raw_target']=5;q.loc[q.index[1],'original_reason']='positive_solar_in_darkness';q.loc[q.index[1],'target']=np.nan
        with patch('app.datasets.dark_hours',return_value=np.ones(8,dtype=bool)):
            self.assertTrue(pd.isna(derive_solar(q,self.s,self.start,self.index[7],1,40).target.iloc[1]))
    def test_effective_limit_and_valid_fallback(self):
        raw=pd.Series(3.,index=self.index[:2]);source=pd.Series('same',index=raw.index)
        q,_=quality_frame(raw,self.s,'load',2,source,background=False,precleaned=(raw,pd.Series('',index=raw.index)))
        self.assertTrue(q.target.isna().all());self.assertTrue(q.original_reason.eq('physical_limit').all())
        sensors=pd.DataFrame({'older':[1,1],'newer':[3,3]},index=raw.index)
        from app.quality import physical
        for col in sensors:sensors[col],_=physical(sensors[col],'load',2)
        self.assertTrue(splice(sensors,list(sensors)).eq(1).all())
        self.assertTrue(splice(sensors,list(sensors),'sum').isna().all())
    def test_units_and_weather(self):
        self.assertEqual(normalized_value(2000,'W','load'),2);self.assertEqual(normalized_value(2000,'Wh','solar'),2)
        with self.assertRaises(ValueError):normalized_value(50,'%','load')
        with self.assertRaises(ValueError):normalized_value(50,'kW','soc')
        weather=pd.DataFrame({'temperature_2m':[10,20,30],'shortwave_radiation':[0,100,200]},index=self.index[:3])
        aligned=weather_intervals(weather);self.assertEqual(aligned.temperature_2m.iloc[0],15);self.assertEqual(aligned.shortwave_radiation.iloc[0],100)
        s=self.s;entity=s.consumption_entity
        class FakeHA(HomeAssistant):
            def states(self):return [{'entity_id':entity,'state':'2','attributes':{'unit_of_measurement':'kW'}}]
            def ws(self,message):
                if message['type']=='recorder/list_statistic_ids':return [{'statistic_id':entity,'unit_of_measurement':'kW','statistics_unit_of_measurement':'W'}]
                assert message['units']['power']=='kW'
                return {entity:[{'start':str(t),'mean':2,'min':1,'max':3} for t in self_test.index[:2]]}
        self_test=self
        frame=FakeHA(s).frame([entity],self.start,self.start+pd.Timedelta(hours=2));self.assertTrue(frame[entity].eq(2).all())
    def test_complete_split_and_quiet_annotation(self):
        end=self.start+pd.Timedelta(hours=1,minutes=30)
        mask=training_mask(self.index[:3],self.start,end);self.assertEqual(mask.tolist(),[True,False,False])
        raw=pd.Series(np.linspace(1.,1.1,len(self.index)),index=self.index);away=pd.Series(True,index=self.index)
        with patch('app.quality.classify_background',return_value=(away,[],[])):
            q,_=quality_frame(raw,self.s)
        self.assertTrue(q.quiet_regime.all());self.assertTrue(q.validation_eligible.all());self.assertTrue(q.original_reason.eq('').all())


class HistoryUnitTests(unittest.TestCase):
    def test_raw_units_change_independently_of_statistics(self):
        s=Settings();t=pd.Timestamp('2026-01-01',tz='UTC');entity=s.consumption_entity
        class FakeHA(HomeAssistant):
            def states(self):return [{'entity_id':entity,'state':'2','attributes':{'unit_of_measurement':'kW'}}]
            def ws(self,message):
                if message['type']=='recorder/list_statistic_ids':return [{'statistic_id':entity,'unit_of_measurement':'kW','statistics_unit_of_measurement':'W'}]
                return {entity:[]}
            def history(self,entity,start,end):return [[{'state':value,'last_updated':(t+pd.Timedelta(minutes=minutes)).isoformat(),'attributes':{'unit_of_measurement':unit}} for minutes,value,unit in [(0,'1000','W'),(30,'3','kW')]]]
        frame=FakeHA(s).frame([entity],t,t+pd.Timedelta(hours=1));self.assertEqual(frame[entity].iloc[0],2)
        self.assertEqual([u['unit'] for u in frame.attrs['sensors'][entity]['history_units']],['W','kW'])

class UnavailableUnitTests(unittest.TestCase):
    def test_unavailable_without_attributes_preserves_gap(self):
        t=pd.Timestamp('2026-01-01',tz='UTC');s=Settings();entity=s.consumption_entity
        class FakeHA(HomeAssistant):
            def states(self):return [{'entity_id':entity,'state':'2','attributes':{'unit_of_measurement':'kW'}}]
            def ws(self,message):return [] if message['type']=='recorder/list_statistic_ids' else {entity:[]}
            def history(self,entity,start,end):return [[{'state':'1','last_updated':t.isoformat(),'attributes':{'unit_of_measurement':'kW'}},{'state':'unavailable','last_updated':(t+pd.Timedelta(seconds=10)).isoformat()},{'state':'1','last_updated':(t+pd.Timedelta(seconds=50)).isoformat(),'attributes':{'unit_of_measurement':'kW'}}]]
        frame=FakeHA(s).frame([entity],t,t+pd.Timedelta(hours=1));self.assertTrue(pd.isna(frame[entity].iloc[0]));self.assertEqual(frame.attrs['sensors'][entity]['durations'].unknown_seconds.iloc[0],40)

class HistoryRequestTests(unittest.TestCase):
    def test_attribute_flags_are_omitted(self):
        class FakeHA(HomeAssistant):
            def get(self,path,params):
                assert 'minimal_response' not in params
                assert 'no_attributes' not in params
                assert 'significant_changes_only' not in params
                return []
        FakeHA(Settings()).history('sensor.example',pd.Timestamp('2026-01-01',tz='UTC'),pd.Timestamp('2026-01-02',tz='UTC'))


class DatasetOutageTests(unittest.TestCase):
    def test_known_outage_is_not_interpolated_by_preparation(self):
        import tempfile
        from pathlib import Path
        from app import datasets
        s=Settings(timezone='UTC');index=pd.date_range('2026-01-01',periods=96,freq='h',tz='UTC');entity=s.banks[0].entity
        values=pd.Series(1.,index=index);values.iloc[12]=np.nan;flags=pd.Series('',index=index);flags.iloc[12]='raw_history_unavailable'
        class FakeHA:
            def __init__(self,settings):pass
            def frame(self,*args,**kwargs):
                f=pd.DataFrame({entity:values});f.attrs={'raw':f.copy(),'sensors':{entity:{'unit':'kW','quality_flags':flags,'original_quality_flags':flags,'statistics':pd.DataFrame({'min':0.,'max':2.},index=index),'details':{}}}};return f
        weather=pd.DataFrame({'temperature_2m':20.,'shortwave_radiation':500.,'tilted_irradiance':500.},index=index)
        req={'kind':'solar','bank_id':s.banks[0].id,'entities':[entity],'train_start':'2026-01-01','train_end':'2026-01-02','validation_start':'2026-01-03','validation_end':'2026-01-04','interpolate_hours':1}
        with tempfile.TemporaryDirectory() as temp,patch.object(datasets,'DATA',Path(temp)),patch.object(datasets,'write_json'),patch.object(datasets,'HomeAssistant',FakeHA),patch.object(datasets,'solar_weather',return_value=weather),patch('app.study.cases',return_value=([],[])),patch('app.datasets.dark_hours',side_effect=lambda settings,ix:np.zeros(len(ix),dtype=bool)):
            datasets.build_dataset(s,req,lambda message:None)
            f=pd.read_pickle(Path(temp)/('solar-'+s.banks[0].id+'.pkl'))['frame']
            self.assertTrue(pd.isna(f.target.iloc[12]));self.assertEqual(f.original_reason.iloc[12],'raw_history_unavailable');self.assertFalse(f.train_eligible.iloc[12])


class CalibrationPolicyTests(unittest.TestCase):
    def test_all_calibrations_require_current_preparation(self):
        from app.models import train
        for kind in ('load','solar','battery'):
            with self.subTest(kind=kind),patch('app.models.read_json',return_value={'sufficient':True,'kind':kind,'quality':{'policy':'old-policy'}}):
                with self.assertRaisesRegex(ValueError,'Prepare this dataset again'):train(Settings(),kind,lambda message:None)


class CorrectionTests(unittest.TestCase):
    def test_restore_undoes_manual_exclusion_without_erasing_quiet(self):
        t=pd.Timestamp('2026-01-01',tz='UTC');index=pd.date_range(t,periods=2,freq='h');s=Settings(timezone='UTC');entity=s.consumption_entity
        edits=[{'start':t.isoformat(),'end':(t+pd.Timedelta(hours=2)).isoformat(),'action':action} for action in ('exclude','restore')]
        class FakeHA(HomeAssistant):
            def states(self):return [{'entity_id':entity,'state':'1','attributes':{'unit_of_measurement':'kW'}}]
            def ws(self,message):return [] if message['type']=='recorder/list_statistic_ids' else {entity:[{'start':str(at),'mean':1,'min':0,'max':2} for at in index]}
        frame=FakeHA(s).frame([entity],t,t+pd.Timedelta(hours=2),corrections=edits)
        self.assertTrue(frame[entity].eq(1).all());self.assertTrue(frame.attrs['sensors'][entity]['original_quality_flags'].eq('').all())
        away=pd.Series(True,index=index)
        with patch('app.quality.classify_background',return_value=(away,[],[])):
            q,_=quality_frame(pd.Series(1.,index=index),s,corrections=edits)
        self.assertTrue(q.target.eq(1).all());self.assertTrue(q.quiet_regime.all());self.assertTrue(q.original_reason.eq('').all())
        bad,_=quality_frame(pd.Series(3.,index=index),s,limit=2,corrections=edits,background=False)
        self.assertTrue(bad.target.isna().all());self.assertTrue(bad.original_reason.eq('physical_limit').all())


class OverrideUnitTests(unittest.TestCase):
    def test_unknown_statistics_unit_uses_explicit_returned_value_basis(self):
        s=Settings();t=pd.Timestamp('2026-01-01',tz='UTC');entity=s.consumption_entity
        class FakeHA(HomeAssistant):
            def states(self):return []
            def ws(self,message):return [] if message['type']=='recorder/list_statistic_ids' else {entity:[{'start':t.isoformat(),'mean':2000,'min':1000,'max':3000}]}
        frame=FakeHA(s).frame([entity],t,t+pd.Timedelta(hours=1),units={entity:'W'})
        self.assertEqual(frame[entity].iloc[0],2)
        info=frame.attrs['sensors'][entity];self.assertEqual(info['normalized_unit'],'kW');self.assertIn('override',info['unit_basis'])


class WeatherRetryTests(unittest.TestCase):
    def test_transient_solar_weather_failure_retries(self):
        from app import sources
        from pathlib import Path
        import tempfile
        calls=[]
        class Response:
            def __init__(self,status):self.status_code=status
            def raise_for_status(self):
                if self.status_code!=200:raise RuntimeError('Temporary weather failure')
            def json(self):return {'hourly':{'time':['2026-01-01T00:00','2026-01-01T01:00','2026-01-01T02:00'],'temperature_2m':[10,20,30],'shortwave_radiation':[0,100,200],'global_tilted_irradiance':[0,120,240]}}
        class Client:
            def __init__(self,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def get(self,*args,**kwargs):calls.append(1);return Response(503 if len(calls)==1 else 200)
        s=Settings();t=pd.Timestamp('2026-01-01',tz='UTC')
        with tempfile.TemporaryDirectory() as temp,patch.object(sources,'DATA',Path(temp)),patch.object(sources.httpx,'Client',Client),patch.object(sources.time,'sleep'):
            frame=sources.solar_weather(s,s.banks[0],t,t+pd.Timedelta(hours=1))
        self.assertEqual(len(calls),2);self.assertEqual(frame.temperature_2m.iloc[0],15);self.assertEqual(frame.tilted_irradiance.iloc[0],120)

if __name__=='__main__':unittest.main()

"""Focused checks for shared issue-time and point/quantile behaviour."""
import unittest
import numpy as np,pandas as pd
from app.forecast_policy import archived_run,ordered,recent_correction,fit_calibration
from app.sources import weather_intervals
from app.study import window
from app.solar_training import solar_predict
from app.datasets import LEGACY_SOLAR_FEATURES
from app.load_training import LoadPredictor,LOAD_FEATURES
from app.quality import LOAD_PIPELINE,QUALITY_POLICY

class Constant:
    def __init__(self,value,columns):self.value=value;self.feature_name_=columns
    def predict(self,x):
        assert list(x)==self.feature_name_
        return np.full(len(x),self.value)

class ForecastPolicy(unittest.TestCase):
    def test_endpoint_alignment_does_not_fill_a_missing_boundary(self):
        idx=pd.date_range('2026-01-01',periods=3,freq='h',tz='UTC')
        f=weather_intervals(pd.DataFrame({'temperature_2m':[10,20,30],'shortwave_radiation':[0,100,200]},index=idx))
        self.assertEqual(f.temperature_2m.iloc[0],15)
        self.assertEqual(f.shortwave_radiation.iloc[0],100)
        self.assertTrue(f.iloc[-1].isna().all())
    def test_partial_sunlight_survives_negative_midpoint(self):
        x=pd.DataFrame({k:[0.] for k in LEGACY_SOLAR_FEATURES});x['solar_elevation']=-1.;x['tilted_irradiance']=2.;x['full_dark']=0.;x['lead_hours']=0.;x['weather_run_age']=np.nan
        artifact={'models':{k:Constant(v,LEGACY_SOLAR_FEATURES) for k,v in [('low',.1),('middle',.2),('high',.3)]},'irradiance_cutoff':0}
        self.assertGreater(solar_predict(artifact,x)['middle'][0],0)
        x['full_dark']=1
        self.assertEqual(solar_predict(artifact,x)['high'][0],0)
    def test_point_mean_is_independent_of_ordered_quantiles(self):
        x=pd.DataFrame({k:[1.] for k in LOAD_FEATURES});x['clock_profile']=4.
        artifact={'pipeline':LOAD_PIPELINE,'quality_policy':QUALITY_POLICY,'models':{k:Constant(v,LOAD_FEATURES) for k,v in [('low',3),('middle',1),('high',2)]},'mean_model':Constant(8,LOAD_FEATURES),'point_method':{'head':'mean','baseline_weight':.5}}
        q=LoadPredictor(artifact).predict(x)
        self.assertEqual(q['point'][0],6)
        self.assertEqual([q[k][0] for k in ('low','middle','high')],[1,2,3])
    def test_sparse_or_future_residuals_cannot_correct_an_issue(self):
        origin=pd.Timestamp('2026-01-04T00:00Z');policy={'days':3,'bound':.5,'taper':24}
        rows=[{'origin':origin.isoformat(),'time':(origin-pd.Timedelta(hours=5)).isoformat(),'actual':100,'point':0,'middle':0}]*100
        np.testing.assert_array_equal(recent_correction(rows,origin,[0,24],policy),[0,0])
        self.assertEqual(fit_calibration(rows[:5]),{})
    def test_publication_allowance_and_partial_window_weights(self):
        origin=pd.Timestamp('2026-01-01T13:30Z');self.assertEqual(archived_run(origin),pd.Timestamp('2026-01-01T00:00Z'))
        index=pd.date_range(origin.floor('h'),periods=25,freq='h')
        c={'origin':origin,'hours':index,'end':origin+pd.Timedelta(hours=24),'valid':np.ones(25,bool)}
        w=window(c,24);self.assertEqual(w.sum(),24);self.assertEqual(w[0],.5);self.assertEqual(w[-1],.5)
        c['valid'][-1]=False;self.assertIsNone(window(c,24))
    def test_legacy_lag_adapter_preserves_semantics_and_excludes_unsettled_hour(self):
        from app.load_training import feature_rows
        from app.datasets import FEATURES
        index=pd.date_range('2026-01-01',periods=250,freq='h',tz='UTC');frame=pd.DataFrame({k:np.ones(len(index)) for k in FEATURES},index=index)
        frame['target']=np.arange(len(index),dtype=float);origin=index[-1]
        current=feature_rows(frame,origin,pd.DatetimeIndex([origin]),'UTC')
        legacy=feature_rows(frame,origin,pd.DatetimeIndex([origin]),'UTC',legacy=True)
        self.assertEqual(current.lag1.iloc[0],247)
        self.assertTrue(np.isnan(legacy.lag1.iloc[0]))
        self.assertEqual(legacy.lag2.iloc[0],247)
        self.assertEqual(legacy.lag24.iloc[0],225)
    def test_replay_and_live_use_identical_quantile_postprocessing(self):
        from app.study import predict
        x=pd.DataFrame({k:[1.] for k in LOAD_FEATURES});x['clock_profile']=4.;x['lead_hours']=0
        models={k:Constant(v,LOAD_FEATURES) for k,v in [('low',-.2),('middle',.5),('high',1.)]}
        calibration={'pooled':{'low':.1,'middle':-.1,'high':.2}}
        artifact={'pipeline':LOAD_PIPELINE,'quality_policy':QUALITY_POLICY,'models':models,'calibration':calibration,'point_method':{'head':'middle','baseline_weight':0}}
        c={'x':x,'origin':pd.Timestamp('2026-01-01T00:00Z'),'daylight':np.array([True])}
        replay=predict(models,c,'load',artifact['point_method'],calibration)
        live=LoadPredictor(artifact).predict(x)
        for key in ('low','middle','high','point'):np.testing.assert_array_equal(replay[key],live[key])
    def test_calibration_orders_then_clips_and_zeroes_darkness(self):
        q=ordered({'low':[-2,1],'middle':[1,2],'high':[0,3]},[0,72],{'pooled':{'low':5,'middle':-2,'high':0}},[False,True])
        self.assertTrue(q['low'][0]<=q['middle'][0]<=q['high'][0]);self.assertEqual(q['low'][0],0);self.assertEqual(q['high'][1],0)

if __name__=='__main__':unittest.main()

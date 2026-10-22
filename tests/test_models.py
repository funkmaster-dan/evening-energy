"""Candidate activation needs only the saved model and relevant settings."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from app import config,models
from app.config import Settings,write_json,read_json,solar_banks
from app.datasets import SOLAR_FEATURES
from app.load_training import LOAD_FEATURES
from app.quality import LOAD_PIPELINE,QUALITY_POLICY,active_policy


class Constant:
    def __init__(self,value,columns):self.value=value;self.feature_name_=columns
    def predict(self,x):
        assert list(x)==self.feature_name_
        return np.full(len(x),self.value)


class Candidates(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for module in (config,models):
            patcher=patch.object(module,'DATA',self.root);patcher.start();self.addCleanup(patcher.stop)
        self.settings=Settings()

    def candidate(self,kind='load',high=3):
        bank=solar_banks(self.settings)[0].id if kind=='solar' else None
        name='solar-'+bank if bank else kind
        columns=SOLAR_FEATURES+(['bank_irradiance_0'] if bank=='__combined' else []) if bank else LOAD_FEATURES
        artifact={'pipeline':LOAD_PIPELINE,'quality_policy':QUALITY_POLICY,
            'models':{k:Constant(v,columns) for k,v in [('low',1),('middle',2),('high',high)]}}
        metadata={'dataset_id':name,'kind':kind,'bank_id':bank,'config_signature':models.signature(self.settings,kind,bank)}
        write_json('candidate-'+name,metadata);joblib.dump(artifact,self.root/(name+'-candidate.joblib'))
        return name

    def test_load_and_solar_activate_without_prepared_data_or_network(self):
        for kind in ('load','solar'):
            with self.subTest(kind=kind):
                name=self.candidate(kind)
                with patch('app.models.pd.read_pickle',side_effect=AssertionError('History must not be read')),patch('app.sources.httpx.Client',side_effect=AssertionError('No network')):
                    models.promote_candidate(self.settings,name)
                self.assertTrue(models.model_ready(self.settings,name,kind,self.settings.banks[0].id if kind=='solar' else None))
                self.assertFalse((self.root/('candidate-'+name+'.json')).exists())

    def test_preparing_new_data_keeps_active_inputs_and_candidate_usable(self):
        name=self.candidate()
        metadata=read_json('candidate-'+name)
        metadata['training_policy']={'max_power_kw':12,'combine':'splice','entities':['sensor.synthetic_history']}
        write_json('candidate-'+name,metadata)
        write_json('dataset-load',{'request':{'max_power_kw':99,'combine':'sum','corrections':[{'action':'exclude'}]}})
        models.promote_candidate(self.settings,name)
        self.assertEqual(active_policy(self.settings,'load')['max_power_kw'],12)
        self.assertEqual(active_policy(self.settings,'load')['combine'],'splice')

    def test_actual_input_change_blocks_activation(self):
        name=self.candidate('solar');changed=self.settings.model_copy(deep=True);changed.banks[0].azimuth=90
        with self.assertRaisesRegex(ValueError,'Configuration has changed'):models.promote_candidate(changed,name)
        self.assertTrue((self.root/(name+'-candidate.joblib')).exists())

    def test_combined_solar_activates_locally_and_checks_panel_counts(self):
        self.settings=Settings(solar_mode='combined')
        name=self.candidate('solar');models.promote_candidate(self.settings,name)
        self.assertTrue(models.model_ready(self.settings,name,'solar','__combined'))
        changed=self.settings.model_copy(deep=True);changed.banks[0].panel_count+=1
        self.assertFalse(models.model_ready(changed,name,'solar','__combined'))

    def test_bad_candidate_keeps_active_artifact(self):
        name=self.candidate(high=np.nan)
        active=self.root/(name+'.joblib');active.write_bytes(b'existing active artifact')
        with self.assertRaisesRegex(ValueError,'finite'):models.promote_candidate(self.settings,name)
        self.assertEqual(active.read_bytes(),b'existing active artifact')
        self.assertTrue((self.root/(name+'-candidate.joblib')).exists())


if __name__=='__main__':unittest.main()

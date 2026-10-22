"""Stopping scores integrate energy windows without using final validation."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from app.config import Settings
from app.fitting import EnergyValidation,development_cases


class Fitting(unittest.TestCase):
    def setUp(self):self.settings=Settings(timezone='UTC')

    def case(self,minute=0):
        origin=pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(minutes=minute)
        hours=pd.date_range(origin.floor('h'),(origin+pd.Timedelta(hours=24)).ceil('h'),freq='h',inclusive='left')
        return {'origin':origin,'hours':hours,'end':origin+pd.Timedelta(hours=24),
                'daily_end':origin.normalize()+pd.Timedelta(days=1),'charge_start':None,
                'valid':np.ones(len(hours),bool),'daylight':np.ones(len(hours),bool),
                'actual':np.ones(len(hours)),'x':pd.DataFrame({'lead_hours':np.arange(len(hours))})}

    def test_energy_score_can_prefer_correct_totals_to_lower_hourly_mae(self):
        validation=EnergyValidation([self.case()],'load',self.settings)
        steady=np.full(24,1.1);balanced=np.tile([.85,1.15],12)
        a=validation.scores(steady);b=validation.scores(balanced)
        self.assertLess(a[2],b[2]);self.assertLess(b[1],a[1]);self.assertLess(b[0],a[0])

    def test_fractional_endpoints_are_integrated(self):
        validation=EnergyValidation([self.case(30)],'load',self.settings)
        prediction=validation.y.copy();prediction[0]+=2;prediction[-1]+=4
        self.assertAlmostEqual(validation.scores(prediction)[1],3)

    def test_missing_measurements_do_not_form_an_energy_window(self):
        case=self.case();case['valid'][12]=False
        with self.assertRaisesRegex(ValueError,'complete energy windows'):EnergyValidation([case],'load',self.settings)

    def test_solar_nights_do_not_dominate_stopping(self):
        case=self.case();case['daylight'][:6]=False;case['daylight'][18:]=False;case['actual'][~case['daylight']]=0
        validation=EnergyValidation([case],'solar',self.settings)
        prediction=np.where(case['daylight'],1,99)
        self.assertEqual(validation.scores(validation.pack([{'point':prediction}]))[0],0)

    def test_development_period_excludes_final_validation(self):
        index=pd.date_range('2026-01-01',periods=60*24,freq='h',tz='UTC')
        frame=pd.DataFrame({'target':1.,'context_target':1.,'train_eligible':True,'validation_eligible':True},index=index)
        train_end=index[40*24]
        def replay(frame,left,right,*args,**kwargs):
            self.assertLessEqual(right,train_end)
            self.assertLessEqual(len(kwargs['issue_times']),16)
            return ([{'origin':left+pd.Timedelta(days=i)} for i in range(3)],[])
        with patch('app.study.cases',side_effect=replay):
            before=development_cases(frame,index[0],train_end,self.settings,'load')
            frame.loc[frame.index>=train_end,'target']=1000000
            after=development_cases(frame,index[0],train_end,self.settings,'load')
        self.assertEqual(before['start'],after['start']);self.assertEqual(before['end'],after['end'])
        self.assertLessEqual(before['folds'][0]['end'],before['folds'][-1]['start'])

    def test_missing_older_archive_does_not_mix_weather_bases(self):
        index=pd.date_range('2026-01-01',periods=60*24,freq='h',tz='UTC');train_end=index[40*24]
        frame=pd.DataFrame({'target':1.,'context_target':1.,'train_eligible':True,'validation_eligible':True},index=index)
        def replay(frame,left,right,*args,**kwargs):
            if right<train_end and kwargs.get('basis','issued')=='issued':return [],[]
            return ([{'origin':left+pd.Timedelta(days=i)} for i in range(3)],[])
        with patch('app.study.cases',side_effect=replay):
            development=development_cases(frame,index[0],train_end,self.settings,'load')
        self.assertEqual(len(development['folds']),1)
        self.assertEqual(development['weather_basis'],'issued')


if __name__=='__main__':unittest.main()

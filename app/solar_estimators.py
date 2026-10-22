"""Small solar point alternatives with one shared learned physical scale."""
import os
import numpy as np
from lightgbm import LGBMRegressor


class SolarEnergyModel:
    def __init__(self,family='physical'):
        self.family=family

    @staticmethod
    def irradiance(x):
        # Temperature remains an input to the residual learner; no panel wattage is assumed.
        return np.maximum(0,x['tilted_irradiance'].to_numpy(float))/1000

    def fit(self,x,y,sample_weight=None):
        self.feature_name_=list(x.columns)
        w=np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight,float)
        sunlight=self.irradiance(x);denominator=np.sum(w*sunlight**2)
        self.scale_=max(0,float(np.sum(w*sunlight*y)/denominator)) if denominator>0 else 0.
        self.residual_=None;self.iterations_=0
        if self.family=='residual':
            self.residual_=LGBMRegressor(objective='regression',n_estimators=120,num_leaves=7,max_depth=3,min_child_samples=50,learning_rate=.03,reg_lambda=10.,reg_alpha=1.,verbosity=-1,n_jobs=max(1,int(os.getenv('ENERGY_TRAIN_THREADS','2'))),random_state=42,deterministic=True,force_col_wise=True)
            self.residual_.fit(x,np.asarray(y)-sunlight*self.scale_,sample_weight=w)
            self.iterations_=120
        self.feature_importances_=self.residual_.feature_importances_ if self.residual_ is not None else np.array([int(c=='tilted_irradiance') for c in x.columns])
        return self

    def predict(self,x):
        prediction=self.irradiance(x)*self.scale_
        if self.residual_ is not None:prediction+=self.residual_.predict(x.reindex(columns=self.feature_name_))
        return np.maximum(0,prediction)

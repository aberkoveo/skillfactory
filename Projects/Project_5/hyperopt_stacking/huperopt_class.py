
import tqdm

import xgboost as xgb
from xgboost import XGBRegressor
import hyperopt 
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor, DMatrix
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool, cv 
import numpy as np

DEVICE='GPU'

class HyperOpt:
    def __init__(self, **kwargs):
        super(HyperOpt, self).__init__()
        self.kwargs = kwargs
       
    def hyperopt_xgb_score(self, params):
        
        model = XGBRegressor(l2_leaf_reg=int(params['l2_leaf_reg']),
                             learning_rate=params['learning_rate'],
                             max_depth=int(params['max_depth']),
                             gamma = params['gamma'],
                             reg_alpha = params['reg_alpha'],
                             reg_lambda = params['reg_lambda'],
                             n_estimators=self.kwargs['n_estimators'],
                             objective='reg:squarederror',
                             verbosity=0,
                             random_seed=42,
                             task_type='GPU')
        fit_params={'early_stopping_rounds': self.kwargs['rounds'], 
                    'eval_metric': 'rmse',
                    'verbose': self.kwargs['verbose'],
                    'eval_set': [[self.kwargs['X_val'],  self.kwargs['y_val']]]}
        
        xgb_cv = cross_val_score(model, self.kwargs['X_train'], self.kwargs['y_train'], 
                                 cv = self.kwargs['cv'], 
                                 scoring = 'neg_mean_squared_error',
                                 fit_params = fit_params)
        best_rmse = np.mean([(-x)**0.5 for x in xgb_cv])
        print(f'Best RMSE: {best_rmse}', params)
        return best_rmse
    
    def hyperopt_catb_score(self, params):
        model = CatBoostRegressor(l2_leaf_reg=int(params['l2_leaf_reg']),
                                  learning_rate=params['learning_rate'],
                                  iterations=self.kwargs['iterations'],
                                  ignored_features = self.kwargs['ignored_features'],
                                  eval_metric='MAPE',
                                  random_seed=42,
                                  task_type='GPU',
                                  logging_level='Silent'
                                 )
    
        cv_data = cv(Pool(self.kwargs['X_train'], self.kwargs['y_train'], 
                          cat_features=self.kwargs['categorical_features_indices']),
                     model.get_params())
        best_rmse = np.min(cv_data['test-RMSE-mean'])
        return best_rmse
    
    def hyperopt_lgbm_score(self, params):
        model = LGBMRegressor(learning_rate=params['learning_rate'],
                              max_depth=int(params['max_depth']),
                              n_estimators=int(self.kwargs['n_estimators']),
                              subsample=params['subsample'],
                              reg_alpha = params['reg_alpha'],
                              reg_lambda = params['reg_lambda'],
                              silent = True,
                              metric='rmse',
                              random_state=42)
        
        fit_params={'early_stopping_rounds': self.kwargs['rounds'], 
                    'eval_metric': 'rmse',
                    'verbose': self.kwargs['verbose'],
                    'eval_set': [[self.kwargs['X_val'],  self.kwargs['y_val']]]}
        
        lgb_cv = cross_val_score(model, self.kwargs['X_train'], self.kwargs['y_train'], 
                                 cv = self.kwargs['cv'], 
                                 scoring = 'neg_mean_squared_error',
                                 fit_params = fit_params)
        
        best_rmse = np.mean([(-x)**0.5 for x in lgb_cv])
        print(f'Best RMSE: {best_rmse}', params)
        return best_rmse
from pandas import Series
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from os import listdir

from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve


class df_scrolling_object():
    def __init__(self, data):
        # input_folder = 'input/'
        self.data = data.copy() #pd.read_csv('input/train.csv')
        self.data.drop(['client_id'],  axis=1, inplace=True, errors='ignore')
        data = self.data.copy()
        self.bin_cols = ['good_work', 'foreign_passport', 'car', 'car_type', 'sex']
        self.num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income']
        self.cat_cols = ['education', 'home_address' , 'work_address']

        self.edu_income_rel = dict({'SCH': list([0, 31960.275039]),
                                    'UGR': list([31960.275039,39391.796251]),
                                    'GRD': list([39391.796251, 54663.948941]),
                                    'PGR': list([54663.948941, 77548.206046]),
                                    'ACD': list([77548.206046, 999999999999])})

        self.data['education_isNAN'] = pd.isna(self.data.education).astype('uint8')
        na_index = self.data[self.data.education.isna()].index                                    
        for index in na_index:
            inc = self.data.income.iloc[index]
            self.data.education.iloc[index] = self.fill_edu(inc, self.edu_income_rel)



    def fill_edu(self, income, edu_income_rel):
        for edu, income_edu in edu_income_rel.items():
            if income_edu[0] < income <  income_edu[1]:
                return edu


    def return_XY(self):
        data = self.data.copy()
        na_index = data[data.education.isna()].index
        
        
        for num_col in self.num_cols:
            data[num_col] = data[num_col].apply(lambda x: np.log(x + 1))

        label_encoder = LabelEncoder()
        for bin_col in self.bin_cols:
            data[bin_col] = label_encoder.fit_transform(data[bin_col])
            
        self.X_cat = OneHotEncoder(sparse = False).fit_transform(data[self.cat_cols].values)
        self.X_num = StandardScaler().fit_transform(data[self.num_cols].values)
        self.X = np.hstack([self.X_num, data[self.bin_cols].values, self.X_cat])
        self.Y = data['default'].values
        return self.X, self.Y


    def return_bal_XY(self, eq=1):
        data = self.data.copy()
        n1 = data[data.default==1].shape[0]
        data_0 = data[data.default==0]
        data_1 = data[data.default==1]
        data_balanced = data_0[:n1*eq].append(data_1)
        X_cat = OneHotEncoder(sparse = False).fit_transform(data_balanced[self.cat_cols].values)
        X_num = StandardScaler().fit_transform(data_balanced[self.num_cols].values)
        X = np.hstack([X_num, data_balanced[self.bin_cols].values, X_cat])
        Y = data_balanced['default'].values
        return X, Y

    

    def normalize(self):
        data = self.data.copy()


    

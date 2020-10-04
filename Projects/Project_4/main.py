from pandas import Series
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class df_scrolling_object():
    def __init__(self, data):
        self.data = data.copy().reindex()
        self.bin_cols = ['good_work', 'foreign_passport', 'car', 'car_type', 'sex' ]
        self.num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income', 'score_bki']
        self.cat_cols = ['education', 'home_address' , 'work_address', 'first_time', 'sna' , 'region_rating']
        self.edu_income_rel = dict({'SCH': list([0, 31960.275039]),
                                    'UGR': list([31960.275039,39391.796251]),
                                    'GRD': list([39391.796251, 54663.948941]),
                                    'PGR': list([54663.948941, 77548.206046]),
                                    'ACD': list([77548.206046, 999999999999])})
        # na_index = self.data[self.data.education.isna()].index       
        nan_educations = self.data[self.data.education.isna()].income.apply(lambda x: self.fill_edu(x, self.edu_income_rel))                             
        self.data.loc[data['education'].isna(), 'education'] = nan_educations

        self.data = self.create_features(data.copy())

        
    def create_features(self, data):
        data = self.data.copy()
        data.score_bki = data.score_bki - data.score_bki.min() 
        data.app_date = pd.to_datetime(data.app_date)
        data['month'] = data.app_date.dt.month
        data['day'] = data.app_date.dt.day
        data['year'] = data.app_date.dt.year
        data['app_date_diff_today'] = pd.datetime.today() - data.app_date
        data['app_date_diff_today'] = data['app_date_diff_today'].dt.days
        data.drop(['app_date'], axis=1, inplace=True)
        data['region_firstt'] = data['region_rating'] / (data['first_time'])
        data['age_first'] = data['age'] - data['first_time'] 
        self.num_cols = self.num_cols + ['region_firstt']
        return data


    def fill_edu(self, income, edu_income_rel):
        for edu, income_edu in edu_income_rel.items():
            if income_edu[0] < income <  income_edu[1]:
                return edu


    def return_XY(self):
        self.preproc_data()
        return self.X, self.Y


    def preproc_data(self):
        data = self.data.copy()
        num_cols = self.num_cols
        cat_cols = self.cat_cols

        data[num_cols] = data[num_cols].apply(lambda x: np.log(x + 1))

        label_encoder = LabelEncoder()
        for bin_col in self.bin_cols:
            data[bin_col] = label_encoder.fit_transform(data[bin_col])
        

        # self.X_cat = OneHotEncoder(sparse = False).fit_transform(data[self.cat_cols].values)
        # cat_cols = self.cat_cols
        # self.X_cat = data[self.cat_cols]
        data = pd.get_dummies(data, columns=cat_cols, drop_first=False)
        #self.X_num = StandardScaler().fit_transform(data[self.num_cols].values)
        data[num_cols] = StandardScaler().fit_transform(data[num_cols].values)

        self.X = data.drop(['default'], axis=1) #np.hstack([self.X_num, data[self.bin_cols].values, self.X_cat])
        self.Y = data['default'].values

    def return_X(self):
        self.preproc_data()
        return self.X

    # def return_bal_XY(self, eq=1):
    #     data = self.data.copy()
    #     n1 = data[data.default==1].shape[0]
    #     data_0 = data[data.default==0]
    #     data_1 = data[data.default==1]
    #     data_balanced = data_0[:n1*eq].append(data_1)
    #     X_cat = OneHotEncoder(sparse = False).fit_transform(data_balanced[self.cat_cols].values)
    #     X_num = StandardScaler().fit_transform(data_balanced[self.num_cols].values)
    #     X = np.hstack([X_num, data_balanced[self.bin_cols].values, X_cat])
    #     Y = data_balanced['default'].values
    #     return X, Y

        # data = self.data.copy()

        # for num_col in self.num_cols:
        #     data[num_col] = data[num_col].apply(lambda x: np.log(x + 1))

        # label_encoder = LabelEncoder()
        # for bin_col in self.bin_cols:
        #     data[bin_col] = label_encoder.fit_transform(data[bin_col])
            
        # self.X_cat = OneHotEncoder(sparse = False).fit_transform(data[self.cat_cols].values)
        # self.X_num = StandardScaler().fit_transform(data[self.num_cols].values)

        # self.X = np.hstack([self.X_num, data[self.bin_cols].values, self.X_cat])
        # self.Y = data['default'].values

        # return self.X, self.Y



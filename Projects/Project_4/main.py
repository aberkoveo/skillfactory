from pandas import Series
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class df_scrolling_object():
    def __init__(self, data):
        print('Data Class created')
        self.data = data.copy().reindex()
        self.bin_cols = ['good_work', 'foreign_passport', 'car', 'car_type', 'sex' ]
        self.num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income', 'score_bki']
        self.cat_cols = ['education', 'home_address' , 'work_address', 'first_time', 'sna' , 
                         'region_rating', 'region_age']
        self.new_num_cols = ['sna_age', 'age_foreign_pass', 
                             'month', 'day', 'year', 'app_date_diff_today', 'age_first',
                             'age_eduaction', 'age_good_work',
                             'work_adr_age', 'bki_cnt_word_adr', 'bki_cnt_first_time', ]
        self.edu_income_rel = dict({'SCH': list([0, 31960.275039]),
                                    'UGR': list([31960.275039,39391.796251]),
                                    'GRD': list([39391.796251, 54663.948941]),
                                    'PGR': list([54663.948941, 77548.206046]),
                                    'ACD': list([77548.206046, 999999999999])})

        lambda_fill = lambda x: self.fill_edu(x, self.edu_income_rel)
        nan_educations = self.data[self.data.education.isna()].income.apply(lambda_fill)                             
        self.data.loc[data['education'].isna(), 'education'] = nan_educations


        edu_dict = dict({'SCH': 0,  #school
                         'GRD': 1,  #out-student
                         'UGR': 2,  #student
                         'PGR': 3,  #aspirant
                         'ACD': 4}) #academic 
        self.data['education'] = self.data['education'].map(edu_dict)
        self.data = self.create_features(self.data)


    def create_features(self, data):
        print('Features Added')
        data = self.data.copy()
        # score_bki приведем к положительным значениям для упрощения логорифмирования
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
        data['sna_age'] =  data['age'] * data['sna'] 
        data['age_eduaction'] = data['age'] / (data['education']+1 )
        data['age_foreign_pass'] = data['age'] / data['foreign_passport'].apply(lambda x: 1/1.1 if x==1 else 1.1)
        data['age_good_work'] = data['age'] / data['good_work'].apply(lambda x: 1/1.2 if x==1 else 1.2)

        data['work_adr_age'] = data['age'] / data['work_address']
        data['bki_cnt_word_adr'] = data['bki_request_cnt'] / data['work_address']
        data['score_bki_work_adr'] = data['score_bki'] - data['work_address']
        data['bki_cnt_first_time'] = data['bki_request_cnt'] * data['first_time']
        data['region_age'] = data['age'] * (data['decline_app_cnt'])
        # добавим новый признак к списку категорий, чтобы он попал на логарифмирование и стандартизацию
        self.num_cols = self.num_cols + ['region_firstt']
        return data


    def fill_edu(self, income, edu_income_rel):
        for edu, income_edu in edu_income_rel.items():
            if income_edu[0] < income <  income_edu[1]:
                return edu


    def preproc_data(self):
        print('Preprocessing')
        data = self.data.copy()
        num_cols = self.num_cols
        cat_cols = self.cat_cols
        bin_cols = self.bin_cols
        data[num_cols] = data[num_cols].apply(lambda x: np.log(x + 1))
        label_encoder = LabelEncoder()
        for bin_col in bin_cols:
            data[bin_col] = label_encoder.fit_transform(data[bin_col])
        # self.X = data.drop(['default'], axis=1)
        # self.Y = data['default'].values
        return data


    def log_std_dummies_data(self, data):
        print('StandardScaler-Dummies done')
        num_cols = self.num_cols
        cat_cols = self.cat_cols
        # data[num_cols] = data[num_cols].apply(lambda x: np.log(x + 1))
        data[num_cols] = StandardScaler().fit_transform(data[num_cols].values)
        data = pd.get_dummies(data, columns=cat_cols, drop_first=False)
        return data


    def return_X(self):
        data = self.log_std_dummies_data(self.preproc_data())
        # data = self.log_std_dummies_data(data)
        X = data.drop(['default'], axis=1)
        return X


    def return_XY(self):
        data = self.log_std_dummies_data(self.preproc_data())
        # data = self.log_std_dummies_data(data)
        X = data.drop(['default'], axis=1)
        Y = data['default'].values
        return X, Y





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



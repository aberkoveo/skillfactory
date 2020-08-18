
"""
Модуль содержит функции, использующие в работе ноутбука
Проект_3_О_вкусной_и_здоровой_пище.ipynb

"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind
from datetime import datetime

from collections import Counter
from datetime import datetime
import re
import ast
from scipy import stats
import my_functions
import importlib
importlib.reload(my_functions)

def get_boxplot(data, column, target):
    """
    функция строит график типа seaborn.boxplot на основе введенных параметров.
    Принимает объекты dataframe, имя колонки для оси Х имя колонки для оси Y
    """
    fig, ax = plt.subplots(figsize = (20, 5))
    sns.boxplot(x=column, y=target, 
                data=data,
               ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel(column, fontsize=20)
    plt.ylabel(target, fontsize=20)
    ax.set_title('Boxplot for ' + column, fontsize=23)
    plt.show()


def review_text_tone(review):
    text_tone_coef = 0
    text_tone_coef_good = 1
    text_tone_coef_bad = 1
    good_words = set(['lovely','very good', 'excellent', 'best', 'nice', 'great','beautifull',
                      'awesome', 'awsom', 'yummy', 'friendly', 'not bad', 'well', 'tasty',
                      'good experience', 'perfect', 'wonderful', 'pleasant', 'helpful', 'cosy',
                      'healthy', 'cute', 'fantastic', 'amazing'])

    bad_words = set(['what the heck', 'waste', 'really bad', 'please try another', 
                     'avoid at all cost', 'terrible', 'worst', 'Not good', 'bad service',
                     'dirty', 'shabby', 'confused', 'disappointing', 'rude', 'no go', 
                     'slow service', 'awful', 'disgusting' , 'horrid', 'horrible'])
    
    for word in good_words:
        for rev in review:
            if word in rev.lower():
                text_tone_coef_good *= 2

    for word in bad_words:
        for rev in review:
            if word in rev.lower():
                text_tone_coef_bad *= 2
    text_tone_coef = text_tone_coef_good - text_tone_coef_bad

    return text_tone_coef


def diff_today(date_list):
    dif_today = 0
    if len(date_list) > 0:
        dif_today = (datetime.today().date() - max(date_list)).days
    return dif_today


def prepoc_data(data_input):

    data = data_input.copy()
    data.columns = data.columns.str.lower()
    data.columns = [name.replace(' ', '_') for name in data.columns]

    data['cuisine_style'] = data['cuisine_style'].fillna('[\'not_specified\']')
    data['Number_of_Reviews_isNAN'] = pd.isna(data.number_of_reviews).astype('uint8')
    data['number_of_reviews'] = data.groupby(by='city').number_of_reviews.apply(lambda x: x.fillna(round(x.mean())))
    data['price_range'].fillna(data.price_range.mode()[0], inplace=True)
    data.reviews.fillna("[[], []]",inplace=True)

    data.drop(['restaurant_id', 'url_ta', 'id_ta'], inplace=True,  axis = 1, errors='ignore')
    data['cuisine_style'] = data['cuisine_style'].apply(ast.literal_eval)
    data['reviews'] = data['reviews'].apply(lambda x: re.sub((r'\bnan\b'), '\'empty_voice\'', x))
    data['reviews'] = data['reviews'].apply(ast.literal_eval)

    data['reviews_text'] =  data['reviews'].apply(lambda x: x[0])
    data['reviews_dates'] = data['reviews'].apply(lambda x: [datetime.strptime(date, '%m/%d/%Y').date() for date in x[1]])
    data['price_range_int'] = data.price_range.apply(lambda x: 1 if x == '$' else 
                                                              (2 if x == '$$ - $$$' else 3))
    data['review_text_tone_coef'] = data['reviews_text'].apply(lambda x: my_functions.review_text_tone(x))

    set_dates = set()
    data['dif_days'] = data.reviews_dates.apply(lambda x: (x[0] - x[-1]) if len(x) > 0 else pd.Timedelta('0 days') ).dt.days

    last_date = set()
    data.reviews_dates.apply(lambda x: last_date.update(x))
    data['diff_last_date'] = data.reviews_dates.apply(my_functions.diff_today)


    cuisine_style_set = set()
    data.cuisine_style.apply(lambda x: cuisine_style_set.update(x))
    cuisine_style_list = list(cuisine_style_set)
    for cuisine in cuisine_style_list:
        data[cuisine] = data.cuisine_style.apply(lambda x: 1 if cuisine in x else 0)

    data['city_orig'] = data['city'] 
    data = pd.get_dummies(data, columns=['city_orig'], dummy_na=True, drop_first=False)

    city_dict = data.groupby(by='city').city.size()
    data['ranking_for_city'] =  data['ranking']  - data.city.apply(lambda x: city_dict[x])

    data['ranking_norm_by_city'] = data.groupby('city').ranking.transform(lambda x: (x - x.mean()) / x.std())
    data['number_of_reviews_norm'] = data.number_of_reviews.apply(lambda x: (x - data.number_of_reviews.mean()) / data.number_of_reviews.std())

    object_columns = [s for s in data.columns if data[s].dtypes == 'object']
    data.drop(object_columns, axis = 1, inplace=True)

    data.drop(['restaurant_id','city', 'city_orig', 'cuisine_style', 'price_range', 
               'reviews', 'url_ta', 'id_ta', 'reviews_text',
               #'ranking',
               #'ranking_for_city',
               'number_of_reviews',
               'reviews_dates'
               ], 
               axis = 1, errors='ignore', inplace=True)
    return data
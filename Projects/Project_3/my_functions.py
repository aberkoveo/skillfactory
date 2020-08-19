
"""
Модуль содержит функции, использующие в работе ноутбука
Проект_3_О_вкусной_и_здоровой_пище.ipynb

"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
    """
    функция подсчитывает примерную тональность текстового комментария:
    находит слова из "хорошего" и "плохого" списков, а соответственно увеличивает
    или уменьшает коеффициент тональности.

    Принимает на вход список текстовых комментарев,
    возвращает int-коэффициент тональности. 
    """


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
    """
    функция считает разницу между сегодняшней датой и самой новой датой из входящего списка дат:
    принимает список дат, возвращает int-количество дней разницы . 

    """
    dif_today = 0
    if len(date_list) > 0:
        dif_today = (datetime.today().date() - max(date_list)).days
    return dif_today


def round_rating(rating_long):
    """
    функция округляет оценку до соотв. шкалы - от 1 до 5 кратной 0.5:
    принимает на вход array не округленных чисел (rating), 
    возвращает для каждого элемента ближашую корректную оценку по соотв. шкале. 
    """
    rating_list = np.arange(1,5.5,0.5)
    rating_dif = abs(rating_list - rating_long)
    return rating_list[np.argmin(rating_dif)]


def prepoc_data(data_input):

    """
    функция проихзводит весь спектр обработок датафрейма, описанный в jupyter-ноутбуке 
    Project_3/Проект_3_О_вкусной_и_здоровой_пище.ipynb , включает:
    - обработку NAN
    - генерацию новых признаков
    - нормирование признаков

    На вход получает исходный датафрейм после импорта,
    возвращает обработанный датафрейм, готовый для обучения модели.

    """

    data = data_input.copy()
    data.columns = data.columns.str.lower()
    data.columns = [name.replace(' ', '_') for name in data.columns]
    
    # Обработка NAN
    data['cuisine_style_isNAN'] = pd.isna(data.cuisine_style).astype('uint8')
    data['cuisine_style'] = data['cuisine_style'].fillna('[\'not_specified\']')
    data['Number_of_Reviews_isNAN'] = pd.isna(data.number_of_reviews).astype('uint8')
    data['number_of_reviews'] = data.groupby(by='city').number_of_reviews.apply(lambda x: x.fillna(round(x.mean())))
    data['price_range_isNAN'] = pd.isna(data.price_range).astype('uint8')
    data['price_range'].fillna(data.price_range.mode()[0], inplace=True)
    data.reviews.fillna("[[], []]",inplace=True)    

    # Предобработка
    data.drop(['restaurant_id', 'url_ta', 'id_ta'], inplace=True,  axis = 1, errors='ignore')
    data['cuisine_style'] = data['cuisine_style'].apply(ast.literal_eval)
    data['reviews'] = data['reviews'].apply(lambda x: re.sub((r'\bnan\b'), '\'empty_voice\'', x))
    data['reviews'] = data['reviews'].apply(ast.literal_eval)

    # добавляем новые признаки
    data['reviews_text'] =  data['reviews'].apply(lambda x: x[0])
    data['reviews_dates'] = data['reviews'].apply(lambda x: [datetime.strptime(date, '%m/%d/%Y').date() for date in x[1]])
    data['empty_reviews'] = data.reviews_text.apply(lambda x: 1 if len(x)==0 else 0)
    data['price_range_int'] = data.price_range.apply(lambda x: 1 if x == '$' else 
                                                              (10 if x == '$$ - $$$' else 100))
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

    data['ranking_by_reviews'] = data['number_of_reviews'] * data['ranking']

    city_data = pd.read_csv('input//csv_city.csv', index_col='name')
    country_dict = city_data.to_dict()['country']
    data['country'] = data.city.apply(lambda x: country_dict[x] )
    data = pd.get_dummies(data, columns=['country'], dummy_na=True, drop_first=False)

    city_data = city_data.to_dict()['pop']
    data['city_pop'] = data.city.apply(lambda x: int(city_data[x]))
    data['ranking_by_pop'] = data['ranking'] / data['city_pop']

    url = 'https://raw.githubusercontent.com/icyrockcom/country-capitals/master/data/country-list.csv'
    capitals = pd.read_csv(url,  error_bad_lines=False, sep=',')
    capitals.drop(['type'],axis=1, inplace=True)
    capitals = capitals.set_index('capital')
    data['capital'] = data.city.apply(lambda x: 1 if x in capitals.index else 0)

    data['ranking_by_city'] = data.groupby(by='city').ranking.transform(lambda x: (len(x) / x) )
    data['number_of_reviews_dif_by_city'] = data.groupby('city').number_of_reviews.transform(lambda x: x / (max(x) - min(x))) 

    # Нормируем признаки
    data['ranking_norm_by_city'] = data.groupby('city').ranking.transform(lambda x: (x - x.mean()) / x.std())
    data['number_of_reviews_norm_by_city'] = data.groupby('city').number_of_reviews.transform(lambda x: (x - x.mean()) / x.std()) 
    data['number_of_reviews_norm'] = data.number_of_reviews.apply(lambda x: (x - data.number_of_reviews.mean()) / 
    data.number_of_reviews.std())
    data['ranking_by_reviews_norm'] = data['ranking_by_reviews'].apply(lambda x: (x - data.ranking_by_reviews.mean()) / data.ranking_by_reviews.std())

    ranking_by_pop_mean = data.ranking_by_pop.mean()
    ranking_by_pop_std = data.ranking_by_pop.std()
    data['ranking_by_pop_norm'] =  data.ranking_by_pop.apply(lambda x: (x - ranking_by_pop_mean) / ranking_by_pop_std)

    # Удаляем колонки типа object
    object_columns = [s for s in data.columns if data[s].dtypes == 'object']
    data.drop(object_columns, axis = 1, inplace=True)

    data.drop([#'rating',                       
               #'ranking',
               #'ranking_norm_by_city',
               'number_of_reviews',
               'ranking_by_reviews_norm',
               #'ranking_by_reviews'
               #'ranking_by_pop',
               #'ranking_by_pop_norm'
               ], 
               axis = 1, errors='ignore', inplace=True)


    return data
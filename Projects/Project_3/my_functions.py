
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
                      'healthy', 'cute', 'fantastic'])

    bad_words = set(['what the heck', 'waste', 'really bad', 'please try another', 
                     'avoid at all cost', 'terrible', 'worst', 'Not good', 'bad service',
                     'dirty', 'shabby', 'confused', 'disappointing', 'rude', 'no go', 
                     'slow service', 'awful', 'disgusting'])
    
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
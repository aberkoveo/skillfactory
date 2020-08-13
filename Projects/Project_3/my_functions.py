
"""
Модуль содержит функции, использующие в работе ноутбука
Проект_3_О_вкусной_и_здоровой_пище.ipynb

"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind


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
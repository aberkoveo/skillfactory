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

from datetime import datetime

def get_boxplot(data, column, target):
    """
    функция строит график типа seaborn.boxplot на основе введенных параметров.
    Принимает объекты dataframe, имя колонки для оси Х имя колонки для оси Y
    """
    fig, ax = plt.subplots(figsize = (10, 5))
    sns.boxplot(x=column, y=target, 
                data=data,
               ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel(column, fontsize=20)
    plt.ylabel(target, fontsize=20)
    ax.set_title('Boxplot for ' + column, fontsize=23)
    plt.show()

def hist_cat_num(data_df, cat_col, num_col, title):
    
    fig, ax = plt.subplots(figsize = (12, 7))
    ax.set_title(title, fontsize=23)
    plt.xlabel(num_col, fontsize=18)
    plt.ylabel(f'{cat_col} count', fontsize=18)
    for cat_uniq in data_df[cat_col].unique():
        data_df[(data_df[cat_col] == cat_uniq)][num_col].hist(bins=100, label=f'{cat_col}:{cat_uniq} / {num_col}')
        plt.title=title
        plt.legend()

def outlier_treatment(datacolumn):
 sorted(datacolumn)
 Q1,Q3 = np.percentile(datacolumn , [25,75])
 IQR = Q3 - Q1
 lower_range = Q1 - (1.5 * IQR)
 upper_range = Q3 + (1.5 * IQR)
 return lower_range,upper_range


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
    fig, ax = plt.subplots(figsize = (20, 7))
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


def plot_cv_metrics(cv_metrics):
    avg_f1_train, std_f1_train = cv_metrics['train_score'].mean(), cv_metrics['train_score'].std()
    avg_f1_valid, std_f1_valid = cv_metrics['test_score'].mean(), cv_metrics['test_score'].std()
    print('[train] F1-score = {:.2f} +/- {:.2f}'.format(avg_f1_train, std_f1_train))
    print('[valid] F1-score = {:.2f} +/- {:.2f}'.format(avg_f1_valid, std_f1_valid))
    
    plt.figure(figsize=(15, 5))

    plt.plot(cv_metrics['train_score'], label='train', marker='.')
    plt.plot(cv_metrics['test_score'], label='valid', marker='.')

    plt.ylim([0., 1.]);
    plt.xlabel('CV iteration', fontsize=15)
    plt.ylabel('F1-score', fontsize=15)
    plt.legend(fontsize=15)



def doc_vectorizer(doc, model):
    # Наивный подход к созданию единого эмбеддинга для документа – средний эмбеддинг по словам
    doc_vector = []
    num_words = 0
    for word in doc:
        try:
            if num_words == 0:
                doc_vector = model[word]
            else:
                doc_vector = np.add(doc_vector, model[word])
            num_words += 1
        except:
            pass
     
    return np.array(doc_vector) / num_words



list_trash_ind = [49684      ,
46129      ,
27658      ,
32423      ,
68900      ,
56013      ,
49956      ,
4574       ,
76485      ,
46127      ,
67957      ,
69268      ,
17848      ,
58794      ,
67320      ,
111515     ,
61630      ,
34170      ,
12391      ,
90039      ,
70726      ,
5401       ,
52304      ,
119691     ,
60051      ,
88462      ,
75242      ,
91285      ,
65812      ,
50541      ,
70588      ,
1507       ,
61913      ,
36359      ,
46006      ,
81002      ,
46950      ,
71317      ,
49818      ,
100889     ,
118856     ,
114785     ,
66206      ,
24566      ,
69564      ,
79517      ,
79481      ,
68143      ,
100324     ]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind


def yn_to_binary(series):
    series = series.map(lambda series: 1 if series == 'yes' else 0 if series == 'no' else series)
    return series


def get_boxplot(data, column, target):
    fig, ax = plt.subplots(figsize = (4, 3))
    sns.boxplot(x=column, y=target, 
                data=data,
               ax=ax)
    plt.xticks(rotation=45)
    plt.xlabel(column, fontsize=20)
    plt.ylabel(target, fontsize=20)
    ax.set_title('Boxplot for ' + column, fontsize=23)
    plt.show()


def get_stat_dif(data, column, target, alpha):
    cols = data.loc[:, column].value_counts().index[:]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(data.loc[data.loc[:, column] == comb[0], target], 
                        data.loc[data.loc[:, column] == comb[1], target], nan_policy='omit').pvalue \
            <= alpha/len(combinations_all): # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break
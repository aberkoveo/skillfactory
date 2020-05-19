"""
Модуль содержит функции, использующие в работе ноутбука
[SF-DST] Movies IMBD v3.0.ipynb 

"""

import pandas as pd
from collections import Counter

# data = pd.read_csv('E:\\DEVgit\\skillfactory\\Projects\\Project_1\\input\\imbd-sf\\data.csv')
# data.drop(['imdb_id', 'tagline', 'overview', 'production_companies'], axis=1, inplace=True)
# data['profit'] = data.revenue - data.budget
# data['cast'] = data.cast.str.split('|')


def expand_df(df, bycolumn):
    """
    Функция добавляет в принимаемый датафрейм новую колонку, в которой значениями являются 
    элементы списка из колонки, имя которой передается как второй параметр.
    
    Функция принимает на вход датафрейм и текстовое имя колонки, содержащей значения в виде списков.
    Разбивает все списки в колонке на отдельный датафрейм со строками по каждому элементу списков всех колонок
    и 2х-уровневой индексацией в виде отдельных колонок. Далее происходит объединение исходного датафрейма
    и нового. 

    Функция возвращает вводимый датафрейм обогащенным новой колонкой, имя которой формируется из имени вводной колонки + '_divided'. 
    Возможно добавление новых строк, если при слиянии суммарное количество всех элементов всех списков оказалось больше, 
    чем количество строк в изначальном (вводимом) датафрейме.  

    """
    df = df.copy()
    column_df = pd.DataFrame(df[bycolumn].to_list()).stack().reset_index() # формируем новый датафрейм на основе элементов всех списков
                                                                           # добавляем колонки с индексами, сбрасывая сами индексы 1 и 2 уровня
    column_df.rename(columns={0: bycolumn + '_divided'}, inplace=True)
    column_df = column_df.merge(df.reset_index(), left_on='level_0', right_index=True, how='inner') # проводим слияние датафреймов по индексам
    column_df.drop(['level_0', 'level_1', 'index'], axis=1, inplace=True) # удаляем ненужные столбцы с индексами
    
    result = column_df

    return result


def season_by_intmonth(intmonth):
    """
    Функция принимает параметр со значением int, означающий номер месяца,
    а возвращает наименование сезона или сообщение, что месяц не определен, если параметр некорректен.

    """

    seasons = {
        'winter':[12,1,2],
        'spring':[3,4,5],
        'summer':[6,7,8],
        'autumn':[9,10,11]
    }
    for season in seasons:
        if intmonth in seasons[season]:
            return season
    return 'season not defined'

# print(expand_df(data[data.release_year==2012], 'cast').info())
# print(expand_df(data, 'cast').groupby(['cast_divided']).profit.sum().sort_values(ascending=False).head(1))
#print(expand_df(data, 'cast'))
#print(expand_df(data, 'director').groupby('director_divided').profit.sum().sort_values(ascending=False).head(1))
# print(data[data.release_year==2012].reset_index())

# data_35 = data.copy()
# data_35 = expand_df(data_35, 'cast')



# c = Counter()


# grouped_data = data_35[['cast_divided', 'cast']].groupby('cast_divided').agg({'cast': sum}).reset_index()


# for index, row in grouped_data.iterrows():
#     for i in range(len(row['cast'])):

#         c[tuple([row['cast_divided'], row['cast'][i]])] += 1


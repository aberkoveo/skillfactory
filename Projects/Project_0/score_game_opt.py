"""
Модуль является результатом оптимизации существующего ранее решения в рамках учебного проекта 0
в SkillFactory (школа веб-курсов).

"""

import numpy as np


def score_game(game_core):
    """Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число"""
    count_ls = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    random_array = np.random.randint(1,101, size=(1000))
    for number in random_array:
        count_ls.append(game_core(number))
    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
    return score


# запускаем
def game_core_v3(number):
    """Сначала устанавливаем любое random число, а потом уменьшаем или увеличиваем его в зависимости от того,
       больше оно или меньше нужного. Уменьшаем границу поиска в зависимости от значения попытки, с каждым шагом ищем
       во вдвое меньшем диапазоне значений. Функция принимает загаданное число и возвращает число попыток"""

    count = 1
    predict = 50  # np.random.randint(1,101) Начинаем угадывать с середины диапазона значений, сокращая кол-во попыток
    predict_array = [0, 100]

    while number != predict:
        count += 1
        if number > predict:
            predict_array[0] = predict
            predict = round(sum(predict_array) / 2)
        elif number < predict:
            predict_array[1] = predict
            predict = round(sum(predict_array) / 2)

    return count  # выход из цикла, если угадали


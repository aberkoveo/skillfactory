import pandas as pd

def yn_to_binary(series):
    series = series.map(lambda series: 1 if series == 'yes' else 0 if series == 'no' else series)
    return series

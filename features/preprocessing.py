import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import pickle
from util.indicators_y import add_indicators
from util.stationarization import log_and_difference
from util.benchmarks import buy_and_hodl, rsi_divergence, sma_crossover


def preprocess(df, initial_balance=10000, commission=0.0025):
    df = df.fillna(method='bfill').reset_index(drop=True).sort_values(['Date'])
    df = add_indicators(df)
    stationary_df = log_and_difference(df, ['Open', 'High', 'Low', 'Close', 'Volume'])
    benchmarks = [
        {
            'label': 'Buy and HODL',
            'values': buy_and_hodl(df['Close'], initial_balance, commission)
        },
        {
            'label': 'RSI Divergence',
            'values': rsi_divergence(df['Close'], initial_balance, commission)
        },
        {
            'label': 'SMA Crossover',
            'values': sma_crossover(df['Close'], initial_balance, commission)
        },
    ]
    return {'df': df, 'stationary_df': stationary_df, 'benchmarks': benchmarks}


data_files = [f for f in listdir('../data/train/') if isfile(join('../data/train/', f))]
data_dict = {}
i = 0
for file in data_files:
    try:
        name = file.split('.')[0]
        train = pd.read_csv('../data/train/'+file)
        test = pd.read_csv('../data/test/'+file)
        data_dict[name] = {'train': preprocess(train),
                           'test': preprocess(test)}
    except Exception as e:
        print(e)
    print(i)
    i += 1

with open('../data/data_dict.pickle', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

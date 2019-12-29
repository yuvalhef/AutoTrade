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


data_dict = {}
path = '../data/txt_data/'
date_split = '2009-10-20'
top_stocks = [x[0] for x in pd.read_csv('../data/top_stocks.csv').values.tolist()]

i = 0
for folder in ['ETFs', 'Stocks']:
    data_files = [f for f in listdir(path+folder) if isfile(join(path+folder, f))]
    for file in data_files:
        try:
            name = file.split('.')[0]
            df = pd.read_csv(path+folder+'/'+file).fillna(method='bfill').reset_index(drop=True).sort_values(['Date'])
            index = df[df['Date'] == date_split].index[0]
            df = df[index:].reset_index(drop=True)

            if not df.shape[0] == 2030 or not name.upper() in top_stocks:
                continue
            else:
                train = df[: int(df.shape[0]*0.8)].reset_index(drop=True)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                test = df[int(df.shape[0]*0.8):].reset_index(drop=True)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                data_dict[name] = {'train': preprocess(train),
                                   'test': preprocess(test)}

                print(str(i) + '  ' + str(df['Date'][int(df.shape[0] * 0.8)]))
                i += 1

        except Exception as e:
            continue

with open('../data/data_dict.pickle', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

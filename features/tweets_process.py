import pandas as pd
import numpy as np
import pickle
from os import listdir
from os.path import isfile, join
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
            'values': buy_and_hodl(df['Close'][10:], initial_balance, commission)
        },
        {
            'label': 'RSI Divergence',
            'values': rsi_divergence(df['Close'][10:], initial_balance, commission)
        },
        {
            'label': 'SMA Crossover',
            'values': sma_crossover(df['Close'][10:], initial_balance, commission)
        },
    ]
    return {'df': df, 'stationary_df': stationary_df, 'benchmarks': benchmarks}


reward_strategy = 'sortino'

with open('../data/sentiment_per_day_dict.pkl', 'rb') as handle:
    sentim_data = pickle.load(handle)

path = '../data/txt_data/'
start_date = '2014-01-02'
end_date = '2016-04-12'
files = [path+'/ETFs/'+f for f in listdir(path+'ETFs') if isfile(join(path+'ETFs', f))] + [path+'/Stocks/'+f for f in listdir(path+'Stocks') if isfile(join(path+'Stocks', f))]
files_dict = {}
for file in files:
    name = file.split('.')[-3].split('/')[-1].upper()
    files_dict[name] = file

sentim_keys = list(set(files_dict.keys()).intersection(set(sentim_data.keys())))

data_dict = {}


def apply_sentiment(name, row):
    date = row['Date']
    if name in sentim_data:
        if date in sentim_data[name]:
            return [sentim_data[name][date][0], sentim_data[name][date][1]]
        else:
            return [-99, -99]


i = 0
prev_dates = []
curr_dates = []
for k in sentim_keys:
    try:
        df = pd.read_csv(files_dict[k]).fillna(method='bfill').reset_index(drop=True).sort_values(['Date'])
        index_start = df[df['Date'] == start_date].index[0]
        index_end = df[df['Date'] == end_date].index[0]
        df = df[index_start: index_end].reset_index(drop=True)

        curr_dates = df['Date'].to_list()

        df['sentiment_1'] = np.zeros((df.shape[0]))
        df['sentiment_2'] = np.zeros((df.shape[0]))
        df[['sentiment_1', 'sentiment_2']] = df.apply(lambda x: pd.Series(apply_sentiment(k, x), index=['sentiment_1', 'sentiment_2']), axis=1)

        if not df.shape[0] == 572 or (i > 0 and not curr_dates == prev_dates):
            continue

        else:
            prev_dates = curr_dates
            train = df[: int(df.shape[0] - 120)].reset_index(drop=True)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_1', 'sentiment_2']]
            test = df[int(df.shape[0] - 120):].reset_index(drop=True)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_1', 'sentiment_2']]
            data_dict[k] = {'train': preprocess(train),
                            'test': preprocess(test)}

            print(str(i) + '  ' + str(df['Date'][int(df.shape[0] - 120)]))
            i += 1

    except Exception as e:
        continue

with open('../data/data_with_sentiment.pickle', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


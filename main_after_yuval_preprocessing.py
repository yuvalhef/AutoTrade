import os
import pickle
import copy
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation, \
    OneD_SymbolicAggregateApproximation


def normalize_data(train, test, cols):
    norm_train = copy.deepcopy(train)
    norm_test = copy.deepcopy(test)

    # Check if cols has nan:
    subset_df = copy.deepcopy(train[cols])
    num_of_rows = subset_df.shape[0]
    no_na_subset_df = subset_df.dropna(axis=0)
    if num_of_rows != no_na_subset_df.shape[0]:
        print("DF with selected cols has nan values, not normalizing the data.")
        return train, test

    # Check if cols are numeric:
    for col in cols:
        if not is_numeric_dtype(train[col]):
            print('Col ' + str(col) + ' is not numeric.')
            cols.remove(col)

    train_mean = norm_train[cols].mean(axis=0)
    train_std = norm_train[cols].std(axis=0)

    norm_train[cols] = copy.deepcopy((norm_train[cols] - train_mean) / train_std)
    norm_test[cols] = copy.deepcopy((norm_test[cols] - train_mean) / train_std)

    return norm_train, norm_test


def main():
    # Configs:
    project_url = os.path.dirname(os.path.realpath(__file__)) + '/'
    input_url = project_url + 'data/'

    # Load pickle:
    file = open(input_url + 'data_with_sentiment.pickle', 'rb')
    data_with_sentiment = pickle.load(file)
    file.close()

    # Clean data:
    for stock in tqdm(list(data_with_sentiment.keys())):
        train_stock_df = data_with_sentiment[stock]['train']['df']
        test_stock_df = data_with_sentiment[stock]['test']['df']
        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_1', 'sentiment_2']
        data_with_sentiment[stock]['train'] = train_stock_df[cols]
        data_with_sentiment[stock]['test'] = test_stock_df[cols]

    stocks_list = list(data_with_sentiment.keys())
    data_with_sentiment = {k: v for k, v in data_with_sentiment.items() if k in stocks_list[0:1]}  # subset

    # # Normalize the stock prices:
    # for stock in tqdm(list(data_with_sentiment.keys())):
    #     train_stock_df = data_with_sentiment[stock]['train']['df']
    #     test_stock_df = data_with_sentiment[stock]['test']['df']
    #     cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_1', 'sentiment_2']
    #     train_stock_df = train_stock_df[cols]
    #     test_stock_df = test_stock_df[cols]
    #     norm_train, norm_test = normalize_data(train=train_stock_df, test=test_stock_df,
    #                                            cols=['Open', 'High', 'Low', 'Close', 'Volume'])
    #     data_with_sentiment[stock]['train'] = norm_train
    #     data_with_sentiment[stock]['test'] = norm_test

    # Sax:

    # Configs:
    min_date = datetime(2014, 1, 2).date()
    max_date = datetime(2015, 10, 19).date()
    diff_in_days = (max_date - min_date).days
    paa_average_days = 5
    n_sax_symbols = 8

    # PAA transform
    days = int(round(diff_in_days / 7, 0)) * paa_average_days
    n_paa_segments = int(round(days / paa_average_days, 0))
    paa = PiecewiseAggregateApproximation(n_segments=n_paa_segments)
    for stock in tqdm(list(data_with_sentiment.keys())):
        train_stock_df = data_with_sentiment[stock]['train']
        test_stock_df = data_with_sentiment[stock]['test']
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols:
            train_stock_df[col] = paa.inverse_transform(paa.fit_transform(train_stock_df[col]))[0]
            test_stock_df[col] = paa.inverse_transform(paa.fit_transform(test_stock_df[col]))[0]

    # SAX transform:
    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    for stock in tqdm(list(data_with_sentiment.keys())):
        train_stock_df = data_with_sentiment[stock]['train']
        test_stock_df = data_with_sentiment[stock]['test']
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols:
            train_stock_df[col] = sax.inverse_transform(paa.fit_transform(train_stock_df[col]))[0]
            test_stock_df[col] = sax.inverse_transform(paa.fit_transform(test_stock_df[col]))[0]

            x =3

    x = 3


    # # 1d-SAX transform
    # n_sax_symbols_avg = 8
    # n_sax_symbols_slope = 8
    # one_d_sax = OneD_SymbolicAggregateApproximation(
    #     n_segments=n_paa_segments,
    #     alphabet_size_avg=n_sax_symbols_avg,
    #     alphabet_size_slope=n_sax_symbols_slope)
    # transformed_data = one_d_sax.fit_transform(norm_value)
    # one_d_sax_dataset_inv = one_d_sax.inverse_transform(transformed_data)

    print('f')


if __name__ == '__main__':
    main()

import os
import pickle
import copy
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm
from datetime import datetime
from tslearn.piecewise import SymbolicAggregateApproximation


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


def compute_sax(stock_dict, cols, n_paa_segments, n_sax_symbols):
    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
    for stock in tqdm(list(stock_dict.keys())):
        train_stock_df = stock_dict[stock]['train']
        test_stock_df = stock_dict[stock]['test']
        for col in cols:
            train_stock_df[col] = sax.inverse_transform(sax.fit_transform(train_stock_df[col]))[0]
            test_stock_df[col] = sax.inverse_transform(sax.fit_transform(test_stock_df[col]))[0]

        stock_dict[stock]['train'] = train_stock_df
        stock_dict[stock]['test'] = test_stock_df

    return stock_dict


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

    # Normalize the stock prices:
    for stock in tqdm(list(data_with_sentiment.keys())):
        train_stock_df = data_with_sentiment[stock]['train']
        test_stock_df = data_with_sentiment[stock]['test']
        norm_train, norm_test = normalize_data(train=train_stock_df, test=test_stock_df,
                                               cols=['Open', 'High', 'Low', 'Close', 'Volume'])
        data_with_sentiment[stock]['train'] = norm_train
        data_with_sentiment[stock]['test'] = norm_test

    # SAX transform:

    # Configs:
    min_date = datetime(2014, 1, 2).date()
    max_date = datetime(2015, 10, 19).date()
    diff_in_days = (max_date - min_date).days
    paa_average_days = 5
    n_sax_symbols = 100
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    days = int(round(452 / 7, 0)) * paa_average_days
    n_paa_segments = int(round(days / paa_average_days, 0))

    data_with_sentiment_sax = compute_sax(stock_dict=data_with_sentiment,
                                          cols=cols,
                                          n_paa_segments=n_paa_segments,
                                          n_sax_symbols=n_sax_symbols)




if __name__ == '__main__':
    main()

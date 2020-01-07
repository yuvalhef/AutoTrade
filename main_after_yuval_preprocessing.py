import os
import pickle
import copy
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from tslearn.piecewise import SymbolicAggregateApproximation
from sklearn.cluster import KMeans
# from fastdtw import fastdtw
from dtaidistance import dtw
from collections import defaultdict


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
    print("\nComputing SAX:")
    sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)

    for stock in tqdm(list(stock_dict.keys())):
        train_stock_df = stock_dict[stock]['train']
        for col in cols:
            train_stock_df[col] = sax.inverse_transform(sax.fit_transform(train_stock_df[col]))[0]

        stock_dict[stock]['train'] = train_stock_df

    return sax, stock_dict


def main():
    # Configs:
    project_url = os.path.dirname(os.path.realpath(__file__)) + '/'
    input_url = project_url + 'data/'

    # Load pickle:
    file = open(input_url + 'data_with_sentiment.pickle', 'rb')
    data_with_sentiment = pickle.load(file)
    file.close()

    stocks_list = list(data_with_sentiment.keys())
    data_with_sentiment = {k: v for k, v in data_with_sentiment.items() if k in stocks_list[0:5]}  # subset

    # Clean data:
    for stock in list(data_with_sentiment.keys()):
        train_stock_df = data_with_sentiment[stock]['train']['df']
        test_stock_df = data_with_sentiment[stock]['test']['df']
        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'sentiment_1', 'sentiment_2']
        data_with_sentiment[stock]['train'] = train_stock_df[cols]
        data_with_sentiment[stock]['test'] = test_stock_df[cols]

    # Normalize the stock prices:
    print("\nNormalize:")
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
    n_sax_symbols = 10  # 5, 10
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    days = int(round(452 / 7, 0)) * paa_average_days
    n_paa_segments = int(round(days / paa_average_days, 0))

    sax, data_with_sentiment_sax = compute_sax(stock_dict=data_with_sentiment,
                                               cols=cols,
                                               n_paa_segments=n_paa_segments,
                                               n_sax_symbols=n_sax_symbols)

    # Compute distance matrix:
    import itertools
    import numpy as np

    distance_dict = defaultdict(lambda: defaultdict(list))

    stocks_list = list(data_with_sentiment.keys())
    combinations = list(itertools.permutations(stocks_list, 2))

    for (stock1, stock2) in tqdm(combinations):

        dtw_dis_list = []
        # sax_dis_list = []

        for col in cols:
            s1 = np.array(data_with_sentiment_sax[stock1]['train'][col])
            s2 = np.array(data_with_sentiment_sax[stock2]['train'][col])

            # Compute DTW distance:
            dtw_distance = dtw.distance(s1, s2)
            dtw_dis_list.append(dtw_distance)

            # Compute SAX distance:
            # sax_distance = np.linalg.norm(s1 - s2)
            # sax_dis_list.append(sax_distance)

        distance_dict['DTW']['stock_1'].append(stock1)
        distance_dict['DTW']['stock_2'].append(stock2)
        distance_dict['DTW']['distance_list'].append(dtw_dis_list)
        # distance_dict['DTW']['distance_average'].append(np.mean(dtw_dis_list))
        distance_dict['DTW']['distance_average'].append(float(0.5))

        # distance_dict['SAX']['stock_1'].append(stock1)
        # distance_dict['SAX']['stock_2'].append(stock2)
        # distance_dict['SAX']['distance_list'].append(sax_dis_list)
        # distance_dict['SAX']['distance_average'].append(np.mean(sax_dis_list))

    dtw_distances_df = pd.DataFrame(distance_dict['DTW'])

    def create_distance_matrix_from_distance(stocks_list, value_col, distances_df):

        dis_matrix = pd.DataFrame(None, columns=stocks_list, index=stocks_list)

        return dis_matrix

    dis_matrix = create_distance_matrix_from_distance(stocks_list=stocks_list,
                                                      value_col='distance_average',
                                                      distances_df=dtw_distances_df)



    # # Cluster the stocks:
    # n_clusters = 3
    # km = KMeans(n_clusters=n_clusters, init='random',max_iter=300, tol=1e-04)
    # y_km = km.fit_predict(X)


if __name__ == '__main__':
    main()

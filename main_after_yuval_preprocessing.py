import os
import pickle
import copy
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from tslearn.piecewise import SymbolicAggregateApproximation
from dtaidistance import dtw
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.spatial.distance as ssd


def save_pickle_object(obj, obj_name, url):
    file = open(url + obj_name + '.obj', 'wb')
    pickle.dump(obj, file)
    print('Info: saved {o} pickle file.'.format(o=str(obj_name)))


def concat_dfs_by_row(list_of_dfs):
    new_list_Of_dfs = []

    if len(list_of_dfs) < 2:
        print('Error: less than two DFs.')
        return None

    col_shape = list_of_dfs[0].shape[1]
    for df in list_of_dfs:
        assert df.shape[1] == col_shape
        df.reset_index(drop=True, inplace=True)
        new_list_Of_dfs.append(df)

    all_df = pd.concat(new_list_Of_dfs, axis=0, ignore_index=True)

    return all_df


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


def compute_distance_matrix(stocks_list, stocks_dict, cols, distance_metric):
    print("\nCompute {d} distance matrix:".format(d=str(distance_metric)))
    distance_dict = defaultdict(lambda: defaultdict(float))
    combinations = list(itertools.permutations(stocks_list, 2))
    values_list = []

    # Remove duplicate combinations:
    no_duplicate_combinations = []
    for (stock1, stock2) in combinations:
        if ((stock1, stock2) in no_duplicate_combinations) or ((stock2, stock1) in no_duplicate_combinations):
            continue
        else:
            no_duplicate_combinations.append((stock1, stock2))

    for (stock1, stock2) in tqdm(combinations):
        distance_list = []
        for col in cols:
            s1 = np.array(stocks_dict[stock1]['train'][col])
            s2 = np.array(stocks_dict[stock2]['train'][col])
            # Compute DTW distance:
            if distance_metric == 'DTW':
                distance = dtw.distance(s1, s2)
            else:
                # Compute SAX distance:
                distance = np.linalg.norm(s1 - s2)
            distance_list.append(distance)

        final_distance = float(np.mean(distance_list))
        values_list.append(final_distance)
        distance_dict[stock1][stock2] = final_distance

    # Create distance matrix:
    min = np.min(values_list)
    max = np.max(values_list)
    distance_matrix_dict = defaultdict(lambda: defaultdict(float))
    for stock1 in stocks_list:
        for stock2 in stocks_list:
            if stock1 == stock2:
                value = 0
            else:
                if stock1 in list(distance_dict.keys()):
                    value = distance_dict[stock1][stock2]
                    value = (value - min) / (max - min)

                else:
                    value = distance_dict[stock2][stock1]
                    value = (value - min) / (max - min)

            distance_matrix_dict[stock1][stock2] = value
    distance_matrix = pd.DataFrame.from_dict(distance_matrix_dict, orient='index')

    return distance_matrix


def cluster_distance_matrix(distance_matrix, cluster_method, cluster_params):
    print("\nClustering distance matrix:")
    if cluster_method == 'linkage':
        stocks = list(distance_matrix.columns)
        dist_array = ssd.squareform(distance_matrix)
        Z = linkage(dist_array)

        if cluster_params['plot']:
            fig = plt.figure(figsize=(25, 10))
            dn = dendrogram(Z, labels=stocks)
            plt.show()

        labels = fcluster(Z=Z, t=cluster_params['t'], criterion=cluster_params['criterion'])
    else:
        cluster = DBSCAN(**cluster_params).fit(distance_matrix)
        labels = cluster.labels_

    clusters_num = len(list(set(labels)))
    print("\nNumber of clusters with method {m}: {n}".format(m=str(cluster_method), n=str(clusters_num)))
    distance_matrix['cluster'] = labels
    print("\nDistribution of stocks per clusters: \n" + str(Counter(distance_matrix['cluster'])))

    distance_matrix = distance_matrix.reset_index()
    distance_matrix.rename(columns={'index': 'stock'}, inplace=True)

    return distance_matrix[['stock', 'cluster']]


def datetime_range(start, end):
    span = end - start
    for i in range(span.days + 1):
        yield start + timedelta(days=i)


def create_features_based_clusters(stocks_per_clusters, list_of_dates, stocks_dict, data_set_type, remove_nan):
    cluster_list = list(set(stocks_per_clusters['cluster']))
    cluster_df = pd.DataFrame(columns=['cluster', 'Date', 'cluster@yesterday_mean_close_price',
                                       'cluster@two_days_ago_mean_close_price',
                                       'cluster@three_days_ago_mean_close_price',
                                       'cluster@four_days_ago_mean_close_price',
                                       'cluster@five_days_ago_mean_close_price'])

    for cluster in tqdm(cluster_list):

        list_of_stocks_per_cluster = list(stocks_per_clusters[stocks_per_clusters['cluster'] == cluster]['stock'])

        for date in list_of_dates:

            yesterday_stock_prices_list = []
            two_days_ago_stock_prices_list = []
            three_days_ago_stock_prices_list = []
            four_days_ago_stock_prices_list = []
            five_days_ago_stock_prices_list = []

            day_before = date - timedelta(days=1)
            two_days_ago = date - timedelta(days=2)
            three_days_ago = date - timedelta(days=3)
            four_days_ago = date - timedelta(days=4)
            five_days_ago = date - timedelta(days=5)

            for stock in list_of_stocks_per_cluster:

                # Create yesterday mean close price feature:
                stock_df = stocks_dict[stock][data_set_type]
                yesterday_stock_df = stock_df[stock_df['Date'] >= pd.Timestamp(day_before)]
                yesterday_stock_df = yesterday_stock_df[yesterday_stock_df['Date'] < pd.Timestamp(date)]
                if yesterday_stock_df.shape[0] > 0:
                    yesterday_stock_prices_list.append(yesterday_stock_df['Close'].values[0])

                # Create two days ago mean close price feature:
                two_days_ago_stock_df = stock_df[stock_df['Date'] >= pd.Timestamp(two_days_ago)]
                two_days_ago_stock_df = two_days_ago_stock_df[
                    two_days_ago_stock_df['Date'] < pd.Timestamp(day_before)]
                if two_days_ago_stock_df.shape[0] > 0:
                    two_days_ago_stock_prices_list.append(two_days_ago_stock_df['Close'].values[0])

                # Create three days ago mean close price feature:
                three_days_ago_stock_df = stock_df[stock_df['Date'] >= pd.Timestamp(three_days_ago)]
                three_days_ago_stock_df = three_days_ago_stock_df[
                    three_days_ago_stock_df['Date'] < pd.Timestamp(two_days_ago)]
                if three_days_ago_stock_df.shape[0] > 0:
                    three_days_ago_stock_prices_list.append(three_days_ago_stock_df['Close'].values[0])

                # Create four days ago mean close price feature:
                four_days_ago_stock_df = stock_df[stock_df['Date'] >= pd.Timestamp(four_days_ago)]
                four_days_ago_stock_df = four_days_ago_stock_df[
                    four_days_ago_stock_df['Date'] < pd.Timestamp(three_days_ago)]
                if four_days_ago_stock_df.shape[0] > 0:
                    four_days_ago_stock_prices_list.append(four_days_ago_stock_df['Close'].values[0])

                # Create five days ago mean close price feature:
                five_days_ago_stock_df = stock_df[stock_df['Date'] >= pd.Timestamp(five_days_ago)]
                five_days_ago_stock_df = five_days_ago_stock_df[
                    five_days_ago_stock_df['Date'] < pd.Timestamp(four_days_ago)]
                if five_days_ago_stock_df.shape[0] > 0:
                    five_days_ago_stock_prices_list.append(five_days_ago_stock_df['Close'].values[0])

            if len(yesterday_stock_prices_list) > 0:
                yesterday_mean_price = np.mean(yesterday_stock_prices_list)
            else:
                yesterday_mean_price = np.nan

            if len(two_days_ago_stock_prices_list) > 0:
                two_days_before_mean_price = np.mean(two_days_ago_stock_prices_list)
            else:
                two_days_before_mean_price = np.nan

            if len(three_days_ago_stock_prices_list) > 0:
                three_days_before_mean_price = np.mean(three_days_ago_stock_prices_list)
            else:
                three_days_before_mean_price = np.nan

            if len(four_days_ago_stock_prices_list) > 0:
                four_days_before_mean_price = np.mean(four_days_ago_stock_prices_list)
            else:
                four_days_before_mean_price = np.nan

            if len(five_days_ago_stock_prices_list) > 0:
                five_days_before_mean_price = np.mean(five_days_ago_stock_prices_list)
            else:
                five_days_before_mean_price = np.nan

            # Append values in cluster df:
            new_row = pd.Series({'cluster': cluster, 'Date': date,
                                 'cluster@yesterday_mean_close_price': yesterday_mean_price,
                                 'cluster@two_days_ago_mean_close_price': two_days_before_mean_price,
                                 'cluster@three_days_ago_mean_close_price': three_days_before_mean_price,
                                 'cluster@four_days_ago_mean_close_price': four_days_before_mean_price,
                                 'cluster@five_days_ago_mean_close_price': five_days_before_mean_price})

            cluster_df = cluster_df.append(new_row, ignore_index=True)

    # Remove nans:
    if remove_nan:
        cols = list(cluster_df.columns)
        cols.remove('cluster')
        cols.remove('Date')
        cluster_df = cluster_df.dropna(subset=cols)

    return cluster_df


def main():
    ########################################### Configs:
    project_url = os.path.dirname(os.path.realpath(__file__)) + '/'
    input_url = project_url + 'data/'

    # Sax:
    preform_sax = True
    train_start_date = datetime(2014, 1, 2, 00, 00, 00, 00).date()
    train_end_date = datetime(2015, 10, 20, 00, 00, 00, 00).date()
    test_start_date = datetime(2015, 10, 19, 00, 00, 00, 00).date()
    test_end_date = datetime(2016, 11, 5, 00, 00, 00, 00).date()
    paa_average_days = 5
    n_sax_symbols = 10  # 5, 10

    # Cluster:
    cluster_method = 'DBSCAN'  # linkage or 'DBSCAN'

    fcluster_params = {

        't': 10,
        'criterion': 'distance',  # inconsistent or distance
        'plot': False

    }

    DBSCAN_params = {
        'eps': 0.5,
        'min_samples': 2,
        'metric': 'euclidean'
    }

    remove_nan = False  # if to remove nan rows when cluster feature is nan

    ###########################################

    # Load pickle:
    file = open(input_url + 'data_with_sentiment.pickle', 'rb')
    data_with_sentiment = pickle.load(file)
    file.close()

    stocks_list = list(data_with_sentiment.keys())
    data_with_sentiment = {k: v for k, v in data_with_sentiment.items() if k in stocks_list[0:10]}  # subset

    # Clean data:
    data_with_sentiment_not_normalized = defaultdict(lambda: defaultdict(list))
    for stock in list(data_with_sentiment.keys()):
        train_stock_df = data_with_sentiment[stock]['train']['df']
        test_stock_df = data_with_sentiment[stock]['test']['df']
        cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Convert 'Date' to date:
        train_stock_df['Date'] = pd.to_datetime(train_stock_df['Date'])
        test_stock_df['Date'] = pd.to_datetime(test_stock_df['Date'])

        data_with_sentiment[stock]['train'] = train_stock_df[cols]
        data_with_sentiment[stock]['test'] = test_stock_df[cols]

        data_with_sentiment_not_normalized[stock]['train'] = train_stock_df[cols]
        data_with_sentiment_not_normalized[stock]['test'] = test_stock_df[cols]

    # Normalize the stock prices:
    print("\nNormalizing the data:")
    for stock in tqdm(list(data_with_sentiment.keys())):
        train_stock_df = data_with_sentiment[stock]['train']
        test_stock_df = data_with_sentiment[stock]['test']
        norm_train, norm_test = normalize_data(train=train_stock_df, test=test_stock_df,
                                               cols=['Open', 'High', 'Low', 'Close', 'Volume'])
        data_with_sentiment[stock]['train'] = norm_train
        data_with_sentiment[stock]['test'] = norm_test

    # Compute distance matrix:
    if preform_sax:
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        days = int(round(452 / 7, 0)) * paa_average_days
        n_paa_segments = int(round(days / paa_average_days, 0))

        sax, data_with_sentiment_sax = compute_sax(stock_dict=data_with_sentiment,
                                                   cols=cols,
                                                   n_paa_segments=n_paa_segments,
                                                   n_sax_symbols=n_sax_symbols)
        distance_matrix = compute_distance_matrix(stocks_list=list(data_with_sentiment_sax.keys()),
                                                  stocks_dict=data_with_sentiment_sax, cols=cols,
                                                  distance_metric='SAX')
    else:
        # Compute DTW distance matrix:
        distance_matrix = compute_distance_matrix(stocks_list=list(data_with_sentiment.keys()),
                                                  stocks_dict=data_with_sentiment,
                                                  cols=cols, distance_metric='DTW')

    # Cluster the stocks:
    if cluster_method == 'linkage':
        cluster_params = fcluster_params
    else:
        cluster_params = DBSCAN_params

    stocks_per_clusters = cluster_distance_matrix(distance_matrix=distance_matrix,
                                                  cluster_method=cluster_method,
                                                  cluster_params=cluster_params)

    # Create features based on clusters:
    print("\nCreate features per clusters:")
    list_of_dates = list(datetime_range(start=train_start_date, end=train_end_date))
    train_cluster_df = create_features_based_clusters(stocks_per_clusters=stocks_per_clusters,
                                                      list_of_dates=list_of_dates,
                                                      stocks_dict=data_with_sentiment_not_normalized,
                                                      data_set_type='train',
                                                      remove_nan=remove_nan)

    list_of_dates = list(datetime_range(start=test_start_date, end=test_end_date))
    test_cluster_df = create_features_based_clusters(stocks_per_clusters=stocks_per_clusters,
                                                     list_of_dates=list_of_dates,
                                                     stocks_dict=data_with_sentiment_not_normalized,
                                                     data_set_type='test',
                                                     remove_nan=remove_nan)

    clusters_df = concat_dfs_by_row(list_of_dfs=[train_cluster_df, test_cluster_df])
    clusters_df["Date"] = pd.to_datetime(clusters_df['Date'])

    # Check if there is duplicates dates between train and test:
    no_duplicates_clusters_df = clusters_df.drop_duplicates(subset=['Date', 'cluster'])
    assert clusters_df.shape[0] == no_duplicates_clusters_df.shape[0]

    # Add cluster feature per stock:
    print('\nAdd cluster features to stocks dict:')
    preprocessed_stocks_dict = copy.deepcopy(data_with_sentiment_not_normalized)
    for stock in tqdm(list(preprocessed_stocks_dict.keys())):
        train_stock_df = preprocessed_stocks_dict[stock]['train']
        test_stock_df = preprocessed_stocks_dict[stock]['test']

        # Extract relevant cluster df:
        cluster = stocks_per_clusters[stocks_per_clusters['stock'] == stock]['cluster'].values[0]
        cluster_df = clusters_df[clusters_df['cluster'] == cluster]

        # Add cluster features:
        train_stock_df_with_cluster_features = pd.merge(train_stock_df, cluster_df, how='left', on='Date')
        test_stock_df_with_cluster_features = pd.merge(test_stock_df, cluster_df, how='left', on='Date')

        preprocessed_stocks_dict[stock]['train'] = train_stock_df_with_cluster_features
        preprocessed_stocks_dict[stock]['test'] = test_stock_df_with_cluster_features

    # Save pickle objects:
    save_pickle_object(obj=clusters_df, obj_name='clusters_df', url=input_url)
    save_pickle_object(obj=stocks_per_clusters, obj_name='stocks_per_clusters', url=input_url)
    save_pickle_object(obj=preprocessed_stocks_dict, obj_name='preprocessed_stocks_dict', url=input_url)


if __name__ == '__main__':
    main()

import os
import pandas as pd
import datetime
from tqdm import tqdm
from datetime import timedelta
from collections import defaultdict
import pickle

def read_data(input_url):
    stocks_dict = defaultdict(list)
    empty_stocks = 0

    stocks_list = os.listdir(input_url)

    for i in tqdm(range(len(stocks_list))):
        filename = stocks_list[i]
        if filename.endswith(".txt"):
            stock_file_name = input_url + "/" + filename
            stock_name = filename.split('.')[0]
            try:
                stock_df = pd.read_csv(stock_file_name, sep=',')
                stock_df["Date"] = pd.to_datetime(stock_df["Date"])
                stocks_dict[stock_name] = stock_df
            except:
                empty_stocks += 1
                continue

    print("Number of empty stock files: " + str(empty_stocks))
    print("Number of stocks: " + str(len(list(stocks_dict.keys()))))
    return stocks_dict


def save_pickle_object(obj, obj_name, url):
    file = open(url + obj_name + '.obj', 'wb')
    pickle.dump(obj, file)
    print('Info: saved {o} pickle file.'.format(o=str(obj_name)))


def get_stocks_stats(stocks_dict):
    stocks_stats_dict = defaultdict(list)
    stocks_list = list(stocks_dict.keys())

    for i in tqdm(range(len(stocks_list))):
        stock_name = stocks_list[i]
        stock_df = stocks_dict[stock_name]
        stocks_stats_dict['name'].append(stock_name)
        stocks_stats_dict['min_date'].append(min(stock_df["Date"]))
        stocks_stats_dict['max_date'].append(max(stock_df["Date"]))
        stocks_stats_dict['logs'].append(stock_df.shape[0])

    stocks_stats_df = pd.DataFrame(stocks_stats_dict)

    return stocks_stats_df


def split_to_train_test(stocks_dict, date):
    data_sets_stocks_dict = {}

    train_dict = defaultdict(list)
    test_dict = defaultdict(list)

    stocks_list = list(stocks_dict.keys())

    for stock_name in stocks_list:
        stock_df = stocks_dict[stock_name]
        train_stock_df = stock_df[stock_df['Date'] <= date]
        test_stock_df = stock_df[stock_df['Date'] > date]

        if train_stock_df.shape[0] == 0 or test_stock_df.shape[0] == 0:
            print('problem in stock: ' + str(stock_name))
            continue

        train_dict[stock_name] = train_stock_df
        test_dict[stock_name] = test_stock_df

    data_sets_stocks_dict['train'] = train_dict
    data_sets_stocks_dict['test'] = test_dict

    return data_sets_stocks_dict


def filtered_dates_stocks(stocks_dict, date):
    filtered_dates_stocks_dict = defaultdict(int)
    stocks_list = list(stocks_dict.keys())

    for i in tqdm(range(len(stocks_list))):
        stock_name = stocks_list[i]
        stock_df = stocks_dict[stock_name]
        filter_days_stock_df = stock_df[stock_df['Date'] >= date]

        if filter_days_stock_df.shape[0] == 0:
            continue

        filtered_dates_stocks_dict[stock_name] = filter_days_stock_df

    return filtered_dates_stocks_dict


def main():
    # Configs:
    project_url = os.path.dirname(os.path.realpath(__file__)) + '/'
    input_url = project_url + 'DB/Stocks'
    min_logs = 2030
    min_date = datetime.date(2014, 1, 2)
    max_date = datetime.date(2016, 4, 12)
    train_prec = 0.8

    # Read Data:
    stocks_dict = read_data(input_url=input_url)
    top_stocks = pd.read_csv(project_url + 'DB/top_stocks.csv', low_memory=False)

    # Filter days in stocks based on date:
    filtered_days_stocks_dict = filtered_dates_stocks(stocks_dict=stocks_dict, date=min_date)
    print("Number of stocks after filtering days: " + str(len(list(filtered_days_stocks_dict.keys()))))
    stocks_stats_df = get_stocks_stats(stocks_dict=filtered_days_stocks_dict)

    # Filter stocks with less then min_logs:
    stocks_stats_df = stocks_stats_df[stocks_stats_df['logs'] >= min_logs]
    stocks_names = list(stocks_stats_df['name'])
    filtered_days_min_logs_stocks_dict = {k: v for k, v in filtered_days_stocks_dict.items() if k in stocks_names}
    print("Number of stocks after filtering min logs: " + str(len(list(filtered_days_min_logs_stocks_dict.keys()))))
    stocks_stats_df = get_stocks_stats(stocks_dict=filtered_days_min_logs_stocks_dict)

    # Filter top stocks:
    stocks_names = list(stocks_stats_df['name'])
    top_stocks_list = [x.lower() for x in list(top_stocks['top_stocks'])]
    top_stocks_names = [x for x in stocks_names if x in top_stocks_list]
    filtered_days_min_logs_top_stocks_dict = {k: v for k, v in filtered_days_min_logs_stocks_dict.items() if
                                              k in top_stocks_names}
    print("Number of top stocks after filtering: " + str(len(list(filtered_days_min_logs_top_stocks_dict.keys()))))

    # Split to train and test:
    stocks_stats_df = get_stocks_stats(stocks_dict=filtered_days_min_logs_top_stocks_dict)
    diff_in_days = (min(stocks_stats_df['max_date']) - max(stocks_stats_df['min_date'])).days
    days_for_train = round(diff_in_days * train_prec)
    end_date_of_train = min_date + timedelta(days=days_for_train)
    print("Train date range: " + str(min_date) + ' - ' + str(end_date_of_train))
    print("Test date range: " + str(end_date_of_train) + ' - ' + str(min(stocks_stats_df['max_date'])))
    stocks_sets_dict = split_to_train_test(stocks_dict=filtered_days_min_logs_top_stocks_dict, date=end_date_of_train)

    


if __name__ == '__main__':
    main()

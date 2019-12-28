import pandas as pd
import os
from os import listdir
from os.path import isfile, join

path = './data/txt_data/'
i = 0
for folder in ['ETFs', 'Stocks']:
    data_files = [f for f in listdir(path+folder) if isfile(join('./data/txt_data/'+folder, f))]
    for file in data_files:
        try:
            name = file.split('.')[0]
            df = pd.read_csv(path+folder+'/'+file).fillna(method='bfill').reset_index().sort_values(['Date'])
            train = df[: int(df.shape[0]*0.8)].reset_index(drop=True)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            test = df[int(df.shape[0]*0.8):].reset_index(drop=True)[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            if test.shape[0] > 61:
                train.to_csv('./data/train/'+name+'.csv', index=False)
                test.to_csv('./data/test/'+name+'.csv', index=False)

            print(i)
            i += 1

        except Exception as e:
            print(e)

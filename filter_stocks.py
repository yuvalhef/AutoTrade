import pickle
import pandas

with open('./data/data_dict.pickle', 'rb') as handle:
    d = pickle.load(handle)

keys = list(d.keys())
for key in keys:
    if d[key]['test']['df'].shape[0] < 100:
        d.pop(key, None)

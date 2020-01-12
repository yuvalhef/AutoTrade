import pandas as pd
import numpy as np
import pickle
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2, ACER
from env.TradingEnv import TradingEnv
import os
import matplotlib.pyplot as plt

best_mean_reward, n_steps = -np.inf, 0


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 2 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print(
                "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True


def get_params(mode='train'):
    env_params = {}
    agent_params = {}
    env_params['mode'] = mode
    env_params['reward_func'] = 'sorinto'
    # env_params['max_steps'] = 120
    agent_params['learning_rate'] = 0.0005
    agent_params['policy'] = MlpLnLstmPolicy
    agent_params['nminibatches'] = 1
    agent_params['verbose'] = 0

    return env_params, agent_params


def get_cluster_features(clusters_df):
    clusters = clusters_df['cluster'].unique()
    clusters_features = {}
    for cluster in clusters:
        c_df = clusters_df[clusters_df['cluster'] == 1]['cluster@yesterday_mean_close_price'].dropna().tolist()
        c_df.append(c_df[-1])
        c_df = np.array(c_df)
        features = []
        for i in range(2, 6):
            feature = []
            for j in range(len(c_df)):
                if j < i:
                    feature.append(0)
                else:
                    feature.append(c_df[j-1] - c_df[j-i])
            features.append(feature)
        features_df = pd.DataFrame(np.array(features).T, columns=['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4'])
        clusters_features[cluster] = features_df
    return clusters_features


def prepare_data(data, clusters_df=None, clusters_dict=None):
    if clusters_df is None:
        return data
    else:
        clustrs_features = get_cluster_features(clusters_df)
        for stock in list(data.keys()):
            cluster_id = clusters_dict[stock.lower()]['train']['cluster'].iloc[0]
            curr_features = clustrs_features[cluster_id]
            feature_dict = {
                'train': curr_features[: int(curr_features.shape[0]*0.8)].reset_index(drop=True),
                'test': curr_features[int(curr_features.shape[0]*0.8):].reset_index(drop=True)
            }
            for set in ['train', 'test']:
                data[stock][set]['df'] = pd.concat([data[stock][set]['df'], feature_dict[set]], axis=1)
                data[stock][set]['stationary_df'] = pd.concat([data[stock][set]['stationary_df'], feature_dict[set]], axis=1)
    return data


def train(data):
    env_params, agent_params = get_params('train')
    env_params['data'] = data
    env_params['max_steps'] = data[list(data.keys())[0]]['test']['df'].shape[0]
    env = TradingEnv(**env_params)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    agent_params['env'] = env

    model = PPO2(**agent_params)
    model.learn(total_timesteps=300000, callback=callback)
    model.save(log_dir + 'last_model.pkl')
    return model


def test(data, render, model=None):
    env_params, agent_params = get_params('test')
    env_params['data'] = data
    env_params['max_steps'] = data[list(data.keys())[0]]['test']['df'].shape[0]
    env = TradingEnv(**env_params)
    env = DummyVecEnv([lambda: env])
    agent_params['env'] = env

    if model is None:
        model = PPO2.load(log_dir + '/best_model.pkl', env=env)
    print('Loaded best model')

    rsi_res = []
    sma_res = []
    buy_hold_res = []
    net_worths = []

    finished = False
    obs, done = env.reset(), False
    while not finished:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        if render:
            env.render()
        if done:
            trade_env = env.envs[0]
            net_worths.append(trade_env.prev_net_worths)
            buy_hold_res.append(trade_env.prev_benchmarks[0]['values'])
            rsi_res.append(trade_env.prev_benchmarks[1]['values'])
            sma_res.append(trade_env.prev_benchmarks[2]['values'])

            if len(trade_env.stocks_names) == 0:
                finished = True

    results_dir = log_dir + '/test_results/'
    os.makedirs(results_dir, exist_ok=True)
    df_buy_hold = pd.DataFrame(buy_hold_res)
    df_buy_hold.to_csv(results_dir + 'buy_hold.csv')
    df_sma = pd.DataFrame(sma_res)
    df_sma.to_csv(results_dir + 'sma.csv')
    df_rsi = pd.DataFrame(rsi_res)
    df_rsi.to_csv(results_dir + 'rsi.csv')
    df_ours = pd.DataFrame(net_worths)
    df_ours.to_csv(results_dir + 'ours.csv')

    mean_results = pd.DataFrame()
    mean_results['time step'] = list(range(len(trade_env.prev_net_worths)))
    mean_results['RSI'] = df_rsi.values.mean(axis=0)
    mean_results['SMA'] = df_sma.values.mean(axis=0)
    # mean_results['BUY_HOLD'] = df_buy_hold.values.mean(axis=0)
    mean_results['AutoTrade'] = df_ours.values.mean(axis=0)
    mean_results.to_csv(results_dir + 'fina_results.csv')

    mean_results.plot(x='time step')
    plt.ylabel('Net Worth')
    plt.title('Average Net Worth - 400 Trading Days')
    plt.savefig(results_dir + 'fina_results.jpg')


def load_pickle_object(obj_name, url):
    file = open(url + obj_name + '.obj', 'rb')
    obj = pickle.load(file)
    print('Info: uploading {o} pickle file.'.format(o=str(obj_name)))
    return obj


def main(mode, exp_type, render):
    print('hello')
    with open('./data/data_dict_new.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)

    if exp_type == 'baseline':
        print('baseline')
    else:
        clusters_df = load_pickle_object('/clusters_df', './data/clustering/'+cluster_type)
        clusters_dict = load_pickle_object('/preprocessed_stocks', './data/clustering/'+cluster_type)
        data_dict = prepare_data(data_dict, clusters_df, clusters_dict)

    if mode == 'train':
        model = train(data_dict)
        test(data_dict, render, model)
    else:
        test(data_dict, render)


if __name__ == '__main__':
    # experiment settings
    mode = 'train'
    exp_type = 'clusters_experiment'
    cluster_types = os.listdir('data/clustering/')
    render = False
    for exp_num in cluster_types:
        cluster_type = cluster_types[0]
        log_dir = exp_type + "/" + cluster_type + "/"
        os.makedirs(log_dir, exist_ok=True)
        main(mode, exp_type, render)

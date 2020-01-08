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
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

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
    env_params['max_steps'] = 120
    agent_params['learning_rate'] = 0.0005
    agent_params['policy'] = MlpLnLstmPolicy
    agent_params['nminibatches'] = 1
    agent_params['verbose'] = 0

    return env_params, agent_params


def prepare_data(data, drop_features):
    if drop_features:
        for stock in list(data.keys()):
            data[stock]['train']['df'].drop(drop_features, axis=1, inplace=True)
            data[stock]['train']['stationary_df'].drop(drop_features, axis=1, inplace=True)
            data[stock]['test']['df'].drop(drop_features, axis=1, inplace=True)
            data[stock]['test']['stationary_df'].drop(drop_features, axis=1, inplace=True)
    else:
        pass
    return data


def train(data):
    env_params, agent_params = get_params('train')
    env_params['data'] = data
    env = TradingEnv(**env_params)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    agent_params['env'] = env

    model = PPO2(**agent_params)
    model.learn(total_timesteps=3000, callback=callback)
    model.save(log_dir+'last_model.pkl')


def test(data, render):
    env_params, agent_params = get_params('test')
    env_params['data'] = data
    env = TradingEnv(**env_params)
    env = DummyVecEnv([lambda: env])
    agent_params['env'] = env

    model = PPO2.load(log_dir+'/last_model.pkl', env=env)
    print('Loaded last model')

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
            buy_hold_res.append(trade_env.prev_benchmarks[0]['values'][-len(trade_env.prev_net_worths):])
            rsi_res.append(trade_env.prev_benchmarks[1]['values'][-len(trade_env.prev_net_worths):])
            sma_res.append(trade_env.prev_benchmarks[2]['values'][-len(trade_env.prev_net_worths):])

            if len(trade_env.stocks_names) == 0:
                finished = True
            else:
                env.reset()

    results_dir = log_dir+'/test_results/'
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
    mean_results['BUY_HOLD'] = df_buy_hold.values.mean(axis=0)
    mean_results['AutoTrade'] = df_ours.values.mean(axis=0)
    mean_results.to_csv(results_dir + 'fina_results.csv')

    mean_results.plot(x='time step')
    plt.ylabel('Net Worth')
    plt.title('Average Net Worth - 110 Trading Days')
    plt.savefig(results_dir + 'fina_results.jpg')


def main(mode, exp_type, render):
    print('hello')
    with open('./data/data_with_sentiment.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)

    if exp_type == 'baseline':
        features_drop = ['sentiment_1', 'sentiment_2']
    else:
        features_drop = None

    data_dict = prepare_data(data_dict, features_drop)

    if mode == 'train':
        train(data_dict)

    test(data_dict, render)


if __name__ == '__main__':
    # experiment settings
    mode = 'train'
    exp_type = 'baseline'
    render = False
    log_dir = exp_type+"/"
    os.makedirs(log_dir, exist_ok=True)
    main(mode, exp_type, render)

import pandas as pd
import numpy as np
import pickle
from stable_baselines.common.policies import MlpLnLstmPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import A2C, ACKTR, PPO2
from env.TradingEnv import TradingEnv
import os

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


def get_params(type='baseline'):
    env_params = {}
    agent_params = {}
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


def train(data, type):
    env_params, agent_params = get_params(type)
    env_params['data'] = data
    env = TradingEnv(**env_params)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    agent_params['env'] = env

    model = PPO2(**agent_params)
    model.learn(total_timesteps=100000, callback=callback)


def main(mode, type):
    print('hello')
    with open('./data/data_with_sentiment.pickle', 'rb') as handle:
        data_dict = pickle.load(handle)

    if type == 'baseline':
        features_drop = ['sentiment_1', 'sentiment_2']
    else:
        features_drop = None

    data_dict = prepare_data(data_dict, features_drop)

    if mode == 'train':
        train(data_dict, type)


if __name__ == '__main__':
    # experiment settings
    mode = 'train'
    type = 'baseline'
    log_dir = mode+"_"+type+"/"
    os.makedirs(log_dir, exist_ok=True)
    main(mode, type)

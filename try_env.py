import pandas as pd
import numpy as np
import pickle

# from stable_baselines.common.policies import MlpLnLstmPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2

from env.TradingEnv import TradingEnv

reward_strategy = 'sortino'

with open('./data/data_with_sentiment.pickle', 'rb') as handle:
    d = pickle.load(handle)
print(len(d))
env = TradingEnv(reward_func=reward_strategy, data=d, mode='train')

for i in range(100000):
    j = np.random.random_integers(12)
    next_state, rew, done, _ = env.step(j)
    env.render()
    print(j)
    if done:
        env.reset()


env.reset()
print('bi')

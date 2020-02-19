import pandas as pd
import numpy as np

# from stable_baselines.common.policies import MlpLnLstmPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2

from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators

input_data_file = 'data/coinbase_hourly.csv'


reward_strategy = 'sortino'

df = pd.read_csv(input_data_file)
df = df.drop(['Symbol'], axis=1)
df = df.sort_values(['Date'])
df = add_indicators(df.reset_index())

train_len = int(len(df) * 0.8)

df = df[:train_len]

validation_len = int(train_len * 0.8)
train_df = df[:validation_len]
test_df = df[validation_len:]


env = BitcoinTradingEnv(train_df, reward_func=reward_strategy)
env.reset()

for i in range(100000):
    j = np.random.random_integers(12)
    next_state, rew, done, _ = env.step(j)
    env.render()
    print(j)
    if done:
        env.reset()


env.reset()
print('bi')


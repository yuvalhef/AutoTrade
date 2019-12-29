import gym
import pandas as pd
import numpy as np
from numpy import inf
from gym import spaces
from sklearn import preprocessing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from empyrical import sortino_ratio, calmar_ratio, omega_ratio

from render.TradingGraph import TradingGraph
import random


# Delete this if debugging
np.warnings.filterwarnings('ignore')


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self, initial_balance=10000, commission=0.0025, reward_func='sortino',
                 max_steps=60, data={}, mode='train', **kwargs):
        super(TradingEnv, self).__init__()

        self.kwargs = kwargs
        self.mode = mode
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_func = reward_func

        self.data = data
        curr_df = random.choice(list(self.data.keys()))
        self.stock_name = curr_df
        self.df = self.data[curr_df][mode]['df']
        self.stationary_df = self.data[curr_df][mode]['stationary_df']
        self.benchmarks = self.data[curr_df][mode]['benchmarks']

        self.balance = self.initial_balance
        self.net_worths = [self.initial_balance]
        self.btc_held = 0
        self.max_steps = max_steps
        self.current_step = np.random.randint(0, len(self.df) - self.max_steps - 2)

        self.account_history = np.array([[self.balance], [0], [0], [0], [0]])
        self.trades = []
        self.forecast_len = kwargs.get('forecast_len', 10)
        self.confidence_interval = kwargs.get('confidence_interval', 0.95)
        self.obs_shape = (1, 5 + len(self.df.columns) - 1 + (self.forecast_len * 3))

        # Actions of the format Buy 1/4, Sell 3/4, Hold (amount ignored), etc.
        self.action_space = spaces.Discrete(12)

        # Observes the price action, indicators, account action, price forecasts
        self.observation_space = spaces.Box(low=0, high=1, shape=self.obs_shape, dtype=np.float16)
        self.max_steps = max_steps
        self.window = 60
        self.reset()

    def _next_observation(self):
        scaler = preprocessing.MinMaxScaler()

        features = self.stationary_df[self.stationary_df.columns.difference(['Date'])]

        scaled = features[:self.current_step + self.forecast_len + 1].values
        scaled[abs(scaled) == inf] = 0
        scaled = scaler.fit_transform(scaled.astype('float32'))
        scaled = pd.DataFrame(scaled, columns=features.columns)

        obs = scaled.values[-1]

        past_df = self.stationary_df['Close'][: self.current_step + self.forecast_len + 1]
        forecast_model = SARIMAX(past_df.values, enforce_stationarity=False, simple_differencing=True)
        model_fit = forecast_model.fit(method='bfgs', disp=False)
        forecast = model_fit.get_forecast(steps=self.forecast_len, alpha=(1 - self.confidence_interval))

        obs = np.insert(obs, len(obs), forecast.predicted_mean, axis=0)
        obs = np.insert(obs, len(obs), forecast.conf_int().flatten(), axis=0)

        scaled_history = scaler.fit_transform(self.account_history.astype('float32'))

        obs = np.insert(obs, len(obs), scaled_history[:, -1], axis=0)

        obs = np.reshape(obs.astype('float16'), self.obs_shape)
        obs[np.bitwise_not(np.isfinite(obs))] = 0

        return obs

    def _current_price(self):
        return self.df['Close'].values[self.current_step + self.forecast_len] + 0.01

    def _take_action(self, action):
        current_price = self._current_price()
        action_type = int(action / 4)
        amount = 1 / (action % 4 + 1)

        btc_bought = 0
        btc_sold = 0
        cost = 0
        sales = 0

        if action_type == 0:
            price = current_price * (1 + self.commission)
            btc_bought = min(self.balance * amount /
                             price, self.balance / price)
            cost = btc_bought * price

            self.btc_held += btc_bought
            self.balance -= cost
        elif action_type == 1:
            price = current_price * (1 - self.commission)
            btc_sold = self.btc_held * amount
            sales = btc_sold * price

            self.btc_held -= btc_sold
            self.balance += sales

        if btc_sold > 0 or btc_bought > 0:
            self.trades.append({'step': self.current_step,
                                'amount': btc_sold if btc_sold > 0 else btc_bought, 'total': sales if btc_sold > 0 else cost,
                                'type': 'sell' if btc_sold > 0 else 'buy'})

        self.net_worths.append(
            self.balance + self.btc_held * current_price)

        self.account_history = np.append(self.account_history, [
            [self.balance],
            [btc_bought],
            [cost],
            [btc_sold],
            [sales]
        ], axis=1)

    def _reward(self):
        length = min(self.current_step, self.forecast_len)
        returns = np.diff(self.net_worths[-length:])

        if np.count_nonzero(returns) < 1:
            return 0

        if self.reward_func == 'sortino':
            reward = sortino_ratio(returns)                  # Change annualization ????????????????????????????????
        elif self.reward_func == 'calmar':
            reward = calmar_ratio(returns)
        elif self.reward_func == 'omega':
            reward = omega_ratio(returns)
        else:
            reward = returns[-1]

        return reward if np.isfinite(reward) else 0

    def _done(self):
        return self.net_worths[-1] < self.initial_balance / 10 or self.current_step == self.max_steps+self.first_step\
               - self.forecast_len - 1

    def reset(self):
        curr_df = random.choice(list(self.data.keys()))
        self.df = self.data[curr_df][self.mode]['df']
        self.stationary_df = self.data[curr_df][self.mode]['stationary_df']
        self.benchmarks = self.data[curr_df][self.mode]['benchmarks']
        self.stock_name = curr_df

        self.balance = self.initial_balance
        self.net_worths = [self.initial_balance]
        self.btc_held = 0

        self.current_step = np.random.randint(0, len(self.df) - self.max_steps - 2)

        self.account_history = np.array([[self.balance], [0], [0], [0], [0]])
        self.first_step = self.current_step
        self.trades = []
        if not self.viewer is None:
            self.viewer.update_df(self.df, self.stock_name)
        return self._next_observation()

    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        obs = self._next_observation()
        reward = self._reward()
        done = self._done()

        return obs, reward, done, {}

    def render(self, mode='human'):
        if mode == 'system':
            print('Price: ' + str(self._current_price()))
            print('Bought: ' + str(self.account_history[2][self.current_step]))
            print('Sold: ' + str(self.account_history[4][self.current_step]))
            print('Net worth: ' + str(self.net_worths[-1]))

        elif mode == 'human':
            if self.viewer is None:
                self.viewer = TradingGraph(self.df, self.stock_name)
            self.viewer.render(self.current_step, self.net_worths, self.benchmarks, self.trades, self.first_step)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

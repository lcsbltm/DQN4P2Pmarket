import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)

TD_DURATION = 3

STEP_SIZE = 5*60 # In seconds
SESSION_DURATION = TD_DURATION*60*60/STEP_SIZE # In steps
PRODUCT_DURATION = 5*60/STEP_SIZE # In steps
MAX_SESSION_QUANTITY = 7
MAX_SESSION_PRICE = 100
MIN_SESSION_PRICE = 0
BOUNDARY = 7.2
CONSTANT_ORDER = 2*BOUNDARY/SESSION_DURATION
MULTIPLIER = 0.05*(TD_DURATION/24)
INITIAL_PRICE = (MAX_SESSION_PRICE+MIN_SESSION_PRICE)/2

class TradingSession(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, action_space_config = 'continous', num_mutual_sessions = 3, dyn_control = 'random', pref_prod = 2, pref_const = 2):
        super(TradingSession, self).__init__()
        # Settings:
        self.pref_prod = pref_prod
        self.pref_const = pref_const
        self.num_mutual_sessions = num_mutual_sessions
        self.action_space_config = action_space_config
        self.max_session_price = MAX_SESSION_PRICE
        self.min_session_price = MIN_SESSION_PRICE
        self.initial_price = INITIAL_PRICE
        self.dyn_control = dyn_control
        self.max_session_quantity = MAX_SESSION_QUANTITY
        self.pos_multiplier = MULTIPLIER
        self.neg_multiplier = -MULTIPLIER
        # Definition of action space:
        if self.action_space_config  == 'continous':
            self.action_space = spaces.Box(low=0, high=1, shape=(self.num_mutual_sessions,), dtype=np.float32)
        elif self.action_space_config  == 'discrete':
            self.constant_order = CONSTANT_ORDER
            self.action_space = spaces.Discrete(self.num_mutual_sessions + 1)
        # Definition of observation space:
        low_space = np.array([0]*self.num_mutual_sessions + [0] + [0])
        high_space = np.array([1]*self.num_mutual_sessions + [self.max_session_quantity*self.num_mutual_sessions/BOUNDARY] + [1])
        self.observation_space = spaces.Box(low=low_space, high=high_space, dtype=np.float32)

    def step(self, action):
        '''
        Executes one time step in the env.
        '''
        # Execute one time step within the environment
        self.current_step += 1
        self.reward = 0
        self._take_action(action)
        reward = self._compute_reward(action)
        obs = self._next_observation()
        done = self._check_if_done()
        if done:
            self.optimal_strategy_reward = self._compute_optimal_strategy_reward()
            return obs, reward, done, {}
        return obs, reward, done, {}

    def reset(self):
        '''
        Reset env to initial conditions.
        '''
        self.sessions_completed = 0
        self.current_step = 0
        self.reward = 0
        self.session_prices = np.full(self.num_mutual_sessions, self.initial_price)
        self._set_case_control()
        self.tendency = self.initial_price
        #for i in range(100):
        #    self._sim_price_dynamics()
        self.session_quantities = np.full(self.num_mutual_sessions, self.max_session_quantity, dtype='float')
        self.session_steps_left = np.arange(SESSION_DURATION, (SESSION_DURATION-self.num_mutual_sessions*PRODUCT_DURATION), -PRODUCT_DURATION)
        self.holdings_quantity = np.zeros(self.num_mutual_sessions, dtype='float')
        self.holdings_quantity_total = 0.0
        self.holdings_cash = np.zeros(self.num_mutual_sessions, dtype='float')
        self.boundary = BOUNDARY
        self.session_idx = None
        self.max_rewards = np.zeros(self.num_mutual_sessions, dtype='float')
        self.optimal_min_prices = np.zeros(int(self.boundary/self.constant_order))
        self.optimal_strategy_reward = 0
        return np.hstack([self.session_prices/self.max_session_price, self.holdings_quantity_total/self.boundary, self.current_step/SESSION_DURATION])

    def _take_action(self, action):
        '''
        Place agent's order and update holdings
        '''
        self.holdings_quantity_previous = self.holdings_quantity.copy()

        if self.action_space_config == 'discrete':
            self.session_idx = action
            action = np.zeros(len(self.session_prices))

            if self.session_idx < len(self.session_prices):
                action[self.session_idx] = self.constant_order

            action_times_quantity = action

        else:
            action_times_quantity = np.multiply(action, self.session_quantities)

        self.holdings_quantity += action_times_quantity
        self.holdings_cash += np.multiply(action_times_quantity, self.session_prices)

        self.holdings_quantity_total = np.sum(self.holdings_quantity)

        for idx in range(self.num_mutual_sessions):
            if action_times_quantity[idx] > 0:
                self.session_quantities[idx] -= action_times_quantity[idx]
            elif action_times_quantity[idx] < 0:
                self.session_quantities[idx] += action_times_quantity[idx]

    def _next_observation(self):
        '''
        Update env and returns formated version of next observation.
        '''
        self._update_session_prices()
        self._update_session_steps_left()
        obs = np.hstack([self.session_prices/self.max_session_price, self.holdings_quantity_total/self.boundary, self.current_step/SESSION_DURATION])
        return obs

    def _update_session_prices(self):
        '''
        Update the price of trading sessions.
        '''
        self._sim_price_dynamics()

        neg_prices_idx = np.argwhere(self.session_prices < self.min_session_price)
        max_prices_idx = np.argwhere(self.session_prices >= self.max_session_price)

        for idx in neg_prices_idx:
            self.session_prices[neg_prices_idx] = self.min_session_price
        for idx in max_prices_idx:
            self.session_prices[max_prices_idx] = self.max_session_price

        self.max_rewards = np.hstack([self.max_rewards, self._compute_all_rewards(self.session_prices).max()])

    def _compute_all_rewards(self, prices):
        all_rewards = []
        for action, _ in enumerate(prices):
            all_rewards.append(self._calculate_reward(prices, action))
        return np.array(all_rewards)

    def _update_session_steps_left(self):
        '''
        Update the progress of trading sessions.
        '''
        self.session_steps_left -= 1

        completed_idx = np.argwhere(self.session_steps_left == 0)

        for idx in completed_idx:
            self._complete_session(idx)

    def _complete_session(self, idx):
        self.sessions_completed +=1
        self.session_steps_left[idx] = 0
        self.session_quantities[idx] = 0

    def _calculate_reward(self, prices, action):
        prices_norm = prices/self.max_session_price
        price_paid = prices_norm[action]

        if action == self.pref_prod:
            return self.pref_const * (1/np.exp(price_paid) - price_paid/np.exp(1))
        else:
            return (1/np.exp(price_paid) - price_paid/np.exp(1))

    def _compute_optimal_strategy_reward(self):
        if self.max_rewards.shape[0] < (int(self.boundary/self.constant_order)):
            return 0
        else:
            return np.sum(np.sort(self.max_rewards)[::-1][0:int(self.boundary/self.constant_order)])

    def _compute_reward(self, action):
        delta_forecast_previous = np.absolute(self.boundary - np.sum(self.holdings_quantity_previous))
        delta_forecast_updated = np.absolute(self.boundary - np.sum(self.holdings_quantity))

        if delta_forecast_previous == delta_forecast_updated:
            self.reward = 0
            return self.reward
        elif delta_forecast_previous < delta_forecast_updated:
            self.reward = -0.1
            return self.reward
        else:
            self.reward = self._calculate_reward(self.session_prices, action)
            return self.reward

    def _check_if_done(self):
        '''
        Check if episode is done.
        '''
        return self.sessions_completed >= self.num_mutual_sessions

    def _sim_price_dynamics(self):
        if self.case_control == 0:
            self.tendency += (24/TD_DURATION)*0.15
        elif self.case_control == 1:
            self.tendency -= (24/TD_DURATION)*0.15
        elif self.case_control == 2:
            self.tendency = self.tendency + (24/TD_DURATION)*(4*np.sin(2.5*np.pi*self.current_step/SESSION_DURATION)*0.02)
        elif self.case_control == 3:
            self.tendency = self.tendency - (24/TD_DURATION)*(4*np.sin(2.5*np.pi*self.current_step/SESSION_DURATION)*0.02)
        elif self.case_control == 4:
            self.tendency = self.tendency
        for i in range(len(self.session_prices)):
            if i == 0:
                self.session_prices[i] = self._ou_process(self.session_prices[i],
                                                               ou_theta = 10e-3,
                                                               ou_mu = self.tendency,
                                                               dt = 24/TD_DURATION,
                                                               ou_sigma = 1)
            else:
                self.session_prices[i] = self._ou_process(self.session_prices[i],
                                                               ou_theta = 10e-3,
                                                               ou_mu = self.session_prices[i-1],
                                                               dt = 24/TD_DURATION,
                                                               ou_sigma = 1)

    def _ou_process(self, x, ou_theta, ou_mu, dt, ou_sigma):
        noise_term = (ou_sigma/np.sqrt(1/dt))*np.random.randn()
        main_term = (ou_theta * (ou_mu - x) * dt)
        dx = main_term + noise_term
        return x + dx

    def _set_case_control(self):
        if self.dyn_control == 'plus':
            self.case_control = 0
        elif self.dyn_control == 'minus':
            self.case_control = 1
        elif self.dyn_control == 'plus_sin':
            self.case_control = 2
        elif self.dyn_control == 'minus_sin':
            self.case_control = 3
        elif self.dyn_control == 'constant':
            self.case_control = 4

## EXTERNAL METHODS:
    def get_current_state(self):
        return np.hstack([self.session_prices/self.max_session_price, self.holdings_quantity_total/self.boundary, self.current_step/SESSION_DURATION])
    def render(self, mode='human', close=False):
        pass

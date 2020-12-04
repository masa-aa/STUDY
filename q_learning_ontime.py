import math
from collections import defaultdict
import gym
from el_agent import ELAgent
import Kyoto_env_ontime
from get_data import get_time, get_happiness
import numpy as np


class QLearningAgent(ELAgent):

    def __init__(self, epsilon=0.1, Q={}):
        super().__init__(epsilon, Q)

    def learn(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0] * len(actions))
        for e in range(episode_count):
            s = env.reset()
            done = False
            sum_reward = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s, actions)
                n_state, reward, done, _ = env.step(a)

                # 報酬+割引率*遷移先の価値
                # 遷移先の価値はValueベース
                gain = reward + gamma * max(self.Q[n_state])
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = n_state
                sum_reward += reward

            else:
                self.log(sum_reward)

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e, interval=report_interval)


def train(episode_count=10, report_interval=1):
    d = get_time()
    happiness = np.array(get_happiness("中国")) * 100
    time_limit = 300
    agent = QLearningAgent(epsilon=0.01)
    env = gym.make("Kyoto_ontime-v0", n=25, start=0, goal=24, happiness=happiness, time_limit=time_limit, d=d)
    agent.learn(env, episode_count=episode_count, report_interval=report_interval)
    # show_q_value(agent.Q)
    agent.show_reward_log(interval=report_interval)


if __name__ == "__main__":
    train(episode_count=1000, report_interval=1)

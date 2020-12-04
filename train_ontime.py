import gym
import Kyoto_env_ontime
from get_data import get_time, get_happiness
import numpy as np
from monte_carlo_ontime import MonteCarloAgent
from q_learning_ontime import QLearningAgent
from play import play


def make_env(n=25, start=12, goal=6, country="中国"):
    """標準は京都駅スタート, 祇園解散"""
    d = get_time(stay=30)
    happiness = get_happiness(country) * 100
    time_limit = 300
    env = gym.make("Kyoto_ontime-v0",
                   n=n, start=start, goal=goal,
                   happiness=happiness,
                   time_limit=time_limit,
                   d=d)
    return env


def train(Agent=MonteCarloAgent,
          episode_count=10,
          epsilon=0.1,
          Q={},
          report_interval=1,
          country="中国",
          save=False):

    agent = Agent(epsilon=epsilon, Q=Q)
    env = make_env(country=country)
    agent.learn(env, episode_count=episode_count, report_interval=report_interval, show_log=False)
    # show_q_value(agent.Q)
    if not save:
        agent.show_reward_log(interval=report_interval)
    else:
        Q = agent.Q
        return Q


if __name__ == "__main__":
    Q = train(Agent=MonteCarloAgent, episode_count=2000000, epsilon=0.1, report_interval=1000, country="中国", save=True)
    reward, route = play(make_env(country="中国"), Q, show_mode=True)
    print(reward)
    print(route)
# report Qでepsilon小さくしすぎると壊れる

import gym
import Kyoto_env_ontime
from get_data import get_time, get_happiness
import numpy as np
from monte_carlo_ontime import MonteCarloAgent
from q_learning_ontime import QLearningAgent
from play import play
from train_ontime import train, make_env
from multiprocessing import Pool, cpu_count


def _train(a):
    episode_count, epsilon = a
    return train(Agent=MonteCarloAgent,
                 episode_count=episode_count,
                 epsilon=epsilon,
                 report_interval=1000,
                 country="中国",
                 save=True)


def _play(Q):
    return play(make_env(country="中国"), Q, show_mode=True)


if __name__ == "__main__":
    n = 8
    p = Pool(cpu_count())
    Q = p.map(_train, [(200000, 0.1)] * n)
    rewards = p.map(_play, Q)
    print(rewards)
    p.close()

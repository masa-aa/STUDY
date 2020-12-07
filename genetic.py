import gym
import Kyoto_env_ontime
from get_data import get_time, get_happiness
import numpy as np
from monte_carlo_ontime import MonteCarloAgent
from q_learning_ontime import QLearningAgent
from play import play
from train_ontime import train, make_env
from multiprocessing import Pool, cpu_count
from time import time
from collections import defaultdict


def _train(a):
    episode_count, epsilon, Q, country = a
    _q = train(Agent=MonteCarloAgent,
               episode_count=episode_count,
               epsilon=epsilon,
               Q=Q,
               report_interval=1000,
               country=country,
               save=True)
    return (play(make_env(country=country), _q, show_mode=True)[0], _q)


def multi_train(n, Qs, episode_count=2000000, epsilon=0.1, country="中国"):
    p = Pool(cpu_count())
    results = p.map(_train, [(episode_count, epsilon, Qs[i], country) for i in range(n)])
    p.close()
    return results


def new_gen():
    return


def _one_cpu(n):
    rewards = []
    for _ in range(n):
        rewards.append(_train((2000000, 0.1, "中国")))
    print(rewards)


if __name__ == "__main__":
    n = 96
    country = "中国"
    rand = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
    env = make_env(country=country)
    Qs = [{} for _ in range(n)]
    t = time()
    results = multi_train(n, Qs, 6000000, np.random.choice(rand), country)
    t = int(time() - t)
    print("{}h {}m {}s".format(t // 3600, (t % 3600) // 60, t % 60))
    results.sort(key=lambda x: x[0], reverse=1)
    # Qs = [v[1] for v in results]
    print([int(v[0]) for v in results])
    for score, q in results[:10]:
        score, route = play(env, q, show_mode=True)
        print("score:{}, route:{}".format(score, route))

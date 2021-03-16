import gym
import numpy as np
from monte_carlo_ontime import MonteCarloAgent
from q_learning_ontime import QLearningAgent
from play import play
from train_ontime import train, make_env
from multiprocessing import Pool, cpu_count
from time import time
from collections import defaultdict
from TSP import convert
cnt = 0


def _train(a):
    """multi_trainに渡す用の関数"""
    global cnt
    episode_count, epsilon, country = a
    print("start:{}".format(cnt))
    _q = train(Agent=MonteCarloAgent,
               episode_count=episode_count,
               epsilon=epsilon,
               report_interval=1000000,
               country=country,
               save=True)
    print("fininsh:{}".format(cnt))
    cnt += 1

    return (play(make_env(country=country), _q, show_mode=0)[0], _q)


def multi_train(n, episode_count=2000000, country="中国"):
    """複数のcpuを用いて学習する．"""
    rand = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
    p = Pool(cpu_count())
    results = p.map(_train, [(episode_count, np.random.choice(rand), country) for i in range(n)])
    p.close()
    return results


def _one_cpu(n):
    """test用の関数"""
    rewards = []
    for _ in range(n):
        rewards.append(_train((2000000, 0.1, "中国")))
    print(rewards)


def main_train(country="中国", num_loop=48, train_loop=1000000):
    env = make_env(country=country)
    t = time()
    results = multi_train(num_loop, train_loop, country)
    t = int(time() - t)
    print("{}h {}m {}s".format(t // 3600, (t % 3600) // 60, t % 60))
    results.sort(key=lambda x: x[0], reverse=1)
    # Qs = [v[1] for v in results]
    routes = set()
    unique_results = []
    for score, q in results:
        score, route = play(env, q, show_mode=2)
        route = convert(env.d, tuple(route))

        if route in routes:
            continue
        unique_results.append((score, route))
        routes.add(route)
    unique_results.sort(key=lambda x: x[0], reverse=True)
    # for score, route in unique_results:
    #     print("score:{}, route:{}".format(score, route))
    return unique_results


def main_train_Q(country="中国", num_loop=48, train_loop=1000000):
    env = make_env(country=country)
    t = time()
    results = multi_train(num_loop, train_loop, country)
    t = int(time() - t)
    print("{}h {}m {}s".format(t // 3600, (t % 3600) // 60, t % 60))
    results.sort(key=lambda x: x[0], reverse=1)
    # Qs = [v[1] for v in results]
    routes = set()
    unique_results = []
    for score, q in results:
        score, route = play(env, q, show_mode=2)
        route = convert(env.d, tuple(route))

        if route in routes:
            continue
        unique_results.append((score, q))
        routes.add(route)
    unique_results.sort(key=lambda x: x[0], reverse=True)
    # for score, route in unique_results:
    #     print("score:{}, route:{}".format(score, route))
    return unique_results


if __name__ == "__main__":
    print(main_train(num_loop=16, train_loop=1_000_000)[0][1])

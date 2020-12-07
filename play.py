import gym
import Kyoto_env_ontime
from get_data import get_time, get_happiness, get_spots
import numpy as np


def play(env, Q, show_mode=False):

    s = env.reset()
    actions = list(range(env.action_space.n))
    done = False
    sum_reward = 0
    reward = 0
    experiece = [env.pos]
    while not done:
        if s not in Q:
            a = np.random.randint(len(actions))
        else:
            a = np.argmax(Q[s])
        n_state, reward, done, _ = env.step(a)
        s = n_state
        experiece.append(s[0])
        sum_reward += reward

    if show_mode:
        g = get_spots()

    return (sum_reward, " -> ".join(map(lambda x: g[x], experiece))) if show_mode \
        else (sum_reward, experiece)

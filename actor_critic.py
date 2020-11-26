import math
from collections import defaultdict
import gym
from el_agent import ELAgent
import Kyoto_env
from get_data import get_time, get_happiness
import numpy as np


class Actor(ELAgent):

    def __init__(self, env):
        super().__init__(epsilon=-1)
        nrow = env.observation_space.n
        ncol = env.action_space.n
        self.actions = list(range(env.action_space.n))
        self.Q = np.random.uniform(0, 1, nrow * ncol).reshape((nrow, ncol))

    def softmax(self, x):
        """softmaxに従い行動する."""
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, s):
        a = np.random.choice(self.actions, 1,
                             p=self.softmax(self.Q[s]))
        return a[0]


class Critic():
    """Vは状態価値"""

    def __init__(self, env):
        states = env.observation_space.n
        self.V = np.zeros(states)


class ActorCritic():

    def __init__(self, actor_class, critic_class):
        self.actor_class = actor_class
        self.critic_class = critic_class

    def train(self, env, episode_count=1000, gamma=0.9,
              learning_rate=0.1, render=False, report_interval=50):
        actor = self.actor_class(env)
        critic = self.critic_class(env)

        actor.init_log()
        for e in range(episode_count):
            s = env.reset()
            done = False
            while not done:
                if render:  # 描画
                    env.render()
                a = actor.policy(s)
                n_state, reward, done, _ = env.step(a)

                # criticの評価値を使ってgainを計算
                gain = reward + gamma * critic.V[n_state]
                estimated = critic.V[s]
                td = gain - estimated
                # TD誤差をactor, criticに反映
                actor.Q[s][a] += learning_rate * td
                critic.V[s] += learning_rate * td
                s = n_state

            else:
                actor.log(reward)

            if e != 0 and e % report_interval == 0:
                actor.show_reward_log(episode=e)

        return actor, critic


def train(episode_count=10, report_interval=1):
    d = get_time()
    happiness = np.array(get_happiness("中国")) * 100
    time_limit = 300
    trainer = ActorCritic(Actor, Critic)
    env = gym.make("Kyoto-v0", n=25, start=0, goal=24, happiness=happiness, time_limit=time_limit, d=d)
    actor, _ = trainer.train(env, episode_count=episode_count, report_interval=report_interval)

    # show_q_value(agent.Q)
    actor.show_reward_log(interval=report_interval)


if __name__ == "__main__":
    train(episode_count=100000, report_interval=4000)

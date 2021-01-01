import math
from collections import defaultdict
from el_agent import ELAgent
import numpy as np


class MonteCarloAgent(ELAgent):

    def __init__(self, env, epsilon=0.1):
        super().__init__(epsilon, env)

    def learn(self, env, episode_count=1000, gamma=0.9,
              render=False, report_interval=50, show_log=True):
        self.init_log()
        actions = list(range(env.action_space.n))
        if not self.Q:
            self.Q = defaultdict(lambda: [0] * len(actions))
        N = defaultdict(lambda: [0] * len(actions))

        for e in range(episode_count):
            s = env.reset()  # 初期化して環境を取得
            done = False
            # Play until the end of episode.
            experience = []
            sum_reward = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s, actions)
                n_state, reward, done, _ = env.step(a)
                experience.append({"state": s, "action": a, "reward": reward})
                s = n_state
                sum_reward += reward
            else:
                self.log(sum_reward)

            # Evaluate each state, action.
            for i, x in enumerate(experience):
                s, a = x["state"], x["action"]

                # Calculate discounted future reward of s.
                G, t = 0, 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                N[s][a] += 1  # count of s, a pair
                alpha = 1 / N[s][a]
                # 更新式
                self.Q[s][a] += alpha * (G - self.Q[s][a])
            if show_log and e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e, interval=report_interval)

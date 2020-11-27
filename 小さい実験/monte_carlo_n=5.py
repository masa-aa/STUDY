import math
from collections import defaultdict
import gym
from el_agent import ELAgent
import Kyoto_env


class MonteCarloAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

    def learn(self, env, episode_count=1000, gamma=0.9,
              render=False, report_interval=50):
        self.init_log()
        actions = list(range(env.action_space.n))
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

            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train():
    d = [[0, 1, 2, 3, 800],
         [1, 0, 3, 4, 7],
         [2, 3, 0, 5, 8],
         [3, 4, 5, 0, 9],
         [8, 7, 8, 9, 0]]
    happiness = [0, 1, 3, 6, 0]
    time_limit = 10
    agent = MonteCarloAgent(epsilon=0.1)
    env = gym.make("Kyoto-v0", n=5, start=0, goal=4, happiness=happiness, time_limit=time_limit, d=d)
    agent.learn(env, episode_count=1000)
    # show_q_value(agent.Q)
    agent.show_reward_log()


if __name__ == "__main__":

    train()

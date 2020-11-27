from collections import defaultdict
import gym
from el_agent import ELAgent
import Kyoto_env


class QLearningAgent(ELAgent):

    def __init__(self, epsilon=0.1):
        super().__init__(epsilon)

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
                self.show_reward_log(episode=e)


def train():
    d = [[0, 1, 2, 3, 8],
         [1, 0, 3, 4, 7],
         [2, 3, 0, 5, 8],
         [3, 4, 5, 0, 9],
         [8, 7, 8, 9, 0]]
    happiness = [0, 2, 6, 12, 0]
    time_limit = 10
    agent = QLearningAgent(epsilon=0.1)
    env = gym.make("Kyoto-v0", n=5, start=0, goal=4, happiness=happiness, time_limit=time_limit, d=d)
    agent.learn(env, episode_count=1000)
    # show_q_value(agent.Q)
    agent.show_reward_log()


if __name__ == "__main__":

    train()

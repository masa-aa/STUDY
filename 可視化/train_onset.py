import gym
import Kyoto_env_onset
from get_data import get_time, get_happiness
import numpy as np
from monte_carlo_onset import MonteCarloAgent
from q_learning_onset import QLearningAgent


def train(Agent=MonteCarloAgent, episode_count=10, report_interval=1, country="中国"):
    d = get_time()
    happiness = np.array(get_happiness(country)) * 100
    time_limit = 300
    agent = Agent(epsilon=0.1)
    env = gym.make("Kyoto_onset-v0", n=25, start=0, goal=24, happiness=happiness, time_limit=time_limit, d=d)
    agent.learn(env, episode_count=episode_count, report_interval=report_interval)
    # show_q_value(agent.Q)
    agent.show_reward_log(interval=report_interval)


if __name__ == "__main__":
    train(Agent=QLearningAgent, episode_count=10000000, report_interval=200000, country="中国")

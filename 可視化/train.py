import gym
import Kyoto_env
from get_data import get_time, get_happiness
import numpy as np
from monte_carlo import MonteCarloAgent
from q_learning import QLearningAgent


def train(Agent=MonteCarloAgent, episode_count=10, report_interval=1, country="中国"):
    d = get_time()
    happiness = np.array(get_happiness(country)) * 100
    time_limit = 300
    agent = Agent(epsilon=0.03)
    env = gym.make("Kyoto-v0", n=25, start=0, goal=24, happiness=happiness, time_limit=time_limit, d=d)
    agent.learn(env, episode_count=episode_count, report_interval=report_interval)
    # show_q_value(agent.Q)
    agent.show_reward_log(interval=report_interval)


if __name__ == "__main__":
    train(Agent=MonteCarloAgent, episode_count=20000, report_interval=1000, country="中国")

import sys
import gym
import numpy as np
import gym.spaces


class Kyoto(gym.Env):
    def __init__(
            self, n: "頂点数",
            start: "始点",
            goal: "終点",
            happiness: "幸福度のリスト",
            time_limit: "時間制限",
            d: "隣接行列"):
        super().__init__()
        self.n = n
        self.start = start
        self.goal = goal
        self.happiness = happiness
        self.time_limit = time_limit
        self.d = d
        self.observation_space = gym.spaces.Box(low=0, high=n, shape=(n,))
        self.action_space = gym.spaces.Discrete(n)
        self.reward_range = [-1., 100.]
        self.reset()

    def reset(self):
        # 諸々の変数を初期化する
        self.pos = self.start
        self.done = False
        self.steps = 0
        self.time = 0
        self.use = [0] * self.n
        self.use[self.start] = 1
        return self._observe()

    def _render(self):
        pass

    def _close(self):
        pass

    def _seed(self, seed=None):
        pass

    def step(self, action):
        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)
        next_pos = action

        if self._is_movable(next_pos):
            self.time += self.d[self.pos][next_pos]
            self.use[next_pos] = 1
            self.pos = next_pos
            moved = True
        else:
            moved = False

        observation = self._observe()
        if self._is_game_over():
            reward = -100
        else:
            reward = self._get_reward(self.pos, moved)
        self.done = self._is_done()
        return observation, reward, self.done, {}

    def _is_movable(self, next_pos):
        """合法手か否か. use[pos]なら既に行った場所なので非合法"""
        return (1 - self.use[next_pos])

    def _get_reward(self, pos, moved):
        """報酬を返す. 再考の余地あり"""
        if not moved:
            return -1
        if self.goal == pos:
            return 0
        return self.happiness[pos]

    def _observe(self):
        return self.pos

    def _is_done(self):
        """時間切れかゴールにたどり着くとdone"""
        return self.pos == self.goal or self.time_limit < self.time

    def _is_game_over(self):
        return self.time_limit < self.time

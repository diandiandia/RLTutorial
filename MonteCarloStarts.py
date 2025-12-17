import gymnasium as gym
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class MCExploringStarts:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.gamma = gamma

        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.N = defaultdict(lambda: np.zeros(env.action_space.n))
        self.pi = defaultdict(lambda: env.action_space.sample())  # 初始随机策略

        # 预定义非终止状态（避免硬编码15）
        self.non_terminal_states = [s for s in range(env.observation_space.n)
                                    if s != 15]  # 假设15是目标状态

    def pi_action(self, state):
        return self.pi[state]

    def generate_episode(self):
        # Exploring Starts
        state = np.random.choice(self.non_terminal_states)
        action = self.env.action_space.sample()

        # 直接设置状态并执行初始动作
        self.env.reset()
        self.env.unwrapped.s = state
        next_state, reward, terminated, truncated, _ = self.env.step(action)

        episode = [(state, action, reward)]

        state = next_state
        done = terminated or truncated
        while not done:
            action = self.pi_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(
                action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        return episode

    def update(self, episode):
        G = 0.0
        visited = set()

        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            key = (state, action)
            if key not in visited:
                visited.add(key)
                self.N[state][action] += 1
                self.Q[state][action] += (G - self.Q[state]
                                          [action]) / self.N[state][action]

        # episode结束后统一改进策略
        for state, _ in visited:
            self.pi[state] = np.argmax(self.Q[state])  # 或加随机打破平局

    def print_tables(self):
        print("\n===== Q Table =====")
        for state, actions in self.Q.items():
            for action, value in enumerate(actions):
                print(
                    f'q_table: {state} -> {action} -> {self.Q[state][action]}')
        print("\n===== N Table =====")
        for state, actions in self.N.items():
            for action, count in enumerate(actions):
                print(f'N: {state} -> {action} -> {self.N[state][action]}')

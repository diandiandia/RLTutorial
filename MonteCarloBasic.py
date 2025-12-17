import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class MonteCarloBasic:
    def __init__(self, env, gamma, seed=10):
        self.env = env
        self.gamma = gamma
        self.seed = seed
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.c_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.mean_q_table = defaultdict(lambda: np.zeros(env.action_space.n))


    def choose_action(self):
        return self.env.action_space.sample()
    
    def generate_episode(self):
        episode = []
        obs, _ = self.env.reset(seed=self.seed)
        done = False
        while not done:
            action = self.choose_action()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((obs, action, reward))
            done = terminated or truncated
            obs = next_obs
        return episode
    
    def update(self, episode, mode='first_visit'):
        G = 0
        visited = set() if mode == 'first_visit' else None
        for state, action, reward in reversed(episode):
            G = reward + self.gamma * G
            if mode == 'first_visit':   
                if (state, action) not in visited:
                    visited.add((state, action))
                    self.q_table[state][action] += G
                    self.c_table[state][action] += 1
                    self.mean_q_table[state][action] = self.q_table[state][action] / self.c_table[state][action]
            if mode == 'every_visit':
                self.q_table[state][action] += G
                self.c_table[state][action] += 1
                self.mean_q_table[state][action] = self.q_table[state][action] / self.c_table[state][action]

    def print_tables(self):
        print("\n===== Q Table =====")
        for state, actions in self.q_table.items():
            for action, value in enumerate(actions):
                print(f'q_table: {state} -> {action} -> {self.q_table[state][action]}')
        
        print("\n===== C Table =====")
        for state, actions in self.c_table.items():
            for action, count in enumerate(actions):
                print(f'c_table: {state} -> {action} -> {self.c_table[state][action]}')
        
        print("\n===== Mean Q Table =====")
        for state, actions in self.mean_q_table.items():
            for action, value in enumerate(actions):
                print(f'mean_q_table: {state} -> {action} -> {self.mean_q_table[state][action]}')





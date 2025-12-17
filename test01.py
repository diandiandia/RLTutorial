from MonteCarloStarts import MCExploringStarts
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm


custom_map = [
    'SFFFF',
    'FFFFF',
    'FFFFF',
    'FFFFF',
    'FFFFG'
]

def main():
    env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=False, render_mode='ansi')
    monte_carlo = MCExploringStarts(env, gamma=0.9)
    num_episodes = 10000
    returns_list = []

    for _ in tqdm(range(num_episodes)):
        episode = monte_carlo.generate_episode()
        monte_carlo.update(episode)
        total_reward = sum(reward for _, _, reward in episode)
        returns_list.append(total_reward)

    monte_carlo.print_tables()




if __name__ == "__main__":
    main()
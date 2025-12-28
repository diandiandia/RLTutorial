import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import collections
import numpy as np
from typing import NamedTuple
import torch
import random
import tqdm


class Params(NamedTuple):
    state_dim: int
    hidden_dim: int
    action_dim: int
    learning_rate: float
    gamma: float
    epsilon: float
    epsilon_decay: float
    epsilon_end: float
    target_update: int
    device: torch.device

class DQNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.network(x)
    
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen= self.buffer_size)

    def push(self, transition_tuple):
        self.buffer.append(transition_tuple)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough elements in the buffer to sample")

        transitions = random.sample(self.buffer, batch_size)
        state, action, rewards, next_state, done = zip(*transitions)
        return np.array(state), action, rewards, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)
    

class DQNLearning:
    def __init__(self, env:gym.Env, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, epsilon_decay, epsilon_end, target_update, device):
        self.env = env
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.device = device

        self.q_network = DQNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.target_network = DQNetwork(state_dim, hidden_dim, action_dim).to(device)

        self.q_network_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.loss_fn = nn.MSELoss()

    def update(self, transition_dict):
        # Convert states and next_states to numpy arrays before creating tensors
        states_np = np.array(transition_dict['states'])
        next_states_np = np.array(transition_dict['next_states'])
        
        tensor_states = torch.tensor(states_np, dtype=torch.float32).to(self.device)
        # Use int64 for actions as they are indices
        tensor_actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).unsqueeze(1).to(self.device)
        tensor_rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).unsqueeze(1).to(self.device)
        tensor_next_states = torch.tensor(next_states_np, dtype=torch.float32).to(self.device)
        tensor_dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).unsqueeze(1).to(self.device)

        q_values = self.q_network(tensor_states).gather(1, tensor_actions)
        max_next_q_values = self.target_network(tensor_next_states).max(1)[0].unsqueeze(1)
        q_targets = tensor_rewards + self.gamma * max_next_q_values * (1 - tensor_dones)
        dqn_loss = self.loss_fn(q_values, q_targets)
        self.q_network_optimizer.zero_grad()
        dqn_loss.backward()
        self.q_network_optimizer.step()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(tensor_state)
            return q_values.argmax().item()
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        """更新目标网络，将Q网络的权复制制到目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        

def main():
    # Use discrete action space environment instead of continuous
    env = gym.make("MountainCar-v0")
    params = Params(
        state_dim=env.observation_space.shape[0],
        hidden_dim=128,
        action_dim=env.action_space.n,  # Use n for discrete action spaces
        learning_rate=1e-3,
        gamma=0.99,  # 折扣因子，接近1以鼓励长期奖励
        epsilon=1.0,
        epsilon_decay=0.995,  # 探索率衰减
        epsilon_end=0.01,
        target_update=10,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    agent = DQNLearning(env, **params._asdict())
    buffer = ReplayBuffer(buffer_size=5000)

    num_episodes = 500  # 增加训练轮数以确保收敛
    batch_size = 64
    best_reward = -float('inf')
    
    for episode in tqdm.tqdm(range(num_episodes)):
        state, _ = env.reset()  # 不固定seed，让起始位置有更多变化
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1

            # 奖励塑造：更加强调速度和位置
            reward_shaped = reward
            
            # 1. 奖励速度大小（取绝对值），速度越快奖励越高
            speed_abs = abs(next_state[1])
            reward_shaped += speed_abs * 100  # 速度奖励，权重10
        
                
            # 3. 到达目标给予巨大奖励
            if next_state[0] >= 0.5:
                reward_shaped += 10000
            
            transition_tuple = (state, action, reward_shaped, next_state, done)
            buffer.push(transition_tuple)

            state = next_state

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                transition_dict = {
                    'states': states,
                    'actions': actions,
                    'rewards': rewards,
                    'next_states': next_states,
                    'dones': dones
                }
                agent.update(transition_dict)
        
        # 每回合结束后更新探索率
        agent.decay_epsilon()
        
        # 打印回合信息
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Steps: {steps}, Epsilon: {agent.epsilon:.4f}")
        
        # 检查是否有改进
        if episode_reward > best_reward:
            best_reward = episode_reward
            print(f"New best reward: {best_reward}")
        
        # 定期更新目标网络
        if (episode + 1) % agent.target_update == 0:
            agent.update_target_network()
            print(f"Target network updated at episode {episode+1}")


if __name__ == "__main__":
    main()
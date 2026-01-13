import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        '''
        DQN神经网络结构
        params:
        state_dim: 状态维度
        action_dim: 动作维度
        '''
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)

class ReplayBuffer:
    def __init__(self, capacity):
        '''
        经验回放缓冲区
        params:
        capacity: 缓冲区容量
        '''
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, terminated):
        '''
        向缓冲区添加经验
        '''
        self.buffer.append((state, action, reward, next_state, terminated))
    
    def sample(self, batch_size):
        '''
        从缓冲区采样经验
        '''
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(terminated)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, env: gym.Env, gamma: float = 0.99, lr: float = 0.001, epsilon_start: float = 1.0, 
                 epsilon_decay: float = 0.995, epsilon_end: float = 0.01, batch_size: int = 64, 
                 buffer_capacity: int = 10000, target_update: int = 10):
        '''
        DQN智能体
        params:
        env: 环境
        gamma: 折扣因子
        lr: 学习率
        epsilon_start: 初始探索率
        epsilon_decay: 探索率衰减率
        epsilon_end: 最小探索率
        batch_size: 批量大小
        buffer_capacity: 经验回放缓冲区容量
        target_update: 目标网络更新频率
        '''
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.target_update = target_update
        
        # 创建DQN网络
        self.policy_net = DQNNetwork(self.state_dim, self.action_dim)
        self.target_net = DQNNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 定义优化器和损失函数
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        
        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
    
    def choose_action(self, state):
        '''
        使用epsilon-greedy策略选择动作
        '''
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
    
    def update(self):
        '''
        更新DQN网络
        '''
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从经验回放缓冲区采样
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = self.replay_buffer.sample(self.batch_size)
        
        # 转换为张量
        state_batch = torch.tensor(state_batch, dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32)
        terminated_batch = torch.tensor(terminated_batch, dtype=torch.float32)
        
        # 计算当前状态的Q值
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        
        # 计算下一个状态的最大Q值（使用目标网络）
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - terminated_batch)
        
        # 计算损失并更新网络
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def decay_epsilon(self):
        '''
        衰减探索率
        '''
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_target_network(self):
        '''
        更新目标网络
        '''
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 训练DQN智能体
if __name__ == "__main__":
    # 创建环境
    env = gym.make("CartPole-v1")
    
    # 初始化智能体
    agent = DQNAgent(env)
    
    # 训练参数
    num_episodes = 500
    max_steps_per_episode = 1000
    
    # 训练过程
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps_per_episode):
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, terminated)
            
            # 更新网络
            agent.update()
            
            # 更新状态
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 衰减探索率
        agent.decay_epsilon()
        
        # 更新目标网络
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # 打印训练信息
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
        
        # 检查是否解决了环境（CartPole-v1要求平均奖励>=475）
        if total_reward >= 500:
            print(f"Environment solved in episode {episode + 1}!")
            break
    
    # 测试训练好的智能体
    print("\nTesting the trained agent...")
    env = gym.make("CartPole-v1", render_mode="human")
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(max_steps_per_episode):
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    print(f"Test Total Reward: {total_reward}")
    env.close()
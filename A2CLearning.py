import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm

class Actor(nn.Module):
    """
    Actor网络：输出每个动作的概率分布
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    """
    Critic网络：输出状态的价值
    """
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) 智能体
    """
    def __init__(self, env, gamma=0.99, lr_actor=0.0003, lr_critic=0.001, hidden_dim=128, n_steps=5):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.n_steps = n_steps  # 使用n-step回报估计
        
        # 创建Actor和Critic网络
        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim)
        self.critic = Critic(self.state_dim, hidden_dim)
        
        # 创建优化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 损失函数
        self.critic_loss_fn = nn.MSELoss()
    
    def choose_action(self, state):
        """
        根据当前策略选择动作
        """
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
        
        # 从概率分布中采样动作
        action = torch.multinomial(action_probs, 1).item()
        return action
    
    def compute_returns(self, rewards, next_state, done):
        """
        计算n-step回报
        """
        returns = []
        next_state_tensor = torch.FloatTensor(next_state)
        
        with torch.no_grad():
            # 计算最后状态的价值
            R = self.critic(next_state_tensor) if not done else 0.0
        
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return torch.FloatTensor(returns)
    
    def update(self, trajectory):
        """
        更新Actor和Critic网络
        使用n-step回报和优势函数
        """
        states, actions, rewards, next_state, done = zip(*trajectory)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.LongTensor(actions)
        returns_tensor = self.compute_returns(rewards, next_state[-1], done[-1])
        
        # 计算当前状态的价值估计
        values = self.critic(states_tensor)
        
        # 计算优势函数
        advantages = returns_tensor - values.squeeze()
        
        # 更新Critic网络
        self.optimizer_critic.zero_grad()
        critic_loss = self.critic_loss_fn(values, returns_tensor.unsqueeze(1))
        critic_loss.backward()
        self.optimizer_critic.step()
        
        # 更新Actor网络
        self.optimizer_actor.zero_grad()
        action_probs = self.actor(states_tensor)
        log_probs = torch.log(action_probs[range(len(actions)), actions_tensor])
        actor_loss = -torch.mean(log_probs * advantages.detach())
        
        actor_loss.backward()
        self.optimizer_actor.step()
    
    def train(self, num_episodes=1000, max_steps_per_episode=1000):
        """
        训练智能体
        """
        rewards_history = []
        
        for episode in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            total_reward = 0
            trajectory = []
            
            for step in range(max_steps_per_episode):
                # 选择动作
                action = self.choose_action(state)
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # 存储轨迹
                trajectory.append((state, action, reward, next_state, done))
                
                # 更新状态和奖励
                state = next_state
                total_reward += reward
                
                # 每n步更新一次网络
                if len(trajectory) == self.n_steps or done:
                    self.update(trajectory)
                    trajectory = []
                
                if done:
                    break
            
            rewards_history.append(total_reward)
            
            # 每100轮打印一次训练信息
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}")
        
        return rewards_history
    
    def test(self, num_episodes=100, max_steps_per_episode=1000, render=False):
        """
        测试训练好的智能体
        """
        env = gym.make(self.env.spec.id, render_mode ='human')
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            
            for step in range(max_steps_per_episode):
                # 选择动作（确定性选择）
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state)
                    action_probs = self.actor(state_tensor)
                    action = torch.argmax(action_probs).item()
                
                # 执行动作
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # 更新状态和奖励
                state = next_state
                total_reward = total_reward + reward
                
                if done:
                    break
            
            total_rewards.append(total_reward)
        
        avg_reward = np.mean(total_rewards)
        print(f"Test Results: Average Reward = {avg_reward:.2f}")
        return avg_reward


# 测试代码
if __name__ == "__main__":
    # 创建环境
    env = gym.make("CartPole-v1")
    
    # 创建并训练智能体
    agent = A2CAgent(env)
    print("开始训练A2C智能体...")
    rewards_history = agent.train(num_episodes=2000)
    
    # 测试智能体
    print("\n开始测试训练好的智能体...")
    agent.test(num_episodes=10, render=False)
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import tqdm
from typing import NamedTuple


class Params(NamedTuple):
    state_dim: int
    action_dim: int
    hidden_dim: int
    actor_lr: float
    critic_lr: float
    gamma: float
    tau: float  # 软更新参数
    noise_std: float  # OU噪声标准差
    noise_theta: float  # OU噪声theta参数
    buffer_size: int  # 经验回放缓冲区大小
    batch_size: int  # 批量大小
    num_episodes: int  # 训练回合数
    max_steps_per_episode: int  # 每回合最大步数
    device: torch.device


class OUNoise:
    """Ornstein-Uhlenbeck噪声过程，用于探索"""
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class Actor(nn.Module):
    """Actor网络：输入状态，输出连续动作"""
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # 动作的范围，用于将输出映射到有效范围

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # tanh输出范围[-1,1]
        x = x * self.action_bound  # 映射到实际动作范围
        return x


class Critic(nn.Module):
    """Critic网络：输入状态和动作，输出Q值"""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        # 状态分支
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # 合并分支
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)  # 合并状态和动作
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, buffer_size):
        self.buffer = collections.deque(maxlen=buffer_size)

    def push(self, transition):
        """添加一个转换样本到缓冲区"""
        self.buffer.append(transition)

    def sample(self, batch_size):
        """随机采样批量样本"""
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)


class DDPG:
    """DDPG算法实现"""
    def __init__(self, params):
        self.params = params
        self.device = params.device

        # 环境信息
        self.state_dim = params.state_dim
        self.action_dim = params.action_dim
        self.action_bound = 2.0  # Pendulum-v1的动作范围是[-2, 2]

        # 噪声生成器
        self.noise = OUNoise(mu=np.zeros(self.action_dim), sigma=params.noise_std, theta=params.noise_theta)

        # 经验回放缓冲区
        self.buffer = ReplayBuffer(params.buffer_size)

        # Actor网络
        self.actor = Actor(self.state_dim, params.hidden_dim, self.action_dim, self.action_bound).to(self.device)
        self.target_actor = Actor(self.state_dim, params.hidden_dim, self.action_dim, self.action_bound).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=params.actor_lr)

        # Critic网络
        self.critic = Critic(self.state_dim, params.hidden_dim, self.action_dim).to(self.device)
        self.target_critic = Critic(self.state_dim, params.hidden_dim, self.action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=params.critic_lr)

        # 将目标网络初始化为与主网络相同
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 损失函数
        self.critic_loss_fn = nn.MSELoss()

    def choose_action(self, state, add_noise=True):
        """选择动作，支持添加探索噪声"""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        if add_noise:
            action += self.noise()
            action = np.clip(action, -self.action_bound, self.action_bound)  # 确保动作在有效范围内
        return action

    def soft_update(self, net, target_net, tau):
        """软更新目标网络参数"""
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self):
        """更新网络参数"""
        if len(self.buffer) < self.params.batch_size:
            return  # 缓冲区样本不足时不更新

        # 从缓冲区采样批量样本
        states, actions, rewards, next_states, dones = self.buffer.sample(self.params.batch_size)

        # 转换为张量
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # 更新Critic网络
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            target_q_values = rewards + self.params.gamma * next_q_values * (1 - dones)
        
        current_q_values = self.critic(states, actions)
        critic_loss = self.critic_loss_fn(current_q_values, target_q_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor网络
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.actor, self.target_actor, self.params.tau)
        self.soft_update(self.critic, self.target_critic, self.params.tau)

        return critic_loss.item(), actor_loss.item()


def main():
    # 创建环境
    env = gym.make("Pendulum-v1")
    
    # 设置设备
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    
    # 定义参数
    params = Params(
        state_dim=env.observation_space.shape[0],  # Pendulum-v1的状态维度是3
        action_dim=env.action_space.shape[0],      # Pendulum-v1的动作维度是1
        hidden_dim=256,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=1e-3,
        noise_std=0.2,
        noise_theta=0.15,
        buffer_size=100000,
        batch_size=64,
        num_episodes=1000,
        max_steps_per_episode=200,  # Pendulum-v1每回合最多200步
        device=device
    )
    
    # 创建DDPG智能体
    agent = DDPG(params)
    
    # 开始训练
    for episode in tqdm.tqdm(range(params.num_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        critic_loss_list = []
        actor_loss_list = []
        
        agent.noise.reset()  # 每个回合重置噪声过程
        
        for step in range(params.max_steps_per_episode):
            # 选择动作
            action = agent.choose_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.buffer.push((state, action, reward, next_state, done))
            
            # 更新状态
            state = next_state
            episode_reward += reward
            
            # 更新网络
            if len(agent.buffer) >= params.batch_size:
                critic_loss, actor_loss = agent.update()
                critic_loss_list.append(critic_loss)
                actor_loss_list.append(actor_loss)
        
        # 打印回合信息
        avg_critic_loss = np.mean(critic_loss_list) if critic_loss_list else 0
        avg_actor_loss = np.mean(actor_loss_list) if actor_loss_list else 0
        print(f"Episode {episode+1}/{params.num_episodes}, Reward: {episode_reward:.2f}, "
              f"Critic Loss: {avg_critic_loss:.4f}, Actor Loss: {avg_actor_loss:.4f}")
        
        # 每100回合保存一次模型
        if (episode + 1) % 100 == 0:
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict()
            }, f"ddpg_pendulum_ckpt_{episode+1}.pt")


if __name__ == "__main__":
    main()
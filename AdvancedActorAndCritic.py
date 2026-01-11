import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class A2C:
    def __init__(self, state_dim, action_dim, hidden_dim=64,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5,
                 max_grad_norm=0.5, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.actor = Actor(state_dim, hidden_dim, action_dim).to(device)
        self.critic = Critic(state_dim, hidden_dim).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

    @torch.no_grad()
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action.item(), log_prob, entropy

    def compute_gae(self, rewards, values, next_value, dones):
        """计算 GAE-Lambda advantage"""
        advantages = []
        gae = 0
        values = values + [next_value]  # 最后补上 next_value

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages, entropies):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.stack(log_probs_old).to(self.device).detach()
        returns = torch.FloatTensor(returns).to(self.device).unsqueeze(-1)
        advantages = torch.FloatTensor(advantages).to(self.device).unsqueeze(-1)
        entropies = torch.stack(entropies).to(self.device).detach()

        # 前向传播
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # ratio = torch.exp(new_log_probs - old_log_probs)  # PPO才需要，A2C不用

        # Actor loss
        actor_loss = -(new_log_probs * advantages.detach()).mean()

        # Critic loss
        values = self.critic(states)
        critic_loss = F.mse_loss(values, returns)

        # Entropy bonus
        entropy_loss = -self.entropy_coef * entropy

        # 总损失
        loss = actor_loss + self.value_loss_coef * critic_loss + entropy_loss

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy.item()


def train():
    env = gym.make('MountainCar-v0')
    device = torch.device('cpu')
    print(f"使用设备: {device}")

    agent = A2C(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=128,
        lr=1e-3,           # A2C 通常学习率稍大一点
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.02,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        device=device
    )

    n_steps = 5              # 每多少步更新一次（A2C典型值5~20）
    num_episodes = 60000
    max_steps = 200

    episode_rewards = []

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        total_reward = 0

        states, actions, log_probs, rewards, dones, values, entropies = [], [], [], [], [], [], []

        for step in range(max_steps):
            action, log_prob, entropy = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 记录轨迹
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(float(done))
            entropies.append(entropy)

            # 记录价值（用于 GAE）
            with torch.no_grad():
                value = agent.critic(torch.FloatTensor(state).unsqueeze(0).to(device)).item()
            values.append(value)

            state = next_state
            total_reward += reward

            # 达到 n_steps 或 episode 结束时更新
            if len(states) >= n_steps or done:
                # 计算 next_value
                with torch.no_grad():
                    next_value = agent.critic(torch.FloatTensor(state).unsqueeze(0).to(device)).item() if not done else 0

                # 计算 GAE 和 returns
                advantages = agent.compute_gae(rewards, values, next_value, dones)
                returns = [adv + val for adv, val in zip(advantages, values)]

                # 更新网络
                agent.update(states, actions, log_probs, returns, advantages, entropies)

                # 清空缓冲区
                states, actions, log_probs, rewards, dones, values, entropies = [], [], [], [], [], [], []

            if done:
                break

        episode_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1:4d} | Avg Reward (last 100): {avg_reward:8.2f}")

    env.close()
    print("训练完成！")


if __name__ == "__main__":
    train()
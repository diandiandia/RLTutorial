import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from typing import NamedTuple
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, state_value_dim=1):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_value_dim)
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic:
    def __init__(self, env, state_dim, hidden_dim, action_dim, epsilon_start, epsilon_end, epsilon_decay_rate, policy_lr, value_lr, gamma):
        self.env = env
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = epsilon_decay_rate
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.gamma = gamma
        self.device = self.to_device()

        self.actor = PolicyNet(state_dim, hidden_dim,
                               action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=policy_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=value_lr)
        self.loss_fn = nn.MSELoss()

    def to_device(self):
        if torch.cuda.is_available():
            device_name = 'cuda'
        elif torch.backends.mps.is_available():
            device_name = 'mps'
        else:
            device_name = 'cpu'
        return device_name

    def take_action(self, state):
        state_tensor = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
        probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def update(self, state, action, reward, next_state, done):
        # 全部变成 batch size=1 的 tensor
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        next_state_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_t = torch.tensor([[action]], dtype=torch.int64).to(self.device)
        reward_t = torch.tensor([[reward]], dtype=torch.float32).to(self.device)
        done_t = torch.tensor([[float(done)]], dtype=torch.float32).to(self.device)

        # ─────── Critic ───────
        v = self.critic(state_t)
        v_next = self.critic(next_state_t)
        td_target = reward_t + self.gamma * v_next * (1 - done_t)
        td_error = td_target - v

        critic_loss = self.loss_fn(v, td_target.detach())   # 也可以用 (v - td_target.detach())**2 .mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ─────── Actor ───────
        probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action_t.squeeze()).unsqueeze(1)

        # 你的 δ_i 就是 td_error
        actor_loss = - (log_prob * td_error.detach()).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def epsilon_decay(self):
        self.epsilon = max(
            self.epsilon_end, self.epsilon * self.epsilon_decay_rate)


class Params(NamedTuple):
    state_dim: int
    hidden_dim: int
    action_dim: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_rate: float
    policy_lr: float
    value_lr: float
    gamma: float


def main():
    env = gym.make('CartPole-v1')

    params = Params(
        state_dim=env.observation_space.shape[0],
        hidden_dim=64,
        action_dim=env.action_space.n,
        epsilon_start=0.95,
        epsilon_end=0.01,
        epsilon_decay_rate=0.995,
        policy_lr=3e-4,
        value_lr=3e-4,
        gamma=0.99,
    )

    agent = ActorCritic(
        env=env,
        state_dim=params.state_dim,
        hidden_dim=params.hidden_dim,
        action_dim=params.action_dim,
        epsilon_start=params.epsilon_start,
        epsilon_end=params.epsilon_end,
        epsilon_decay_rate=params.epsilon_decay_rate,
        policy_lr=params.policy_lr,
        value_lr=params.value_lr,
        gamma=params.gamma
    )

    num_episodes = 5000
    episode_rewards = []
    episode_actor_losses = []
    episode_critic_losses = []

    for episode in tqdm(range(num_episodes)):
        episode_reward = 0
        state, _ = env.reset()  # 修复：获取元组的第一个元素
        done = False

        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 只更新当前这一步！！
            al, cl = agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            episode_actor_losses.append(al)
            episode_critic_losses.append(cl)

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            agent.epsilon_decay()
            print(f"Episode {episode + 1}: epsilon {agent.epsilon:.4f}")
            

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Average Reward = {np.mean(episode_rewards[-100:]):.2f}, Average Actor Loss = {np.mean(episode_actor_losses):.4f}, Average Critic Loss = {np.mean(episode_critic_losses):.4f}")

    plot_rewards(episode_rewards)


def plot_rewards(rewards):
    plt.figure(figsize=(10,5))
    plt.plot(rewards, alpha=0.3, label='raw')
    if len(rewards) >= 50:
        avg = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(rewards)), avg, label='moving avg 50')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('CartPole-v1 Training Curve')
    plt.savefig('rewards.png', dpi=150)
    plt.close()

if __name__ == '__main__':
    main()

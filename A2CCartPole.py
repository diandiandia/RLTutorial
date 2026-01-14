import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class A2CAgent:
    def __init__(
        self,
        env,
        device,
        hidden_dim=128,
        gamma=0.99,
        beta=0.01,
        actor_lr=5e-4,
        critic_lr=1e-3,
    ):
        self.env = env
        self.device = device
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.beta = beta

        self.actor = PolicyNet(self.state_dim, hidden_dim, self.action_dim).to(
            self.device
        )
        self.critic = ValueNet(self.state_dim, hidden_dim).to(self.device)

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.loss_fn = nn.MSELoss()

    def take_action(self, state):
        # 确保状态张量在正确设备上
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state_tensor)

        # 从概率分布中采样动作
        action = torch.multinomial(action_probs, 1).item()
        return action

    def compute_returns(self, rewards, next_vs, dones):
        G_returns = []

        G = 0.0 if dones[-1] else next_vs[-1]
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            G_returns.insert(0, G)

        # 返回与next_vs相同设备的张量
        return torch.FloatTensor(G_returns).to(self.device)

    def compute_actor_loss(self, states_t, actions_t, advantages):
        self.optim_actor.zero_grad()
        action_probs = self.actor(states_t)

        # 计算所有动作的log概率（用于熵计算）
        all_log_probs = torch.log(action_probs + 1e-8)

        # 计算采样动作的log概率（用于策略梯度）
        log_probs = all_log_probs.gather(1, actions_t.unsqueeze(1))

        # 计算熵正则化项：H(π) = -Σ p(a) log p(a)
        entropy = -torch.sum(action_probs * all_log_probs, dim=1)
        beta_h = self.beta * entropy

        # 策略目标函数：E[ln π(a|s)A + βH(π)]
        actor_loss = -torch.mean(log_probs * advantages.detach() + beta_h)
        actor_loss.backward()
        self.optim_actor.step()
        return actor_loss

    def compute_critic_loss(self, current_vs, G_returns):
        self.optim_critic.zero_grad()
        critic_loss = self.loss_fn(current_vs, G_returns)
        critic_loss.backward()
        self.optim_critic.step()
        return critic_loss

    def update(self, states, actions, rewards, next_states, dones):
        # 确保所有张量在正确设备上
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(np.array(dones)).to(self.device)

        # 计算V(s)
        current_vs = self.critic(states_t).squeeze(dim=1)
        with torch.no_grad():
            next_vs = self.critic(next_states_t).squeeze(dim=1)

        G_returns = self.compute_returns(
            rewards_t.detach(), next_vs.detach(), dones_t.detach()
        )
        advantages = G_returns - current_vs

        # 更新Critic：L_w = E[(V_w(s_t) - G_t)^2]
        critic_loss = self.compute_critic_loss(current_vs, G_returns)

        # 更新Actor：J(θ) = E[ln π(a|s)A + βH(π)]
        actor_loss = self.compute_actor_loss(states_t, actions_t, advantages)

        return actor_loss.item(), critic_loss.item()


def main():
    env = gym.make("CartPole-v1")
    # 自动检测可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = A2CAgent(env, device=device)

    num_episodes = 2000
    total_rewards = []
    actor_losses = []
    critic_losses = []
    n_steps = 10  # 控制网络更新频率，而不是episode长度

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        trajectory = []
        episode_reward = 0

        # 让episode持续运行直到失败，而不是固定步数
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            trajectory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            # 每n_steps更新一次网络
            if len(trajectory) == n_steps:
                actor_loss, critic_loss = agent.update(*zip(*trajectory))
                trajectory = []
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

        # 如果episode结束时还有未处理的轨迹，也要更新网络
        if len(trajectory) > 0:
            actor_loss, critic_loss = agent.update(*zip(*trajectory))
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        total_rewards.append(episode_reward)

        # 每100轮打印一次训练信息
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(
                f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}"
            )

    plot(total_rewards)


def plot(rewards):
    plt.plot(rewards, alpha=0.3)
    avg = np.convolve(rewards, np.ones(50) / 50, mode="valid")
    plt.plot(range(49, len(rewards)), avg)
    plt.title("A2C on CartPole")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.savefig("a2c_cartpole.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

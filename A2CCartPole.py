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
        self, env: gym.Env, hidden_dim=128, gamma=0.99, actor_lr=0.0003, critic_lr=0.001
    ):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.actor = PolicyNet(self.state_dim, hidden_dim, self.action_dim).to(
            self.device
        )
        self.critic = ValueNet(self.state_dim, hidden_dim).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.loss_fn = nn.MSELoss()

    def take_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.actor(state_t)
        action = torch.multinomial(probs, num_samples=1).item()
        return action

    def compute_returns(self, rewards, next_state, done):
        returns = []

        with torch.no_grad():
            # 计算最后状态的价值
            R = self.critic(next_state) if not done else 0.0

        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        return torch.FloatTensor(returns).to(self.device)

    def update(self, trajectory):

        states, actions, rewards, next_states, dones = zip(*trajectory)
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards_t = (
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        )
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # ----- Critic -----
        current_values = self.critic(states_t)
        next_values = self.compute_returns(rewards_t, next_states_t[-1], dones_t[-1])

        advantages = next_values - current_values.squeeze()

        critic_loss = self.loss_fn(current_values, next_values.unsqueeze(1))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ----- Actor -----
        probs = self.actor(states_t)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)
        actor_loss = -(log_probs * advantages.detach()).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return actor_loss.item(), critic_loss.item()


# ------------------------
# Training
# ------------------------
def main():
    env = gym.make("CartPole-v1")

    agent = A2CAgent(env)

    n_steps = 2000
    num_episodes = 4000

    rewards_history = []

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        trajectory = []

        for _ in range(n_steps):
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            trajectory.append((state, action, reward, next_state, done))

            episode_reward += reward
            state = next_state

            if len(trajectory) == n_steps or done:
                agent.update(trajectory)

            if done:
                break

        rewards_history.append(episode_reward)

        # 每100轮打印一次训练信息
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(
                f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}"
            )

    plot(rewards_history)


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

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ------------------------
# Networks
# ------------------------
class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------
# A2C Agent
# ------------------------
class A2C:
    def __init__(self, state_dim, action_dim, hidden_dim=64, gamma=0.99, lr=3e-4):
        self.gamma = gamma

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def take_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)
        return dist.sample().item()

    def update(self, states, actions, returns):
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64).to(self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32).unsqueeze(1).to(self.device)

        # ----- Critic -----
        values = self.critic(states_t)
        advantages = returns_t - values

        critic_loss = advantages.pow(2).mean()
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ----- Actor -----
        probs = self.actor(states_t)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions_t)

        entropy = dist.entropy().mean()

        actor_loss = -(log_probs.unsqueeze(1) * advantages.detach()).mean() - 0.01 * entropy

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return actor_loss.item(), critic_loss.item()


# ------------------------
# Training
# ------------------------
def main():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2C(state_dim, action_dim)

    n_steps = 5
    num_episodes = 2000

    rewards_history = []

    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            states, actions, rewards = [], [], []

            for _ in range(n_steps):
                states.append(state)
                action = agent.take_action(state)
                actions.append(action)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                rewards.append(reward)
                episode_reward += reward
                state = next_state

                if done:
                    break

            # Bootstrap
            with torch.no_grad():
                if done:
                    R = 0
                else:
                    R = agent.critic(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)).item()

            returns = []
            for r in reversed(rewards):
                R = r + agent.gamma * R
                returns.insert(0, R)

            agent.update(states, actions, returns)

        rewards_history.append(episode_reward)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1}, Avg reward(50) = {np.mean(rewards_history[-50:]):.1f}")

    plot(rewards_history)


def plot(rewards):
    plt.plot(rewards, alpha=0.3)
    avg = np.convolve(rewards, np.ones(50)/50, mode="valid")
    plt.plot(range(49, len(rewards)), avg)
    plt.title("A2C on CartPole")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.savefig("a2c_cartpole.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

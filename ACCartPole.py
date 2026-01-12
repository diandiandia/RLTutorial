import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from typing import NamedTuple


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
    def __init__(self, env, state_dim, hidden_dim, action_dim, epsilon_start, epsilon_end, epsilon_decay_rate, policy_lr, value_lr, gamma, device='cpu'):
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
        self.device = self.to_devices()

        self.actor = PolicyNet(state_dim, hidden_dim,
                               action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=policy_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=value_lr)
        self.loss_fn = nn.MSELoss()

    def to_devices(self):
        if torch.cuda.is_available():
            device_name = 'cuda'
        elif torch.backends.mps.is_available():
            device_name = 'mps'
        else:
            device_name = 'cpu'
        return torch.device(device_name)

    def take_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float16).to(self.device)
        probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(
            np.array(transition_dict['states']), dtype=torch.float16).to(self.device)
        actions = torch.tensor(
            transition_dict['actions'], dtype=torch.int8).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            transition_dict['rewards'], dtype=torch.float16).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            np.array(transition_dict['next_states']), dtype=torch.float16).to(self.device)
        dones = torch.tensor(
            transition_dict['dones'], dtype=torch.float16).view(-1, 1).to(self.device)

        # 计算TD
        current_v = self.critic(states)
        next_v = self.critic(next_states)
        td_target = rewards + (1-dones) * self.gamma * next_v
        td_error = td_target - current_v

        # 更新critic
        critic_loss = self.loss_fn(current_v, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新actor
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = -torch.mean(log_probs * td_error.detach())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def epsilon_decay(self):
        self.epsilon = max(
            self.epsilon_end, self.epsilon_start * self.epsilon_decay_rate)


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
    device: str


def main():
    env = gym.make('CartPole-v1')
    params = Params(
        state_dim=4,
        hidden_dim=64,
        action_dim=2,
        epsilon_start=0.95,
        epsilon_end=0.01,
        epsilon_decay_rate=0.995,
        policy_lr=3e-4,
        value_lr=3e-4,
        gamma=0.99,
        device='cpu',
    )
    params = params._replace(state_dim=env.observation_space.shape[0])
    params = params._replace(action_dim=env.action_space.n)

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
        gamma=params.gamma,
        device=params.device,
    )

    num_episodes = 1000
    episode_rewards = []
    episode_actor_losses = []
    episode_critic_losses = []

    for episode in tqdm(range(num_episodes)):
        episode_reward = 0
        state, _ = env.reset()
        done = False
        transition_dict = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }

        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(done)
            state = next_state
            episode_reward += reward
            episode_rewards.append(episode_reward)

            actor_loss, critic_loss = agent.update(transition_dict)
            episode_actor_losses.append(actor_loss)
            episode_critic_losses.append(critic_loss)

        if (episode + 1) % 10 == 0:
            agent.epsilon_decay()

        if (episode + 1) % 100 == 0:
            print(
                f'Episode {episode + 1}: reward = {episode_reward:.2f}, actor_loss = {actor_loss:.4f}, critic_loss = {critic_loss:.4f}')


def test():
    pass


if __name__ == '__main__':
    main()
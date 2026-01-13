from torch import nn
import gymnasium as gym
from torch import optim
import torch


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        '''
        Docstring for __init__
        
        :param self: Description
        :param state_dim: Description
        :param hidden_dim: Description
        :param action_dim: Description
        softmax 输出动作概率分布
        '''
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)
    
    
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim=1):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)
    
class ACAgent:
    def __init__(self, env: gym.Env, gamma: float, hidden_dim: int, lr_actor:float, lr_critic:float) -> None:
        self.env = env
        self.gamma = gamma
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        
        self.actor = Actor(self.state_dim, hidden_dim, self.action_dim)
        self.critic = Critic(self.state_dim, hidden_dim, 1)
        
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.loss_fn = nn.MSELoss()
        
        
    
    
    

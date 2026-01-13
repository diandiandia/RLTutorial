import numpy as np
import gymnasium as gym
from collections import defaultdict


class QLearning:
    def __init__(self, env:gym.Env, gamma:float, alpha:float, epsilon_start:float, epsilon_decay_rate:float=0.0099, epsilon_end:float=0.01):
        '''
        创建 QLearning 智能体
        params:
        env: 环境
        gamma: 折扣因子
        alpha: 学习率
        epsilon_start: 初始探索率
        epsilon_decay_rate: 探索率衰减率
        epsilon_end: 最小探索率
        '''
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_end = epsilon_end
        
        # 确保动作空间是Discrete类型并获取动作数量
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_space_n = env.action_space.n
        else:
            raise ValueError("动作空间必须是Discrete类型")
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_n))
            

    def choose_action(self, obs:tuple[int, int, bool])->int:
        """
        使用epsilon-greedy贪婪策略选择动作

        :param obs: 当前状态
        """
        n_actions = len(self.q_table[obs])

        prob_others = self.epsilon / n_actions
        prob_greedy = 1 - self.epsilon + prob_others
        # 贪婪动作
        greedy_action = np.argmax(self.q_table[obs])
        # 随机动作
        probs = np.full(n_actions, prob_others)
        probs[greedy_action] = prob_greedy

        action = np.random.choice(n_actions, p=probs)

        return action

    def update(self, obs:int, action:int, reward:float, obs_next:int, terminated:bool):
        """
        使用 Q-learning 的 Bellman optimality equation
        更新动作价值函数 Q(s,a)
        :param obs: 当前状态
        :param action: 当前动作
        :param reward: 奖励
        :param obs_next: 下一个状态
        :param terminated: 是否终止
        """
        if terminated:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[obs_next])
        td_error = td_target - self.q_table[obs][action]
        self.q_table[obs][action] = self.q_table[obs][action] + self.alpha * td_error
        
    def decay_epsilon(self):
        """
        衰减 epsilon
        """
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay_rate)
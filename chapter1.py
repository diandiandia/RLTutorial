import numpy as np
import math
from collections import defaultdict
import gymnasium as gym


class Config:
    """配置类，用于管理所有超参数"""
    def __init__(self):
        self.env_name = "CliffWalking-v1"
        self.lr = 0.1  # 学习率
        self.gamma = 0.9  # 折扣因子
        self.epsilon_start = 0.9  # 初始探索率
        self.epsilon_end = 0.01  # 最终探索率
        self.epsilon_decay = 300  # 探索率衰减步数
        self.train_eps = 500  # 训练回合数
        self.test_eps = 10  # 测试回合数
        self.sample = 0  # 样本计数
        self.epsilon = self.epsilon_start  # 当前探索率


class QLearning(object):
    def __init__(self, n_states, n_actions, cfg):
        self.n_states = n_states
        self.n_actions = n_actions  # 添加这一行，保存动作数量为实例变量
        self.lr = cfg.lr
        self.gamma = cfg.gamma
        self.epsilon = cfg.epsilon
        self.sample_count = cfg.sample
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda: np.zeros(n_actions))

    def sample_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay)
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[state])
        else:
            action = np.random.choice(range(self.n_actions))
        return action
    
    def predict_action(self, state):
        action = np.argmax(self.Q_table[state])
        return action
    
    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[state][action]
        if done:
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[next_state])
        self.Q_table[state][action] += self.lr * (Q_target - Q_predict)


def train(cfg, env, agent):
    print('Start training...')
    rewards = []
    for i_episode in range(cfg.train_eps):
        state, _ = env.reset(seed=cfg.train_eps)  # gymnasium中reset返回(state, info)
        done = False
        episode_reward = 0
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            # 如果truncated为True，表示时间步已用完，结束回合
            if truncated:
                break
        rewards.append(episode_reward)
        if (i_episode+1) % 10 == 0:
            print(f"Episode: {i_episode+1}/{cfg.train_eps}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
    print('Finish training...')
    return rewards


def test(cfg, env, agent):
    print('Start testing...')
    rewards = []
    for i_episode in range(cfg.test_eps):
        state, _ = env.reset(seed=cfg.test_eps)  # gymnasium中reset返回(state, info)
        done = False
        episode_reward = 0
        while not done:
            action = agent.predict_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            # 如果truncated为True，表示时间步已用完，结束回合
            if truncated:
                break
        rewards.append(episode_reward)
        if (i_episode+1) % 10 == 0:
            print(f"Episode: {i_episode+1}/{cfg.test_eps}, Reward: {episode_reward:.2f}")
    print('Finish testing...')
    return rewards


def CliffWalkingEnv(cfg):
    env = gym.make(cfg.env_name)
    return env


def main():
    """主函数"""
    cfg = Config()
    env = CliffWalkingEnv(cfg)
    
    # 打印环境信息
    print(f"环境名称: {cfg.env_name}")
    print(f"状态空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"动作数量: {env.action_space.n}")
    
    # 初始化智能体
    agent = QLearning(env.observation_space.n, env.action_space.n, cfg)
    
    # 训练智能体
    train_rewards = train(cfg, env, agent)
    
    # 测试智能体
    test_rewards = test(cfg, env, agent)
    
    # 打印结果统计
    print(f"训练平均奖励: {np.mean(train_rewards):.2f}")
    print(f"测试平均奖励: {np.mean(test_rewards):.2f}")
    
    # 关闭环境
    env.close()


if __name__ == "__main__":
    main()
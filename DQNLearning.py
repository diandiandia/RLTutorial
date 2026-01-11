# 导入强化学习环境库
import gymnasium as gym
# 导入PyTorch的神经网络模块
import torch.nn as nn
# 导入PyTorch的优化器模块
import torch.optim as optim
# 导入Python的collections模块，用于创建双端队列
import collections
# 导入NumPy科学计算库
import numpy as np
# 从typing模块导入NamedTuple，用于创建命名元组
from typing import NamedTuple
# 导入PyTorch深度学习框架
import torch
# 导入Python的random模块，用于生成随机数
import random
# 导入tqdm模块，用于显示进度条
import tqdm


# 定义参数类，使用NamedTuple方便参数管理和传递
class Params(NamedTuple):
    state_dim: int  # 状态空间维度
    hidden_dim: int  # 神经网络隐藏层维度
    action_dim: int  # 动作空间维度
    learning_rate: float  # 学习率
    gamma: float  # 折扣因子，用于计算未来奖励的现值
    epsilon: float  # 探索率，用于ε-贪婪策略
    epsilon_decay: float  # 探索率衰减因子
    epsilon_end: float  # 最小探索率
    target_update: int  # 目标网络更新频率
    device: torch.device = torch.device("mps" if torch.mps.is_available() else "cpu")  # Mac MPS设备

# 定义Q网络类，继承自PyTorch的nn.Module
class DQNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(DQNetwork, self).__init__()  # 调用父类的初始化方法
        # 定义神经网络结构：输入层 -> 隐藏层(ReLU激活) -> 输出层
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # 线性变换，输入层到隐藏层
            nn.ReLU(),  # ReLU激活函数，增加非线性
            nn.Linear(hidden_dim, action_dim)  # 线性变换，隐藏层到输出层
        )

    def forward(self, x):
        # 定义前向传播过程，输入状态x，输出各动作的Q值
        return self.network(x)
    
# 定义经验回放缓冲区类
class ReplayBuffer:  
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size  # 缓冲区最大容量
        # 创建双端队列作为缓冲区，超过容量时自动移除旧样本
        self.buffer = collections.deque(maxlen= self.buffer_size)

    def push(self, transition_tuple):
        # 将一个转换样本(状态,动作,奖励,下一状态,是否结束)添加到缓冲区
        self.buffer.append(transition_tuple)

    def sample(self, batch_size):
        # 从缓冲区中随机采样batch_size个样本
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough elements in the buffer to sample")  # 缓冲区样本不足时抛出错误

        transitions = random.sample(self.buffer, batch_size)  # 随机采样
        # 将采样的样本按状态、动作、奖励、下一状态、是否结束分别提取并打包
        state, action, rewards, next_state, done = zip(*transitions)
        return np.array(state), action, rewards, np.array(next_state), done  # 返回numpy数组形式的样本

    def __len__(self):
        # 返回缓冲区当前的样本数量
        return len(self.buffer)
    

# 定义DQN学习算法类
class DQNLearning:
    def __init__(self, env:gym.Env, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, epsilon_decay, epsilon_end, target_update, device):
        self.env = env  # 强化学习环境
        
        # 保存算法参数
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.device = device

        # 初始化Q网络和目标Q网络
        self.q_network = DQNetwork(state_dim, hidden_dim, action_dim).to(device)  # 主Q网络，用于选择动作和更新
        self.target_network = DQNetwork(state_dim, hidden_dim, action_dim).to(device)  # 目标Q网络，用于计算目标Q值

        # 为Q网络创建优化器
        self.q_network_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 将Q网络的权复制制到目标Q网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        # 设置目标网络为评估模式（不进行梯度更新）
        self.target_network.eval()
        # 定义损失函数，使用均方误差损失
        self.loss_fn = nn.MSELoss()

    def update(self, transition_dict):
        # 从转换字典中获取批量样本并转换为numpy数组
        states_np = transition_dict['states']
        next_states_np = transition_dict['next_states']
        
        # 将numpy数组转换为PyTorch张量，并移动到指定设备
        tensor_states = torch.tensor(states_np, dtype=torch.float32).to(self.device)
        # 动作需要为int64类型，因为它们是用于索引的
        tensor_actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).unsqueeze(1).to(self.device)
        tensor_rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).unsqueeze(1).to(self.device)
        tensor_next_states = torch.tensor(next_states_np, dtype=torch.float32).to(self.device)
        tensor_dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).unsqueeze(1).to(self.device)

        # 计算当前状态下的Q值，使用gather函数选择实际执行动作的Q值
        q_values = self.q_network(tensor_states).gather(1, tensor_actions)
        # 使用目标网络计算下一状态的最大Q值
        max_next_q_values = self.target_network(tensor_next_states).max(1)[0].unsqueeze(1)
        # 计算目标Q值：奖励 + 折扣因子 * 下一状态的最大Q值（如果不是终止状态）
        q_targets = tensor_rewards + self.gamma * max_next_q_values * (1 - tensor_dones)
        # 计算Q值与目标Q值之间的均方误差损失
        dqn_loss = self.loss_fn(q_values, q_targets)
        # 梯度清零
        self.q_network_optimizer.zero_grad()
        # 反向传播计算梯度
        dqn_loss.backward()
        # 更新Q网络的参数
        self.q_network_optimizer.step()

    def choose_action(self, state):
        # ε-贪婪策略选择动作
        if np.random.rand() < self.epsilon:
            # 以ε的概率随机选择动作（探索）
            return self.env.action_space.sample()
        else:
            # 以1-ε的概率选择当前Q值最大的动作（利用）
            # 将状态转换为张量并添加批次维度
            tensor_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():  # 不计算梯度，提高效率
                q_values = self.q_network(tensor_state)  # 计算当前状态下的所有动作Q值
            return q_values.argmax().item()  # 返回Q值最大的动作索引
        
    def decay_epsilon(self):
        # 衰减探索率，使智能体逐渐从探索转向利用
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        """更新目标网络，将Q网络的权复制制到目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def save_model(self, path):
        # 保存Q网络的参数到文件
        torch.save(self.q_network.state_dict(), path)

    def load_model(self, path):
        # 从文件加载Q网络的参数
        self.q_network.load_state_dict(torch.load(path))
        self.q_network.eval()  # 设置为评估模式
        

def main():
    # 创建强化学习环境（山车问题，离散动作空间）
    env = gym.make("MountainCar-v0")
    # 定义算法参数
    params = Params(
        state_dim=env.observation_space.shape[0],  # 状态维度：山车问题有2个状态（位置和速度）
        hidden_dim=128,  # 隐藏层神经元数量
        action_dim=env.action_space.n,  # 动作维度：山车问题有3个离散动作
        learning_rate=1e-3,  # 学习率：1e-3是常用值
        gamma=0.99,  # 折扣因子：接近1，重视长期奖励
        epsilon=1.0,  # 初始探索率：1表示完全探索
        epsilon_decay=0.995,  # 探索率衰减因子：每步衰减5%
        epsilon_end=0.01,  # 最小探索率：保证一定的探索
        target_update=10,  # 目标网络更新频率：每10个回合更新一次
        device=torch.device("mps" if torch.mps.is_available() else "cpu")  # 计算设备：优先使用MPS（Apple Silicon），否则使用CPU
    )

    # 创建DQN智能体
    agent = DQNLearning(env, **params._asdict())
    # 创建经验回放缓冲区，容量为5000
    buffer = ReplayBuffer(buffer_size=5000)

    # 定义训练参数
    num_episodes = 5000  # 训练回合数
    batch_size = 64  # 每次更新的样本数
    
    # 开始训练循环
    for episode in tqdm.tqdm(range(num_episodes)):
        # 重置环境，获取初始状态
        state, _ = env.reset(seed=43)
        done = False  # 回合是否结束的标志
        episode_reward = 0  # 当前回合的总奖励
        
        # 开始回合内的循环
        while not done:
            # 根据当前状态选择动作
            action = agent.choose_action(state)
            # 执行动作，获取下一状态、奖励、是否终止、是否截断等信息
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward  # 累加回合奖励
            done = terminated or truncated  # 回合结束的条件

            # 将转换样本添加到经验回放缓冲区
            transition_tuple = (state, action, reward, next_state, done)
            buffer.push(transition_tuple)

            # 更新当前状态
            state = next_state

            # 当缓冲区样本足够时，进行网络更新
            if len(buffer) >= batch_size:
                # 从缓冲区采样批量样本
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                # 组织成转换字典
                transition_dict = {
                    'states': states,
                    'actions': actions,
                    'rewards': rewards,
                    'next_states': next_states,
                    'dones': dones
                }
                # 更新Q网络
                agent.update(transition_dict)
                # 衰减探索率
                agent.decay_epsilon()
        
        # 打印当前回合的信息
        print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}")
        
        # 定期更新目标网络
        if (episode + 1) % agent.target_update == 0:
            agent.update_target_network()
            print(f"Target network updated at episode {episode+1}")


# 程序入口，当直接运行该脚本时执行main函数
if __name__ == "__main__":
    main()
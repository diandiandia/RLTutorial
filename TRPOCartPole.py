import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import torch.nn.functional as F
import copy
from tqdm import tqdm


class PolicyNet(torch.nn.Module):
    '''
    策略网络：输入状态，输出动作概率分布
    这是一个简单的两层全连接神经网络
    '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        # 第一层：将状态维度映射到隐藏层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层：将隐藏层映射到动作空间维度
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # 使用ReLU激活函数增加非线性
        x = F.relu(self.fc1(x))
        # 使用softmax函数将输出转换为概率分布，确保所有动作概率之和为1
        return F.softmax(self.fc2(x), dim=-1)


class ValueNet(torch.nn.Module):
    '''
    价值网络：输入状态，输出该状态的状态价值函数V(s)
    这是一个简单的两层全连接神经网络
    '''
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        # 第一层：将状态维度映射到隐藏层
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        # 第二层：将隐藏层映射到1维价值估计
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 使用ReLU激活函数增加非线性
        x = F.relu(self.fc1(x))
        # 输出状态价值
        return self.fc2(x)


class TRPOAgent:
    def __init__(self, env: gym.Env, hidden_dim, lmbda, kl_constraint, alpha, critic_lr, gamma, device):
        '''
        TRPO智能体初始化
        :param env: 环境对象
        :param hidden_dim: 隐藏层维度
        :param lmbda: GAE中的lambda参数，控制折扣权重
        :param kl_constraint: KL散度约束，限制策略更新幅度
        :param alpha: 线搜索中的步长缩放因子
        :param critic_lr: 价值网络学习率
        :param gamma: 折扣因子
        :param device: 计算设备
        '''
        self.env = env
        self.state_dim = env.observation_space.shape[0]  # 状态空间维度
        self.action_dim = env.action_space.n  # 动作空间维度

        # 初始化策略网络（Actor）
        self.actor = PolicyNet(self.state_dim, hidden_dim,
                               self.action_dim).to(device)
        # 初始化价值网络（Critic）
        self.critic = ValueNet(self.state_dim, hidden_dim).to(device)

        # 初始化价值网络优化器
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr)

        # TRPO相关超参数
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE参数
        self.kl_constraint = kl_constraint  # KL散度约束阈值
        self.alpha = alpha  # 线搜索步长缩放因子
        self.device = device

    def take_action(self, state):
        '''
        根据当前策略选择动作
        :param state: 当前状态
        :return: 选中的动作
        '''
        # 将状态转换为tensor并移到指定设备
        state_tensor = torch.tensor(state, dtype=torch.float).to(self.device)
        # 通过策略网络得到动作概率分布
        prob = self.actor(state_tensor)

        # 使用多项式采样从概率分布中采样动作
        action = torch.multinomial(prob, num_samples=1)  # 采样动作
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        '''
        计算海森矩阵与向量的乘积（Fisher信息矩阵向量积）
        这是TRPO算法的核心部分，用于计算策略更新方向
        :param states: 状态张量
        :param old_action_dists: 旧策略的动作分布
        :param vector: 输入向量
        :return: 海森矩阵与向量的乘积结果
        '''
        # 计算当前策略的动作概率分布
        prob = self.actor(states)
        new_action_dists = torch.distributions.Categorical(prob)
        
        # 计算新旧策略间的KL散度
        kl = torch.mean(torch.distributions.kl.kl_divergence(
            old_action_dists, new_action_dists))
        
        # 计算KL散度相对于策略参数的一阶导数
        kl_grad = torch.autograd.grad(
            kl, self.actor.parameters(), create_graph=True)
        # 将梯度展平成向量
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # 计算梯度向量与输入向量的点积
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        # 计算二阶梯度（即海森矩阵与向量的乘积）
        grad2 = torch.autograd.grad(
            kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        '''
        共轭梯度法求解线性方程组 Hx = g
        其中H是Fisher信息矩阵，g是策略梯度
        :param grad: 策略梯度
        :param states: 状态张量
        :param old_action_dists: 旧策略动作分布
        :return: 解向量（策略更新方向）
        '''
        # 初始化解向量为零
        x = torch.zeros_like(grad)
        # 初始化残差为梯度
        r = grad.clone()
        # 初始化搜索方向为梯度
        p = grad.clone()
        # 计算初始残差的平方范数
        r_dot_old = torch.dot(r, r)
        
        # 最多迭代10次
        for _ in range(10):
            # 计算Hp（Fisher矩阵与搜索方向的乘积）
            Hp = self.hessian_matrix_vector_product(
                states, old_action_dists, p)
            # 计算步长alpha
            alpha = r_dot_old / torch.dot(p, Hp)
            # 更新解向量
            x += alpha * p
            # 更新残差
            r -= alpha * Hp
            # 计算新的残差平方范数
            r_dot_new = torch.dot(r, r)
            # 如果残差很小，则收敛，退出循环
            if r_dot_new < 1e-10:
                break
            # 更新搜索方向
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        '''
        计算替代目标函数（代理目标函数）
        这是TRPO用来近似策略改进的目标函数
        :param states: 状态张量
        :param actions: 动作张量
        :param advantage: 优势函数值
        :param old_log_probs: 旧策略的动作对数概率
        :param actor: 策略网络
        :return: 代理目标函数值
        '''
        # 计算当前策略下动作的对数概率
        log_probs = torch.log(actor(states).gather(1, actions))
        # 计算重要性采样比率
        ratio = torch.exp(log_probs - old_log_probs)
        # 计算代理目标函数（期望的重要性采样优势）
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs, old_action_dists, mav_vec):
        '''
        线搜索：寻找合适的步长以确保策略更新满足约束
        :param states: 状态张量
        :param actions: 动作张量
        :param advantage: 优势函数值
        :param old_log_probs: 旧策略动作对数概率
        :param old_action_dists: 旧策略动作分布
        :param mav_vec: 策略更新向量
        :return: 新的参数向量
        '''
        # 获取当前策略参数
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        # 计算当前策略的代理目标函数值
        old_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor)

        # 尝试不同的步长系数
        for i in range(15):
            # 计算步长系数（按alpha的幂次递减）
            coef = self.alpha ** i
            # 计算新参数
            new_para = old_para + coef * mav_vec
            # 创建临时策略网络
            new_actor = copy.deepcopy(self.actor)
            # 将新参数应用到临时网络
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            # 计算新策略的动作分布
            new_action_dists = torch.distributions.Categorical(
                new_actor(states))
            # 计算新旧策略间的KL散度
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(
                old_action_dists, new_action_dists))
            # 计算新策略的代理目标函数值
            new_obj = self.compute_surrogate_obj(
                states, actions, advantage, old_log_probs, new_actor)
            # 如果新策略更好且KL散度满足约束，则接受更新
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        # 如果所有尝试都不满足条件，返回原参数
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs, advantage):
        '''
        TRPO策略学习：计算并执行策略更新
        :param states: 状态张量
        :param actions: 动作张量
        :param old_action_dists: 旧策略动作分布
        :param old_log_probs: 旧策略动作对数概率
        :param advantage: 优势函数值
        '''
        # 计算代理目标函数
        surrogate_obj = self.compute_surrogate_obj(
            states, actions, advantage, old_log_probs, self.actor)
        # 计算代理目标函数相对于策略参数的梯度
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        # 将梯度展平成向量
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 使用共轭梯度法求解策略更新方向
        descent_direction = self.conjugate_gradient(
            obj_grad, states, old_action_dists)
        # 计算Fisher矩阵与下降方向的乘积
        Hd = self.hessian_matrix_vector_product(
            states, old_action_dists, descent_direction)
        # 计算最大步长系数，确保KL散度约束
        max_coef = torch.sqrt(2 * self.kl_constraint /
                              (torch.dot(descent_direction, Hd) + 1e-8))
        # 通过线搜索找到合适的参数更新
        new_para = self.line_search(
            states, actions, advantage, old_log_probs, old_action_dists, descent_direction * max_coef)
        # 应用新的参数到策略网络
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())

    def compute_advantage(self, td_delta):
        '''
        计算广义优势估计（GAE）
        GAE结合了TD误差和折扣累积奖励，提供更稳定的梯度估计
        :param td_delta: TD误差（即时奖励+折扣后继状态价值-当前状态价值）
        :return: 优势函数值
        '''
        # 将TD误差转为numpy数组以便处理
        td_delta = td_delta.detach().numpy()
        advantage_list = []
        advantage = 0.0
        # 从后往前计算累积优势
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 将结果反转，使其与原始时间顺序对应
        advantage_list.reverse()
        # 转换为tensor并返回
        return torch.tensor(advantage_list, dtype=torch.float)

    def update(self, transition_dict):
        '''
        更新策略网络和价值网络
        :param transition_dict: 包含状态转移信息的字典
        '''
        # 将经验数据转换为tensor并移到指定设备
        states = torch.tensor(
            transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(
            transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(
            transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(
            transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(
            transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 计算目标价值（即时奖励+折扣后继状态价值）
        td_target = rewards + self.gamma * \
            self.critic(next_states) * (1 - dones)
        # 计算TD误差（目标价值-当前价值估计）
        td_delta = td_target - self.critic(states)
        # 计算优势函数
        advantage = self.compute_advantage(td_delta.cpu()).to(self.device)

        # 更新策略网络
        # 计算旧策略下动作的对数概率
        old_log_probs = torch.log(self.actor(
            states).gather(1, actions)).detach()
        # 创建旧策略的动作分布
        old_action_dists = torch.distributions.Categorical(
            self.actor(states).detach())
        # 计算价值网络损失
        critic_loss = torch.mean(F.mse_loss(
            self.critic(states), td_target.detach()))
        # 清空价值网络梯度
        self.critic_optimizer.zero_grad()
        # 反向传播计算价值网络梯度
        critic_loss.backward()
        # 更新价值网络参数
        self.critic_optimizer.step()
        # 使用TRPO算法更新策略网络
        self.policy_learn(states, actions, old_action_dists,
                          old_log_probs, advantage)


def main():
    # 设置训练参数
    num_episodes = 3000  # 训练回合数
    hidden_dim = 128  # 神经网络隐藏层维度
    gamma = 0.98  # 折扣因子
    lmbda = 0.95  # GAE参数
    critic_lr = 1e-2  # 价值网络学习率
    kl_constraint = 0.0005  # KL散度约束（TRPO的关键参数）
    alpha = 0.5  # 线搜索步长缩放因子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备选择

    env_name = 'CartPole-v1'  # 环境名称
    total_rewards = []  # 存储每回合总奖励
    env = gym.make(env_name)  # 创建环境
    # 初始化TRPO智能体
    agent = TRPOAgent(env, hidden_dim, lmbda, kl_constraint,
                      alpha, critic_lr, gamma, device)

    # 开始训练循环
    for i_episode in tqdm(range(num_episodes)):
        # 重置环境
        state, _ = env.reset()
        episode_return = 0
        done = False
        # 存储一个回合的经验
        transition_dict = {'states': [], 'actions': [],
                           'rewards': [], 'next_states': [], 'dones': []}
        # 执行一个回合
        while not done:
            # 根据当前策略选择动作
            action = agent.take_action(state)
            # 执行动作，获得下一个状态和奖励
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 判断回合是否结束

            # 存储经验数据
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)
            transition_dict['next_states'].append(next_state)
            transition_dict['dones'].append(done)

            state = next_state
            episode_return += reward

            if done:
                break

        # 使用收集到的经验更新策略
        agent.update(transition_dict)

        total_rewards.append(episode_return)
        # 每100轮打印一次训练信息
        if (i_episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])
            print(
                f"Episode {i_episode + 1}/{num_episodes}, Average Reward: {avg_reward:.2f}"
            )

    # 使用训练好的模型进行测试
    test_episodes = 10

    env = gym.make(env_name, render_mode="human")  # 创建渲染环境
    for _ in range(test_episodes):
        episode_return = 0
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_return += reward
        print(f"Test Episode Return: {episode_return}")

    plot(total_rewards)


def plot(rewards):
    '''
    绘制奖励曲线图
    :param rewards: 奖励序列
    '''
    # 绘制原始奖励曲线
    plt.plot(rewards, alpha=0.3)
    # 计算并绘制移动平均奖励曲线（窗口大小为50）
    avg = np.convolve(rewards, np.ones(50) / 50, mode="valid")
    plt.plot(range(49, len(rewards)), avg)
    plt.title("TRPO on CartPole-v1")  # 修改标题以反映实际使用的算法
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    plt.savefig("trpo_cartpole.png", dpi=150)  # 修改保存的文件名
    plt.close()


if __name__ == "__main__":
    main()
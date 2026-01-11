import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # Softmax会在forward方法中添加稳定性处理
        )

    def forward(self, x):
        # 添加数值稳定性处理：减去最大值防止指数爆炸
        logits = self.net(x)
        logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        return nn.functional.softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, state_value_dim=1):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_value_dim)  # 输出状态值V(s)
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic:
    def __init__(self, env: gym.Env, device, lr_actor, lr_critic, epsilon, epsilon_decay_rate, epsilon_end, gamma, state_dim, hidden_dim, action_dim):
        self.env = env
        self.device = device
        self.actor_lr = lr_actor
        self.critic_lr = lr_critic
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # 初始化网络并移动到指定设备
        self.actor = Actor(state_dim=state_dim, hidden_dim=hidden_dim, action_dim=action_dim).to(device)
        self.critic = Critic(state_dim=state_dim, hidden_dim=hidden_dim).to(device)

        # 创建优化器，降低学习率以提高稳定性
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 定义损失函数
        self.critic_loss_fn = nn.MSELoss()

    def choose_action(self, state):
        # 将状态转换为张量并添加批次维度
        tensor_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_probs = self.actor(tensor_state)

            # 确保概率分布有效
            action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
            # 重新归一化概率
            action_probs = action_probs / torch.sum(action_probs, dim=-1, keepdim=True)

        # ε-贪婪策略选择动作
        if np.random.rand() < self.epsilon:
            # 探索：随机选择动作
            action = np.random.randint(self.action_dim)
        else:
            try:
                # 利用：根据概率分布采样动作
                action = torch.multinomial(action_probs, num_samples=1).item()
            except RuntimeError:
                # 如果采样失败，回退到随机选择
                print("概率采样失败，回退到随机选择")
                action = np.random.randint(self.action_dim)

        return action

    def update(self, transition_dict):
        # 先将列表转换为numpy数组，再转换为张量，提高效率
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float32).to(self.device)

        # 计算当前状态和下一状态的价值
        current_v = self.critic(states)
        next_v = self.critic(next_states)

        # 计算TD目标和TD误差
        td_target = rewards + (1 - dones) * self.gamma * next_v
        td_error = td_target - current_v

        # 更新Critic网络
        critic_loss = self.critic_loss_fn(current_v, td_target.detach()) # Detach td_target here
        self.optim_critic.zero_grad()
        critic_loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.optim_critic.step()

        # 计算Actor网络的损失
        action_probs = self.actor(states)
        # 确保概率分布有效
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        log_probs = torch.log(action_probs)

        selected_log_probs = log_probs[range(len(actions)), actions]
        actor_loss = -torch.mean(selected_log_probs * td_error.detach()) # Detach td_error here

        # 更新Actor网络
        self.optim_actor.zero_grad()
        actor_loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.optim_actor.step()

        return actor_loss.item(), critic_loss.item()

    def epsilon_decay(self):
      self.epsilon = max(self.epsilon_end, self.epsilon_start * (self.epsilon_decay_rate ** self.epsilon_start))

    def save_model(self, path):
        """保存模型参数"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.optim_actor.state_dict(),
            'critic_optimizer_state_dict': self.optim_critic.state_dict(),
        }, path)

    def load_model(self, path):
        """加载模型参数"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optim_actor.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.optim_critic.load_state_dict(checkpoint['critic_optimizer_state_dict'])


def main():
    # 创建环境
    env = gym.make('MountainCar-v0')

    # 设置设备（M1 Mac使用mps，否则使用cpu）
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print(f"使用设备: {device}")

    # 设置参数，降低学习率以提高稳定性
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 64  # 减小隐藏层大小，提高训练稳定性
    lr_actor = 1e-4  # 降低Actor学习率
    lr_critic = 1e-4  # 降低Critic学习率
    epsilon = 0.1  # 初始ε-贪婪策略的探索率
    epsilon_min = 0.01  # 最小探索率
    epsilon_decay_rate = 0.999  # 探索率衰减因子
    gamma = 0.99  # 折扣因子

    # 创建AC智能体
    actor_critic = ActorCritic(
        env=env,
        device=device,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        epsilon=epsilon,
        epsilon_decay_rate=epsilon_decay_rate,
        epsilon_end=epsilon_min,
        gamma=gamma,
        state_dim=state_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim
    )

    # 训练参数
    num_episodes = 5000  # 增加训练回合数
    max_steps_per_episode = 200 # Note: MountainCar-v0 has a default step limit of 200, so this line isn't strictly necessary but good for clarity.

    # 开始训练
    for episode in tqdm(range(num_episodes)):
        

        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_actor_losses = []
        episode_critic_losses = []

        while not done:
            # 选择动作
            action = actor_critic.choose_action(state)

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 构建转换字典
            transition_dict = {
                'states': [state],
                'actions': [action],
                'rewards': [reward],
                'next_states': [next_state],
                'dones': [done]
            }

            # 更新网络
            actor_loss, critic_loss = actor_critic.update(transition_dict)
            episode_actor_losses.append(actor_loss)
            episode_critic_losses.append(critic_loss)

            # 更新状态
            state = next_state
            total_reward += reward

        # 每100回合打印一次训练信息
        if (episode + 1) % 100 == 0:
            avg_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0
            avg_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0
            print(f"Episode {episode+1}: Total Reward = {total_reward}, "
                  f"Actor Loss = {avg_actor_loss:.6f}, Critic Loss = {avg_critic_loss:.6f}, Epsilon = {actor_critic.epsilon:.4f}")
        # 更新 epsilon 以平衡探索与利用
        actor_critic.epsilon_decay()

    # 保存模型
    actor_critic.save_model('ac_mountaincar.pth')
    print("模型已保存为 ac_mountaincar.pth")

    # 测试训练好的模型
    try:
        test_env = gym.make('MountainCar-v0', render_mode='human')
        actor_critic.epsilon = 0.0  # 测试时不使用探索

        print("\n开始测试模型...")
        for episode in range(5):
            state, _ = test_env.reset()
            done = False
            total_reward = 0

            while not done:
                action = actor_critic.choose_action(state)
                next_state, reward, terminated, truncated, _ = test_env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward

            print(f"测试回合 {episode+1}: Total Reward = {total_reward}")

        test_env.close()
    except Exception as e:
        print(f"测试时出错: {e}")


if __name__ == '__main__':
    main()
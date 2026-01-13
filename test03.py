import gymnasium as gym
from DQNLearning import DQNAgent
from DQNLearning import Params
from matplotlib import pyplot as plt
import torch


def main():
    env = gym.make("CartPole-v1")

    params = Params(
        gamma=0.99,
        lr=0.001,
        epsilon_start=1.0,
        epsilon_decay=0.05,
        epsilon_end=0.0001,
        batch_size=64,
        buffer_capacity=10000,
        target_update=10,
        state_size=env.observation_space.shape[0],
        hidden_size=128,
        action_size=env.action_space.n,
    )

    agent = DQNAgent(
        env,
        state_size=params.state_size,
        hidden_size=params.hidden_size,
        action_size=params.action_size,
        gamma=params.gamma,
        lr=params.lr,
        epsilon_start=params.epsilon_start,
        epsilon_decay=params.epsilon_decay,
        epsilon_end=params.epsilon_end,
        batch_size=params.batch_size,
        buffer_capacity=params.buffer_capacity,
        target_update=params.target_update,
    )

    num_episodes = 500
    max_steps_per_episode = 1000
    total_reward_list = []

    for episode in range(num_episodes):
        state, _ = env.reset(seed=42)
        total_reward = 0
        for step in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, terminated)
            agent.update()
            state = next_state
            total_reward += reward

            if done:
                break

        agent.epsilon_decay_step()
        if episode % agent.target_update == 0:
            agent.update_target_network()

        print(
            f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}"
        )
        total_reward_list.append(total_reward)
        
        if total_reward >= 500:
            agent.save_model("dqn_cartpole_model.pth")

    plot_awards(total_reward_list)
    
    # 使用模型测试智能体
    agent.policy_net.load_state_dict(torch.load("dqn_cartpole_model.pth"))
    test_episodes = 100
    env = gym.make("CartPole-v1")
    total_reward_list = []
    for episode in range(test_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
        total_reward_list.append(total_reward)
        print(f"Test Episode {episode + 1}/{test_episodes}, Total Reward: {total_reward}")
        
    plot_awards(total_reward_list, "dqn_cartpole_rewards_test.png")
    
    


def plot_awards(total_reward_list, filename="dqn_cartpole_rewards.png"):
    plt.plot(total_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN on CartPole-v1")
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    main()

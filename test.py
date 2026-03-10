import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from agents.dqn_agent import DQNAgent


def train():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)

    episode_rewards = []
    best_reward = 0

    for episode in range(200):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 每步都更新模型
            agent.update()


        episode_rewards.append(total_reward)



        # 打印进度
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"Episode {episode:4d} | Avg Reward (last 20): {avg_reward:6.2f} | Epsilon: {agent.epsilon:.3f}")

        # 判断是否收敛（连续100轮平均≥195）
        if len(episode_rewards) >= 100:
            avg_100 = np.mean(episode_rewards[-100:])
            if avg_100 >= 195:
                print(f"Solved in {episode} episodes! Avg 100 reward: {avg_100:.2f}")
                break

    env.close()

    # 绘制奖励曲线
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training on CartPole-v1')
    plt.show()

if __name__ == "__main__":
    train()
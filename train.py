from matplotlib import pyplot as plt

from DDQNAgent import DDQNAgent, DQNAgent
import numpy as np
from Env import RadarEnvironment, Radar, StealthUAV


def showTrainResults(env, agent):
    """展示训练结果"""
    state = env.reset()
    done = False
    total_reward = 0
    # 开启交互模式
    plt.ion()

    # 创建图形和坐标轴
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('X-axis(Km)')
    ax.set_ylabel('Y-axis(Km)')
    ax.set_title('Pre-Flight Path')

    plt.legend(["Flight Path"])  # 图例

    # 初始化数据存储
    x_data, y_data = [], []
    line, = ax.plot([], [], 'b-', marker='o')  # 折线+圆点


    while not done:
        action = agent.select_action(state)
        x, y = state
        # 更新数据
        x_data.append(x)
        y_data.append(y)
        # 更新线条数据
        line.set_data(x_data, y_data)
        # 刷新图形
        plt.pause(0.01)  # 暂停 0.01 秒，同时刷新图形

        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward

    plt.savefig("figures/pre_flight_path.png")
    # 关闭交互模式（可选）
    plt.ioff()
    plt.show()  # 保持窗口开启



def train():
    """训练主函数"""
    # agent = DDQNAgent()
    agent = DQNAgent()
    env = RadarEnvironment()

    # 阶段1: 预训练（无雷达威胁）
    print("Phase 1: Pre-training without radar threats...")
    env.set_radar_enabled(False)

    for episode in range(800):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            # action = 0
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay * agent.epsilon)
        if (episode + 1) % 50 == 0:
            print(f"Pre-training Episode {episode + 1}, Return: {total_reward:.2f}")

    # 展示预训练结果
    showTrainResults(env, agent)


    # 阶段2: 正式训练（四雷达环境）
    print("\nPhase 2: Training with radar threats...")
    env.set_radar_enabled(True)

    returns = []
    # for episode in range(800):
    #     state = env.reset()
    #     done = False
    #     total_reward = 0
    #
    #     while not done:
    #         action = agent.select_action(state)
    #         next_state, reward, done, info = env.step(action)
    #         agent.store_transition(state, action, reward, next_state, done)
    #         agent.update()
    #         state = next_state
    #         total_reward += reward
    #
    #     returns.append(total_reward)
    #
    #     if (episode + 1) % 100 == 0:
    #         avg_return = np.mean(returns[-100:])
    #         print(f"Episode {episode + 1}, Avg Return: {avg_return:.2f}, Status: {info['status']}")

    return agent, returns


if __name__ == "__main__":
    agent, returns = train()
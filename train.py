from matplotlib import pyplot as plt

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
import numpy as np
from envs.Env import RadarEnvironment, Radar, StealthUAV
from utils.visualization import plot_convergence_curve

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
    agent = DDQNAgent()
    # agent = DQNAgent(state_dim=2, action_dim=5)
    env = RadarEnvironment()

    # 阶段1: 预训练（无雷达威胁）
    print("Phase 1: Pre-training without radar threats...")
    env.set_radar_enabled(False)

    returns = []

    for episode in range(400):
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

        returns.append(total_reward)
        if (episode + 1) % 50 == 0:
            print(f"Pre-training Episode {episode + 1}, Return: {total_reward:.2f}")
            avg_reward = np.mean(returns[-20:])
            print(f"Episode {episode + 1:4d} | Avg Reward (last 50): {avg_reward:6.2f} | Epsilon: {agent.epsilon:.3f}")

    # 展示预训练结果
    showTrainResults(env, agent)
    save_path = "./figures"
    print("Generating Figure : Convergence (No Threat)...")
    fig = plot_convergence_curve(returns,
                                  "Figure : Optimal Convergence Rate (No Threat)",
                                  with_threat=False)
    fig.savefig(f"{save_path}/figure_convergence_no_threat.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 阶段2: 正式训练（四雷达环境）
    # print("\nPhase 2: Training with radar threats...")
    # env.set_radar_enabled(True)
    #
    # returns = []
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
    # 展示正式训练结果
    # save_path = "./figures"
    # print("Generating Figure : Convergence ...")
    # fig = plot_convergence_curve(returns,
    #                              "Figure : Optimal Convergence Rate ",
    #                              with_threat=False)
    # fig.savefig(f"{save_path}/figure_convergence_threat.png", dpi=300, bbox_inches='tight')
    # plt.close(fig)

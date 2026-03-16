from matplotlib import pyplot as plt

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
import numpy as np
from envs.environment import RadarEnvironment
from utils.visualization import plot_convergence_curve, plot_real_time_metrics


def pre_train(env, agent):
    """预训练"""
    # 阶段1: 预训练（无雷达威胁）
    print("Phase 1: Pre-training without radar threats...")
    env.set_radar_enabled(False)
    q_values = []
    losses = []
    returns = []

    # 开启交互模式
    plt.ion()
    plt.figure(figsize=(10, 8))

    for episode in range(1000):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

        returns.append(total_reward)
        # 收集Q值和损失数据
        if agent.q_value_history:
            q_values.append(np.mean(agent.q_value_history[-10:]))  # 取最近10个Q值的平均值
        if agent.loss_history:
            losses.append(np.mean(agent.loss_history[-10:]))  # 取最近10个损失的平均值

        if (episode + 1) % 10 == 0:  # 每10个episode更新一次图表
            print(f"Pre-training Episode {episode + 1}, Return: {total_reward:.2f}")
            avg_reward = np.mean(returns[-20:])
            print(f"Episode {episode + 1:4d} | Avg Reward (last 20): {avg_reward:6.2f} | Epsilon: {agent.epsilon:.3f}")
            # 实时绘制指标
            plot_real_time_metrics(returns, q_values, losses, episode + 1)
    # 关闭交互模式
    plt.ioff()

    # 保存预训练结果图
    save_path = "./figures"
    print("Generating Figure : Convergence (No Threat)...")
    fig = plot_convergence_curve(returns,
                                 "Figure : Optimal Convergence Rate (No Threat)",
                                 with_threat=False)
    fig.savefig(f"{save_path}/figure_convergence_no_threat_dqn.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    return agent


def train(env, agent):
    """训练主函数"""
    agent = DDQNAgent()
    # agent = DQNAgent(state_dim=2, action_dim=5)
    env = RadarEnvironment()
    returns = []

    # 加载预训练模型
    # agent.load_model("models/ddqn_pretrained.pt")
    agent.epsilon = 1.0
    # agent.epsilon_decay = 0.999

    # 阶段2: 正式训练（四雷达环境）
    print("\nPhase 2: Training with radar threats...")
    env.set_radar_enabled(True)

    q_values = []
    losses = []

    # 开启交互模式
    plt.ion()
    plt.figure(figsize=(10, 8))

    for episode in range(1000):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

        returns.append(total_reward)
        # 收集Q值和损失数据
        if agent.q_value_history:
            q_values.append(np.mean(agent.q_value_history[-10:]))  # 取最近10个Q值的平均值
        if agent.loss_history:
            losses.append(np.mean(agent.loss_history[-10:]))  # 取最近10个损失的平均值

        if (episode + 1) % 10 == 0:  # 每10个episode更新一次图表
            print(f"Pre-training Episode {episode + 1}, Return: {total_reward:.2f}")
            avg_reward = np.mean(returns[-20:])
            print(f"Episode {episode + 1:4d} | Avg Reward (last 20): {avg_reward:6.2f} | Epsilon: {agent.epsilon:.3f}")
            # 实时绘制指标
            plot_real_time_metrics(returns, q_values, losses, episode + 1)

        if (episode + 1) % 100 == 0:
            avg_return = np.mean(returns[-100:])
            print(f"Episode {episode + 1}, Avg Return: {avg_return:.2f}, Status: {info['status']}")

    # 关闭交互模式
    plt.ioff()
    return agent, returns


if __name__ == "__main__":
    agent = DDQNAgent()
    env = RadarEnvironment()
    # 预训练
    agent = pre_train(env, agent)
    # 保存预训练模型
    # agent.save_model("models/ddqn_pretrained.pt")
    # agent.save_model("models/dqn_pretrained.pt")

    # 正式训练
    # agent = train(env, agent)
    # 保存正式训练模型
    # agent.save_model("models/ddqn_trained.pt")
    # 展示正式训练结果
    # save_path = "./figures"
    # print("Generating Figure : Convergence ...")
    # fig = plot_convergence_curve(returns,
    #                              "Figure : Optimal Convergence Rate (With Threat)",
    #                              with_threat=True)
    # fig.savefig(f"{save_path}/figure_convergence_threat.png", dpi=300, bbox_inches='tight')
    # plt.close(fig)

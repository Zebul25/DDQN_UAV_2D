import numpy as np
from matplotlib import pyplot as plt
from agents.ddqn_agent import DDQNAgent
from envs.Env import RadarEnvironment
from utils.visualization import plot_convergence_curve


def test_model(model_path):
    """测试保存的模型"""
    # 创建环境和智能体
    env = RadarEnvironment()
    agent = DDQNAgent()
    env.set_radar_enabled(False)

    # 加载模型
    agent.load_model(model_path)

    returns = []

    for episode in range(800):
        # 测试模型
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward

        returns.append(total_reward)
        if episode % 20 == 0:
            print(f"Episode {episode:4d} | Total Reward: {total_reward:6.2f}")

    save_path = "./figures"
    print("Generating Figure : PreTestModel...")
    fig = plot_convergence_curve(returns,
                                 "Figure : PreTestModel",
                                 with_threat=False)
    fig.savefig(f"{save_path}/pre_test_model.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    # 测试预训练模型
    test_model("../models/ddqn_pretrained.pt")
    # 测试正式训练模型
    # test_model("models/ddqn_trained.pt")

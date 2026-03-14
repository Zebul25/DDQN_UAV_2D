from matplotlib import pyplot as plt


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

    plt.savefig("figures/flight_path.png")
    # 关闭交互模式（可选）
    plt.ioff()
    plt.show()  # 保持窗口开启

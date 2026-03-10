"""
基于DDQN的隐身无人机路径规划 - 可视化模块
论文: Stealth UAV Path Planning Based on DDQN Against Multi-Radar Detection
Aerospace 2025, 12, 774

本模块实现论文中所有图表的可视化：
- Figure 3: 检测惩罚与检测概率关系
- Figure 4: 固定RCS的雷达检测概率分布
- Figure 7: 动态RCS的雷达检测概率分布  
- Figure 8: 四雷达环境下的路径规划对比
- Figure 5/6/9: 路径与收敛曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import erfc
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# =============================================================================
# 第一部分：雷达模型与检测概率计算
# =============================================================================

class RadarSystem:
    """雷达系统类 - 用于计算检测概率"""
    
    def __init__(self):
        # 雷达参数 (Table 2)
        self.G = 10 ** (20/10)      # 20 dB → 线性
        self.Pt = 30e6              # 30 MW
        self.f0 = 9e9               # 9 GHz
        self.c = 3e8
        self.wavelength = self.c / self.f0
        self.k = 1.38e-23           # 玻尔兹曼常数
        self.Te = 290               # K
        self.B = 100e6              # 100 MHz
        self.Fn = 1
        self.L = 1
        self.Pfa = 1e-6             # 虚警概率
    
    def calculate_SNR(self, R, sigma):
        """
        计算信噪比 (公式10)
        R: 距离 (m)
        sigma: RCS (m²)
        """
        if R < 1:
            return float('inf')
        
        numerator = self.Pt * (self.G ** 2) * (self.wavelength ** 2) * sigma
        denominator = ((4 * np.pi) ** 3) * (R ** 4) * self.k * self.Te * self.B * self.Fn * self.L
        
        return numerator / denominator if denominator > 0 else 0
    
    def calculate_Pd(self, R, sigma):
        """
        计算检测概率 (公式8)
        R: 距离 (m)
        sigma: RCS (m²)
        """
        SNR = self.calculate_SNR(R, sigma)
        z = np.sqrt(-np.log(self.Pfa)) - np.sqrt(SNR + 0.5)
        Pd = 0.5 * erfc(z)
        return np.clip(Pd, 0, 1)


def get_dynamic_RCS(uav_position, uav_heading, radar_position):
    """
    计算动态RCS
    
    Args:
        uav_position: 无人机位置 (x, y)
        uav_heading: 无人机航向角 (度)
        radar_position: 雷达位置 (x, y)
    
    Returns:
        RCS值 (m²)
    """
    direction = np.array(radar_position) - np.array(uav_position)
    angle_to_radar = np.arctan2(direction[1], direction[0])
    relative_angle = np.degrees(angle_to_radar) - uav_heading
    relative_angle = relative_angle % 360
    
    # RCS模型参数
    sigma_min = 0.01   # m², 正面最小RCS
    sigma_max = 1.0    # m², 侧面最大RCS
    
    # 根据相对角度计算RCS
    if relative_angle < 30 or relative_angle > 330:
        return sigma_min
    elif 60 < relative_angle < 120 or 240 < relative_angle < 300:
        return sigma_max
    else:
        # 使用余弦函数平滑过渡
        angle_rad = np.radians(relative_angle)
        return sigma_min + (sigma_max - sigma_min) * (0.5 + 0.5 * np.cos(2 * angle_rad))


# =============================================================================
# 第二部分：Figure 3 - 检测惩罚与检测概率关系
# =============================================================================

def plot_detection_penalty():
    """
    绘制Figure 3: 检测惩罚与检测概率的关系
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 定义检测概率区间和对应惩罚值
    Pd_ranges = [
        (0.0, 0.3, 0),
        (0.3, 0.5, -5),
        (0.5, 0.6, -10),
        (0.6, 0.7, -20),
        (0.7, 0.8, -40),
        (0.8, 0.9, -80),
        (0.9, 1.0, -100)
    ]
    
    # 绘制阶梯函数
    for start, end, penalty in Pd_ranges:
        ax.hlines(y=penalty, xmin=start, xmax=end, colors='blue', linewidth=2)
        if end < 1.0:
            ax.plot(end, penalty, 'bo', markersize=6)  # 实心点表示包含
            ax.plot(end, Pd_ranges[Pd_ranges.index((start, end, penalty)) + 1][2], 
                   'bo', fillstyle='none', markersize=6)  # 空心点表示不包含
    
    ax.set_xlabel('Detection probability', fontsize=12)
    ax.set_ylabel('$R_b$', fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-100, 10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Figure 3: Detection Penalty vs Detection Probability', fontsize=12)
    
    plt.tight_layout()
    return fig


# =============================================================================
# 第三部分：Figure 4 - 固定RCS的雷达检测概率分布
# =============================================================================

def plot_radar_detection_fixed_RCS(sigma=0.1):
    """
    绘制Figure 4: 固定RCS时的雷达空域检测概率
    当RCS为常数时，检测概率分布呈同心圆形状
    
    Args:
        sigma: 固定RCS值 (m²)，论文中使用0.1
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    radar = RadarSystem()
    radar_position = (50, 50)  # 雷达位于中心
    
    # 创建网格
    x = np.linspace(0, 100, 500)
    y = np.linspace(0, 100, 500)
    X, Y = np.meshgrid(x, y)
    
    # 计算每个点的检测概率
    Pd = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            distance = np.sqrt((X[i,j] - radar_position[0])**2 + 
                              (Y[i,j] - radar_position[1])**2) * 1000  # km转m
            Pd[i,j] = radar.calculate_Pd(distance, sigma)
    
    # 创建自定义颜色映射（蓝色渐变）
    colors = ['#FFFFFF', '#E6F3FF', '#CCE7FF', '#99CFFF', '#66B7FF', 
              '#339FFF', '#0087FF', '#0066CC', '#004C99', '#003366']
    cmap = LinearSegmentedColormap.from_list('radar_blue', colors)
    
    # 绘制热力图
    im = ax.contourf(X, Y, Pd, levels=50, cmap=cmap, vmin=0, vmax=1)
    
    # 添加等高线
    contours = ax.contour(X, Y, Pd, levels=[0.2, 0.4, 0.6, 0.8], 
                         colors='white', linewidths=0.5, linestyles='--')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='Detection probability')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 标记雷达位置
    ax.plot(radar_position[0], radar_position[1], 'k^', markersize=10, label='Radar')
    
    ax.set_xlabel('X-axis(Km)', fontsize=12)
    ax.set_ylabel('Y-axis(Km)', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.set_title(f'Figure 4: Radar Detection Probability (σ = {sigma} m²)', fontsize=12)
    
    plt.tight_layout()
    return fig


# =============================================================================
# 第四部分：Figure 7 - 动态RCS的雷达检测概率分布
# =============================================================================

def plot_radar_detection_dynamic_RCS():
    """
    绘制Figure 7: 动态RCS时的雷达空域检测概率
    当考虑动态RCS时，检测概率分布不再是同心圆，而是呈现不规则形状
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    radar = RadarSystem()
    
    # 四部雷达位置
    radar_positions = [(30, 30), (30, 70), (70, 40), (70, 80)]
    
    # 创建网格
    x = np.linspace(0, 100, 300)
    y = np.linspace(0, 100, 300)
    X, Y = np.meshgrid(x, y)
    
    # 假设无人机航向固定为45度（朝向目标）
    uav_heading = 45
    
    # 计算最大检测概率（任一雷达检测到即可）
    Pd_max = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            uav_pos = (X[i,j], Y[i,j])
            max_pd = 0
            
            for radar_pos in radar_positions:
                # 计算动态RCS
                sigma = get_dynamic_RCS(uav_pos, uav_heading, radar_pos)
                
                # 计算距离和检测概率
                distance = np.sqrt((uav_pos[0] - radar_pos[0])**2 + 
                                  (uav_pos[1] - radar_pos[1])**2) * 1000
                pd = radar.calculate_Pd(distance, sigma)
                max_pd = max(max_pd, pd)
            
            Pd_max[i,j] = max_pd
    
    # 创建颜色映射
    colors = ['#FFFFFF', '#E6F3FF', '#CCE7FF', '#99CFFF', '#66B7FF', 
              '#339FFF', '#0087FF', '#0066CC', '#004C99', '#003366']
    cmap = LinearSegmentedColormap.from_list('radar_blue', colors)
    
    # 绘制热力图
    im = ax.contourf(X, Y, Pd_max, levels=50, cmap=cmap, vmin=0, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='Detection probability')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 标记雷达位置
    for pos in radar_positions:
        ax.plot(pos[0], pos[1], 'k^', markersize=10)
    
    ax.set_xlabel('X-axis(Km)', fontsize=12)
    ax.set_ylabel('Y-axis(Km)', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.set_title('Figure 7: Radar Detection Probability (Dynamic RCS)', fontsize=12)
    
    plt.tight_layout()
    return fig


# =============================================================================
# 第五部分：Figure 5 - 无威胁环境下的直线路径
# =============================================================================

def plot_path_without_threat():
    """
    绘制Figure 5: 无威胁环境下的无人机飞行路径
    在没有雷达威胁时，无人机应该沿直线飞向目标
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建空白检测概率背景（全为0）
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)
    Pd = np.zeros_like(X)
    
    # 绘制淡蓝色背景
    ax.contourf(X, Y, Pd, levels=[0, 0.1], colors=['#F0F8FF'], alpha=0.5)
    
    # 起点和终点
    start = (0, 0)
    target = (100, 100)
    
    # 绘制直线路径
    ax.plot([start[0], target[0]], [start[1], target[1]], 
            'b-', linewidth=2, label='Flight path')
    
    # 标记起点和终点
    ax.plot(start[0], start[1], 'ko', markersize=12, label='Departure position')
    ax.plot(target[0], target[1], 'o', color='orange', markersize=12, label='Target position')
    
    # 添加颜色条（用于与其他图保持一致）
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Detection probability')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    ax.set_xlabel('X-axis(Km)', fontsize=12)
    ax.set_ylabel('Y-axis(Km)', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.legend(loc='upper left')
    ax.set_title('Figure 5: Stealth UAV Path Without Anti-aircraft Fire', fontsize=12)
    
    plt.tight_layout()
    return fig


# =============================================================================
# 第六部分：Figure 6 & 9 - 收敛曲线
# =============================================================================

def plot_convergence_curve(returns, title="Convergence Curve", with_threat=False):
    """
    绘制收敛曲线 (Figure 6 或 Figure 9)
    
    Args:
        returns: 每轮的累计回报列表
        title: 图表标题
        with_threat: 是否有雷达威胁
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = np.arange(len(returns))
    
    # 绘制原始曲线
    ax.plot(episodes, returns, 'b-', linewidth=1, alpha=0.7)
    
    # 添加移动平均线（平滑曲线）
    window = 20
    if len(returns) >= window:
        smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], smoothed, 'r-', linewidth=2, label='Moving Average')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('cumulative rewardReturn' if not with_threat else 'Return', fontsize=12)
    ax.set_xlim(0, len(returns))
    
    if with_threat:
        ax.set_ylim(-1000, 100)
    else:
        ax.set_ylim(-200, 400)
    
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    return fig


def simulate_convergence_data(with_threat=False, episodes=800):
    """
    模拟生成收敛曲线数据
    用于演示可视化效果
    """
    np.random.seed(42)
    
    if with_threat:
        # 有威胁时的收敛过程
        returns = []
        for i in range(episodes):
            if i < 100:
                # 初期：大量失败
                base = -800 + i * 5
            elif i < 300:
                # 中期：逐渐学习
                base = -300 + (i - 100) * 1.2
            else:
                # 后期：趋于稳定
                base = -60 + 10 * np.sin(i / 50)
            
            noise = np.random.normal(0, 50)
            returns.append(base + noise)
    else:
        # 无威胁时的收敛过程（更快收敛）
        returns = []
        for i in range(episodes):
            if i < 50:
                base = -80 + i * 1.2
            else:
                base = -20 + 5 * np.sin(i / 30)
            
            noise = np.random.normal(0, 10)
            returns.append(base + noise)
    
    return returns


def plot_real_time_metrics(returns, q_values, losses, episode):
    """实时绘制训练指标"""
    plt.clf()

    # 创建3个子图
    plt.subplot(3, 1, 1)
    plt.plot(returns)
    plt.title(f'Training Metrics - Episode {episode}')
    plt.ylabel('Reward')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    if q_values:
        plt.plot(q_values)
        plt.ylabel('Q Value')
        plt.grid(True)

    plt.subplot(3, 1, 3)
    if losses:
        plt.plot(losses)
        plt.ylabel('Loss')
        plt.xlabel('Update Step')
        plt.grid(True)

    plt.tight_layout()
    plt.pause(0.01)  # 暂停以更新图表

# =============================================================================
# 第七部分：Figure 8 - 四雷达环境下的路径对比
# =============================================================================

def plot_path_comparison():
    """
    绘制Figure 8: 四雷达环境下三种算法的路径对比
    - DDQN + 动态RCS (本文方法)
    - DDQN + 固定RCS
    - A*算法
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    radar = RadarSystem()
    radar_positions = [(30, 30), (30, 70), (70, 40), (70, 80)]
    
    # 创建网格计算检测概率
    x = np.linspace(0, 100, 300)
    y = np.linspace(0, 100, 300)
    X, Y = np.meshgrid(x, y)
    
    # 假设无人机航向固定为45度
    uav_heading = 45
    Pd_max = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            uav_pos = (X[i,j], Y[i,j])
            max_pd = 0
            
            for radar_pos in radar_positions:
                sigma = get_dynamic_RCS(uav_pos, uav_heading, radar_pos)
                distance = np.sqrt((uav_pos[0] - radar_pos[0])**2 + 
                                  (uav_pos[1] - radar_pos[1])**2) * 1000
                pd = radar.calculate_Pd(distance, sigma)
                max_pd = max(max_pd, pd)
            
            Pd_max[i,j] = max_pd
    
    # 绘制检测概率热力图
    colors = ['#FFFFFF', '#E6F3FF', '#CCE7FF', '#99CFFF', '#66B7FF', 
              '#339FFF', '#0087FF', '#0066CC', '#004C99', '#003366']
    cmap = LinearSegmentedColormap.from_list('radar_blue', colors)
    
    im = ax.contourf(X, Y, Pd_max, levels=50, cmap=cmap, vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, label='Detection probability')
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 模拟三种算法的路径
    # 路径1: DDQN + 动态RCS (蓝色，本文方法)
    path_ddqn_dynamic = [
        (0, 0), (5, 5), (10, 10), (15, 15), (20, 20),
        (25, 22), (30, 20), (35, 18), (40, 20),
        (45, 25), (50, 35), (55, 45), (60, 55),
        (62, 58), (65, 62), (68, 68), (72, 75),
        (78, 82), (85, 88), (92, 95), (100, 100)
    ]
    
    # 路径2: A*算法 (绿色)
    path_astar = [
        (0, 0), (7, 7), (14, 14), (21, 21),
        (28, 25), (35, 23), (42, 22), (49, 28),
        (52, 35), (55, 42), (58, 50), (62, 58),
        (68, 65), (75, 72), (82, 80), (88, 88),
        (94, 94), (100, 100)
    ]
    
    # 路径3: DDQN + 固定RCS (红色)
    path_ddqn_fixed = [
        (0, 0), (5, 5), (10, 10), (15, 15),
        (18, 18), (20, 25), (22, 35), (25, 45),
        (30, 52), (38, 55), (48, 55), (58, 52),
        (65, 55), (72, 62), (78, 72), (82, 80),
        (88, 88), (95, 95), (100, 100)
    ]
    
    # 绘制路径
    path_ddqn_dynamic = np.array(path_ddqn_dynamic)
    path_astar = np.array(path_astar)
    path_ddqn_fixed = np.array(path_ddqn_fixed)
    
    ax.plot(path_ddqn_dynamic[:, 0], path_ddqn_dynamic[:, 1], 
            'b-', linewidth=2.5, label='DDQN algorithm with dynamic RCS')
    ax.plot(path_astar[:, 0], path_astar[:, 1], 
            'g-', linewidth=2.5, label='A* algorithm')
    ax.plot(path_ddqn_fixed[:, 0], path_ddqn_fixed[:, 1], 
            'r-', linewidth=2.5, label='DDQN algorithm with fixed RCS')
    
    # 标记起点和终点
    ax.plot(0, 0, 'ko', markersize=12, label='Departure position')
    ax.plot(100, 100, 'o', color='orange', markersize=12, label='Target position')
    
    # 标记雷达位置
    for pos in radar_positions:
        ax.plot(pos[0], pos[1], 'k^', markersize=10)
    
    ax.set_xlabel('X-axis(Km)', fontsize=12)
    ax.set_ylabel('Y-axis(Km)', fontsize=12)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_title('Figure 8: Stealth UAV Path When Four Air Defense Radars Are Deployed', 
                fontsize=12)
    
    plt.tight_layout()
    return fig


# =============================================================================
# 第八部分：实时可视化类（用于训练过程中）
# =============================================================================

class RealtimeVisualizer:
    """
    实时可视化器 - 用于训练过程中的动态显示
    """
    
    def __init__(self, radar_positions, airspace_size=(100, 100)):
        self.radar_positions = radar_positions
        self.airspace_size = airspace_size
        self.radar = RadarSystem()
        
        # 预计算检测概率网格
        self._precompute_detection_grid()
        
        # 设置交互模式
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 6))
        
    def _precompute_detection_grid(self, resolution=100):
        """预计算检测概率网格以加速可视化"""
        x = np.linspace(0, self.airspace_size[0], resolution)
        y = np.linspace(0, self.airspace_size[1], resolution)
        self.X, self.Y = np.meshgrid(x, y)
        
        uav_heading = 45
        self.Pd_grid = np.zeros_like(self.X)
        
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                uav_pos = (self.X[i,j], self.Y[i,j])
                max_pd = 0
                
                for radar_pos in self.radar_positions:
                    sigma = get_dynamic_RCS(uav_pos, uav_heading, radar_pos)
                    distance = np.sqrt((uav_pos[0] - radar_pos[0])**2 + 
                                      (uav_pos[1] - radar_pos[1])**2) * 1000
                    pd = self.radar.calculate_Pd(distance, sigma)
                    max_pd = max(max_pd, pd)
                
                self.Pd_grid[i,j] = max_pd
    
    def update(self, path, returns, episode):
        """
        更新可视化
        
        Args:
            path: 当前轮次的路径点列表 [(x1,y1), (x2,y2), ...]
            returns: 历史回报列表
            episode: 当前轮次
        """
        # 清除之前的图
        self.axes[0].clear()
        self.axes[1].clear()
        
        # 左图：路径可视化
        ax1 = self.axes[0]
        
        colors = ['#FFFFFF', '#E6F3FF', '#CCE7FF', '#99CFFF', '#66B7FF', 
                  '#339FFF', '#0087FF', '#0066CC', '#004C99', '#003366']
        cmap = LinearSegmentedColormap.from_list('radar_blue', colors)
        
        ax1.contourf(self.X, self.Y, self.Pd_grid, levels=50, cmap=cmap, vmin=0, vmax=1)
        
        # 绘制路径
        if len(path) > 0:
            path = np.array(path)
            ax1.plot(path[:, 0], path[:, 1], 'b-', linewidth=2)
            ax1.plot(path[-1, 0], path[-1, 1], 'go', markersize=8)  # 当前位置
        
        # 标记起点终点和雷达
        ax1.plot(0, 0, 'ko', markersize=10)
        ax1.plot(100, 100, 'o', color='orange', markersize=10)
        for pos in self.radar_positions:
            ax1.plot(pos[0], pos[1], 'k^', markersize=8)
        
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X (km)')
        ax1.set_ylabel('Y (km)')
        ax1.set_title(f'Episode {episode}: UAV Path')
        
        # 右图：收敛曲线
        ax2 = self.axes[1]
        
        if len(returns) > 0:
            ax2.plot(returns, 'b-', linewidth=1, alpha=0.7)
            
            # 移动平均
            window = min(20, len(returns))
            if len(returns) >= window:
                smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(returns)), smoothed, 'r-', linewidth=2)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Return')
        ax2.set_title('Training Convergence')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def close(self):
        """关闭可视化窗口"""
        plt.ioff()
        plt.close(self.fig)


# =============================================================================
# 第九部分：生成所有论文图表
# =============================================================================

def generate_all_figures(save_path="../figures"):
    """
    生成论文中的所有图表
    
    Args:
        save_path: 图片保存路径
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    print("Generating Figure 3: Detection Penalty vs Probability...")
    fig3 = plot_detection_penalty()
    fig3.savefig(f"{save_path}/figure3_detection_penalty.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("Generating Figure 4: Radar Detection (Fixed RCS)...")
    fig4 = plot_radar_detection_fixed_RCS(sigma=0.1)
    fig4.savefig(f"{save_path}/figure4_fixed_rcs.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print("Generating Figure 5: Path Without Threat...")
    fig5 = plot_path_without_threat()
    fig5.savefig(f"{save_path}/figure5_no_threat_path.png", dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    print("Generating Figure 6: Convergence (No Threat)...")
    returns_no_threat = simulate_convergence_data(with_threat=False, episodes=800)
    fig6 = plot_convergence_curve(returns_no_threat, 
                                  "Figure 6: Optimal Convergence Rate (No Threat)", 
                                  with_threat=False)
    fig6.savefig(f"{save_path}/figure6_convergence_no_threat.png", dpi=300, bbox_inches='tight')
    plt.close(fig6)
    
    print("Generating Figure 7: Radar Detection (Dynamic RCS)...")
    fig7 = plot_radar_detection_dynamic_RCS()
    fig7.savefig(f"{save_path}/figure7_dynamic_rcs.png", dpi=300, bbox_inches='tight')
    plt.close(fig7)
    
    print("Generating Figure 8: Path Comparison...")
    fig8 = plot_path_comparison()
    fig8.savefig(f"{save_path}/figure8_path_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig8)
    
    print("Generating Figure 9: Convergence (With Threat)...")
    returns_with_threat = simulate_convergence_data(with_threat=True, episodes=800)
    fig9 = plot_convergence_curve(returns_with_threat, 
                                  "Figure 9: Convergence Rate (With Radar Threat)", 
                                  with_threat=True)
    fig9.savefig(f"{save_path}/figure9_convergence_with_threat.png", dpi=300, bbox_inches='tight')
    plt.close(fig9)
    
    print(f"\nAll figures saved to: {save_path}/")
    print("Generated files:")
    print("  - figure3_detection_penalty.png")
    print("  - figure4_fixed_rcs.png")
    print("  - figure5_no_threat_path.png")
    print("  - figure6_convergence_no_threat.png")
    print("  - figure7_dynamic_rcs.png")
    print("  - figure8_path_comparison.png")
    print("  - figure9_convergence_with_threat.png")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("DDQN Stealth UAV Path Planning - Visualization Module")
    print("=" * 60)
    
    # 生成所有图表
    generate_all_figures(save_path="./figures")
    
    print("\nDone!")

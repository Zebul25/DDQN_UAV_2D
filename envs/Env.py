import numpy as np
from scipy.special import erfc


class Radar:
    """雷达类"""

    def __init__(self, position):
        """
        初始化雷达

        Args:
            position: 雷达位置 (x, y) in km
        """
        self.position = np.array(position)

        # 雷达参数 (Table 2)
        self.G = 10 ** (20 / 10)  # 20 dB → 线性
        self.Pt = 30e6  # 30 MW
        self.f0 = 9e9  # 9 GHz
        self.c = 3e8
        self.wavelength = self.c / self.f0
        self.k = 1.38e-23  # 玻尔兹曼常数
        self.Te = 290  # K
        self.B = 100e6  # 100 MHz
        self.Fn = 1  # 噪声因子
        self.L = 1  # 损耗因子
        self.Pfa = 1e-6  # 虚警概率

    def calculate_distance(self, uav_position):
        """计算到无人机的距离 (m)"""
        return np.linalg.norm(self.position - np.array(uav_position)) * 1000

    def calculate_SNR(self, distance, sigma):
        """计算信噪比"""
        numerator = self.Pt * (self.G ** 2) * (self.wavelength ** 2) * sigma
        denominator = ((4 * np.pi) ** 3) * (distance ** 4) * self.k * self.Te * self.B * self.Fn * self.L
        return numerator / denominator if denominator > 0 else 0

    def calculate_detection_probability(self, uav_position, sigma):
        """计算检测概率"""
        R = self.calculate_distance(uav_position)
        if R < 1:
            return 1.0

        SNR = self.calculate_SNR(R, sigma)
        z = np.sqrt(-np.log(self.Pfa)) - np.sqrt(SNR + 0.5)
        Pd = 0.5 * erfc(z)
        return np.clip(Pd, 0, 1)


class StealthUAV:
    """隐身无人机类"""

    def __init__(self, start_position):
        self.position = np.array(start_position, dtype=float)
        self.heading = 45  # 初始航向角
        self.step_size = 4  # 每步飞行距离 (km)

    def get_dynamic_RCS(self, radar_position):
        """计算动态RCS"""
        # 计算从无人机到雷达的方向向量
        direction = np.array(radar_position) - self.position

        # 计算相对于雷达的角度（弧度转换为角度）
        angle_to_radar = np.arctan2(direction[1], direction[0])

        # 计算雷达方向与无人机航向之间的相对角度
        relative_angle = np.degrees(angle_to_radar - np.radians(self.heading)) % 360

        # 定义RCS的最小值和最大值
        sigma_min = 0.01  # 最小RCS值（隐身效果最好时）
        sigma_max = 1.0  # 最大RCS值（最容易被探测到）

        # 根据相对角度确定当前的RCS值：（---相对角度是以无人机来说的---）
        # - 当雷达位于无人机前方(±30度范围内)或后方(330-360度)时，返回最小RCS值（隐身效果最佳）
        # - 当雷达位于无人机侧方特定角度范围(60-120度或240-300度)时，返回最大RCS值（容易被探测）
        # - 其他角度返回中等RCS值（平均值）
        if relative_angle < 30 or relative_angle > 330:
            return sigma_min
        elif 60 < relative_angle < 120 or 240 < relative_angle < 300:
            return sigma_max
        else:
            return (sigma_min + sigma_max) / 2

    def move(self, action_index):
        """执行动作"""
        action_angles = [0, 45, 90, -45, -90]
        turn_angle = action_angles[action_index]

        self.heading = (self.heading + turn_angle) % 360
        dx = self.step_size * np.cos(np.radians(self.heading))
        dy = self.step_size * np.sin(np.radians(self.heading))
        self.position = self.position + np.array([dx, dy])

        return self.position.copy()


class RadarEnvironment:
    """雷达对抗环境"""

    def __init__(self):
        # 空域参数
        self.airspace_size = (100, 100)
        self.start_point = (0, 0)
        self.target_point = (100, 100)
        self.target_threshold = 5  # 到达判定距离 (km)

        # 雷达配置
        self.radar_positions = [(30, 30), (30, 70), (70, 40), (70, 80)]
        self.radars = [Radar(pos) for pos in self.radar_positions]
        self.radar_enabled = True

        # 检测阈值
        self.Pd_destroy_threshold = 0.9

        # 无人机
        self.uav = None

    def reset(self):
        """重置环境"""
        self.uav = StealthUAV(self.start_point)
        return self.uav.position.copy()

    def set_radar_enabled(self, enabled):
        """设置雷达开关（用于预训练）"""
        self.radar_enabled = enabled

    def get_max_detection_probability(self):
        """获取最大检测概率"""
        if not self.radar_enabled:
            return 0.0

        max_Pd = 0.0
        for radar in self.radars:
            sigma = self.uav.get_dynamic_RCS(radar.position)
            Pd = radar.calculate_detection_probability(self.uav.position, sigma)
            max_Pd = max(max_Pd, Pd)

        return max_Pd

    def calculate_reward(self, Pd_max):
        """计算奖励"""
        Ra, Rb, Rc = 0, 0, -1

        # 到达奖励
        dist = np.linalg.norm(self.uav.position - np.array(self.target_point))
        if dist < self.target_threshold:
            Ra = 100

        # 检测惩罚（---有待调整---）
        if Pd_max < 0.3:
            Rb = 0
        elif Pd_max < 0.5:
            Rb = -5
        elif Pd_max < 0.6:
            Rb = -10
        elif Pd_max < 0.7:
            Rb = -20
        elif Pd_max < 0.8:
            Rb = -40
        elif Pd_max < 0.9:
            Rb = -80
        else:
            Rb = -100

        return Ra + Rb + Rc

    def check_boundary(self):
        """检查边界"""
        x, y = self.uav.position
        return 0 <= x <= self.airspace_size[0] and 0 <= y <= self.airspace_size[1]

    def check_arrival(self):
        """检查是否到达"""
        dist = np.linalg.norm(self.uav.position - np.array(self.target_point))
        return dist < self.target_threshold

    def step(self, action):
        """执行一步"""
        new_position = self.uav.move(action)
        Pd_max = self.get_max_detection_probability()
        reward = self.calculate_reward(Pd_max)

        done = False
        info = {"status": "flying", "Pd_max": Pd_max}

        if self.check_arrival():
            done = True
            info["status"] = "arrived"
        elif Pd_max >= self.Pd_destroy_threshold:
            done = True
            info["status"] = "destroyed"
        elif not self.check_boundary():
            done = True
            reward = -100
            info["status"] = "out_of_bounds"

        return new_position, reward, done, info
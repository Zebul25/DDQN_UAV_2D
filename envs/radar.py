from scipy.special import erfc

import numpy as np


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
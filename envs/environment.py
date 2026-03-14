import numpy as np
from scipy.special import erfc

from envs.radar import Radar
from envs.uav import StealthUAV


class RadarEnvironment:
    """雷达对抗环境"""

    def __init__(self):
        # 空域参数
        self.airspace_size = (100, 100)
        self.start_point = (0, 0)
        self.target_point = (100, 100)
        self.target_threshold = 4  # 到达判定距离 (km)

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
        Ra, Rb, Rc = 0, 0, -0.1

        # 到达奖励
        dist = np.linalg.norm(self.uav.position - np.array(self.target_point))
        if dist < self.target_threshold:
            Ra = 200

        # 检测惩罚（---有待调整---）
        if Pd_max < 0.3:
            Rb = 0
        elif Pd_max < 0.5:
            Rb = -2
        elif Pd_max < 0.6:
            Rb = -4
        elif Pd_max < 0.7:
            Rb = -8
        elif Pd_max < 0.8:
            Rb = -16
        elif Pd_max < 0.9:
            Rb = -32
        else:
            Rb = -64

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
        old_position, new_position = self.uav.move(action)
        Pd_max = self.get_max_detection_probability()
        reward = self.calculate_reward(Pd_max)

        # 设计1
        # 距离越近得到奖励相反得到惩罚
        old_dist = np.linalg.norm(old_position - np.array(self.target_point))
        new_dist = np.linalg.norm(new_position - np.array(self.target_point))
        reward = reward + (old_dist - new_dist) * 0.3

        # 设计2
        # 计算距离变化
        # dist_change = old_dist - new_dist
        # 基础奖励系数，随距离减小而增大
        # 使用倒数函数，确保离终点越近，相同距离变化的奖励越大
        # min_dist = 1.0  # 避免除以零
        # old_reward_factor = 10.0 / max(old_dist, min_dist)
        # new_reward_factor = 10.0 / max(new_dist, min_dist)
        #
        # # 计算距离奖励
        # distance_reward = dist_change * (old_reward_factor + new_reward_factor) / 2
        # reward = reward + distance_reward
        # 设计3
        # dist_current = np.linalg.norm(new_position - np.array(self.target_point))
        # dist_start = np.linalg.norm(np.array(self.start_point) - np.array(self.target_point))
        # distance_reward = 5 * (1 - dist_current / dist_start)
        # reward = reward + distance_reward
        done = False
        info = {"status": "flying", "Pd_max": Pd_max}

        if self.check_arrival():
            done = True
            info["status"] = "arrived"
        elif Pd_max >= self.Pd_destroy_threshold:
            done = True
            reward = -50
            info["status"] = "destroyed"
        elif not self.check_boundary():
            done = True
            reward = -100
            info["status"] = "out_of_bounds"


        return new_position, reward, done, info

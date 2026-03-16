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
        self.init_dist = np.linalg.norm(self.start_point - np.array(self.target_point))

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

    def angle_to_goal(self):
        """计算到目标的方位角"""
        dx = self.target_point[0] - self.uav.position[0]
        dy = self.target_point[1] - self.uav.position[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 360

    def step(self, action):
        """执行一步"""
        old_position, new_position = self.uav.move(action)
        Pd_max = self.get_max_detection_probability()
        reward = self.calculate_reward(Pd_max)
        r_path, r_heading, r_collision, r_boundary = 0, 0, 0, 0

        # 路径奖励
        # 距离越近得到奖励相反得到惩罚
        old_dist = np.linalg.norm(old_position - np.array(self.target_point))
        new_dist = np.linalg.norm(new_position - np.array(self.target_point))
        dist_change = old_dist - new_dist
        # 归一化奖励
        r_path = dist_change / self.init_dist * 10

        # 航向奖励
        # 计算航向偏差
        ideal_heading = self.angle_to_goal()
        heading_diff = abs(self.uav.heading - ideal_heading)
        heading_diff = min(heading_diff, 360 - heading_diff)  # 取较小的角度差
        # 归一化到 [0, 1]，角度差越小奖励越大
        r_heading = 1.0 - (heading_diff / 180.0)

        done = False
        info = {"status": "flying", "Pd_max": Pd_max}

        if self.check_arrival():
            done = True
            info["status"] = "arrived"
        elif Pd_max >= self.Pd_destroy_threshold:
            done = True
            r_collision = -50
            info["status"] = "destroyed"
        elif not self.check_boundary():
            done = True
            r_boundary = -100
            info["status"] = "out_of_bounds"

        # 总奖励
        reward = r_path + r_heading + r_collision + r_boundary

        return new_position, reward, done, info

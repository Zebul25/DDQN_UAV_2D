import math

import numpy as np
from scipy.special import erfc

from envs.radar import Radar
from envs.uav import StealthUAV


def _normalize_angle(angle):
    """将弧度角归一化到 [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def distance_to_goal(state):
    """计算到目标的距离"""
    return state.dist_goal


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

    def get_state(self):
        """获取当前状态"""
        # 位置归一化
        x_norm = self.uav.position[0] / 100.0
        y_norm = self.uav.position[1] / 100.0

        # 航向 (sin/cos 避免角度跳变)
        heading_rad = np.radians(self.uav.heading)

        # 目标信息
        dx = 100 - self.uav.position[0]
        dy = 100 - self.uav.position[1]
        dist_goal = np.sqrt(dx ** 2 + dy ** 2) / 141.4  # 归一化

        goal_angle = np.arctan2(dy, dx)
        relative_angle = _normalize_angle(goal_angle - heading_rad)

        # 雷达信息（关键！）
        radar_info = []
        for radar in self.radars:
            dist = np.sqrt((self.uav.position[0] - radar.position[0]) ** 2 +
                           (self.uav.position[1] - radar.position[1]) ** 2) / 100.0
            prob = self.get_detection_probability(radar)
            radar_info.extend([dist, prob])

        return np.array([
            x_norm, y_norm,  # 2
            np.sin(heading_rad), np.cos(heading_rad),  # 2
            dist_goal, relative_angle / np.pi,  # 2
            *radar_info  # 8
        ], dtype=np.float32)  # 共14维

    def reset(self):
        """重置环境"""
        self.uav = StealthUAV(self.start_point)

        return self.get_state()

    def set_radar_enabled(self, enabled):
        """设置雷达开关（用于预训练）"""
        self.radar_enabled = enabled

    def get_detection_probability(self, radar: Radar):
        """获取最大检测概率"""
        if not self.radar_enabled:
            return 0.0
        sigma = self.uav.get_dynamic_RCS(radar.position)
        return radar.calculate_detection_probability(self.uav.position, sigma)

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

    def calculate_reward(self, state, next_state):
        reward = 0

        # 1. 到达目标（保持）
        if self.check_arrival():
            return 100  # 降低一点，让其他奖励更有意义

        # 2. 撞边界
        if self.check_boundary():
            return -20  # 降低，不要比雷达惩罚还重

        # 3. 路径进展奖励（关键改进）
        dist_prev = state[4]
        dist_curr = next_state[4]
        dist_change = dist_prev - dist_curr  # 正值=接近目标

        # 归一化到合理范围
        r_progress = dist_change * 2  # 步长5，最大变化5，奖励最大10

        # 4. 雷达惩罚（大幅降低）
        detection_prob = self.get_max_detection_probability()
        if detection_prob < 0.3:
            r_radar = 0
        elif detection_prob < 0.5:
            r_radar = -1
        elif detection_prob < 0.7:
            r_radar = -2
        elif detection_prob < 0.9:
            r_radar = -4
        else:
            r_radar = -6

        # 5. 移除每步固定惩罚（已有进展奖励）
        # r_step = -0.1  # 删除或降到 -0.01

        # 6. 航向奖励（简化）
        goal_direction = math.atan2(100 - next_state[1], 100 - next_state[0])
        heading_diff = abs(_normalize_angle(next_state[2] - goal_direction))
        r_heading = 0.5 * (1 - heading_diff / 180)  # 范围[0, 0.5]

        reward = r_progress + r_radar + r_heading
        return reward

    # def calculate_reward(self, Pd_max):
    #     """计算奖励"""
    #     Ra, Rb, Rc = 0, 0, -0.1
    #
    #     # 到达奖励
    #     dist = np.linalg.norm(self.uav.position - np.array(self.target_point))
    #     if dist < self.target_threshold:
    #         Ra = 100
    #
    #     # 检测惩罚（---有待调整---）
    #     if Pd_max < 0.3:
    #         Rb = 0
    #     elif Pd_max < 0.5:
    #         Rb = -5
    #     elif Pd_max < 0.6:
    #         Rb = -10
    #     elif Pd_max < 0.7:
    #         Rb = -20
    #     elif Pd_max < 0.8:
    #         Rb = -30
    #     elif Pd_max < 0.9:
    #         Rb = -40
    #     else:
    #         Rb = -50
    #
    #     return Ra + Rb + Rc

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
        state = self.get_state()
        old_position, new_position = self.uav.move(action)
        next_state = self.get_state()
        Pd_max = self.get_max_detection_probability()
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
            info["status"] = "out_of_bounds"

        reward = self.calculate_reward(state, next_state)

        return next_state, reward, done, info

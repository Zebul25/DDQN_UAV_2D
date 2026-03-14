import numpy as np


class StealthUAV:
    """隐身无人机类"""

    def __init__(self, start_position):
        self.position = np.array(start_position, dtype=float)
        self.heading = 45  # 初始航向角
        self.step_size =5  # 每步飞行距离 (km)

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
        old_position = self.position.copy()
        self.position = self.position + np.array([dx, dy])

        return old_position, self.position.copy()
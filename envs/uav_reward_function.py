"""
UAV强化学习奖励函数设计
环境: 100x100 2D空间
任务: 从(0,0)到(100,100)，同时规避雷达检测

Author: Claude
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class RadarConfig:
    """雷达配置"""
    position: Tuple[float, float]  # 雷达位置
    detection_radius: float = 30.0  # 基础探测半径
    max_power: float = 1.0  # 最大探测强度


@dataclass
class UAVState:
    """无人机状态"""
    x: float
    y: float
    heading: int  # 航向角度 (0, 45, 90, 135, 180, 225, 270, 315)


class RCSModel:
    """
    RCS (Radar Cross Section) 角度模型
    不同角度下的雷达散射截面积不同
    """
    
    # RCS角度系数 (相对于无人机航向的角度)
    # 0° = 正面, 90° = 侧面, 180° = 尾部
    RCS_COEFFICIENTS = {
        0: 1.0,      # 正面 - 最大RCS
        45: 0.8,     # 前侧面
        90: 0.5,     # 侧面 - 中等RCS
        135: 0.3,    # 后侧面
        180: 0.2,    # 尾部 - 最小RCS
        225: 0.3,    # 后侧面
        270: 0.5,    # 侧面
        315: 0.8,    # 前侧面
    }
    
    @classmethod
    def get_rcs(cls, relative_angle: float) -> float:
        """
        根据相对角度获取RCS系数
        
        Args:
            relative_angle: 无人机相对于雷达的角度 (度)
            
        Returns:
            RCS系数 (0-1)
        """
        # 归一化角度到 [0, 360)
        angle = relative_angle % 360
        
        # 使用插值计算RCS
        angles = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])
        rcs_values = np.array([1.0, 0.8, 0.5, 0.3, 0.2, 0.3, 0.5, 0.8, 1.0])
        
        return np.interp(angle, angles, rcs_values)


class RewardFunction:
    """
    综合奖励函数设计
    
    R_total = α·R_goal + β·R_heading + γ·R_radar + δ·R_boundary + ε·R_arrival + λ·R_step
    """
    
    def __init__(
        self,
        # 环境参数
        env_size: Tuple[float, float] = (100, 100),
        start_pos: Tuple[float, float] = (0, 0),
        goal_pos: Tuple[float, float] = (100, 100),
        goal_radius: float = 5.0,
        
        # 雷达配置
        radars: List[RadarConfig] = None,
        
        # 奖励权重
        alpha: float = 1.0,   # 目标距离奖励权重
        beta: float = 0.5,    # 航向奖励权重
        gamma: float = 2.0,   # 雷达惩罚权重 (较大以强调安全)
        delta: float = 5.0,   # 边界惩罚权重
        epsilon: float = 100.0,  # 到达目标奖励
        lambda_: float = 0.1,    # 时间步惩罚
        
        # 其他参数
        detection_threshold: float = 0.7,  # 检测概率阈值
    ):
        self.env_size = env_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.goal_radius = goal_radius
        
        # 默认雷达配置
        if radars is None:
            self.radars = [
                RadarConfig(position=(30, 30), detection_radius=25),
                RadarConfig(position=(60, 60), detection_radius=25),
                RadarConfig(position=(30, 60), detection_radius=25),
                RadarConfig(position=(60, 30), detection_radius=25),
            ]
        else:
            self.radars = radars
            
        # 权重
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.lambda_ = lambda_
        
        self.detection_threshold = detection_threshold
        
        # 计算初始距离用于归一化
        self.initial_distance = np.sqrt(
            (goal_pos[0] - start_pos[0])**2 + 
            (goal_pos[1] - start_pos[1])**2
        )
        
    def _distance_to_goal(self, state: UAVState) -> float:
        """计算到目标的距离"""
        return np.sqrt(
            (self.goal_pos[0] - state.x)**2 + 
            (self.goal_pos[1] - state.y)**2
        )
    
    def _angle_to_goal(self, state: UAVState) -> float:
        """计算到目标的方位角"""
        dx = self.goal_pos[0] - state.x
        dy = self.goal_pos[1] - state.y
        angle = np.degrees(np.arctan2(dy, dx))
        return angle % 360
    
    def _compute_detection_probability(
        self, 
        state: UAVState, 
        radar: RadarConfig
    ) -> float:
        """
        计算被单个雷达检测到的概率
        
        P_d = σ(θ) · exp(-d²/2R²)
        
        其中:
        - σ(θ) 是RCS角度函数
        - d 是到雷达的距离
        - R 是雷达探测半径
        """
        # 计算到雷达的距离
        distance = np.sqrt(
            (radar.position[0] - state.x)**2 + 
            (radar.position[1] - state.y)**2
        )
        
        # 如果距离超过探测半径的2倍，检测概率接近0
        if distance > radar.detection_radius * 2:
            return 0.0
        
        # 计算无人机相对于雷达的角度
        radar_angle = np.degrees(np.arctan2(
            state.y - radar.position[1],
            state.x - radar.position[0]
        ))
        
        # 计算相对角度 (无人机航向相对于雷达方向)
        # 这决定了雷达"看到"无人机的哪个面
        relative_angle = (state.heading - radar_angle + 180) % 360
        
        # 获取RCS系数
        rcs = RCSModel.get_rcs(relative_angle)
        
        # 计算基于距离的检测概率
        distance_factor = np.exp(-(distance**2) / (2 * radar.detection_radius**2))
        
        # 总检测概率
        detection_prob = rcs * distance_factor * radar.max_power
        
        return np.clip(detection_prob, 0.0, 1.0)
    
    def reward_goal_distance(self, state: UAVState, prev_state: UAVState) -> float:
        """
        R_goal: 目标距离奖励
        
        鼓励无人机接近目标，基于距离变化给予奖励
        """
        current_dist = self._distance_to_goal(state)
        prev_dist = self._distance_to_goal(prev_state)
        
        # 距离减少则奖励为正
        distance_change = prev_dist - current_dist
        
        # 归一化奖励
        reward = distance_change / self.initial_distance * 10
        
        return reward
    
    def reward_heading(self, state: UAVState) -> float:
        """
        R_heading: 航向奖励
        
        鼓励无人机航向朝向目标方向
        """
        # 计算理想航向 (指向目标的角度)
        ideal_heading = self._angle_to_goal(state)
        
        # 计算航向偏差
        heading_diff = abs(state.heading - ideal_heading)
        heading_diff = min(heading_diff, 360 - heading_diff)  # 取较小的角度差
        
        # 归一化到 [0, 1]，角度差越小奖励越大
        reward = 1.0 - (heading_diff / 180.0)
        
        return reward
    
    def penalty_radar(self, state: UAVState) -> float:
        """
        R_radar: 雷达检测惩罚
        
        基于RCS和检测概率的综合惩罚
        """
        total_penalty = 0.0
        
        for radar in self.radars:
            detection_prob = self._compute_detection_probability(state, radar)
            
            # 分级惩罚
            if detection_prob > self.detection_threshold:
                # 高检测概率: 严重惩罚
                penalty = -10.0 * detection_prob
            elif detection_prob > 0.3:
                # 中等检测概率: 中等惩罚
                penalty = -3.0 * detection_prob
            else:
                # 低检测概率: 轻微惩罚
                penalty = -0.5 * detection_prob
            
            total_penalty += penalty
        
        return total_penalty
    
    def penalty_boundary(self, state: UAVState) -> float:
        """
        R_boundary: 边界惩罚
        
        防止无人机越界
        """
        penalty = 0.0
        margin = 5.0  # 边界缓冲区
        
        # 检查X边界
        if state.x < margin:
            penalty -= (margin - state.x) / margin * 5
        elif state.x > self.env_size[0] - margin:
            penalty -= (state.x - (self.env_size[0] - margin)) / margin * 5
            
        # 检查Y边界
        if state.y < margin:
            penalty -= (margin - state.y) / margin * 5
        elif state.y > self.env_size[1] - margin:
            penalty -= (state.y - (self.env_size[1] - margin)) / margin * 5
        
        # 完全越界: 严重惩罚
        if state.x < 0 or state.x > self.env_size[0] or \
           state.y < 0 or state.y > self.env_size[1]:
            penalty = -50.0
        
        return penalty
    
    def reward_arrival(self, state: UAVState) -> float:
        """
        R_arrival: 到达目标奖励
        """
        distance = self._distance_to_goal(state)
        
        if distance <= self.goal_radius:
            return self.epsilon  # 大额奖励
        return 0.0
    
    def penalty_step(self) -> float:
        """
        R_step: 时间步惩罚
        
        鼓励无人机快速到达目标
        """
        return -self.lambda_
    
    def compute_total_reward(
        self, 
        state: UAVState, 
        prev_state: UAVState,
        done: bool = False
    ) -> Dict[str, float]:
        """
        计算总奖励
        
        Returns:
            包含各分项奖励的字典
        """
        rewards = {
            'goal_distance': self.alpha * self.reward_goal_distance(state, prev_state),
            'heading': self.beta * self.reward_heading(state),
            'radar': self.gamma * self.penalty_radar(state),
            'boundary': self.delta * self.penalty_boundary(state),
            'arrival': self.reward_arrival(state),
            'step': self.penalty_step(),
        }
        
        rewards['total'] = sum(rewards.values())
        
        return rewards
    
    def get_state_info(self, state: UAVState) -> Dict:
        """
        获取当前状态的详细信息 (用于调试和可视化)
        """
        info = {
            'position': (state.x, state.y),
            'heading': state.heading,
            'distance_to_goal': self._distance_to_goal(state),
            'angle_to_goal': self._angle_to_goal(state),
            'radar_detections': [],
        }
        
        for i, radar in enumerate(self.radars):
            detection_prob = self._compute_detection_probability(state, radar)
            info['radar_detections'].append({
                'radar_id': i,
                'position': radar.position,
                'detection_probability': detection_prob,
            })
        
        return info


class AdaptiveRewardFunction(RewardFunction):
    """
    自适应奖励函数
    
    根据训练进度动态调整权重
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_count = 0
        
    def update_weights(self, episode: int, success_rate: float):
        """
        根据训练进度调整权重
        
        Args:
            episode: 当前训练轮次
            success_rate: 最近的成功率
        """
        self.episode_count = episode
        
        # 早期训练: 强调目标导向
        if episode < 1000:
            self.alpha = 2.0
            self.gamma = 1.0
        # 中期训练: 平衡目标和安全
        elif episode < 5000:
            self.alpha = 1.5
            self.gamma = 2.0
        # 后期训练: 强调安全规避
        else:
            self.alpha = 1.0
            self.gamma = 3.0
        
        # 如果成功率较高，增加安全惩罚以优化路径
        if success_rate > 0.8:
            self.gamma *= 1.5


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    
    # 创建奖励函数
    reward_fn = RewardFunction(
        env_size=(100, 100),
        start_pos=(0, 0),
        goal_pos=(100, 100),
        alpha=1.0,
        beta=0.5,
        gamma=2.0,
        delta=5.0,
        epsilon=100.0,
        lambda_=0.1,
    )
    
    # 模拟状态转移
    prev_state = UAVState(x=10, y=10, heading=45)
    curr_state = UAVState(x=15, y=15, heading=45)
    
    # 计算奖励
    rewards = reward_fn.compute_total_reward(curr_state, prev_state)
    
    print("=" * 50)
    print("奖励函数计算结果:")
    print("=" * 50)
    for key, value in rewards.items():
        print(f"{key:20s}: {value:+.4f}")
    print("=" * 50)
    
    # 获取状态信息
    info = reward_fn.get_state_info(curr_state)
    print("\n状态信息:")
    print(f"位置: {info['position']}")
    print(f"航向: {info['heading']}°")
    print(f"到目标距离: {info['distance_to_goal']:.2f}")
    print(f"到目标角度: {info['angle_to_goal']:.2f}°")
    print("\n雷达检测概率:")
    for radar_info in info['radar_detections']:
        print(f"  雷达 {radar_info['radar_id']} @ {radar_info['position']}: "
              f"{radar_info['detection_probability']:.3f}")


def test_rcs_model():
    """测试RCS模型"""
    print("\n" + "=" * 50)
    print("RCS角度模型测试:")
    print("=" * 50)
    
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    for angle in angles:
        rcs = RCSModel.get_rcs(angle)
        bar = "█" * int(rcs * 20)
        print(f"角度 {angle:3d}°: RCS = {rcs:.2f} {bar}")


if __name__ == "__main__":
    example_usage()
    test_rcs_model()

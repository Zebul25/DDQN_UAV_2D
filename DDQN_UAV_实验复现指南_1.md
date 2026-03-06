# 基于DDQN的隐身无人机多雷达对抗路径规划

## 实验复现指南

**论文来源**: Aerospace 2025, 12, 774  
**作者**: Lei Bao, Zhengtao Guo, Xianzhong Gao, Chaolong Li  
**单位**: 国防科技大学测试中心  
**DOI**: https://doi.org/10.3390/aerospace12090774

---

## 目录

1. [研究概述](#一研究概述)
2. [实验平台](#二实验平台)
3. [环境模型创建](#三环境模型创建)
4. [核心算法实现](#四核心算法实现)
5. [奖励函数设计](#五奖励函数设计)
6. [雷达检测概率模型](#六雷达检测概率模型)
7. [动态RCS模型](#七动态rcs模型)
8. [训练流程](#八训练流程)
9. [完整代码实现](#九完整代码实现)
10. [复现建议与注意事项](#十复现建议与注意事项)

---

## 一、研究概述

### 1.1 研究背景

无人机突防雷达防空区域的问题可建模为路径约束、火力威胁约束和飞行时间约束下的路径规划问题。传统方法通常假设无人机雷达散射截面(RCS)为常数，忽略了隐身无人机不同角度RCS差异显著的特性。

本文提出了一种基于双深度Q网络(DDQN)的隐身无人机路径规划算法，考虑了动态RCS特性对雷达检测概率的影响，实现了路径规划方案的迭代优化。

### 1.2 核心创新点

- **动态RCS建模**：首次将隐身无人机角度相关的RCS特性融入强化学习路径规划
- **目标导向复合奖励函数**：解决稀疏奖励问题，显著提升收敛速度
- **动作空间离散化**：将连续转向角离散为5个固定值，简化网络训练
- **双网络解耦**：DDQN用训练网络选动作、目标网络估值，避免Q值高估

---

## 二、实验平台

### 2.1 平台信息

论文**未明确说明**具体使用的编程语言和深度学习框架，根据DDQN算法主流实现推断如下：

| 项目 | 推测/信息 |
|------|-----------|
| **编程语言** | Python (DDQN算法主流实现语言) |
| **深度学习框架** | PyTorch 或 TensorFlow |
| **代码获取** | 联系作者获取 (论文声明) |
| **联系邮箱** | baolei20@nudt.edu.cn |
| **联系邮箱** | zhengtaoguo@mail.nwpu.edu.cn |

### 2.2 依赖环境

推荐安装以下Python依赖包：

```bash
pip install torch numpy scipy matplotlib
```

### 2.3 推荐硬件配置

| 硬件 | 推荐配置 |
|------|----------|
| GPU | NVIDIA GPU (支持CUDA) |
| 内存 | ≥ 8GB |
| 存储 | ≥ 1GB (用于经验回放池) |

---

## 三、环境模型创建

### 3.1 空域环境参数

| 参数 | 设置值 |
|------|--------|
| 空域尺寸 | 100km × 100km 方形封闭空域 |
| 起点坐标 | (0, 0) |
| 目标终点 | (100km, 100km) |
| 边界约束 | 无人机不能飞出空域边界 |

```python
# 空域配置
airspace_config = {
    "size": (100, 100),         # 100km × 100km
    "start_point": (0, 0),      # 起点
    "target_point": (100, 100), # 目标终点
    "boundary": "closed"        # 封闭边界
}
```

### 3.2 雷达部署配置

环境中部署四部独立的防空雷达：

| 雷达编号 | 位置坐标 |
|----------|----------|
| 雷达1 | (30km, 30km) |
| 雷达2 | (30km, 70km) |
| 雷达3 | (70km, 40km) |
| 雷达4 | (70km, 80km) |

```python
# 雷达位置配置
radar_positions = [
    (30, 30),
    (30, 70),
    (70, 40),
    (70, 80)
]
```

### 3.3 雷达系统参数 (Table 2)

| 参数 | 符号 | 数值 |
|------|------|------|
| 天线增益 | G | 20 dB |
| 峰值发射功率 | Pt | 30 MW |
| 工作频率 | f₀ | 9 GHz |
| 光速 | c | 3×10⁸ m/s |
| 信号波长 | λ | c/f₀ |
| 玻尔兹曼常数 | k | 1.38×10⁻²³ J/K |
| 有效噪声温度 | Te | 290 K |
| 雷达带宽 | B | 100 MHz |

```python
# 雷达参数配置
radar_params = {
    "G": 10 ** (20/10),      # 20 dB 转线性
    "Pt": 30e6,              # 30 MW
    "f0": 9e9,               # 9 GHz
    "c": 3e8,                # 光速
    "k": 1.38e-23,           # 玻尔兹曼常数
    "Te": 290,               # 有效噪声温度 K
    "B": 100e6,              # 带宽 100 MHz
    "Fn": 1,                 # 噪声因子
    "L": 1,                  # 损耗因子
    "Pfa": 1e-6              # 虚警概率
}
```

---

## 四、核心算法实现

### 4.1 DDQN超参数设置 (Table 1)

| 参数名称 | 符号 | 参数值 |
|----------|------|--------|
| 学习率 | lr | 2×10⁻³ |
| 折扣因子 | γ | 0.98 |
| 贪婪概率 | ε | 0.01 |
| 总轮数 | E | 800 |
| 目标网络更新间隔 | Ntrain | 10 |
| 经验池最大容量 | Nmax | 10,000 |
| 最小采样数量 | Nmin | 1,000 |
| 采样批次大小 | N | 64 |

```python
# DDQN超参数
hyperparams = {
    "learning_rate": 2e-3,
    "discount_factor": 0.98,
    "epsilon": 0.01,
    "total_episodes": 800,
    "target_update_interval": 10,
    "replay_buffer_size": 10000,
    "min_samples": 1000,
    "batch_size": 64
}
```

### 4.2 状态空间与动作空间

**状态空间**：无人机二维位置坐标 s = (x, y)

**动作空间**：离散化转向角，共5个动作

```python
# 动作空间定义
action_space = [0, 45, 90, -45, -90]  # 度
# 对应含义：
# 0°   - 保持当前航向
# 45°  - 左转45度
# 90°  - 左转90度
# -45° - 右转45度
# -90° - 右转90度
```

### 4.3 Q网络结构

训练网络和目标网络具有相同结构：

- **输入层**：位置信息 (x, y)，维度为2
- **隐藏层**：3层全连接层，每层128个节点
- **激活函数**：ReLU
- **输出层**：5个动作对应的Q值

```python
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    Q网络结构
    输入: 状态 (x, y) - 2维
    输出: 5个动作的Q值
    """
    def __init__(self, state_dim=2, action_dim=5, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        return self.network(state)
```

### 4.4 DDQN损失函数

**DQN损失函数 (公式1)**：
```
θ* = argmin_θ (1/2)[Q_θ(s,a) - (R + γ max_a' Q_θ(s',a'))]²
```

**DDQN损失函数 (公式2)**：
```
θ* = argmin_θ (1/2)[Q_θ(s,a) - (R + γ Q_θ-(s', argmax_a' Q_θ(s',a')))]²
```

DDQN的关键改进：用训练网络选择动作，用目标网络计算Q值，避免过估计。

---

## 五、奖励函数设计

### 5.1 复合奖励函数

总奖励由三个子奖励函数组成 (公式6)：

```
R = Ra + Rb + Rc
```

### 5.2 到达奖励 Ra (公式3)

```python
# 到达奖励
if state == target_position:
    Ra = 1  # r1 = 1
else:
    Ra = 0
```

### 5.3 检测惩罚 Rb (公式4)

根据雷达最大检测概率分级惩罚：

| 检测概率范围 | 惩罚值 | 说明 |
|--------------|--------|------|
| Pd < 0.3 | 0 | 无惩罚 |
| 0.3 ≤ Pd < 0.5 | r₂,₀ | 轻微惩罚 |
| 0.5 ≤ Pd < 0.6 | r₂,₁ | - |
| 0.6 ≤ Pd < 0.7 | r₂,₂ | - |
| 0.7 ≤ Pd < 0.8 | r₂,₃ | - |
| 0.8 ≤ Pd < 0.9 | r₂,₄ | 最大惩罚 |
| Pd ≥ 0.9 | r₂,₅ | 被击落，路径终止 |

### 5.4 步进惩罚 Rc (公式5)

```python
Rc = -1  # r3 = -1，每步飞行的固定惩罚
```

### 5.5 参考惩罚值设置

> **注意**：论文未给出具体惩罚数值，以下为根据实验经验的推荐设置：

```python
def calculate_reward(state, target, Pd_max):
    """
    计算复合奖励函数
    
    Args:
        state: 当前位置
        target: 目标位置
        Pd_max: 最大检测概率
    
    Returns:
        总奖励值
    """
    import numpy as np
    
    Ra = 0  # 到达奖励
    Rb = 0  # 检测惩罚
    Rc = -1  # 步进惩罚
    
    # 到达奖励
    dist_to_target = np.linalg.norm(np.array(state) - np.array(target))
    if dist_to_target < 5:  # 假设5km为到达判定距离
        Ra = 1
    
    # 检测惩罚 (分级设置)
    if Pd_max < 0.3:
        Rb = 0
    elif 0.3 <= Pd_max < 0.5:
        Rb = -5
    elif 0.5 <= Pd_max < 0.6:
        Rb = -10
    elif 0.6 <= Pd_max < 0.7:
        Rb = -20
    elif 0.7 <= Pd_max < 0.8:
        Rb = -40
    elif 0.8 <= Pd_max < 0.9:
        Rb = -80
    elif Pd_max >= 0.9:
        Rb = -100  # 被击落
    
    return Ra + Rb + Rc
```

---

## 六、雷达检测概率模型

### 6.1 信噪比计算 (公式10)

$$SNR = \frac{P_t G^2 \lambda^2 \sigma}{(4\pi)^3 R^4 k T_e B F_n L}$$

其中：
- Pt：峰值发射功率
- G：天线增益
- λ：信号波长
- σ：雷达散射截面(RCS)
- R：检测距离
- k：玻尔兹曼常数
- Te：有效噪声温度
- B：雷达带宽
- Fn：噪声因子
- L：损耗因子

### 6.2 检测概率计算 (公式8)

$$P_d \approx 0.5 \times erfc\left(\sqrt{-\ln P_{fa}} - \sqrt{SNR + 0.5}\right)$$

其中：
- Pfa：虚警概率（通常设置为10⁻⁶）
- erfc(·)：互补误差函数

### 6.3 互补误差函数 (公式9)

$$erfc(z) = 1 - \frac{2}{\sqrt{\pi}} \int_0^z e^{-v^2} dv$$

### 6.4 检测概率计算代码

```python
import numpy as np
from scipy.special import erfc

def calculate_SNR(Pt, G, wavelength, sigma, R, k, Te, B, Fn=1, L=1):
    """
    计算信噪比 (公式10)
    
    Args:
        Pt: 峰值发射功率 (W)
        G: 天线增益 (线性值)
        wavelength: 信号波长 (m)
        sigma: RCS (m²)
        R: 检测距离 (m)
        k: 玻尔兹曼常数
        Te: 有效噪声温度 (K)
        B: 雷达带宽 (Hz)
        Fn: 噪声因子
        L: 损耗因子
    
    Returns:
        SNR值
    """
    numerator = Pt * (G ** 2) * (wavelength ** 2) * sigma
    denominator = ((4 * np.pi) ** 3) * (R ** 4) * k * Te * B * Fn * L
    
    if denominator <= 0:
        return 0
    
    return numerator / denominator


def calculate_detection_probability(SNR, Pfa=1e-6):
    """
    计算检测概率 (公式8)
    
    Args:
        SNR: 信噪比
        Pfa: 虚警概率
    
    Returns:
        检测概率 Pd
    """
    z = np.sqrt(-np.log(Pfa)) - np.sqrt(SNR + 0.5)
    Pd = 0.5 * erfc(z)
    return np.clip(Pd, 0, 1)
```

---

## 七、动态RCS模型

### 7.1 动态RCS特性

**这是本文的核心创新点**。隐身无人机的RCS随其与雷达的相对角度动态变化：

- **正面**：RCS最小（隐身效果最好）
- **侧面**：RCS最大（容易被探测）
- **斜向**：RCS介于两者之间

与传统固定RCS假设相比，动态RCS更接近真实战场环境。

### 7.2 RCS模型实现

> **⚠️ 重要提示**：论文未提供具体RCS函数，以下为示例实现，复现时建议联系作者获取实际RCS数据。

```python
import numpy as np

class StealthUAV:
    """隐身无人机类"""
    
    def __init__(self, start_position):
        self.position = np.array(start_position, dtype=float)
        self.heading = 45  # 初始航向角（朝向目标，东北方向）
        self.step_size = 5  # 每步飞行距离 (km)，论文未明确
    
    def get_dynamic_RCS(self, radar_position):
        """
        计算动态RCS
        
        Args:
            radar_position: 雷达位置
        
        Returns:
            RCS值 (m²)
        """
        # 计算无人机相对于雷达的角度
        direction = np.array(radar_position) - self.position
        angle_to_radar = np.arctan2(direction[1], direction[0])
        
        # 相对角度 = 雷达视角 - 无人机航向
        relative_angle = angle_to_radar - np.radians(self.heading)
        relative_angle = np.degrees(relative_angle) % 360
        
        # 示例RCS模型（需要根据实际隐身无人机特性调整）
        sigma_min = 0.01   # m², 正面最小RCS
        sigma_max = 1.0    # m², 侧面最大RCS
        
        # 根据相对角度计算RCS
        if relative_angle < 30 or relative_angle > 330:
            # 正面 (±30°)
            sigma = sigma_min
        elif 60 < relative_angle < 120 or 240 < relative_angle < 300:
            # 侧面 (60°-120° 或 240°-300°)
            sigma = sigma_max
        else:
            # 斜向
            sigma = (sigma_min + sigma_max) / 2
        
        return sigma
    
    def move(self, action_index):
        """
        执行动作，更新位置
        
        Args:
            action_index: 动作索引 (0-4)
        
        Returns:
            新位置
        """
        action_angles = [0, 45, 90, -45, -90]
        turn_angle = action_angles[action_index]
        
        # 更新航向
        self.heading = (self.heading + turn_angle) % 360
        
        # 更新位置
        dx = self.step_size * np.cos(np.radians(self.heading))
        dy = self.step_size * np.sin(np.radians(self.heading))
        self.position = self.position + np.array([dx, dy])
        
        return self.position.copy()
```

---

## 八、训练流程

### 8.1 两阶段训练策略

| 阶段 | 雷达状态 | 训练目的 |
|------|----------|----------|
| **预训练** | 关闭 | 学习最短路径先验经验 |
| **正式训练** | 开启（四部雷达） | 学习规避威胁的最优路径 |

### 8.2 DDQN算法伪代码 (Algorithm 1)

```
Algorithm 1: DDQN Path Planning

输入: 超参数 lr, γ, ε, E, Ntrain, Nmax, Nmin, N

初始化:
    初始化训练网络 Qθ (随机权重)
    初始化经验回放池 R (空)
    设置目标网络 Qθ- = Qθ

For 轮数 e = 1 → E do
    获取初始状态 s (起点位置)
    设置 d = True (路径未结束)
    
    While d = True do
        // 动作选择 (ε-greedy策略)
        以概率 ε 随机选择动作 at
        以概率 1-ε 选择 at = argmax_a Qθ(st, a)
        
        // 执行动作
        执行动作 at
        获得奖励 Rt
        状态转移到 st+1
        更新终止标志 d
        
        // 存储经验
        将 (st, at, Rt, st+1, d) 存入经验回放池 R
        
        // 网络更新
        if |R| > Nmin then
            从 R 中随机采样 N 组数据 {(si, ai, Ri, si+1, di)}
            
            // 计算TD目标 (DDQN核心)
            For each sample i do
                a'i = argmax_a Qθ(si+1, a)  // 训练网络选动作
                yi = Ri + γ * Qθ-(si+1, a'i) * (1 - di)  // 目标网络估值
            end
            
            // 更新训练网络
            L = (1/N) Σ (yi - Qθ(si, ai))²
            使用梯度下降最小化损失 L
            
            // 更新目标网络
            if 更新次数 % Ntrain == 0 then
                θ- ← θ
            end
        end
        
        st ← st+1
    end
end for

输出: 训练好的Q网络 Qθ
```

### 8.3 ε-greedy策略 (公式12)

```python
def select_action(q_network, state, epsilon=0.01):
    """
    ε-greedy动作选择策略
    
    Args:
        q_network: Q网络
        state: 当前状态
        epsilon: 探索概率
    
    Returns:
        选择的动作索引
    """
    import torch
    import numpy as np
    
    if np.random.random() < epsilon:
        # 随机探索
        return np.random.randint(5)
    else:
        # 贪婪利用
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_network(state_tensor)
            return q_values.argmax().item()
```

### 8.4 终止条件

每轮训练在以下任一条件满足时终止：

1. **到达目标**：无人机到达目标位置（获得正奖励 Ra = 1）
2. **被击落**：检测概率 Pd > 0.9（获得最大惩罚，d = False）
3. **越界**：无人机飞出空域边界

---

## 九、完整代码实现

### 9.1 雷达类

```python
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
        self.G = 10 ** (20/10)      # 20 dB → 线性
        self.Pt = 30e6              # 30 MW
        self.f0 = 9e9               # 9 GHz
        self.c = 3e8
        self.wavelength = self.c / self.f0
        self.k = 1.38e-23           # 玻尔兹曼常数
        self.Te = 290               # K
        self.B = 100e6              # 100 MHz
        self.Fn = 1                 # 噪声因子
        self.L = 1                  # 损耗因子
        self.Pfa = 1e-6             # 虚警概率
    
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
```

### 9.2 隐身无人机类

```python
class StealthUAV:
    """隐身无人机类"""
    
    def __init__(self, start_position):
        self.position = np.array(start_position, dtype=float)
        self.heading = 45  # 初始航向角
        self.step_size = 5  # 每步飞行距离 (km)
    
    def get_dynamic_RCS(self, radar_position):
        """计算动态RCS"""
        direction = np.array(radar_position) - self.position
        angle_to_radar = np.arctan2(direction[1], direction[0])
        relative_angle = np.degrees(angle_to_radar - np.radians(self.heading)) % 360
        
        sigma_min = 0.01
        sigma_max = 1.0
        
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
```

### 9.3 环境类

```python
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
            Ra = 1
        
        # 检测惩罚
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
```

### 9.4 DDQN智能体

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DDQNAgent:
    """DDQN智能体"""
    
    def __init__(self):
        self.q_network = QNetwork()
        self.target_network = QNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=2e-3)
        self.replay_buffer = deque(maxlen=10000)
        self.update_count = 0
        
        self.gamma = 0.98
        self.epsilon = 0.01
        self.batch_size = 64
        self.min_samples = 1000
        self.target_update_interval = 10
    
    def select_action(self, state):
        """选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(5)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def update(self):
        """更新网络"""
        if len(self.replay_buffer) < self.min_samples:
            return
        
        # 随机采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # DDQN: 训练网络选动作，目标网络估值
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))
        
        # 计算损失并更新
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
```

### 9.5 训练主函数

```python
def train():
    """训练主函数"""
    agent = DDQNAgent()
    env = RadarEnvironment()
    
    # 阶段1: 预训练（无雷达威胁）
    print("Phase 1: Pre-training without radar threats...")
    env.set_radar_enabled(False)
    
    for episode in range(200):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        
        if (episode + 1) % 50 == 0:
            print(f"Pre-training Episode {episode + 1}, Return: {total_reward:.2f}")
    
    # 阶段2: 正式训练（四雷达环境）
    print("\nPhase 2: Training with radar threats...")
    env.set_radar_enabled(True)
    
    returns = []
    for episode in range(800):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        
        returns.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_return = np.mean(returns[-100:])
            print(f"Episode {episode + 1}, Avg Return: {avg_return:.2f}, Status: {info['status']}")
    
    return agent, returns

if __name__ == "__main__":
    agent, returns = train()
```

---

## 十、复现建议与注意事项

### 10.1 获取源码

论文声明代码可通过联系作者获取，这是最直接的复现方式：

- **邮箱1**: baolei20@nudt.edu.cn
- **邮箱2**: zhengtaoguo@mail.nwpu.edu.cn
- **邮箱3**: lichaolong13@nudt.edu.cn

### 10.2 论文未明确的参数

以下参数论文未明确给出，复现时需自行设定或联系作者确认：

| 参数 | 说明 | 建议值 |
|------|------|--------|
| 无人机飞行步长 | 每步飞行的距离(km) | 5 km |
| 动态RCS具体函数 | 不同角度对应的RCS值 | 联系作者 |
| 惩罚值r₂的具体数值 | 各检测概率区间的惩罚值 | 见5.5节 |
| 目标到达判定半径 | 多近算到达目标 | 5 km |

### 10.3 验证方法

通过复现论文中的图表来验证实现的正确性：

1. **Figure 5**: 无威胁环境下的直线路径 → 无人机应从(0,0)直线飞向(100,100)
2. **Figure 6**: 无威胁环境下的收敛曲线 → 应快速收敛
3. **Figure 8**: 四雷达环境下的规避路径 → 应绕开高威胁区域
4. **Figure 9**: 四雷达环境下的收敛曲线 → 约500轮后趋于稳定

### 10.4 预期结果

根据论文实验结果，正确复现后应获得以下路径长度：

| 算法 | 路径长度 |
|------|----------|
| **DDQN + 动态RCS (本文方法)** | **173.05 km** |
| DDQN + 固定RCS | 178.91 km |
| A*算法 | 167.20 km |

> 注：虽然A*算法路径最短，但存在多次穿越威胁区域、转弯次数多等问题。

### 10.5 常见问题与解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 收敛慢 | 奖励函数设计不当 | 调整惩罚值梯度 |
| 无法到达目标 | 步长过大或过小 | 调整step_size |
| 陷入局部最优 | ε值过小 | 增大探索概率 |
| Q值爆炸 | 学习率过大 | 减小learning_rate |

### 10.6 未来改进方向

论文提出的后续研究方向：

1. **三维路径规划**：扩展至3D空间并结合3D RCS特性
2. **实测验证**：进行实际飞行测试
3. **动态环境**：研究雷达位置或参数动态变化时的在线重规划
4. **多机协同**：扩展至编队突防场景

---

## 附录：参考文献

1. Bao, L.; Guo, Z.; Gao, X.; Li, C. Stealth UAV Path Planning Based on DDQN Against Multi-Radar Detection. *Aerospace* **2025**, 12, 774.

2. Hasselt, V.H.; Guez, A.; Silver, D. Deep reinforcement learning with double q-learning. *AAAI* **2016**.

3. Mnih, V. et al. Human-level control through deep reinforcement learning. *Nature* **2015**, 518, 529-533.

4. Zhang, Z. et al. A Novel Real-Time Penetration Path Planning Algorithm for Stealth UAV in 3D Complex Dynamic Environment. *IEEE Access* **2020**, 8, 122757-122771.

---

## 十一、实验可视化

### 11.1 论文图表概述

论文中包含以下主要图表：

| 图表 | 内容 | 实现方法 |
|------|------|----------|
| Figure 3 | 检测惩罚与检测概率关系 | 阶梯函数绘制 |
| Figure 4 | 固定RCS的雷达检测概率分布 | 热力图 (contourf) |
| Figure 5 | 无威胁环境下的飞行路径 | 直线路径绘制 |
| Figure 6 | 无威胁环境收敛曲线 | 折线图 |
| Figure 7 | 动态RCS的雷达检测概率分布 | 热力图 (考虑角度) |
| Figure 8 | 四雷达环境路径对比 | 热力图 + 多条路径 |
| Figure 9 | 四雷达环境收敛曲线 | 折线图 |

### 11.2 雷达检测概率热力图实现原理

**核心思路**：
1. 创建100×100的空间网格
2. 对每个网格点计算到所有雷达的距离
3. 根据距离和RCS计算检测概率
4. 使用 `matplotlib.contourf` 绘制热力图

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import erfc

def plot_radar_detection_probability():
    """绘制雷达检测概率热力图"""
    
    # 雷达参数
    G = 10 ** (20/10)
    Pt = 30e6
    f0 = 9e9
    c = 3e8
    wavelength = c / f0
    k = 1.38e-23
    Te = 290
    B = 100e6
    Pfa = 1e-6
    
    # 雷达位置
    radar_positions = [(30, 30), (30, 70), (70, 40), (70, 80)]
    
    # 创建网格
    x = np.linspace(0, 100, 300)
    y = np.linspace(0, 100, 300)
    X, Y = np.meshgrid(x, y)
    
    # 计算每点的最大检测概率
    Pd_max = np.zeros_like(X)
    sigma = 0.1  # 固定RCS (或使用动态RCS函数)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            max_pd = 0
            for radar_pos in radar_positions:
                # 计算距离 (km转m)
                R = np.sqrt((X[i,j]-radar_pos[0])**2 + 
                           (Y[i,j]-radar_pos[1])**2) * 1000
                
                if R < 1:
                    pd = 1.0
                else:
                    # 计算SNR (公式10)
                    SNR = (Pt * G**2 * wavelength**2 * sigma) / \
                          ((4*np.pi)**3 * R**4 * k * Te * B)
                    
                    # 计算检测概率 (公式8)
                    z = np.sqrt(-np.log(Pfa)) - np.sqrt(SNR + 0.5)
                    pd = 0.5 * erfc(z)
                    pd = np.clip(pd, 0, 1)
                
                max_pd = max(max_pd, pd)
            
            Pd_max[i,j] = max_pd
    
    # 创建蓝色渐变色彩映射
    colors = ['#FFFFFF', '#E6F3FF', '#CCE7FF', '#99CFFF', 
              '#66B7FF', '#339FFF', '#0087FF', '#003366']
    cmap = LinearSegmentedColormap.from_list('radar', colors)
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(X, Y, Pd_max, levels=50, cmap=cmap, vmin=0, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label='Detection probability')
    
    # 标记雷达位置
    for pos in radar_positions:
        ax.plot(pos[0], pos[1], 'k^', markersize=10)
    
    ax.set_xlabel('X-axis(Km)')
    ax.set_ylabel('Y-axis(Km)')
    ax.set_aspect('equal')
    
    return fig
```

### 11.3 动态RCS的检测概率分布

与固定RCS不同，动态RCS需要考虑无人机与雷达的相对角度：

```python
def get_dynamic_RCS(uav_position, uav_heading, radar_position):
    """
    计算动态RCS
    
    无人机正面RCS最小，侧面最大
    """
    direction = np.array(radar_position) - np.array(uav_position)
    angle_to_radar = np.arctan2(direction[1], direction[0])
    relative_angle = np.degrees(angle_to_radar) - uav_heading
    relative_angle = relative_angle % 360
    
    sigma_min = 0.01  # 正面
    sigma_max = 1.0   # 侧面
    
    if relative_angle < 30 or relative_angle > 330:
        return sigma_min  # 正面
    elif 60 < relative_angle < 120 or 240 < relative_angle < 300:
        return sigma_max  # 侧面
    else:
        return (sigma_min + sigma_max) / 2  # 斜向
```

**关键区别**：
- 固定RCS：检测概率分布呈**同心圆**形状
- 动态RCS：检测概率分布呈**不规则**形状（取决于无人机航向）

### 11.4 路径可视化

```python
def plot_path_with_radar(path, radar_positions):
    """
    绘制带雷达威胁的路径图
    
    Args:
        path: 路径点列表 [(x1,y1), (x2,y2), ...]
        radar_positions: 雷达位置列表
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 先绘制检测概率热力图作为背景
    # ... (热力图代码)
    
    # 绘制路径
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, 
            label='DDQN with dynamic RCS')
    
    # 标记起点和终点
    ax.plot(0, 0, 'ko', markersize=12, label='Start')
    ax.plot(100, 100, 'o', color='orange', markersize=12, label='Target')
    
    # 标记雷达
    for pos in radar_positions:
        ax.plot(pos[0], pos[1], 'k^', markersize=10)
    
    ax.legend()
    return fig
```

### 11.5 收敛曲线可视化

```python
def plot_convergence(returns):
    """
    绘制训练收敛曲线
    
    Args:
        returns: 每轮累计回报列表
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 原始曲线
    ax.plot(returns, 'b-', alpha=0.7, linewidth=1)
    
    # 移动平均（平滑）
    window = 20
    smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(returns)), smoothed, 'r-', linewidth=2,
            label='Moving Average')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig
```

### 11.6 实时训练可视化

在训练过程中实时显示路径和收敛情况：

```python
class RealtimeVisualizer:
    """实时可视化器"""
    
    def __init__(self):
        plt.ion()  # 开启交互模式
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))
    
    def update(self, path, returns, episode):
        """每轮训练后调用"""
        self.axes[0].clear()
        self.axes[1].clear()
        
        # 左图：当前路径
        # ... 绘制热力图和路径
        
        # 右图：收敛曲线
        self.axes[1].plot(returns)
        
        plt.pause(0.01)  # 短暂暂停以更新显示
    
    def close(self):
        plt.ioff()
        plt.close()
```

### 11.7 运行可视化代码

```bash
# 生成所有论文图表
python visualization.py

# 输出文件：
# ./figures/figure3_detection_penalty.png
# ./figures/figure4_fixed_rcs.png
# ./figures/figure5_no_threat_path.png
# ./figures/figure6_convergence_no_threat.png
# ./figures/figure7_dynamic_rcs.png
# ./figures/figure8_path_comparison.png
# ./figures/figure9_convergence_with_threat.png
```

### 11.8 可视化效果说明

**Figure 4 vs Figure 7 对比**：

| 特征 | Figure 4 (固定RCS) | Figure 7 (动态RCS) |
|------|-------------------|-------------------|
| 形状 | 同心圆 | 不规则 |
| 颜色分布 | 均匀渐变 | 角度相关 |
| 物理意义 | 只与距离相关 | 与距离和角度都相关 |

**颜色映射说明**：
- 白色/浅蓝：低检测概率（安全区域）
- 深蓝：高检测概率（危险区域）
- 检测概率 > 0.9 的区域为禁飞区

---

## 十二、完整项目结构

```
ddqn_uav_path_planning/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包列表
├── config.py                 # 配置参数
├── models/
│   └── q_network.py          # Q网络定义
├── envs/
│   ├── radar.py              # 雷达类
│   ├── uav.py                # 无人机类
│   └── environment.py        # 环境类
├── agents/
│   └── ddqn_agent.py         # DDQN智能体
├── utils/
│   └── visualization.py      # 可视化模块
├── train.py                  # 训练脚本
├── evaluate.py               # 评估脚本
└── figures/                  # 生成的图表
    ├── figure3_detection_penalty.png
    ├── figure4_fixed_rcs.png
    └── ...
```

---

*文档版本: 1.1*  
*整理日期: 2026年3月*  
*更新: 添加可视化章节*

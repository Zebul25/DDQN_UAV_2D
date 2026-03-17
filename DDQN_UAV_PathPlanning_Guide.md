# DDQN 无人机路径规划训练优化指南

## 目录

- [一、问题诊断](#一问题诊断)
- [二、状态空间设计](#二状态空间设计)
- [三、奖励函数设计](#三奖励函数设计)
- [四、课程学习策略](#四课程学习策略)
- [五、探索策略优化](#五探索策略优化)
- [六、网络与超参数](#六网络与超参数)
- [七、专家演示](#七专家演示)
- [八、调试与监控](#八调试与监控)
- [九、完整代码示例](#九完整代码示例)

---

## 一、问题诊断

### 1.1 环境设置回顾

```
空域: 100×100
起点: (0, 0)
终点: (100, 100)
步长: 5
雷达位置: (30,30), (30,70), (70,40), (70,80)
航向: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
动作空间: 转向角度 [0, 45, 90, -45, -90]
```

### 1.2 核心问题分析

| 问题 | 描述 | 影响 |
|------|------|------|
| **奖励尺度失衡** | 穿越雷达累计惩罚 -600~-900，到达奖励仅 +200 | 成功到达反而是负回报，智能体学会撞墙 |
| **状态信息不足** | 只有 (x, y, heading) 3维 | 网络不知道雷达和目标位置 |
| **探索困难** | 随机探索几乎不可能走出安全路径 | 从未见过"成功"样本 |
| **Q值传播慢** | 30步路径，需要多轮训练传播 | 收敛极慢 |
| **预训练策略固化** | 预训练强化了"直走"策略 | 加入雷达后仍坚持直走 |

### 1.3 奖励计算示例

```
路径A - 直线穿越雷达:
  步数: ~30步
  到达奖励: +200
  雷达惩罚: -30 × 20步 = -600
  总计: ≈ -400 ❌

路径B - 绕行成功:
  步数: ~50步  
  到达奖励: +200
  雷达惩罚: -5 × 10步 = -50
  总计: ≈ +145 ✓

路径C - 快速撞墙:
  步数: ~5步
  撞墙惩罚: -50
  总计: ≈ -50 (比穿越雷达"更优"!)
```

**结论**: 智能体会学会撞墙，因为这比穿越雷达到达终点的回报更高。

---

## 二、状态空间设计

### 2.1 原始状态 (不推荐)

```python
def get_state(self):
    return np.array([self.uav.x, self.uav.y, self.uav.heading])
```

**问题**: 神经网络不知道雷达在哪、目标在哪，必须通过海量样本"死记硬背"。

### 2.2 改进状态 (推荐)

```python
def get_state(self):
    # 1. 位置归一化
    x_norm = self.uav.x / 100.0
    y_norm = self.uav.y / 100.0
    
    # 2. 航向 (用sin/cos避免角度不连续性)
    heading_rad = np.radians(self.uav.heading)
    h_sin = np.sin(heading_rad)
    h_cos = np.cos(heading_rad)
    
    # 3. 目标信息
    dx = self.target_point[0] - self.uav.x
    dy = self.target_point[1] - self.uav.y
    dist_goal = np.sqrt(dx**2 + dy**2) / self.init_dist  # 归一化
    goal_angle = np.arctan2(dy, dx)
    relative_angle = self._normalize_angle(goal_angle - heading_rad)
    angle_norm = relative_angle / np.pi  # [-1, 1]
    
    # 4. 雷达信息 (关键!)
    radar_features = []
    for radar in self.radars:
        dist = np.sqrt((self.uav.x - radar.x)**2 + 
                       (self.uav.y - radar.y)**2) / 100.0
        prob = radar.get_detection_prob(self.uav.x, self.uav.y)
        radar_features.extend([dist, prob])
    
    state = np.array([
        x_norm, y_norm,           # 位置 (2)
        h_sin, h_cos,             # 航向 (2)
        dist_goal, angle_norm,    # 目标信息 (2)
        *radar_features           # 雷达信息 (8): 4雷达 × (距离, 概率)
    ], dtype=np.float32)
    
    return state  # 共14维
```

### 2.3 辅助函数

```python
def _normalize_angle(self, angle):
    """将弧度角归一化到 [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi
```

### 2.4 状态维度对比

| 信息 | 原状态 (3维) | 新状态 (14维) |
|------|-------------|---------------|
| 自己位置 | ✓ (原始值) | ✓ (归一化) |
| 航向 | ✓ (角度值) | ✓ (sin/cos) |
| 目标方向 | ❌ | ✓ |
| 目标距离 | ❌ | ✓ |
| 雷达距离 | ❌ | ✓ (4个) |
| 检测概率 | ❌ | ✓ (4个) |

---

## 三、奖励函数设计

### 3.1 原始奖励问题

```python
# 原始奖励 (问题多)
到达目标: +200
每步惩罚: -0.1
雷达检测: -5 到 -50 (太重!)
碰到边界: -50
```

### 3.2 改进奖励函数

```python
def calculate_reward(self, state, next_state, done, info):
    reward = 0
    
    # 1. 到达目标 (降低绝对值,让其他奖励更有意义)
    if info.get('reached_goal'):
        return 50.0
    
    # 2. 撞边界 (不要比雷达惩罚还重)
    if info.get('hit_boundary'):
        return -10.0
    
    # 3. 被雷达摧毁
    if info.get('destroyed'):
        return -20.0
    
    # 4. 路径进展奖励 (核心!)
    dist_prev = self._distance_to_goal(state)
    dist_curr = self._distance_to_goal(next_state)
    dist_change = dist_prev - dist_curr  # 正值=接近目标
    r_progress = dist_change * 2.0  # 步长5, 最大奖励10
    
    # 5. 雷达惩罚 (大幅降低!)
    detection_prob = self._get_max_detection_prob(next_state)
    if detection_prob < 0.3:
        r_radar = 0
    elif detection_prob < 0.5:
        r_radar = -0.5
    elif detection_prob < 0.7:
        r_radar = -1.0
    elif detection_prob < 0.9:
        r_radar = -2.0
    else:
        r_radar = -3.0
    
    # 6. 安全奖励 (鼓励远离雷达)
    min_radar_dist = self._min_radar_distance(next_state)
    if min_radar_dist > 30:  # 安全距离
        r_safety = 0.3
    else:
        r_safety = 0
    
    # 7. 航向奖励 (简化)
    heading_diff = self._get_heading_diff_to_goal(next_state)
    r_heading = 0.2 * (1 - abs(heading_diff) / np.pi)
    
    reward = r_progress + r_radar + r_safety + r_heading
    return reward
```

### 3.3 奖励设计原则

1. **保证成功路径总回报为正**: 绕行成功的累计奖励必须 > 0
2. **保证失败路径总回报为负**: 撞墙、被摧毁的回报 < 绕行成功
3. **密集奖励**: 每步都有反馈，不要只在终点给奖励
4. **尺度平衡**: 各项奖励在同一数量级

### 3.4 奖励塑形 (Reward Shaping)

```python
def potential_based_shaping(self, state, next_state):
    """
    基于势能的奖励塑形
    保证最优策略不变: F = γ·Φ(s') - Φ(s)
    """
    def potential(s):
        dist_goal = self._distance_to_goal(s)
        min_radar_dist = self._min_radar_distance(s)
        safety_bonus = max(0, min_radar_dist - 20)
        return -dist_goal + 0.3 * safety_bonus
    
    return self.gamma * potential(next_state) - potential(state)
```

---

## 四、课程学习策略

### 4.1 为什么需要课程学习?

直接在完整环境训练的问题:
- 4个雷达覆盖大部分直线路径
- 随机探索几乎不可能发现安全路径
- 智能体从未见过"成功"样本

### 4.2 渐进式课程学习

```python
class CurriculumTrainer:
    def __init__(self):
        self.stages = [
            {
                "radars": [],
                "target_success_rate": 0.7,
                "description": "无雷达，学习到达目标"
            },
            {
                "radars": [(50, 50)],
                "target_success_rate": 0.5,
                "description": "1个中心雷达"
            },
            {
                "radars": [(30, 50), (70, 50)],
                "target_success_rate": 0.4,
                "description": "2个雷达"
            },
            {
                "radars": [(30, 30), (30, 70), (70, 40), (70, 80)],
                "target_success_rate": 0.3,
                "description": "完整4个雷达"
            },
        ]
        self.current_stage = 0
        self.episode_results = []
        
    def get_current_config(self):
        return self.stages[self.current_stage]
    
    def record_result(self, success):
        self.episode_results.append(success)
        if len(self.episode_results) > 100:
            self.episode_results.pop(0)
    
    def should_advance(self):
        if len(self.episode_results) < 100:
            return False
        
        success_rate = sum(self.episode_results) / len(self.episode_results)
        target = self.stages[self.current_stage]["target_success_rate"]
        return success_rate >= target
    
    def advance_stage(self):
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.episode_results = []
            print(f"进入阶段 {self.current_stage}: "
                  f"{self.stages[self.current_stage]['description']}")
            return True
        return False
```

### 4.3 训练循环中使用课程学习

```python
curriculum = CurriculumTrainer()
epsilon = 1.0

for episode in range(total_episodes):
    # 设置当前阶段的雷达
    config = curriculum.get_current_config()
    env.set_radars(config["radars"])
    
    # 运行episode
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state, epsilon)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
    
    # 记录结果
    success = info.get('reached_goal', False)
    curriculum.record_result(success)
    
    # 检查是否升级
    if curriculum.should_advance():
        if curriculum.advance_stage():
            # 关键: 升级时重置探索率!
            epsilon = max(0.5, epsilon)
            print(f"重置 epsilon = {epsilon}")
    
    # 正常衰减epsilon
    epsilon = max(0.05, epsilon * 0.9995)
```

### 4.4 多难度混合训练 (替代方案)

```python
def sample_difficulty():
    """随机采样不同难度"""
    r = np.random.random()
    if r < 0.1:
        return []  # 10% 无雷达
    elif r < 0.3:
        return [(50, 50)]  # 20% 1个雷达
    elif r < 0.6:
        return random.sample(all_radars, 2)  # 30% 2个雷达
    else:
        return all_radars  # 40% 全部雷达

# 训练时
for episode in range(total_episodes):
    env.set_radars(sample_difficulty())
    # ... 训练 ...
```

---

## 五、探索策略优化

### 5.1 ε-greedy 改进

```python
class AdaptiveEpsilonGreedy:
    def __init__(self, start=1.0, end=0.05, decay=0.9995):
        self.epsilon = start
        self.end = end
        self.decay = decay
        self.stage_resets = 0
        
    def select_action(self, q_values):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(q_values))
        return np.argmax(q_values)
    
    def decay_step(self):
        self.epsilon = max(self.end, self.epsilon * self.decay)
    
    def reset_for_new_stage(self, min_epsilon=0.5):
        """阶段升级时重置探索率"""
        self.epsilon = max(min_epsilon, self.epsilon)
        self.stage_resets += 1
        print(f"探索率重置为 {self.epsilon}")
```

### 5.2 噪声网络 (NoisyNet)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            return F.linear(x, 
                          self.weight_mu + self.weight_sigma * self.weight_epsilon,
                          self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)
```

### 5.3 优先经验回放 (PER)

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def add(self, state, action, reward, next_state, done, priority=None):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if priority is None:
            priority = max_priority
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        # 计算采样概率
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]
        
        # 重要性采样权重
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
```

---

## 六、网络与超参数

### 6.1 网络结构

```python
import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, state_dim=14, action_dim=5, hidden_dim=128):
        super().__init__()
        
        # 特征提取层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
```

### 6.2 推荐超参数

```python
# 训练参数
CONFIG = {
    # 网络
    "state_dim": 14,
    "action_dim": 5,
    "hidden_dim": 128,
    
    # 学习
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "batch_size": 128,
    "buffer_size": 100000,
    
    # 目标网络
    "target_update_freq": 1000,  # 硬更新频率
    "tau": 0.005,                # 软更新系数 (二选一)
    
    # 探索
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.9995,
    
    # 环境
    "max_steps_per_episode": 200,
    "total_episodes": 100000,
    
    # 训练稳定性
    "grad_clip": 10.0,
    "reward_clip": (-10, 10),
}
```

### 6.3 训练稳定性技巧

```python
def train_step(self, batch):
    states, actions, rewards, next_states, dones = batch
    
    # 计算当前Q值
    q_values = self.policy_net(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Double DQN: 用policy网络选动作，target网络评估
    with torch.no_grad():
        next_actions = self.policy_net(next_states).argmax(1)
        next_q_values = self.target_net(next_states)
        next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target_q = rewards + self.gamma * next_q_values * (1 - dones)
    
    # 计算损失
    loss = F.smooth_l1_loss(q_values, target_q)  # Huber Loss更稳定
    
    # 梯度更新
    self.optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
    
    self.optimizer.step()
    
    return loss.item()
```

---

## 七、专家演示

### 7.1 生成安全路径

```python
def generate_safe_path(start, goal, radars, safety_radius=25):
    """
    使用简单规则生成安全路径
    实际可用 A* 或 RRT 替代
    """
    waypoints = [start]
    
    # 简单策略: 先沿边界走，再到终点
    # 路径: (0,0) -> (0, 100) -> (100, 100)
    waypoints.append((0, 50))
    waypoints.append((0, 95))
    waypoints.append((50, 100))
    waypoints.append(goal)
    
    return waypoints

def interpolate_path(waypoints, step_size=5):
    """将waypoints插值为细粒度路径"""
    full_path = []
    for i in range(len(waypoints) - 1):
        start = np.array(waypoints[i])
        end = np.array(waypoints[i + 1])
        dist = np.linalg.norm(end - start)
        n_steps = max(1, int(dist / step_size))
        
        for j in range(n_steps):
            t = j / n_steps
            point = start + t * (end - start)
            full_path.append(tuple(point))
    
    full_path.append(waypoints[-1])
    return full_path
```

### 7.2 添加专家演示到Buffer

```python
def add_expert_demonstrations(buffer, env, n_demos=100):
    """将专家轨迹加入经验回放池"""
    safe_waypoints = generate_safe_path(
        env.start_point, env.target_point, env.radar_positions
    )
    full_path = interpolate_path(safe_waypoints, step_size=5)
    
    for _ in range(n_demos):
        state = env.reset()
        
        for i in range(len(full_path) - 1):
            current_pos = full_path[i]
            next_pos = full_path[i + 1]
            
            # 计算动作
            action = compute_action(
                current_pos, next_pos, 
                env.uav.heading, env.action_space
            )
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存入buffer (高优先级)
            buffer.add(
                state, action, reward, next_state, done,
                priority=10.0  # 高优先级
            )
            
            state = next_state
            if done:
                break
    
    print(f"添加了 {n_demos} 条专家演示")

def compute_action(current, target, heading, action_space):
    """计算从current到target需要的动作"""
    dx = target[0] - current[0]
    dy = target[1] - current[1]
    target_angle = np.degrees(np.arctan2(dy, dx))
    
    # 计算需要的转向
    turn_needed = target_angle - heading
    # 归一化到 [-180, 180]
    turn_needed = (turn_needed + 180) % 360 - 180
    
    # 找最接近的动作
    actions = [0, 45, 90, -45, -90]  # 动作空间
    best_action = min(range(len(actions)), 
                      key=lambda i: abs(actions[i] - turn_needed))
    return best_action
```

---

## 八、调试与监控

### 8.1 监控指标

```python
class TrainingMonitor:
    def __init__(self, log_dir="./logs"):
        self.metrics = {
            'episode_rewards': [],
            'episode_steps': [],
            'success_rate': [],
            'avg_q_value': [],
            'loss': [],
            'epsilon': [],
            'collision_rate': [],
        }
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def log_episode(self, reward, steps, success, collision, epsilon):
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_steps'].append(steps)
        self.metrics['epsilon'].append(epsilon)
        
        # 计算最近100个episode的成功率
        recent = self.metrics['episode_rewards'][-100:]
        # ... 更多统计
    
    def log_training(self, loss, q_values):
        self.metrics['loss'].append(loss)
        self.metrics['avg_q_value'].append(q_values.mean().item())
    
    def plot_metrics(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Episode奖励
        axes[0, 0].plot(self.metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        
        # 成功率
        axes[0, 1].plot(self.metrics['success_rate'])
        axes[0, 1].set_title('Success Rate')
        
        # Q值
        axes[0, 2].plot(self.metrics['avg_q_value'])
        axes[0, 2].set_title('Average Q Value')
        
        # 损失
        axes[1, 0].plot(self.metrics['loss'])
        axes[1, 0].set_title('Loss')
        
        # Epsilon
        axes[1, 1].plot(self.metrics['epsilon'])
        axes[1, 1].set_title('Epsilon')
        
        # 步数
        axes[1, 2].plot(self.metrics['episode_steps'])
        axes[1, 2].set_title('Steps per Episode')
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/training_metrics.png")
        plt.close()
```

### 8.2 可视化轨迹

```python
def visualize_episode(env, agent, save_path=None):
    """可视化一个episode的轨迹"""
    state = env.reset()
    trajectory = [(env.uav.x, env.uav.y)]
    
    for _ in range(200):
        action = agent.select_action(state, epsilon=0)  # 贪婪
        state, reward, done, _ = env.step(action)
        trajectory.append((env.uav.x, env.uav.y))
        if done:
            break
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制轨迹
    traj = np.array(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='UAV Path')
    ax.scatter(traj[0, 0], traj[0, 1], c='green', s=200, marker='o', label='Start')
    ax.scatter(traj[-1, 0], traj[-1, 1], c='blue', s=100, marker='x')
    
    # 绘制终点
    ax.scatter(100, 100, c='red', s=200, marker='*', label='Goal')
    
    # 绘制雷达
    for rx, ry in env.radar_positions:
        circle = plt.Circle((rx, ry), 20, color='orange', alpha=0.3)
        ax.add_patch(circle)
        ax.scatter(rx, ry, c='orange', s=100, marker='^')
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('UAV Trajectory')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
```

### 8.3 Q值诊断

```python
def diagnose_q_values(agent, env):
    """诊断起点处各动作的Q值"""
    state = env.reset()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor).squeeze().numpy()
    
    actions = ['直行(0°)', '左转45°', '左转90°', '右转45°', '右转90°']
    
    print("\n========== Q值诊断 ==========")
    print(f"当前位置: ({env.uav.x}, {env.uav.y})")
    print(f"当前航向: {env.uav.heading}°")
    print("-" * 30)
    for i, (action, q) in enumerate(zip(actions, q_values)):
        print(f"{action}: Q = {q:.2f}")
    print("-" * 30)
    print(f"最优动作: {actions[np.argmax(q_values)]}")
    print(f"Q值范围: [{q_values.min():.2f}, {q_values.max():.2f}]")
    print("=" * 30)
    
    # 异常检测
    if q_values.max() > 500:
        print("⚠️ 警告: Q值过大，可能存在过估计问题")
    if q_values.max() < -100:
        print("⚠️ 警告: Q值全为负，策略可能崩溃")
    if np.std(q_values) < 0.1:
        print("⚠️ 警告: Q值差异太小，策略可能未学习")
```

### 8.4 快速诊断流程

```
Q值是否正常?
├─ 爆炸 (>500) → 降低学习率，检查奖励尺度
├─ 全为负 (<-100) → 奖励设计问题，确保成功路径正回报
├─ 差异太小 → 检查状态表示，可能信息不足
└─ 正常范围 (0~100)
    ↓
智能体行为?
├─ 完全随机 → ε衰减太慢 或 状态表示问题
├─ 原地转圈 → 奖励冲突，检查每步惩罚
├─ 直线撞墙 → 预训练固化，重置探索率
├─ 直线穿雷达 → 雷达信息未加入状态
└─ 朝目标但卡住 → 探索不足，考虑PER或专家演示
```

---

## 九、完整代码示例

### 9.1 环境类

```python
import numpy as np
import math

class Radar:
    def __init__(self, position, detection_range=40):
        self.x, self.y = position
        self.detection_range = detection_range
    
    def get_detection_prob(self, uav_x, uav_y):
        dist = np.sqrt((uav_x - self.x)**2 + (uav_y - self.y)**2)
        if dist > self.detection_range:
            return 0.0
        # 简单线性模型，可替换为RCS模型
        return max(0, 1 - dist / self.detection_range)


class UAV:
    def __init__(self, x=0, y=0, heading=45):
        self.x = x
        self.y = y
        self.heading = heading  # 角度制
    
    def move(self, turn_angle, step_size=5):
        self.heading = (self.heading + turn_angle) % 360
        rad = np.radians(self.heading)
        self.x += step_size * np.cos(rad)
        self.y += step_size * np.sin(rad)


class UAVEnvironment:
    def __init__(self):
        # 空域参数
        self.airspace_size = (100, 100)
        self.start_point = np.array([0, 0])
        self.target_point = np.array([100, 100])
        self.target_threshold = 5
        self.step_size = 5
        self.max_steps = 200
        
        self.init_dist = np.linalg.norm(self.target_point - self.start_point)
        
        # 雷达
        self.radar_positions = [(30, 30), (30, 70), (70, 40), (70, 80)]
        self.radars = [Radar(pos) for pos in self.radar_positions]
        
        # 动作空间
        self.action_space = [0, 45, 90, -45, -90]
        self.action_dim = len(self.action_space)
        self.state_dim = 14
        
        # 状态
        self.uav = None
        self.steps = 0
    
    def reset(self):
        self.uav = UAV(0, 0, 45)
        self.steps = 0
        return self.get_state()
    
    def set_radars(self, radar_positions):
        """用于课程学习"""
        self.radar_positions = radar_positions
        self.radars = [Radar(pos) for pos in radar_positions]
    
    def get_state(self):
        x_norm = self.uav.x / 100.0
        y_norm = self.uav.y / 100.0
        
        heading_rad = np.radians(self.uav.heading)
        h_sin = np.sin(heading_rad)
        h_cos = np.cos(heading_rad)
        
        dx = self.target_point[0] - self.uav.x
        dy = self.target_point[1] - self.uav.y
        dist_goal = np.sqrt(dx**2 + dy**2) / self.init_dist
        goal_angle = np.arctan2(dy, dx)
        relative_angle = self._normalize_angle(goal_angle - heading_rad)
        angle_norm = relative_angle / np.pi
        
        radar_features = []
        for radar in self.radars:
            dist = np.sqrt((self.uav.x - radar.x)**2 + 
                          (self.uav.y - radar.y)**2) / 100.0
            prob = radar.get_detection_prob(self.uav.x, self.uav.y)
            radar_features.extend([dist, prob])
        
        # 如果雷达数量不足4个，填充0
        while len(radar_features) < 8:
            radar_features.extend([1.0, 0.0])
        
        return np.array([
            x_norm, y_norm,
            h_sin, h_cos,
            dist_goal, angle_norm,
            *radar_features[:8]
        ], dtype=np.float32)
    
    def step(self, action_idx):
        turn_angle = self.action_space[action_idx]
        
        prev_state = self.get_state()
        prev_dist = self._distance_to_goal()
        
        self.uav.move(turn_angle, self.step_size)
        self.steps += 1
        
        curr_dist = self._distance_to_goal()
        next_state = self.get_state()
        
        # 检查终止条件
        done = False
        info = {}
        
        if curr_dist < self.target_threshold:
            done = True
            info['reached_goal'] = True
        elif self._hit_boundary():
            done = True
            info['hit_boundary'] = True
        elif self._get_max_detection_prob() > 0.9:
            done = True
            info['destroyed'] = True
        elif self.steps >= self.max_steps:
            done = True
            info['timeout'] = True
        
        reward = self._calculate_reward(prev_dist, curr_dist, info)
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, prev_dist, curr_dist, info):
        if info.get('reached_goal'):
            return 50.0
        if info.get('hit_boundary'):
            return -10.0
        if info.get('destroyed'):
            return -20.0
        
        # 进展奖励
        dist_change = prev_dist - curr_dist
        r_progress = dist_change * 2.0
        
        # 雷达惩罚
        max_prob = self._get_max_detection_prob()
        if max_prob < 0.3:
            r_radar = 0
        elif max_prob < 0.5:
            r_radar = -0.5
        elif max_prob < 0.7:
            r_radar = -1.0
        else:
            r_radar = -2.0
        
        # 安全奖励
        min_dist = self._min_radar_distance()
        r_safety = 0.2 if min_dist > 25 else 0
        
        return r_progress + r_radar + r_safety
    
    def _distance_to_goal(self):
        return np.sqrt((self.uav.x - self.target_point[0])**2 + 
                      (self.uav.y - self.target_point[1])**2)
    
    def _hit_boundary(self):
        return (self.uav.x < 0 or self.uav.x > 100 or 
                self.uav.y < 0 or self.uav.y > 100)
    
    def _get_max_detection_prob(self):
        if not self.radars:
            return 0.0
        return max(r.get_detection_prob(self.uav.x, self.uav.y) 
                   for r in self.radars)
    
    def _min_radar_distance(self):
        if not self.radars:
            return 100.0
        return min(np.sqrt((self.uav.x - r.x)**2 + (self.uav.y - r.y)**2) 
                   for r in self.radars)
    
    def _normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
```

### 9.2 DDQN Agent

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)


class DDQNAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = config.get('gamma', 0.99)
        self.lr = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 128)
        self.buffer_size = config.get('buffer_size', 100000)
        self.target_update_freq = config.get('target_update_freq', 1000)
        self.grad_clip = config.get('grad_clip', 10.0)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.buffer = deque(maxlen=self.buffer_size)
        self.train_step_count = 0
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def train(self):
        if len(self.buffer) < self.batch_size:
            return None
        
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # 当前Q值
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = F.smooth_l1_loss(q_values, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()
        
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
```

### 9.3 主训练循环

```python
def train():
    config = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'batch_size': 128,
        'buffer_size': 100000,
        'target_update_freq': 1000,
        'grad_clip': 10.0,
    }
    
    env = UAVEnvironment()
    agent = DDQNAgent(env.state_dim, env.action_dim, config)
    curriculum = CurriculumTrainer()
    
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    
    total_episodes = 100000
    
    for episode in range(total_episodes):
        # 课程学习
        config = curriculum.get_current_config()
        env.set_radars(config["radars"])
        
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # 记录结果
        success = info.get('reached_goal', False)
        curriculum.record_result(success)
        
        # 课程升级
        if curriculum.should_advance():
            if curriculum.advance_stage():
                epsilon = max(0.5, epsilon)  # 重置探索率
        
        # 衰减epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 日志
        if episode % 100 == 0:
            recent_rewards = # ... 计算最近100个episode的平均
            print(f"Episode {episode}, Reward: {total_reward:.1f}, "
                  f"Epsilon: {epsilon:.3f}, Stage: {curriculum.current_stage}")
        
        # 保存模型
        if episode % 5000 == 0:
            agent.save(f"models/ddqn_ep{episode}.pt")

if __name__ == "__main__":
    train()
```

---

## 十、检查清单

训练前确认:

- [ ] 状态包含目标信息 (距离、相对角度)
- [ ] 状态包含雷达信息 (距离、检测概率)
- [ ] 所有状态特征已归一化到 [0,1] 或 [-1,1]
- [ ] 航向用 sin/cos 表示
- [ ] 成功路径的总回报 > 0
- [ ] 失败路径的总回报 < 成功路径
- [ ] 雷达惩罚尺度合理 (建议 -0.5 ~ -3)
- [ ] 课程学习从简单开始
- [ ] 阶段升级时重置探索率
- [ ] 添加了专家演示 (可选但推荐)

训练中监控:

- [ ] Q值在合理范围 (0 ~ 100)
- [ ] Loss在下降
- [ ] 成功率在上升
- [ ] 可视化轨迹确认行为正常

---

## 参考资料

- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [Double DQN](https://arxiv.org/abs/1509.06461)
- [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Curriculum Learning](https://arxiv.org/abs/2003.04960)

from collections import deque
import random
import numpy as np
import torch
from torch import optim, nn

from models.q_network import QNetwork


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=10000)
        self.update_count = 0
        # 添加Q值和损失历史记录
        self.q_value_history = []
        self.loss_history = []

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.min_samples = 1000
        self.target_update_interval = 10

    def select_action(self, state):
        """选择动作"""
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
            # return 0
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

        # DQN: 训练网络选动作，目标网络估值
        # DQN: 使用目标网络同时完成动作选择和估值（与DDQN的核心区别）
        with torch.no_grad():
            # DQN：直接用目标网络计算所有动作的Q值并选择最大Q值的动作
            next_q_values = self.target_network(next_states)  # 目标网络输出所有动作Q值
            next_actions = next_q_values.argmax(dim=1)  # 目标网络选择最优动作
            next_q = next_q_values.gather(1, next_actions.unsqueeze(1))  # 目标网络评估该动作的Q值
            target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))
        # 计算损失并更新
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # 添加梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']
        print(f"Model loaded from {path}")
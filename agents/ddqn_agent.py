import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
from q_network import QNetwork


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
        self.end_epsilon = 0.01
        self.decay_rate = 0.995
        self. batch_size = 64
        self.min_samples = 1000
        self.target_update_interval = 10

    def select_action(self, state):
        """选择动作"""
        if np.random.random() < self.epsilon:
            # self.epsilon = max(self.end_epsilon, self.epsilon * self.decay_rate)
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
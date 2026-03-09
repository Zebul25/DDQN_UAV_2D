from collections import deque
from random import random
import numpy as np
import torch
from torch import optim, nn

from q_network import QNetwork

class DQNAgent:
    def __init__(self):
        self.q_network = QNetwork()
        self.target_network = QNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=2e-3)
        self.replay_buffer = deque(maxlen=10000)
        self.update_count = 0

        self.gamma = 0.98
        self.epsilon = 0.99
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
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
        self.optimizer.step()

        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

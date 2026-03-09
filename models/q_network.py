from torch import nn


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
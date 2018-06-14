import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, available_actions_count):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(288, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 288)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
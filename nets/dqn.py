import torch
import torch.nn as nn
import torch.nn.functional as F
from noisy_layer import NoisyLinear

class DQN(nn.Module):
    def __init__(self, available_actions_count, noisy = False):
        self.noisy_mode = noisy
        self.num_actions = available_actions_count

        if self.noisy_mode:
            self.fully_connected = NoisyLinear()
        else:
            self.fully_connected = nn.Linear()
    
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.fc1 = self.fully_connected(288, 128)
        self.fc2 = self.fully_connected(128, available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 288)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def reset_noise(self):
    if self.noisy_mode:
        for layer_name, layer in self.named_children():
          if 'fc' in layer_name:
            layer.reset_noise()
    else:
        raise ValueError('The network was not initialized in noisy mode')
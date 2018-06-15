import torch
import torch.nn as nn
import torch.nn.functional as F
from noisy_layer import NoisyLinear

class DDQN(nn.Module):
    def __init__(self, available_actions_count, noisy = False):
        self.noisy_mode = noisy
        self.num_actions = available_actions_count

        if self.noisy_mode:
            self.fully_connected = NoisyLinear()
        else:
            self.fully_connected = nn.Linear()

        super(DDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.fc1_v = self.fully_connected(288, 128)
        self.fc1_a = self.fully_connected(288, 128)        
        self.fc2_v = self.fully_connected(128, self.num_actions)
        self.fc2_a = self.fully_connected(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, 288)

        adv = F.relu(self.fc1_a(x))
        val = F.relu(self.fc1_v(x))

        adv = self.fc2_a(adv)
        val = self.fc2_v(val).expand(x.size(0), self.num_actions)
        
        q = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        
        return q

    def reset_noise(self):
    if self.noisy_mode:
        for layer_name, layer in self.named_children():
          if 'fc' in layer_name:
            layer.reset_noise()
    else:
        raise ValueError('The network was not initialized in noisy mode')
import numpy as np
import skimage.color, skimage.transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange

class DQNAgent():

	def __init__(self, stateCount, actionCount):
		self.stateCount = stateCount
		self.actionCount = actionCount
		self.Net = DQNet
		self.Memory = ReplayMemory

	def act(self, state):

	def observe(self, sample):  
        self.memory.add_transition(sample)

    def learn(self):

    def replay(self):
    	if self.memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q = self.Net.get_q_values(s2).data.numpy()
        q2 = np.max(q, axis=1)
        target_q = self.Net.get_q_values(s1).data.numpy()
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        learn(s1, target_q)
        pass



class Model(nn.Module):
    def __init__(self, available_actions_count):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, available_actions_count)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 192)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNet:

	def __init__(self, learning_rate = 0.00025):
		self.model = Model
		self.learning_rate = learning_rate
		self.discount_factor = 0.99
		self.batch_size


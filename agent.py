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

    def __init__(self, action_count, replay_memory):
        #self.stateCount = stateCount
        self.action_count = action_count
        self.model = DQNet
        self.memory = replay_memory

    def get_q_values(self, state):
        state = torch.from_numpy(state)
        state = Variable(state)
        return self.model(state)

    def load_saved_agent(self, checkpoint_file):
        self.model.load_saved_agent(checkpoint_file)

    def get_best_action(state):
        q = self.get_q_values(state)
        m, index = torch.max(q, 1)
        action = index.data.numpy()[0]
        return action

    def act(self, state):
        return

    def observe(self, sample):  
        self.memory.add_transition(sample)

    def learn(self, environment, epochs = 1000):
        self.model.train(epochs, environment)

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

class DQNet():

    def __init__(self, learning_rate = 0.00025):
        self.model = Model
        self.learning_rate = learning_rate
        self.discount_factor = 0.99
        self.batch_size = 64
        self.criteria = nn.SmoothL1Loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate)

    
    def learn(self, state, target_q):
        s1 = torch.from_numpy(s1)
        target_q = torch.from_numpy(target_q)
        s1, target_q = Variable(s1), Variable(target_q)
        output = model(s1)
        loss = criterion(output, target_q)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def load_saved_agent(self,checkpoint_file):
        self.model = torch.load(checkpoint_file)

    def train(self, environment):
        return
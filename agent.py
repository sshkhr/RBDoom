import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import trange
from random import sample, randint, random
from time import time

class CNN(nn.Module):
    def __init__(self, available_actions_count):
        super(CNN, self).__init__()
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

class DQNet():

    def __init__(self, action_count, learning_rate = 0.00025):
        self.model = CNN(action_count)
        self.learning_rate = learning_rate
        self.batch_size = 64
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), learning_rate)
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            print("Using GPU")
            self.model = self.model.cuda()

    def process_state(self,state):
        return self.model(state)
  
    def learn(self, state, target_q):
        state = torch.from_numpy(state)
        target_q = torch.from_numpy(target_q)

        if self.use_gpu:
            state, target_q = state.cuda(), target_q.cuda()

        state, target_q = Variable(state), Variable(target_q)
        
        output = self.model(state)
        loss = self.criterion(output, target_q)
        
        # compute gradient and do SGD step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def save_agent(self, checkpoint_file):
        print("Saving the network weigths to:", checkpoint_file)
        torch.save(self.model, checkpoint_file)

    def load_saved_agent(self,checkpoint_file):
        self.model = torch.load(checkpoint_file)

class DQNAgent():

    def __init__(self, action_count, replay_memory):
        #self.stateCount = stateCount
        self.action_count = action_count
        self.model = DQNet(action_count)
        self.memory = replay_memory
        self.discount_factor = 0.99
        self.epochs = 100
        self.learning_steps_per_epoch = 2000
        self.test_episodes_per_epoch = 100

    def load_saved_agent(self, checkpoint_file):
        self.model.load_saved_agent(checkpoint_file)

    def exploration_rate(self, epoch):
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * self.epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * self.epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                               (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    def get_q_values(self, state):
        state = torch.from_numpy(state)

        if self.model.use_gpu:
            state = state.cuda()

        state = Variable(state)

        q_values = self.model.process_state(state)

        if self.model.use_gpu:
            q_values = q_values.cpu()
        
        return q_values

    def get_best_action(self, state):
        q = self.get_q_values(state)
        m, index = torch.max(q, 1)
        
        action = index.data.numpy()[0]
        return action

    def learn_from_memory(self):
        if self.memory.size > self.model.batch_size:
            state, a, next_state, isterminal, r = self.memory.get_sample(self.model.batch_size)

            q_values = self.get_q_values(next_state).data.numpy()
            q_next = np.max(q_values, axis=1)
            target_q = self.get_q_values(state).data.numpy()

            # target differs from q only for the selected action. The following means:
            # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
            target_q[np.arange(target_q.shape[0]), a] = r + self.discount_factor * (1 - isterminal) * q_next
            
            self.model.learn(state, target_q)

    def perform_learning_step(self, environment, epoch):
        """ Makes an action according to eps-greedy policy, observes the result
        (next state, reward) and learns from the transition"""

        eps = self.exploration_rate(epoch)

        state = environment.get_state()
        actions = environment.get_actions()

        if random() <= eps:
            a = randint(0, len(actions) - 1)
        else:
        # Choose the best action according to the network
            a = self.get_best_action(state)

        reward = environment.game.make_action(actions[a], environment.frame_repeat)

        isterminal = environment.game.is_episode_finished()
        next_state = environment.get_state() if not isterminal else None

        # Remember the transition that was just experienced.
        self.memory.add_transition(state, a, next_state, isterminal, reward)

        self.learn_from_memory()

    def train(self, environment, savefile = "saved_models/RBDoom_DQN.pth", test_each_epoch = True):
        
        time_start = time()
        for epoch in range(self.epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training............")
            environment.game.new_episode()

            for learning_step in trange(self.learning_steps_per_epoch, leave=False):
                self.perform_learning_step(environment, epoch)

                if environment.game.is_episode_finished():
                    score = environment.game.get_total_reward()
                    train_scores.append(score)
                    environment.game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Train Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())
            
            if test_each_epoch:

                print("\nTesting...........")

                test_episode = []
                test_scores = []

                for test_episode in trange(self.test_episodes_per_epoch, leave=False):
                    environment.game.new_episode()
                    actions = environment.get_actions()

                    while not environment.game.is_episode_finished():
                        state = environment.get_state() #environment.preprocess(environment.game.get_state().screen_buffer)
                        #state = state.reshape([1, state.shape[2], state.shape[0], state.shape[1]])
                        best_action_index = self.get_best_action(state)
                        environment.game.make_action(actions[best_action_index], environment.frame_repeat)
                    
                    r = environment.game.get_total_reward()
                    test_scores.append(r)

                test_scores = np.array(test_scores)
                print("Test Results: mean: %.1f +/- %.1f," % (test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),"max: %.1f" % test_scores.max())

            
            self.model.save_agent(savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))



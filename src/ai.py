# AI for Self Driving Car

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

"""
BRO!! fc1 is the structure b/w 1st and 2nd layer. By giving it state we are providing values
to first layer. Then we apply the ReLU activation function to it and get the output. The o/p
of 1st layer is the i/p of the 2nd layer.
"""


class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)  # Creates structure b/w layers of NN
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


"""
Action Replay is important to avoid out ANN being too much inclined towards the test-data/environment.
It will basically learn a path to travel b/w targets but not be dynamic enough to handle any changes.
Not exactly used to counter the overfitting case but sort of.
Overfitting -> The model starts learning from the noise & learns the underlying trends of training data.
Underfitting -> Opposite of over fitting.  It reduces the accuracy. Basically means that ANN could not 
fit to the data well enough.
"""


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            # Since everything is an object in python, DEL is used to delete any object.
            del self.memory[0]

    def sample(self, batch_size):
        # Memory contains each element as [state, reward, action].
        # We need samples as [[states], [rewards], [actions]]
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


"""
Optimizer-> The function that modifies the attribute of ANN like weights and learning rate.
Eg: GD, SGD, MBGD
Softmax basically takes in a i/p vector and generates a vector of probabilities as its o/p. 
This probabilistic o/p tells us the probability of each action. 
Why we used temp??
softmax([1,2,3]) = [0.04, 0.11, 0.85] & softmax([1,2,3]*3) = [0.0, 0.02, 0.98] 
Thus with temp we can regulate the surety of action to take vs exploration. 
NOTE: We used Variable as the tensor automatically contains gradients which we wont be using. Thus 
      volatile=True. 
"""
class Dqn:

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        # Track of last 100 rewards to take their mean and check if ai is improving.
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        temp = 90  # This represents our bias for exploitation rather than exploration.
        # probs = F.softmax(self.model(torch.autograd.Variable(state, volatile=True) * temp))
        # probs = F.softmax(self.model(Variable(state, volatile=True)) * temp, dim=1)
        probs = F.softmax(self.model(torch.autograd.Variable(state, volatile=True) * temp))
        probs = F.softmax(self.model(Variable(state, volatile=True)) * temp, dim=1)

        action = probs.multinomial(num_samples=1)
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()  # Reinitializing at starting of each iteration of the loop.
        td_loss.backward(retain_graph=True)  # Back propagation
        self.optimizer.step()  # Update the ANN wights.

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            # batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)


import math
import random
import math
#import roboschool
import pybullet_envs
import gym
try:
    import roboschool
except:
    pass
import numpy as np
import argparse
from running_state  import ZFilter
# import matplotlib
from utils import *
import json
from multiprocessing import Pool
#import matplotlib

#matplotlib.use('agg')
import torch.multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from datetime import datetime


class OptionsValueNetwork(nn.Module):
    def __init__(self, state_dim, num_options, hidden_dim, init_w=3e-3):
        super(OptionsValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_options)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class DiscretePolicyNetwork(nn.Module):
    def __init__(self, num_input, num_actions, hidden_size, option_dim=1, init_w=3e-3, log_std_min=-20, log_std_max=2,
                 temp=1.0): #0.1
        super(DiscretePolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.counter = 0
        self.temp_min = 0.5
        self.ANNEAL_RATE = 0.00003
        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_input, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.linear2.weight.data.uniform_(-init_w, init_w)

        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        torch.nn.init.xavier_uniform(self.linear1.weight)
        torch.nn.init.xavier_uniform(self.linear2.weight)
        torch.nn.init.xavier_uniform(self.linear3.weight)

        self.temp = temp
        self.to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        return x

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        last_dim = logits.shape[-1]
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, 1 * last_dim)

    def anneal_temp(self):
        self.counter += 1
        if self.counter % 100 == 1:
            self.temp = np.maximum(self.temp * np.exp(-self.ANNEAL_RATE * self.counter), self.temp_min)

    def evaluate(self, state, epsilon=0.01):
        # state = self.body(state)
        #log_values = self.forward(state)
        #values = torch.exp(log_values)
        values = self.forward(state)
        values_y = values.view(values.size(0), 1, self.num_actions)

        actions = self.gumbel_softmax(values_y, self.temp)
        prob = F.softmax(values,-1)

        self.anneal_temp()
        return actions, torch.log(prob+1e-20), None, None, None

    def get_action(self, state, epsilon=0.01):
        state = state.unsqueeze(0)
        #values = torch.exp(self.forward(state))
        values = self.forward(state)
        values_y = values.view(values.size(0), 1, self.num_actions)
        actions = self.gumbel_softmax(values_y, self.temp)
        prob = F.softmax(values,-1)
        self.anneal_temp()
        return torch.argmax(actions, -1),prob


class OptionsSoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(OptionsSoftQNetwork, self).__init__()
        num_inputs = num_inputs
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.to(device)

    def forward(self, state, action):
        # action = action.float()
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class BetaBody(nn.Module):
    def __init__(self, state_dim, option_dim, hidden_size=64, gate=F.relu, init_w=3e-3):
        super(BetaBody, self).__init__()
        self.feature_dim = state_dim
        self.linear1 = nn.Linear(state_dim, hidden_size)
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, option_dim)

        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        self.to(device)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = F.sigmoid(self.linear4(x))
        return x

class OptionsPolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, minm=-1,
                 maxm=1, scale=False):

        super(OptionsPolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        print(num_actions)
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        self.minm = minm
        self.maxm = maxm
        self.scale = scale
        self.to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def scale_action(self, action):

        action = self.minm + (action + 1.0) * 0.5 * (self.maxm - self.minm)
        action = np.clip(action, self.minm, self.maxm)

        return np.array([int(round(float(action)))])

    def get_action(self, state):
        state = tensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        if (torch.isnan(action[0][0])):
            import ipdb;
            ipdb.set_trace()
        if (self.scale == True):
            return self.scale_action(action.detach())
        else:
            return action[0].detach()

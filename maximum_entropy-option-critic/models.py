
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

from replay_buffer import *
def softmax(values, temp=0.3):
    if len(values.shape)==1:
        values = values.reshape(-1,values.shape[0])
    actual_values=values
    values = values / temp
    probs = F.softmax(values,dim=-1)
    selected = torch.multinomial(probs,num_samples=1)
    return selected, probs


try:
    matplotlib.use('TkAgg')

except:
    pass
# import roboschool

print(gym.envs.registry.all())
# import matplotlib
from scipy.special import expit
#import matplotlib.pyplot as plt

rng = np.random.RandomState(1234)

import torch

torch.set_num_threads(8)
print(torch.get_num_threads())
# print(torch.get_num_threads())
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

use_cuda = torch.cuda.is_available()
device = "cpu" #torch.device("cuda" if use_cuda else "cpu")
from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

# import matplotlib

# import matplotlib.pyplot as plt
import numpy as np

colours = ['y', 'r', 'g', 'm', 'b', 'k', 'w', 'c']

import pickle

class DifferentPolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_options, num_actions, state_hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, minm=-1,
                 maxm=1, scale=False):
        hidden_size = state_hidden_size
        hidden_size = int(state_hidden_size/num_options)

        super(DifferentPolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max



        self.linear1 = nn.Parameter(torch.zeros(num_options,num_inputs,state_hidden_size))

        self.linear2 = nn.Parameter(torch.zeros(num_options,state_hidden_size,hidden_size))
        self.mean_linear = nn.Parameter(torch.ones(num_options, hidden_size, num_actions).uniform_(-init_w, init_w))
        self.log_std_linear = nn.Parameter(torch.ones(num_options, hidden_size, num_actions).uniform_(-init_w, init_w))


        self.minm = minm
        self.maxm = maxm
        self.scale = scale
        self.action_dim= num_actions
        self.to(device)

    def forward(self, state, option):
        option = option.detach()
        x= F.relu(torch.matmul(state.unsqueeze(1), self.linear1[option.long().flatten(), :, :]))
        x= F.relu(torch.matmul(x, self.linear2[option.long().flatten(), :, :]))
        #x=x.unsqueeze(1)
        mean = torch.matmul(x, self.mean_linear[option.long().flatten(), :, :]).reshape(-1,self.action_dim)
        log_std = torch.matmul(x, self.log_std_linear[option.long().flatten(), :, :]).reshape(-1,self.action_dim)


        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, option, epsilon=1e-6):
        mean, log_std = self.forward(state,option)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)



        return action, log_prob, z, mean, log_std

    def scale_action(self, action):

        action = self.minm + (action + 1.0) * 0.5 * (self.maxm - self.minm)
        action = np.clip(action, self.minm, self.maxm)

        return np.array([int(round(float(action)))])

    def get_action(self, state, option):
        state = tensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state,option)
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

class DifferentPolicyNetworkType2(nn.Module):
    def __init__(self, num_inputs, num_options, num_actions, state_hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, minm=-1,
                 maxm=1, scale=False):
        hidden_size = int(state_hidden_size/num_options)

        super(DifferentPolicyNetworkType2, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max



        self.linear1 = nn.Linear(num_inputs, state_hidden_size)
        self.linear2 = nn.Linear(state_hidden_size, hidden_size)
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        #self.linear1 = torch.ones(num_options,num_inputs,state_hidden_size).uniform_(-init_w, init_w)

        #self.linear2 = torch.ones(num_options,state_hidden_size,hidden_size).uniform_(-init_w, init_w)
        self.mean_linear = nn.Parameter(torch.ones(num_options, hidden_size, num_actions).uniform_(-init_w, init_w))
        self.log_std_linear = nn.Parameter(torch.ones(num_options, hidden_size, num_actions).uniform_(-init_w, init_w))


        self.minm = minm
        self.maxm = maxm
        self.scale = scale
        self.action_dim= num_actions
        self.to(device)

    def forward(self, state, option):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        #x= F.relu(torch.matmul(state.unsqueeze(1), self.linear1[option.long().flatten(), :, :]))
        #x= F.relu(torch.matmul(x, self.linear2[option.long().flatten(), :, :]))
        x=x.unsqueeze(1)
        mean = torch.matmul(x, self.mean_linear[option.long().flatten(), :, :]).reshape(-1,self.action_dim)
        log_std = torch.matmul(x, self.log_std_linear[option.long().flatten(), :, :]).reshape(-1,self.action_dim)

        #mean = torch.matmul(x.unsqueeze(1), self.mean_linear[option.long().flatten(), :, :]).reshape(-1,self.action_dim)
        #log_std = torch.matmul(x.unsqueeze(1), self.log_std_linear[option.long().flatten(), :, :]).reshape(-1,self.action_dim)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, option, epsilon=1e-6):
        mean, log_std = self.forward(state,option)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return action, log_prob, z, mean, log_std

    def scale_action(self, action):

        action = self.minm + (action + 1.0) * 0.5 * (self.maxm - self.minm)
        action = np.clip(action, self.minm, self.maxm)

        return np.array([int(round(float(action)))])

    def get_action(self, state, option):
        state = tensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state,option)
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

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

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


class OptionsSoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_options, num_actions, hidden_size, init_w=3e-3):
        super(OptionsSoftQNetwork, self).__init__()
        num_inputs = num_inputs
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_options)

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

class DifferentSoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_options, num_actions, state_hidden_size, init_w=3e-3):
        super(DifferentSoftQNetwork, self).__init__()

        hidden_size = int(state_hidden_size/num_options)
        num_inputs = num_inputs + num_actions
        self.linear1 = nn.Parameter(torch.ones(num_options, num_inputs, state_hidden_size).uniform_(-init_w, init_w))

        self.linear2 = nn.Parameter(torch.ones(num_options, state_hidden_size, hidden_size).uniform_(-init_w, init_w))
        self.linear3 = nn.Parameter(torch.ones(num_options, hidden_size, 1).uniform_(-init_w, init_w))




    def forward(self, state, option, action):
        option = option.detach()
        x = torch.cat([state, action], 1)
        x = F.relu(torch.matmul(x.unsqueeze(1), self.linear1[option.long().flatten(), :, :]))
        x = F.relu(torch.matmul(x, self.linear2[option.long().flatten(), :, :]))
        x = torch.matmul(x, self.linear3[option.long().flatten(), :, :]).reshape(-1, 1)
        return x





import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_options, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, minm=-1,
                 maxm=1, scale=False):

        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        num_inputs = num_inputs+ num_options
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

    def evaluate(self, state,epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

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


class OptionsPolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_options, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, minm=-1,
                 maxm=1, scale=False):

        super(OptionsPolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        num_inputs = num_inputs
        self.num_actions = num_actions
        print(num_actions)
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions*num_options)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions*num_options)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        self.minm = minm
        self.maxm = maxm
        self.scale = scale
        self.to(device)
        self.num_options = num_options

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state,epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)


        log_prob = log_prob.reshape(-1,self.num_options,self.num_actions)
        action = action.reshape(-1,self.num_options,self.num_actions)
        mean = mean.reshape(-1,self.num_options,self.num_actions)
        log_std = log_std.reshape(-1,self.num_options,self.num_actions)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True).reshape(-1,self.num_options)

        return action, log_prob, z, mean, log_std

    def scale_action(self, action):

        action = self.minm + (action + 1.0) * 0.5 * (self.maxm - self.minm)
        action = np.clip(action, self.minm, self.maxm)

        return np.array([int(round(float(action)))])

    def get_action(self, state,option):
        state = tensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        action = action.reshape(-1,self.num_options,self.num_actions)
        if (torch.isnan(action[0][0])):
            import ipdb;
            ipdb.set_trace()
        if (self.scale == True):
            return self.scale_action(action.detach())
        else:
            return action[0][option].detach()


class DifferentOptionsPolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_options, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, minm=-1,
                 maxm=1, scale=False):

        super(DifferentOptionsPolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        num_inputs = num_inputs
        self.num_actions = num_actions
        print(num_actions)


        self.linear1 = nn.Parameter(torch.zeros(num_options,num_inputs,hidden_size).uniform_(-init_w, init_w))

        self.linear2 = nn.Parameter(torch.zeros(num_options,hidden_size,int(hidden_size/num_options)).uniform_(-init_w, init_w))
        self.mean_linear = nn.Parameter(torch.ones(num_options, int(hidden_size/num_options), num_actions).uniform_(-init_w, init_w))
        self.log_std_linear = nn.Parameter(torch.ones(num_options, int(hidden_size/num_options), num_actions).uniform_(-init_w, init_w))
        self.mean_bias = nn.Parameter(torch.ones(num_options, num_actions).uniform_(-init_w, init_w))
        self.log_std_bias = nn.Parameter(torch.ones(num_options, num_actions).uniform_(-init_w, init_w))




        self.minm = minm
        self.maxm = maxm
        self.scale = scale
        self.to(device)
        self.num_options = num_options

    def forward(self, state,option):

        option = option.detach()
        x = F.relu(torch.matmul(state.unsqueeze(1), self.linear1[option.long().flatten(), :, :]))
        x = F.relu(torch.matmul(x, self.linear2[option.long().flatten(), :, :]))
        # x=x.unsqueeze(1)
        mean = torch.matmul(x, self.mean_linear[option.long().flatten(), :, :]).reshape(-1, self.num_actions)
        mean = mean + self.mean_bias[option.long().flatten(),:]

        log_std = torch.matmul(x, self.log_std_linear[option.long().flatten(), :, :]).reshape(-1, self.num_actions)

        log_std = log_std + self.log_std_bias[option.long().flatten(),:]

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

        #x = F.relu(self.linear1(state))
        #x = F.relu(self.linear2(x))



    def evaluate(self, state,option,epsilon=1e-6):
        mean, log_std = self.forward(state,option)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        action = torch.tanh(mean + std * z.to(device))
        log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(1 - action.pow(2) + epsilon)


        log_prob = log_prob.reshape(-1,self.num_actions)
        action = action.reshape(-1,self.num_actions)
        mean = mean.reshape(-1,self.num_actions)
        log_std = log_std.reshape(-1,self.num_actions)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True).reshape(-1,1)

        return action, log_prob, z, mean, log_std

    def scale_action(self, action):

        action = self.minm + (action + 1.0) * 0.5 * (self.maxm - self.minm)
        action = np.clip(action, self.minm, self.maxm)

        return np.array([int(round(float(action)))])

    def get_action(self, state,option):
        state = tensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state,option)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample().to(device)
        action = torch.tanh(mean + std * z)

        action = action.cpu()  # .detach().cpu().numpy()
        action = action.reshape(-1,self.num_actions)

        return action[0].detach()



class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_options, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        num_inputs = num_inputs + num_options
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

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_options, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, minm=-1,
                 maxm=1, scale=False):

        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        num_inputs = num_inputs + num_options
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
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

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

    def get_action_probs(self, state):
        values = self.forward(state)
        #values = torch.exp(self.forward(state))
        values_y = values.view(values.size(0), 1, self.num_actions)

        actions = self.gumbel_softmax(values_y, self.temp)
        prob = F.softmax(values,-1)

        return prob,torch.argmax(actions,-1)








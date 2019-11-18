
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
import json
from multiprocessing import Pool
#import matplotlib

#matplotlib.use('agg')
import torch.multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from datetime import datetime


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



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # else:
        #     return
        self.buffer[self.position] = (
            state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size,flag=False):
        batch = random.sample(self.buffer, batch_size)
        if flag==True:
            batch = self.buffer[0:batch_size]
        #batch = self.buffer[0:batch_size]
        state, action, reward, next_state, done = map(
            np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class OptionsReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done,p):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        # else:
        #     return
        self.buffer[self.position] = (
            state, action, reward, next_state, done,p)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size,flag=False):
        batch = random.sample(self.buffer, batch_size)
        if flag==True:
            batch = self.buffer[0:batch_size]
        #batch = self.buffer[0:batch_size]
        state, action, reward, next_state, done, p = map(
            np.stack, zip(*batch))
        return state, action, reward, next_state, done, p

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.__init__(self.capacity)

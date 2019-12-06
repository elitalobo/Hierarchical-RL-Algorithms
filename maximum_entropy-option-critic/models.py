
import math
import random
import math
#import roboschool
import pybullet_envs
import gym
from torch.distributions import Normal
try:
    import roboschool
except:
    pass
import numpy as np
import argparse
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

class ActorNetwork(nn.Module):
    def __init__(self,state_dim,num_actions, hidden_dim,init_w=1e-3):
        super(ActorNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim,hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0],hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1],num_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)


        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)







    def forward(self,x):
        #print("state")
        #print(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        #print(x)
        return x

class CriticNetwork(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim,init_w=1e-3):
        super(CriticNetwork, self).__init__()
        self.q1_linear1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.q1_linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.q1_linear3 = nn.Linear(hidden_dim[1], 1)
        self.q1_linear3.weight.data.uniform_(-init_w, init_w)

        # self.q1_linear1.weight.data.uniform_(-init_w, init_w)
        # self.q1_linear2.weight.data.uniform_(-init_w, init_w)
        # self.q1_linear3.weight.data.uniform_(-init_w, init_w)

        torch.nn.init.xavier_uniform_(self.q1_linear1.weight)
        torch.nn.init.xavier_uniform_(self.q1_linear2.weight)

        #torch.nn.init.xavier_uniform_(self.q1_linear3.weight)

        self.q2_linear1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.q2_linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.q2_linear3 = nn.Linear(hidden_dim[1], 1)
        torch.nn.init.xavier_uniform_(self.q2_linear1.weight)
        torch.nn.init.xavier_uniform_(self.q2_linear2.weight)
        self.q2_linear3.weight.data.uniform_(-init_w, init_w)


        #torch.nn.init.xavier_uniform_(self.q2_linear3.weight)

    def forward(self, state,action):
        input = torch.cat((state,action),-1)
        x = F.relu(self.q1_linear1(input))
        x = F.relu(self.q1_linear2(x))
        q1_value = self.q1_linear3(x)

        x = F.relu(self.q2_linear1(input))
        x = F.relu(self.q2_linear2(x))
        q2_value = self.q2_linear3(x)
        return q1_value,q2_value

class OptionNetwork(nn.Module):
    def __init__(self,state_dim,action_dim,options_num,hidden_dim,vat_noise,init_w=1e-3):
        super(OptionNetwork, self).__init__()
        self.enc_linear1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.enc_linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.enc_linear3 = nn.Linear(hidden_dim[1], options_num)

        torch.nn.init.xavier_uniform_(self.enc_linear1.weight)
        torch.nn.init.xavier_uniform_(self.enc_linear2.weight)
        self.enc_linear3.weight.data.uniform_(-init_w, init_w)


        #torch.nn.init.xavier_uniform_(self.enc_linear3.weight)


        self.dec_linear3 = nn.Linear(hidden_dim[0],state_dim + action_dim)
        self.dec_linear2 = nn.Linear(hidden_dim[1], hidden_dim[0])
        self.dec_linear1 = nn.Linear(options_num , hidden_dim[1])
        self.dec_linear3.weight.data.uniform_(-init_w, init_w)


        torch.nn.init.xavier_uniform_(self.dec_linear1.weight)
        torch.nn.init.xavier_uniform_(self.dec_linear2.weight)


        #torch.nn.init.xavier_uniform_(self.dec_linear3.weight)



        self.vat_noise = vat_noise


    def forward(self, state,action):


        input = torch.cat((state,action),-1)
        x = F.relu(self.enc_linear1(input))
        x = F.relu(self.enc_linear2(x))
        enc_output = self.enc_linear3(x)

        output_options = F.softmax(enc_output,-1)
        inpshape = input.shape
        epsilon = Normal(0,1).sample(inpshape)
        x2 = input + self.vat_noise * epsilon * torch.abs(input)

        input_noise = self.enc_linear1(x2)
        hidden2_noise = self.enc_linear2(input_noise)
        out_noise = self.enc_linear3(hidden2_noise)

        output_option_noise = F.softmax(out_noise,-1)


        dec_1 = F.relu(self.dec_linear1(enc_output))
        dec_2 = F.relu(self.dec_linear2(dec_1))
        dec_output = self.dec_linear3(dec_2)
        return enc_output, output_options, output_option_noise, dec_output, input



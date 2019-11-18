import math
import random
import math
try:
    import roboschool
except:
    pass
import pybullet_envs

from running_state import ZFilter

import gym
from collections import deque

import numpy as np
import argparse
# import matplotlib
import json
from multiprocessing import Pool
#import matplotlib

#matplotlib.use('agg')
import torch.multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from datetime import datetime

import numpy as np
import torch


class HyperParams:
    gamma = 0.99
    lamda = 0.98
    hidden = 128
    critic_lr = 0.0003
    actor_lr = 0.0003
    batch_size = 64
    l2_rate = 0.001
    max_kl = 0.01
    clip_param = 0.2

hp = HyperParams()


def softmax(values, temp=0.3):
    values = values / temp
    #import ipdb; ipdb.set_trace()
    values = values - torch.max(values,dim=-1)[0].reshape(-1,1)
    values_exp = torch.exp(values)

    probs = values - torch.log(torch.sum(values_exp,dim=-1).reshape(-1,1))
    probs = torch.exp(probs)
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


def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


def cov(m, rowvar=False):
    # https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def softmax_loss_vectorized(X, y):
    # Taken from
    loss = 0.0
    num_train = X.shape[0]

    score = X
    score = score - torch.max(score, dim=-1, keepdim=True)[0]
    exp_score = torch.exp(score)
    sum_exp_score_col = torch.sum(exp_score, dim=-1, keepdim=True)

    loss = torch.log(sum_exp_score_col + 1e-20)
    loss = loss - score[np.arange(num_train), y.flatten()].unsqueeze(-1)
    # loss = torch.sum(loss) / float(num_train)

    return torch.mean(loss);


def save_weights(fname, input):
    order_expected = ["option_terminations", "policy", "critic", "action_critic", "policy_improvement",
                      "termination_improvement", "action_weights"]
    output_file_sup = fname
    idx = 0
    for item in order_expected:
        output = open("model/" + fname + str(item) + '.pkl', 'wb')
        pickle.dump(input[idx], output, pickle.HIGHEST_PROTOCOL)

        output.close()
        with open("model/" + fname + str(item) + '.pkl', 'rb') as pickle_file:
            content = pickle.load(pickle_file)
        idx += 1


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    x = torch.tensor(x, device=device)
    return x.to(device).float()


def get_vector(num, len, order):
    vec = []
    res = order
    for cnt in np.arange(len):
        vec.append(int(num % res))
        num = int(num / order)
    vec.reverse()
    return np.array(vec)


def initialize_fourier_vector(order, num_features):
    fourier_vector = np.zeros((np.power((order + 1), (num_features)), num_features))
    lent = fourier_vector.shape[0]
    # print(lent)
    for cnt in np.arange(lent):
        fourier_vector[cnt] = get_vector(cnt, num_features, order + 1)
    return fourier_vector


def apply_reshape(arr):
    arr = arr.reshape(arr.shape[0], 1)
    return arr


def apply_cos(arr):
    import math
    # res = [cos(y) for y in arr]
    res = np.sum(arr)
    return np.cos(res * math.pi)


def normalize_state(state, state_limit):
    state = (state - state_limit[:, 0]) / (state_limit[:, 1] - state_limit[:, 0])
    return state


def evaluate_fourier(fourier_vector, features, state_limit):
    shape = features.shape
    features = features.reshape(1, -1)
    features = normalize_state(features, state_limit)
    # dist, vel = self.get_normalized(dist, vel)
    req = np.array(features)

    evaluated_fourier = np.cos((np.matmul(fourier_vector, req.transpose())).transpose() * np.pi)
    # res = np.apply_along_axis(self.apply_cos, 1, evaluated_fourier)
    # print(evaluated_fourier.shape)
    np.apply_along_axis(apply_reshape, 1, evaluated_fourier)
    # print(evaluated_fourier.shape)
    # res = evaluated_fourier.reshape(evaluated_fourier.shape[0], 1)
    return evaluated_fourier


class DiscreteNormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = 0
        high = self.action_space.n - 1

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        return int(round(float(action)))

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


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
        x = tensor(x).float()
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))
        x = F.sigmoid(self.linear4(x))
        return x


'''
def sample_softmax(values):
    values = values / temp
    probs = values - logsumexp(values)
    probs = np.exp(probs)

    selected = np.array([rng.choice(values.shape[0], p=probs)])
    return tensor(selected)
'''



class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, num_outputs)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std, logstd

class DifferentPolicyNetwork(nn.Module):
    def __init__(self, num_options, num_inputs, num_actions, state_hidden_size, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, minm=-1,
                 maxm=1, scale=False):

        super(DifferentPolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        #self.linear1 = torch.ones(num_options,num_inputs,hidden_size).uniform_(-init_w, init_w)
        #self.linear2 = torch.ones(num_options,hidden_size,hidden_size).uniform_(-init_w, init_w)


        self.linear1 = nn.Linear(num_inputs, state_hidden_size)
        self.linear2 = nn.Linear(state_hidden_size, hidden_size)

        self.mean_linear = nn.Parameter(torch.ones(num_options, hidden_size, num_actions).uniform_(-init_w, init_w))
        self.log_std_linear = nn.Parameter(torch.ones(num_options, hidden_size, num_actions).uniform_(-init_w, init_w))

        #
        # self.mean_linear = nn.Linear(hidden_size, num_actions)
        # self.mean_linear.weight.data.uniform_(-init_w, init_w)
        # self.mean_linear.bias.data.uniform_(-init_w, init_w)
        #
        # self.log_std_linear = nn.Linear(hidden_size, num_actions)
        # self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        # self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        self.minm = minm
        self.maxm = maxm
        self.scale = scale
        self.action_dim= num_actions
        self.to(device)

    def forward(self, state, option):
        x = F.tanh(self.linear1(state))
        x = F.tanh(self.linear2(x))

        mean = torch.matmul(x.unsqueeze(1), self.mean_linear[option.long().flatten(), :, :]).reshape(-1,self.action_dim)
        log_std = torch.matmul(x.unsqueeze(1), self.log_std_linear[option.long().flatten(), :, :]).reshape(-1,self.action_dim)

        log_std = torch.zeros_like(mean)
        std = torch.exp(log_std)


        return mean, std, log_std


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hp.hidden)
        self.fc2 = nn.Linear(hp.hidden, hp.hidden)
        self.fc3 = nn.Linear(hp.hidden, 1)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        v = self.fc3(x)
        return v

def sample_sigmoid(value):
    return int(rng.uniform() < value)


def scale_action(num_actions, action):
    low = 0
    high = num_actions - 1  # self.action_space.n-1

    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    # print(int(round(float(action))))
    # print(int(round(float(action))))

    return int(round(float(action)))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, option, action, reward, next_state, done, next_option, extended_state, extended_next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            state, option, action, reward, next_state, done, next_option, extended_state, extended_next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, option, action, reward, next_state, done, next_option, extended_state, extended_next_state = map(
            np.stack, zip(*batch))
        return state, option, action, reward, next_state, done, next_option, extended_state, extended_next_state

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = -1  # self.action_space.low
        high = 1  # self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


'''
def plot(frame_idx, rewards):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
'''


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


class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

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

class DifferentSoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_options, num_actions, hidden_size, init_w=3e-3):
        super(DifferentSoftQNetwork, self).__init__()



        self.linear1 = nn.Parameter(torch.ones(num_options, num_inputs, hidden_size).uniform_(-init_w, init_w))
        self.linear2 = nn.Parameter(torch.ones(num_options,  hidden_size, hidden_size).uniform_(-init_w, init_w))
        self.linear3 = nn.Parameter(torch.ones(num_options, hidden_size, 1).uniform_(-init_w, init_w))
        #self.linear3 = nn.Linear(hidden_size, 1)

        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)
        self.to(device)

    def forward(self, state, option, action):
        # action = action.float()
        #x = torch.cat([state, action], 1)

        x = torch.matmul(state.unsqueeze(1), self.linear1[option.long().flatten(), :, :])
        x = torch.matmul(x, self.linear2[option.long().flatten(), :, :])
        x = torch.matmul(x, self.linear3[option.long().flatten(), :, :]).reshape(-1,1)
        #x = self.linear3(x)
        return x



import torch.nn.functional as F


class DiscretePolicyNetwork(nn.Module):
    def __init__(self, num_input, num_actions, hidden_size, option_dim=1, init_w=3e-3, log_std_min=-20, log_std_max=2,
                 temp=1):
        super(DiscretePolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.counter = 0
        self.temp_min = 0.5
        self.ANNEAL_RATE = 0.000003
        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_input, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear3 = nn.Linear(hidden_size, hidden_size)
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

        x = F.log_softmax(x, dim=-1)
        # print(x[0])
        return x

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)
        return -(torch.log(-torch.log(U + eps) + eps))

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())  # .cuda()
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature):
        # taken from https: // github.com / dev4488 / VAE_gumble_softmax
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, 1 * self.num_actions)

    def anneal_temp(self):
        self.counter += 1
        if self.counter % 100 == 1:
            self.temp = np.maximum(self.temp * np.exp(-self.ANNEAL_RATE * self.counter), self.temp_min)

    def evaluate(self, state, epsilon=0.01):
        # state = self.body(state)
        log_values = self.forward(state)
        values = torch.exp(log_values)
        values_y = values.view(values.size(0), 1, self.num_actions)
        actions = self.gumbel_softmax(values_y, self.temp)

        self.anneal_temp()
        # print(values[0])
        # return values, log_values, None, None, None
        return actions, log_values, None, None, None

    def get_action(self, state, epsilon=0.01):
        state = tensor(state).unsqueeze(0).to(device)
        values = torch.exp(self.forward(state))
        values_y = values.view(values.size(0), 1, self.num_actions)
        actions = self.gumbel_softmax(values_y, self.temp)
        self.anneal_temp()
        return torch.argmax(actions, -1), values_y

    def get_action_probs(self, state):
        state = tensor(state).to(device)
        values = torch.exp(self.forward(state))
        return values


def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)


def squeeze_all(input):
    next_state, reward, done, _ = input
    next_state = np.array(next_state)
    reward = np.array(reward)
    done = np.array(done)

    return next_state[0], reward[0], done[0], _


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-20, log_std_max=2, minm=-1,
                 maxm=1, scale=False):

        super(PolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

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
        if (self.scale == True):
            return self.scale_action(action.detach())
        else:
            return action[0].detach()


class SoftActorCritic():
    def __init__(self, args):
        self.reset()

    def reset(self):

        self.env = gym.make(args.env_name)
        print(args)
        self.env_name = args.env_name
        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filename = args.log_dir + "/" + args.env_name + "_" + self.timestamp + str(
            args.options_cnt)
        self.soft_tau = args.soft_tau
        self.model_dir = args.model_dir
        self.plot_dir = args.plot_dir

        self.le = args.le
        self.lb = args.lb
        self.lv = args.lv
        self.mi_penality = args.mi_penalty

        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.log = open(self.filename, 'w+')
        self.log.write("bipedal_walker " + str(args) + "\n")
        self.log.write(self.filename + "\n")
        self.log.write(str(args) + "\n")
        self.trial = args.trial


        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.hidden_dim = args.hidden_dim
        self.num_actions = args.num_actions  # 2
        self.options_cnt = args.options_cnt
        self.temp = 1.0
        self.options_tensor = torch.Tensor(np.arange(self.options_cnt)).unsqueeze(-1)
        self.tau = 0.5  # 1.0 / self.options_cnt  # args.tau
        self.replay_buffer_size = 1000000
        self.total_steps = 0.0
        self.max_frames = args.max_frames
        self.max_steps = args.max_steps
        self.frame_idx = 0
        self.rewards = []
        self.entropy_lr = 1e-2
        self.batch_size = args.batch_size
        self.alpha_temp =0.001
        self.running_state = ZFilter((self.state_dim,), clip=5)

        self.decay = 1.0
        self.target_entropy = args.target_entropy or None
        self.target_entropy_ = args.target_entropy_ or None
        print(self.target_entropy_)
        self.option_log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = None
        self.option_alpha = self.option_log_alpha.exp().detach()
        self.log_alphas = [torch.zeros(1, requires_grad=True) for x in range(self.options_cnt)]
        self.alpha_optimizers = [None for x in range(self.options_cnt)]
        self.evaluate_iter = args.evaluate_iter
        self.discriminator_init = args.discriminator_init
        self.update_discriminator = args.update_discriminator
        self.discriminator_lr = self.discriminator_init
        self.discriminator_lr_max = args.discriminator_lr
        self.running_state = ZFilter((self.state_dim,), clip=5)

        self.alphas = [self.log_alphas[x].exp().detach() for x in range(self.options_cnt)]

        if self.target_entropy is not None:
            for idx in range(self.options_cnt):
                self.log_alphas[idx] = torch.zeros(1, requires_grad=True)
                self.alphas[idx] = self.log_alphas[idx].exp().detach()
                self.alpha_optimizers[idx] = optim.Adam(
                    [self.log_alphas[idx]],
                    lr=args.option_alpha_lr,
                )
        else:
            for idx in range(self.options_cnt):
                self.alphas[idx] = self.log_alphas[idx].exp().detach() * 0.0
                self.log_alphas[idx] = torch.zeros(1, requires_grad=False)

        if self.target_entropy_ is not None:
            self.option_alpha = self.option_log_alpha.exp().detach()
            self.option_alpha_optimizer = optim.Adam(
                [self.option_log_alpha],
                lr=args.alpha_lr, )
        else:
            self.option_log_alpha = torch.zeros(1, requires_grad=False)
            self.option_alpha = self.option_log_alpha.exp().detach() * 0.0

        config = [args.value_lr, args.soft_q_lr, args.policy_lr, args.option_policy_lr, args.beta_lr, args.alpha_lr]
        print("Config " + str(config))

        self.runs = args.runs

        self.value_lr = args.value_lr  # 3e-4 for all pendulum
        self.soft_q_lr = args.soft_q_lr  # 3e-4**
        self.policy_lr = args.policy_lr  # 3e-5
        self.beta_lr = args.beta_lr  # 3e-5
        self.option_policy_lr = args.option_policy_lr  # 3e-4
        self.option_alpha_lr = args.option_alpha_lr
        self.alpha_lr = args.alpha_lr

        self.is_continuous = args.is_continuous  # True


        self.option_policy_net = DiscretePolicyNetwork(self.state_dim, self.options_cnt, hidden_size=self.hidden_dim).to(device)

        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.options_cnt, self.action_dim, self.hidden_dim).to(device)

        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.options_cnt, self.action_dim, self.hidden_dim).to(device)
        self.beta_net = BetaBody(self.state_dim + self.options_cnt,1, self.hidden_dim).to(device)


        # for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
        #     target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)


        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.soft_q_lr)

        self.beta_optimizer = optim.Adam(self.beta_net.parameters(), lr=self.beta_lr)
        self.option_policy_optimizer = optim.Adam(self.option_policy_net.parameters(), lr=self.option_policy_lr)
        self.actor = DifferentPolicyNetwork(self.options_cnt, self.state_dim, self.action_dim,hp.hidden,hp.hidden)
        self.critic = Critic(self.state_dim + self.options_cnt)
        self.target_critic = Critic(self.state_dim + self.options_cnt)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=hp.critic_lr,weight_decay=hp.l2_rate)

        # self.options_prob = torch.zeros(self.options_cnt)

        self.option_indices = tensor(np.ones((self.batch_size, self.options_cnt)) * np.arange(self.options_cnt))
        self.option_indices = self.option_indices.unsqueeze(-1)

        self.batch_iter = np.arange(self.batch_size)  # trying to cache as much as possible
        self.batch_iter_t = tensor(self.batch_iter).long()
        self.embeddings = torch.eye(self.options_cnt)

        self.option_indices = tensor(np.ones((self.batch_size, self.options_cnt)) * np.arange(self.options_cnt))
        self.option_indices = self.option_indices.unsqueeze(-1)

        self.batch_iter = np.arange(self.batch_size)  # trying to cache as much as possible
        self.batch_iter_t = tensor(self.batch_iter).long()

        self.mini_batch_size = 16
        self.mini_batch_options_tensor = self.encode(self.options_tensor.repeat(16, 1))

    def save_weights(self):
        print('saving weights')
        torch.save(self.critic.state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-value_net")
        torch.save(self.beta_net.state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-beta_net")
        torch.save(self.actor.state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-policy_net")
        torch.save(self.soft_q_net1.state_dict(),
                   self.model_dir + self.env_name + "_" + self.timestamp + "-soft_q_net1")
        torch.save(self.soft_q_net2.state_dict(),
                   self.model_dir + self.env_name + "_" + self.timestamp + "-soft_q_net2")
        torch.save(self.option_policy_net.state_dict(),
                   self.model_dir + self.env_name + "_" + self.timestamp + "-option_policy_net")

        torch.save(self.option_log_alpha, self.model_dir + self.env_name + "_" + self.timestamp + "-option-log-alpha")
        torch.save(self.option_alpha, self.model_dir + self.env_name + "_" + self.timestamp + "-option-alpha")
        torch.save(self.log_alphas, self.model_dir + self.env_name + "_" + self.timestamp + "-log-alphas")
        torch.save(self.alphas, self.model_dir + self.env_name + "_" + self.timestamp + "-alphas")
        with open(self.model_dir + self.env_name + "_" + self.timestamp + '-zfilter', 'wb') as f:
            pickle.dump(self.running_state, f)

    def encode(self, options):
        return self.embeddings[options.flatten().long()]

    def decode(self, options):
        return torch.argmax(options, -1).reshape(-1,1)

    def load_weights(self, prefix=None):
        if prefix==None:
            prefix = self.env_name + "_" + self.timestamp
        self.critic.load_state_dict(torch.load(self.model_dir + prefix + "-value_net"))
        self.actor.load_state_dict(torch.load(self.model_dir  + prefix + "-beta_net"))
        self.critic.load_state_dict(torch.load(self.model_dir + prefix + "-policy_net"))
        self.soft_q_net1.load_state_dict(torch.load(self.model_dir + prefix + "-soft_q_net1"))
        self.soft_q_net2.load_state_dict(torch.load(self.model_dir + prefix + "-soft_q_net2"))
        self.option_policy_net.load_state_dict(torch.load(self.model_dir + prefix + "-option_policy_net"))
        self.option_log_alpha = torch.load(self.model_dir + prefix + "-option-log-alpha")
        self.option_alpha = torch.load(self.model_dir + prefix + "-option-alpha")
        self.log_alphas = torch.load(self.model_dir + prefix + "-log-alphas")
        self.alphas = torch.load(self.model_dir + prefix + "-alphas")
        with open(self.model_dir + prefix + '-zfilter', 'rb') as f:
            self.running_state = pickle.load(f)


    def get_option_vals(self, states, option_dim=1):
        option_indices = self.option_indices
        all_input = self.batch_iter
        shape = self.batch_size
        if states.shape[0] != self.batch_size:
            shape = states.shape[0]
            all_input = np.arange(states.shape[0])
            option_indices = tensor(np.ones((shape, self.options_cnt)) * np.arange(self.options_cnt)).unsqueeze(-1)

        option_vals = tensor(np.zeros((shape, self.options_cnt)))
        for option_idx in range(self.options_cnt):
            extended_states = torch.cat((states, self.embeddings[option_idx].repeat(shape, 1)), -1)
            option_vals[all_input, option_idx] = self.critic(extended_states).squeeze(-1)
        return option_vals

    def get_target_vals(self, states, option_dim=1):
        # num_input = states.shape[0]
        option_vals = tensor(np.zeros((self.batch_size, self.options_cnt)))
        for option_idx in range(self.options_cnt):
            extended_states = torch.cat((states, self.embeddings[option_idx].repeat(self.batch_size, 1)), -1)
            option_vals[self.batch_iter, option_idx] = self.target_critic(extended_states).squeeze(-1)
        return option_vals

    def get_beta_vals(self, state):
        beta_vals = self.beta_net(state)
        return beta_vals

    def get_expected_q_values(self, states, actions):
        soft_q1_values = torch.zeros(self.batch_size, self.num_actions)
        soft_q2_values = torch.zeros(self.batch_size, self.num_actions)
        for idx in range(self.num_actions):
            action_idx = torch.ones(self.batch_size, 1) * idx
            soft_q1_values[self.batch_iter_t, idx] = self.soft_q_net1(states, action_idx).flatten()
            soft_q2_values[self.batch_iter_t, idx] = self.soft_q_net2(states, action_idx).flatten()
        soft_q1_values = torch.sum(actions * soft_q1_values, -1, keepdim=True)
        soft_q2_values = torch.sum(actions * soft_q2_values, -1, keepdim=True)
        return soft_q1_values, soft_q2_values

    def get_expected_option_values(self, states, options, target=False):
        extended_states = torch.cat((states, options), -1)
        soft_option_q_values = None
        if target == False:
            soft_option_q_values = self.value_net(extended_states)
        else:
            soft_option_q_values = self.target_value_net(extended_states)
        return soft_option_q_values

    def stable_softmax(self, X):
        exps = np.exp(X - np.max(X, 1).reshape(-1, 1))
        return exps / np.sum(exps, 1).reshape(-1, 1)

    def get_expected_log_prob(self, actions, log_probs):
        expected_log_probs = log_probs[np.arange(self.batch_size), torch.argmax(actions, -1)]
        # expected_log_probs = log_probs* actions
        # expected_log_probs = torch.sum(expected_log_probs, -1, keepdim=True)
        return expected_log_probs.reshape(-1, 1)

    def get_expected_option_log_prob(self, options, log_probs):
        expected_log_probs = log_probs[np.arange(self.batch_size), torch.argmax(options, -1)]
        # expected_log_probs = torch.sum(expected_log_probs, -1, keepdim=True)
        return expected_log_probs.reshape(-1, 1)

    def get_mutual_information_penalty(self, actions_obs):
        penalty = 0.0

        for idx in range(self.action_dim):
            covariance = cov(actions_obs[idx, ::])
            # print(covariance)
            variance = torch.sum(covariance * torch.eye(self.options_cnt), -1).reshape(-1, 1)
            sigma_ij = torch.sqrt(torch.mm(variance, variance.t()))
            mutual_information = covariance / sigma_ij

            penalty = penalty + torch.sum(-0.5 * torch.log(1 - mutual_information ** 2 + 1e-20))

        return penalty

    def get_returns(self,rewards, masks):
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)

        running_returns = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
            returns[t] = running_returns

        returns = (returns - returns.mean()) / returns.std()
        return returns

    def get_loss(self, returns, states,options, actions):
        mu, std, logstd = self.actor(states,options)
        log_policy = log_density(torch.Tensor(actions), mu, std, logstd)
        returns = returns.unsqueeze(1)

        objective = returns * log_policy
        objective = objective.mean()
        return - objective

    def train_critic(self, extended_states, returns):

        criterion = torch.nn.MSELoss()
        n = len(extended_states)
        arr = np.arange(n)

        for epoch in range(5):
            np.random.shuffle(arr)

            for i in range(n // hp.batch_size):
                batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                inputs = torch.Tensor(extended_states)[batch_index]
                target = returns.unsqueeze(1)[batch_index]

                values = self.critic(inputs)
                loss = criterion(values, target)
                self.critic_optimizer.zero_grad()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
                self.critic_optimizer.step()


    def train_actor(self, returns, states,options, actions):
        loss = self.get_loss(returns, states,options, actions)
        self.actor_optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optimizer.step()


    def train_beta(self, next_states,next_states_a):

        adv = self.critic(next_states) - torch.max(self.critic(next_states_a).reshape(-1,self.options_cnt),-1)[0].reshape(-1,1)
        beta_loss = (adv.detach()*self.beta_net(next_states)).mean()
        self.beta_optimizer.zero_grad()
        beta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.beta_net.parameters(), 10)
        self.beta_optimizer.step()

    def train_model(self,memory):
        memory = np.array(memory)
        extended_states = np.vstack(memory[:, 0])
        actions = list(memory[:, 1])
        rewards = list(memory[:, 2])
        masks = list(memory[:, 3])
        next_states = np.vstack(memory[:, 4])
        next_states_a = np.vstack(memory[:, 5])
        options = np.vstack(memory[:, 7])
        states = np.vstack(memory[:, 6])


        returns = self.get_returns(rewards, masks)
        extended_states = torch.Tensor(extended_states)
        states = torch.Tensor(states)
        options = torch.Tensor(options)
        next_states = torch.Tensor(next_states)
        next_states_a = torch.Tensor(next_states_a)
        self.train_beta(next_states,next_states_a)
        self.train_critic(extended_states, returns)
        self.train_actor(returns, states, options, actions)


        return returns

    def update(self, batch_size, gamma=0.99, soft_tau=0.01, frame_idx=0, options_prob_episode=None, flag=False,
               actions_obs=None,memory=None,actions_probs=None):











        self.train_model(memory)
        print("trained")

    def test(self,max_episode_len=1000):
        env = gym.make(self.env_name)
        frame_idx = 0
        run = self.trial
        options_used = ""
        total_rewards = []
        episode_cnt = 0
        total_steps=0
        options_probs = torch.zeros(self.options_cnt, max_episode_len)
        while episode_cnt < 10:
            state = np.array(env.reset())
            state = self.running_state(state)
            state = tensor(state)
            episode_reward = 0

            option, options_prob = softmax(self.get_option_vals(state.unsqueeze(0)))
            option = tensor(option[0])
            old_option = option
            for step in range(self.max_steps):
                option_idx = int(option.cpu().numpy()[0])
                if episode_cnt==0:
                    options_probs[:, step] = options_prob[0]
                    total_steps+=1
                    options_used+=str(option_idx)

                extended_states = torch.cat(
                    (state.repeat(self.options_cnt, 1), self.encode(self.options_tensor)), -1)
                extended_state = extended_states[option_idx]
                mu, std, _ = self.actor(torch.Tensor(extended_state).unsqueeze(0))
                action = get_action(mu, std)[0]

                next_state, reward, done, _ = self.env.step(action)

                next_state = self.running_state(next_state)
                next_state = tensor(next_state)



                extended_next_states = torch.cat(
                    (next_state.repeat(self.options_cnt, 1), self.encode(self.options_tensor)), -1)
                extended_next_state = extended_next_states[option_idx]

                beta_next = self.beta_net(extended_next_state.unsqueeze(0))
                beta_next_np = beta_next.detach().cpu().numpy()[0]
                new_option = option
                option_terminations = sample_sigmoid(beta_next_np)
                #if True:
                if option_terminations == 1:
                    new_option, options_prob = softmax(self.get_option_vals(next_state.unsqueeze(0)))
                    new_option = tensor(new_option[0])
                    old_option = option
                    option = new_option

                state = next_state

                episode_reward += reward

                if done:
                    total_rewards.append(episode_reward)
                    break

            episode_cnt += 1
        env.close()
        return np.mean(total_rewards), options_used, torch.mean(options_probs[:,:total_steps],-1).detach().cpu().numpy().tolist()



    def train(self, run, max_episode_len=1000):

        # import time
        # time.sleep(10)
        frame_idx = 0
        run = self.trial
        self.env = gym.make(self.env_name)
        self.rewards = []

        log_buffer = []
        duration = 1
        option_switches = 0
        episode_cnt = 0
        options_used=""
        step = 0.0


        while frame_idx < self.max_frames:

            memory = deque()
            kdx=0
            iter_rewards = []
            while kdx < 2048:
                state = np.array(self.env.reset())
                state = self.running_state(state)
                state = tensor(state)
                episode_reward = 0
                options_probs = torch.zeros(self.options_cnt, max_episode_len)


                current_vals = self.get_option_vals(state.unsqueeze(0))
                option, options_prob = softmax(current_vals)

                option = tensor(option[0])
                duration = 1
                option_switches = 1
                avgduration = 0.

                done = False
                for step in range(self.max_steps):
                    option_idx = int(option.cpu().numpy()[0])
                    options_probs[:, step] = options_prob[0]  # beta_next[0][option_idx]

                    if frame_idx >=0:

                        extended_states = torch.cat(
                            (state.repeat(self.options_cnt, 1), self.encode(self.options_tensor)), -1)
                        extended_state = extended_states[option_idx]
                        mu, std, _ = self.actor(torch.Tensor(state.unsqueeze(0)),option.unsqueeze(0))
                        action = get_action(mu, std)[0]

                        next_state, reward, done, _ = self.env.step(action)
                        next_state = self.running_state(next_state)
                        next_state = tensor(next_state)


                        extended_next_states = torch.cat(
                            (next_state.repeat(self.options_cnt, 1), self.encode(self.options_tensor)), -1)
                        extended_next_state = extended_next_states[option_idx]


                        memory.append([extended_state.cpu().numpy(), action, reward, 1-done, extended_next_state.cpu().numpy(), extended_next_states.cpu().numpy(),state,self.options_tensor[option_idx]])



                    action = tensor(action)
                    next_state = tensor(next_state)

                    beta_next = self.beta_net(extended_next_state.unsqueeze(0))
                    beta_next_np = beta_next.detach().cpu().numpy()[0]
                    new_option=option
                    option_terminations = sample_sigmoid(beta_next_np)
                    if option_terminations == 1:
                        current_vals = self.get_option_vals(next_state.unsqueeze(0))
                        new_option, options_prob = softmax(current_vals)
                        new_option = tensor(new_option[0])
                        if new_option != option:
                            option_switches += 1
                        avgduration += (1. / option_switches) * (duration - avgduration)
                        duration = 1
                    old_option = option
                    option = new_option


                    state = next_state

                    episode_reward += reward
                    kdx+=1



                    frame_idx += 1
                    if (frame_idx % self.evaluate_iter == 0):
                        test_rewards, options_used, test_option_probs = self.test()
                        print("Test rewards: " + str(test_rewards) + " " + str(test_option_probs) + " " + str(options_used))
                        self.save_weights()
                        log_buffer.append(
                            "Test rewards for frame_idx " + str(frame_idx) + " : " + str(test_rewards) + "\n" + str(
                            options_used) +  " " + str(test_option_probs) + "\n")

                    if done:
                        iter_rewards.append(episode_reward)
                        break


                episode_cnt += 1

                k = ' run {} episode {} steps {} cumreward {} avg. duration {} switches {} alpha {} option_alpha {} frame_idx {}'.format(
                    run, episode_cnt, step,
                    episode_reward, avgduration,
                    option_switches,
                    self.alphas[0], self.option_alpha, frame_idx)
                log_buffer.append(k + "\n")
                if (len(log_buffer) > 100):
                    self.log.writelines("%s" % item for item in log_buffer)
                    log_buffer.clear()
                    self.log.flush()

                self.rewards.append(episode_reward)

            print(
                ' run {} episode {} steps {} cumreward {} avg. duration {} switches {} alpha {} option_alpha {} frame_idx {} avg_rewards {}'.format(
                    run, episode_cnt, step,
                    episode_reward, avgduration,
                    option_switches,
                    self.alphas, self.option_alpha, frame_idx,np.mean(iter_rewards)))

            self.update(self.batch_size, flag=done, frame_idx=frame_idx,
                            options_prob_episode=None,
                            soft_tau=self.soft_tau, actions_obs=None, memory=memory,
                            actions_probs=None)


        if len(log_buffer) != 0:
            self.log.writelines("%s" % item for item in log_buffer)
            log_buffer.clear()
            self.log.flush()
        # self.log.close()
        self.env.close()
        return np.mean(self.rewards[-5:])


def train_helper(idx, extra=None):
    sac = SoftActorCritic(args)
    sac.train(idx)


def multiprocessing(func, args, workers):
    import time
    begin_time = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func, args, [begin_time for i in range(len(args))])
    return list(res)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--value_lr', help='Discount factor', type=float, default=3e-4)
    parser.add_argument('--soft_q_lr', help='Discount factor', type=float, default=3e-4)
    parser.add_argument('--policy_lr', help='Discount factor', type=float, default=3e-4)  # 3e-5
    parser.add_argument('--beta_lr', help='Discount factor', type=float, default=1e-2)
    parser.add_argument('--option_policy_lr', help='Discount factor', type=float, default=3e-4)
    parser.add_argument('--option_alpha_lr', help='Discount factor', type=float, default=3e-4)
    parser.add_argument('--alpha_lr', help='Discount factor', type=float, default=1e-3)
    parser.add_argument('--env_name', help='Discount factor', type=str, default="HopperBulletEnv-v0")
    parser.add_argument('--log_dir', help='Log directory', type=str, default="log_dir")
    parser.add_argument('--model_dir', help='Model directory', type=str, default="model/")
    parser.add_argument('--plot_dir', help='Model directory', type=str, default="plots/")
    parser.add_argument('--evaluate_iter', help='num of episodes before evaluation', type=int, default=4000000)

    parser.add_argument('--hidden_dim', help='Hidden dimension', type=int, default=256)
    parser.add_argument('--num_actions', help='Action dimension', type=int, default=2)
    parser.add_argument('--options_cnt', help='Option count', type=int, default=4)
    parser.add_argument('--runs', help='Runs', type=int, default=5)

    parser.add_argument('--is_continuous', help='is_continuous', type=bool, default=True)
    parser.add_argument('--temp', help='temp', type=float, default=1.0)
    parser.add_argument('--replay_buffer_size', help='Replay buffer size', type=float, default=1000000)
    parser.add_argument('--max_frames', help='Maximum no of frames', type=int, default=150000000)
    parser.add_argument('--max_steps', help='Maximum no of steps', type=int, default=150000000)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=64)
    parser.add_argument('--discriminator_lr', help='Discriminator lr', type=float, default=0.00)
    parser.add_argument('--discriminator_init', help='Discriminator init', type=float, default=0.00)
    parser.add_argument('--update_discriminator', help='Update discriminator after this many episodes', type=int,
                        default=1000)
    parser.add_argument('--decay', help='Decay', type=float, default=0.0)
    parser.add_argument('--target_entropy', help='target_entropy', type=int, default=0)
    parser.add_argument('--target_entropy_', help='target_entropy_', type=int, default=0)
    parser.add_argument('--trial', help='trial', type=int, default=0)
    parser.add_argument('--soft_tau', help='soft_tau', type=float, default=0.01)
    parser.add_argument('--tau', help='tau', type=float, default=0.5)
    parser.add_argument('--le', help='le', type=float, default=0)
    parser.add_argument('--lb', help='lb', type=float, default=0.0) #0.1
    parser.add_argument('--lv', help='lv', type=float, default=0)
    parser.add_argument('--mi_penalty', help='mi_penalty', type=float, default=0)  # 10

    option_policy_lrs = [3e-4, 3e-3, 1e-2]
    policy_lrs = [3e-4, 3e-3, 3e-5]
    options_cnts = [2, 3, 4]
    discriminator_lrs = [0.01, 0.001, 0.0001, 0.0]
    temps = [0.2, 0.01, 0.001, 1.0]
    option_alpha_lrs = [1e-3, 1e-2]
    option_policy_lrs = [3e-3, 3e-4, 3e-5]
    les = [10, 1, 0.1, 0.01]
    lbs = [0.1, 0.01, 1, 10]
    lvs = [1, 0.1, 0.01, 10]
    mi_penalties = [1, 10, 20, 0.1, 0.01]

    best_policy_lr = policy_lrs[0]
    best_option_policy_lr = option_policy_lrs[0]
    best_options_cnt = options_cnts[0]
    best_discriminator_lr = discriminator_lrs[0]
    best_temp = temps[0]
    best_option_alpha_lr = option_alpha_lrs[0]
    best_le = les[0]
    best_lbs = lbs[0]
    best_lvs = lvs[0]
    best_mi_penalty = mi_penalties[0]

    args = parser.parse_args()
    for idx in range(5):
        sac = SoftActorCritic(args)
        sac.save_weights()
        sac.train(1)
        del sac





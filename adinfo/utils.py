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
# import matplotlib
import json
from multiprocessing import Pool
#import matplotlib

#matplotlib.use('agg')
import torch.multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from datetime import datetime

def sample_sigmoid(value):
    return int(rng.uniform() < value)


# def softmax(values, temp=0.3):
#     if len(values.shape)==1:
#         values = values.reshape(-1,values.shape[0])
#
#     if values.shape[-1]==1:
#         return torch.zeros_like(values), torch.ones_like(values)
#     actual_values=values
#     values = values - torch.max(actual_values,-1)[0].reshape(-1,1)
#     values = values / temp
#     probs = F.softmax(values,dim=-1)
#     selected = torch.multinomial(probs,num_samples=1)
#     return selected, probs
def softmax(values, temp=0.3):
    values = values / temp
    #import ipdb; ipdb.set_trace()
    values = values - torch.max(values,dim=-1)[0].reshape(-1,1)
    values_exp = torch.exp(values)

    probs = values - torch.log(torch.sum(values_exp,dim=-1).reshape(-1,1))
    probs = torch.exp(probs)
    selected = torch.multinomial(probs,num_samples=1)
    return selected, probs

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    x = torch.tensor(x, device=device)
    return x.to(device).float()

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


def plot_performance(self, rewards):
    plt.figure()
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.title('Task %s : Returns:' % (self.env_name))
    y = np.arange(len(rewards))
    # plt.plot(rewards,y)
    plt.scatter(y, rewards, color='r', marker='.')
    plt.savefig(self.plot_dir + self.env_name + "_returns.png")
    plt.clf()


from mpl_toolkits import mplot3d

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
    m = m- torch.mean(m, dim=1, keepdim=True)
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

def inverse_tanh(y,epsilon=1e-6):
    return 0.5 * torch.log(((1+y+epsilon)/(1-y+epsilon))+epsilon)


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
    features = features.reshape(1, -1)
    features = normalize_state(features, state_limit)
    req = np.array(features)

    evaluated_fourier = np.cos((np.matmul(fourier_vector, req.transpose())).transpose() * np.pi)
    np.apply_along_axis(apply_reshape, 1, evaluated_fourier)


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


def generate_points(self, cnt):
        u = np.random.uniform(-5.0, 5.0, cnt)
        v = np.random.uniform(-1.0, 1.0, cnt)
        x = np.random.uniform(-5.0, 5.0, cnt)
        y = np.random.uniform(-5.0, 5.0, cnt)
        z = np.random.uniform(-5.0, 5, cnt)

        return u, v, x, y, z

def plot_options(self, x, y, z, options, beta):

        for idx in range(self.options_cnt):
            plt.clf()
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)

            im = ax.scatter3D(x, y, z, c=options[:, idx], cmap=plt.get_cmap("viridis"), norm=normalize, marker='.')
            ax.set_xlabel('Cos theta')
            ax.set_ylabel('Sin theta')
            ax.set_zlabel('Theta dot')
            fig.colorbar(im, ax=ax)

            ax.set_title('Option space')
            plt.savefig(self.plot_dir + self.timestamp + self.env_name + "options_space_" + str(idx) + ".png",
                        dpi=72)
            plt.close(fig)
            plt.clf()
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)

            im = ax.scatter3D(x, y, z, c=beta[:, idx], cmap=plt.get_cmap("viridis"), norm=normalize, marker='.')
            ax.set_xlabel('Cos theta')
            ax.set_ylabel('Sin theta')
            ax.set_zlabel('Theta dot')
            fig.colorbar(im, ax=ax)

            ax.set_title('Beta space')
            plt.savefig(self.plot_dir + self.timestamp + self.env_name + "beta_space_" + str(idx) + ".png",
                        dpi=72)
            plt.close(fig)

def plot_option_space(self, num_points):
        u, v, x, y, z = self.generate_points(num_points)
        points = tensor([np.array([u[idx], v[idx], x[idx], y[idx], z[idx]]) for idx in np.arange(num_points)])
        option_values = self.option_policy_net.get_action_probs(points)  # get_beta_vals(points)
        # option_values = get_option_vals(points)
        beta_vals = self.get_beta_vals(points)
        option_vals_np = option_values.detach().cpu().numpy()
        beta_vals_np = beta_vals.detach().cpu().numpy()
        # option_vals_np = stable_softmax(option_vals_np)
        self.plot_options(u, x, y, option_vals_np, beta_vals_np)

def plot_temporal_activations(self, activations, time_steps):
        plt.figure()
        activations = activations.detach().cpu().numpy()

        timesteps = np.arange(time_steps)
        plt.ylim(-1, 1)
        plt.xlabel("Time steps")
        plt.ylabel("Option selected")
        plt.title('Task %s : Options selected :' % (self.env_name))
        for idx in range(self.options_cnt):
            # plt.plot(rewards[idx], idx, colours[options[idx]]+'.')
            plt.plot(timesteps, activations[idx, 0:time_steps], color=colours[idx], linestyle='-')
        plt.savefig(self.plot_dir + self.timestamp + self.env_name + "_options_selected.png", format="png", dpi=720)
        plt.clf()


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


def log_losses(writer,losses,iter):
    writer.add_scalar('soft_q_net1_loss', losses["sq1"],iter )
    writer.add_scalar('soft_q_net2_loss', losses["sq2"],iter)
    writer.add_scalar('option_policy_loss', losses["op"],iter)
    writer.add_scalar('beta_loss', losses["b"],iter)
    writer.add_scalar('policy_loss', losses["p"],iter)
    writer.add_scalar('sparsity_loss', losses["sp"], iter)
    writer.add_scalar('variance_loss', losses["v"], iter)
    writer.add_scalar('mutual_info_loss', losses["mi"], iter)



def log_episode_summary(writer,episode_idx,steps,alphas,switches,frame_idx):
    writer.add_scalar('episode_idx', episode_idx, episode_idx)
    writer.add_scalar('steps', steps, episode_idx)
    writer.add_scalar('frame_idx',frame_idx, episode_idx)
    for idx in range(len(alphas)):
        writer.add_scalar('alpha_'+str(idx),alphas[idx],episode_idx)
    writer.add_scalar('switches',switches , episode_idx)




def log_test_summary(writer,test_rewards):
    writer.add_scalar('test_rewards', test_rewards)


def squeeze_all(input):
    next_state, reward, done, _ = input
    next_state = np.array(next_state)
    reward = np.array(reward)
    done = np.array(done)

    return next_state[0], reward[0], done[0], _







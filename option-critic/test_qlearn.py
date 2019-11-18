# correct with beta

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

from running_state import ZFilter
#matplotlib.use('agg')
import torch.multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from datetime import datetime




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
        print(self.filename)

        self.log = open(self.filename, 'w+')
        self.log.write(str(args) + "\n")
        self.log.write(self.filename + "\n")
        self.trial = args.trial
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.hidden_dim = args.hidden_dim
        self.num_actions = args.num_actions  # 2
        self.options_cnt = args.options_cnt
        self.temp = 1.0
        self.options_tensor = torch.Tensor(np.arange(self.options_cnt)).unsqueeze(-1)
        self.tau = 1.0 / self.options_cnt  # args.tau
        self.tau_2 = 1.0 / self.options_cnt
        self.replay_buffer_size = 1000000
        self.total_steps = 0.0
        self.max_frames = args.max_frames
        self.max_steps = args.max_steps
        self.frame_idx = 0
        self.rewards = []
        self.entropy_lr = 1e-2
        self.batch_size = args.batch_size

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

        self.value_net = ValueNetwork(self.state_dim + self.options_cnt, self.hidden_dim).to(device)
        self.target_value_net = ValueNetwork(self.state_dim + self.options_cnt, self.hidden_dim).to(device)

        self.option_policy_net = DiscretePolicyNetwork(self.state_dim, self.options_cnt, hidden_size=self.hidden_dim).to(device)

        self.soft_q_net1 = SoftQNetwork(self.state_dim + self.options_cnt, self.action_dim, self.hidden_dim).to(device)

        self.soft_q_net2 = SoftQNetwork(self.state_dim + self.options_cnt, self.action_dim, self.hidden_dim).to(device)
        self.beta_net = BetaBody(self.state_dim, self.options_cnt, self.hidden_dim).to(device)

        self.policy_net = PolicyNetwork(self.state_dim + self.options_cnt, self.action_dim, self.hidden_dim, scale=False).to(
                device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)
        self.beta_optimizer = optim.Adam(self.beta_net.parameters(), lr=self.beta_lr)
        self.option_policy_optimizer = optim.Adam(self.option_policy_net.parameters(), lr=self.option_policy_lr)
        # self.options_prob = torch.zeros(self.options_cnt)
        self.embeddings = torch.eye(self.options_cnt)
        self.running_state = ZFilter((self.state_dim,), clip=5)

        self.option_indices = tensor(np.ones((self.batch_size, self.options_cnt)) * np.arange(self.options_cnt))
        self.option_indices = self.option_indices.unsqueeze(-1)

        self.batch_iter = np.arange(self.batch_size)  # trying to cache as much as possible
        self.batch_iter_t = tensor(self.batch_iter).long()

        self.mini_batch_size = 16
        self.mini_batch_options_tensor = self.encode(self.options_tensor.repeat(16, 1))

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

    def save_weights(self):
        print('saving weights')
        torch.save(self.value_net.state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-value_net")
        torch.save(self.beta_net.state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-beta_net")
        torch.save(self.policy_net.state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-policy_net")
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
        with open(self.model_dir + self.env_name + "_" + self.timestamp + '-zfilter',  'wb') as f:
            pickle.dump(self.running_state, f)




    def encode(self,options):
        return self.embeddings[options.flatten().long()]

    def decode(self,options):
        return torch.sum(options,-1)

    def load_weights(self, prefix=None):
        if prefix==None:
            prefix = self.env_name + "_" + self.timestamp
        self.value_net.load_state_dict(torch.load(self.model_dir + prefix + "-value_net"))
        self.beta_net.load_state_dict(torch.load(self.model_dir  + prefix + "-beta_net"))
        self.policy_net.load_state_dict(torch.load(self.model_dir + prefix + "-policy_net"))
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
        option_indices=self.option_indices
        all_input=self.batch_iter
        shape = self.batch_size
        if states.shape[0]!= self.batch_size:
            shape = states.shape[0]
            all_input = np.arange(states.shape[0])
            option_indices= tensor(np.ones((shape, self.options_cnt)) * np.arange(self.options_cnt)).unsqueeze(-1)

        option_vals = tensor(np.zeros((shape, self.options_cnt)))
        for option_idx in range(self.options_cnt):
            extended_states = torch.cat((states, self.embeddings[option_idx].repeat(shape,1)), -1)
            option_vals[all_input, option_idx] = self.value_net(extended_states).squeeze(-1)
        return option_vals

    def get_target_vals(self, states, option_dim=1):
        # num_input = states.shape[0]
        option_vals = tensor(np.zeros((self.batch_size, self.options_cnt)))
        for option_idx in range(self.options_cnt):
            extended_states = torch.cat((states, self.embeddings[option_idx].repeat(self.batch_size,1)), -1)
            option_vals[self.batch_iter, option_idx] = self.target_value_net(extended_states).squeeze(-1)
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
        extended_states = torch.cat((states,options),-1)
        soft_option_q_values = None
        if target==False:
            soft_option_q_values = self.value_net(extended_states)
        else:
            soft_option_q_values = self.target_value_net(extended_states)
        return soft_option_q_values

    def stable_softmax(self, X):
        exps = np.exp(X - np.max(X, 1).reshape(-1, 1))
        return exps / np.sum(exps, 1).reshape(-1, 1)

    def get_expected_log_prob(self, actions, log_probs):
        expected_log_probs = log_probs[np.arange(self.batch_size), torch.argmax(actions, -1)]
        #expected_log_probs = log_probs* actions
        #expected_log_probs = torch.sum(expected_log_probs, -1, keepdim=True)
        return expected_log_probs.reshape(-1,1)

    def get_expected_option_log_prob(self, options, log_probs):
        expected_log_probs = log_probs[np.arange(self.batch_size), torch.argmax(options, -1)]
        #expected_log_probs = torch.sum(expected_log_probs, -1, keepdim=True)
        return expected_log_probs.reshape(-1,1)

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


    def update(self, batch_size, gamma=0.99, soft_tau=0.01, frame_idx=0, options_prob_episode=None, flag=False,
               actions_obs=None):
        state, option, action, reward, next_state, done, next_option, extended_state, extended_next_state = self.replay_buffer.sample(
            batch_size)

        state = tensor(state).to(device)
        option = tensor(option).to(device)
        next_state = tensor(next_state).to(device)
        action = tensor(action).to(device)
        extended_state = tensor(extended_state).to(device)
        extended_next_state = tensor(extended_next_state).to(device)
        reward = tensor(reward).unsqueeze(1).to(device)
        done = tensor(np.float32(done)).unsqueeze(1).to(device)
        beta = self.get_beta_vals(state)


        predicted_q_value1 = self.soft_q_net1(extended_state, action)

        predicted_q_value2 = self.soft_q_net2(extended_state, action)

        predicted_value = self.value_net(extended_state)

        target_predicted_next_value = self.target_value_net(extended_next_state)
        predicted_next_value = self.value_net(extended_next_state)

        predicted_next_values = self.get_option_vals(next_state)



        new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(extended_state)

        option_idx = option.detach().long().flatten().cpu().numpy()

        beta_next = self.get_beta_vals(next_state)
        option_vals = self.get_option_vals(state)
        next_option_vals = self.get_option_vals(next_state)

        new_option, option_prob = softmax(option_vals)

        next_new_option, next_option_prob = softmax(next_option_vals)
        option_log_prob = torch.log(option_prob+1e-20)
        next_option_log_prob = torch.log(next_option_prob+1e-20)

        expected_predicted_next_value = torch.mean(next_option_log_prob.exp() * next_option_vals,-1)





        predicted_next_new_option_q_value  = torch.max(self.get_option_vals(next_state),-1)[0].reshape(-1,1) # makes it unstable
        target_predicted_next_new_option_q_value  = torch.max(self.get_target_vals(next_state),-1)[0].reshape(-1,1)






        target_q_value = reward + (1 - done) * gamma * (((1 - beta_next[
            self.batch_iter, option_idx]).reshape(-1, 1) * (target_predicted_next_value)) + beta_next[
                                                            self.batch_iter, option_idx].reshape(-1, 1) * (
                                                                target_predicted_next_new_option_q_value))


        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), 10)
        self.soft_q_optimizer1.step()


        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), 10)
        self.soft_q_optimizer2.step()

        advantage = (predicted_next_values[self.batch_iter, option_idx].reshape(-1, 1) - (
         predicted_next_new_option_q_value.reshape(-1, 1)))

        beta_loss = (beta_next[self.batch_iter, option_idx].reshape(-1, 1) * (
            advantage.detach())).mean()

        self.beta_optimizer.zero_grad()
        beta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.beta_net.parameters(), 10)
        self.beta_optimizer.step()

        alphas_option = tensor([self.alphas[option_idx[0].long()] for option_idx in option])
            # print("Here")
        predicted_q_value1, predicted_q_value2 = self.soft_q_net1(extended_state, new_action), self.soft_q_net2(
                extended_state,
                new_action)

        predicted_new_q_value = torch.min(predicted_q_value1,
                                          predicted_q_value2)
        target_value_func = predicted_new_q_value - alphas_option.view(-1, 1) * log_prob  # target_q_value - log_prob #

        value_loss = self.value_criterion(predicted_value, target_value_func.detach())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 10)
        self.value_optimizer.step()



        option_policy_loss = None



        #if flag:
        #    print(softmax(self.get_option_vals(state))[1][1])
        policy_loss = (alphas_option * log_prob - predicted_new_q_value).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)

        self.policy_optimizer.step()

        if self.target_entropy is not None:
            log_alphas_option = [self.log_alphas[option_idx[0].long()] for option_idx in option]
            log_alphas_opt = torch.stack(log_alphas_option)
            alpha_loss = -(log_alphas_opt.exp() * (log_prob + self.target_entropy).detach()).mean()
            for idx in range(self.options_cnt):
                self.alpha_optimizers[idx].zero_grad()
                if idx == self.options_cnt - 1:
                    alpha_loss.backward()
                else:
                    alpha_loss.backward(retain_graph=True)
                self.alpha_optimizers[idx].step()

                self.alphas[idx] = self.log_alphas[idx].detach().exp()
                # if(episode> 100):
                #    alphas[option] = 0.0
        option_alpha_loss = None
        if self.target_entropy_ is not None:

            option_alpha_loss = -(
                    self.option_log_alpha.exp() * (option_log_prob + self.target_entropy_).detach()).mean()

            self.option_alpha_optimizer.zero_grad()
            option_alpha_loss.backward()
            self.option_alpha_optimizer.step()
            self.option_alpha = self.option_log_alpha.detach().exp()
            # if(episode>100):
            #    option_alpha=0.0
        else:
            alpha_loss = 0

        if frame_idx % 1 == 0:
            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )
        return policy_loss, beta_loss, value_loss, option_policy_loss, option_alpha_loss, q_value_loss2, q_value_loss1


    def test(self, max_episode_len=1000):
        env = gym.make(self.env_name)

        frame_idx = 0
        run = self.trial
        options_used = ""
        total_rewards = []
        total_steps = 0
        episode_cnt = 0
        mean_activation=[]
        while episode_cnt < 10:
            state = np.array(env.reset())
            #state = self.running_state(state)
            state = tensor(state)
            episode_reward = 0
            options_probs = torch.zeros(self.options_cnt, max_episode_len)

            # option, options_prob,__,___,___ = self.option_policy_net.evaluate(state.reshape(1,-1))
            # option = torch.argmax(option,-1)
            option, options_prob = softmax(self.get_option_vals(state.unsqueeze(0)))
            option = tensor(option[0])
            option = tensor(option)
            options_used=""
            total_steps=0

            for step in range(self.max_steps):
                option_idx = int(option.cpu().numpy()[0])
                options_probs[:, step] = options_prob[0]
                total_steps+=1
                options_used+=str(option_idx)

                extended_state = torch.cat((state, self.encode(option).squeeze()), -1)
                action = self.policy_net.get_action(extended_state).detach()
                # print(action)

                next_state, reward, done, _ = env.step(action.cpu().numpy())
                #next_state = self.running_state(next_state)

                next_state = tensor(next_state)

                beta_next = self.get_beta_vals(next_state.unsqueeze(0))
                beta_next_np = beta_next.detach().cpu().numpy()[0]
                option_terminations = sample_sigmoid(beta_next_np[option_idx])
                if option_terminations == 1:
                    # new_option, options_prob, __, ___, ___ = self.option_policy_net.evaluate(next_state.reshape(1, -1))
                    # new_option = torch.argmax(new_option, -1)

                    new_option, options_prob = softmax(self.get_option_vals(next_state.unsqueeze(0)))
                    new_option = tensor(new_option[0])

                    new_option = tensor(new_option)
                    option = new_option

                state = next_state

                episode_reward += reward

                if done:
                    total_rewards.append(episode_reward)
                    mean_activation.append(torch.mean(options_probs[:,:total_steps],-1))
                    break

            episode_cnt += 1
        env.close()

        return np.mean(total_rewards), options_used, torch.mean(torch.stack(mean_activation),0).detach().cpu().numpy().tolist()

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
        options_probs = torch.zeros(self.options_cnt, max_episode_len)
        step = 0.0

        while frame_idx < self.max_frames:
            state = np.array(self.env.reset())
            #state = self.running_state(state)
            state = tensor(state)
            episode_reward = 0
            options_probs = torch.zeros(self.options_cnt, max_episode_len)
            actions_obs = torch.zeros(self.action_dim, max_episode_len, self.options_cnt)
            beta_next = self.get_beta_vals(state.unsqueeze(0))


            #option, options_prob, __, ___, ___ = self.option_policy_net.evaluate(state.reshape(1, -1))
            # option = torch.argmax(option, -1)
            option, options_prob = softmax(self.get_option_vals(state.unsqueeze(0)))
            option = tensor(option[0])
            # option = sample_softmax(option_vals.detach().cpu().numpy()[0])
            duration = 1
            option_switches = 1
            avgduration = 0.
            old_option = option
            new_option = option
            done = False
            for step in range(self.max_steps):
                option_idx = int(option.cpu().numpy()[0])
                options_probs[:, step] = options_prob[0].exp() # beta_next[0][option_idx]

                if frame_idx >=0 :

                    extended_states = torch.cat((state.repeat(self.options_cnt, 1), self.encode(self.options_tensor)), -1)
                    extended_state = extended_states[option_idx]
                    actions, _, __, ___, _____ = self.policy_net.evaluate(extended_states)
                    actions_obs[:, step, :] = actions.t()
                    action = actions[option_idx].detach().cpu()

                    # print(action)

                    next_state, reward, done, _ = self.env.step(action.cpu().numpy())
                    #next_state = self.running_state(next_state)
                # next_state = features(next_state)


                action = tensor(action)
                next_state = tensor(next_state)
                # next_state = features(next_state)

                beta_next = self.get_beta_vals(next_state.unsqueeze(0))
                beta_next_np = beta_next.detach().cpu().numpy()[0]
                new_option=option
                option_terminations = sample_sigmoid(beta_next_np[option_idx])
                #if True:
                if option_terminations == 1:
                    new_option, options_prob = softmax(self.get_option_vals(next_state.unsqueeze(0)))
                    new_option = tensor(new_option[0])

                if new_option != option:
                    option_switches += 1
                    avgduration += (1. / option_switches) * (duration - avgduration)
                    duration = 1
                old_option = option
                option = new_option
                self.replay_buffer.push(state.detach().cpu().numpy(),
                                        tensor(old_option).cpu().numpy(), action.cpu().numpy(),
                                        reward, next_state.cpu().numpy(), done,
                                        tensor(option).cpu().numpy(),
                                        torch.cat((state, self.encode(old_option).squeeze()), -1).cpu().numpy(),
                                        torch.cat((next_state, self.encode(old_option).squeeze()), -1).cpu().numpy())

                # env.render()

                state = next_state

                episode_reward += reward

                if len(self.replay_buffer) > self.batch_size:
                    if done:

                        policy_loss, beta_loss, value_loss, option_policy_loss, option_alpha_loss, q_value_loss2, q_value_loss1 = self.update(self.batch_size, flag=done, frame_idx=frame_idx,
                                    options_prob_episode=options_probs[:, :step],
                                    soft_tau=self.soft_tau, actions_obs=None)
                        log_buffer.append(" policy_loss " + str(policy_loss) + " beta_loss " + str(beta_loss) + " value_loss " + str(value_loss) + " option_policy_loss " + str(option_policy_loss)
                                         + " option_alpha_loss " + str(option_alpha_loss) + " q_value_loss " + str(q_value_loss1) + " q_value_loss_2 " + str(q_value_loss2) + "\n")
                        options_probs = options_probs*0.0
                        actions_obs = actions_obs*0.0


                    else:
                        self.update(self.batch_size, frame_idx=frame_idx, options_prob_episode=None,
                                    soft_tau=self.soft_tau,actions_obs=None)

                frame_idx += 1
                if (frame_idx % self.evaluate_iter == 0):
                    test_rewards, options_used, test_option_probs = self.test()
                    print("Test rewards: " + str(test_rewards) + " " + str(test_option_probs))
                    print(options_used)
                    self.save_weights()
                    log_buffer.append(
                        "Test rewards for frame_idx " + str(frame_idx) + " : " + str(test_rewards) + " " + str(
                            options_used) + " " + str(test_option_probs) + "\n")

                if done:
                    # action = actions[0].detach()
                    # self.plot_temporal_activations(options_probs,step)
                    break

            episode_cnt += 1
            if self.discriminator_lr < self.discriminator_lr_max and episode_cnt > self.update_discriminator:
                self.discriminator_lr = min(self.discriminator_lr * 1.1, self.discriminator_lr_max)
            print(
                ' run {} episode {} steps {} cumreward {} avg. duration {} switches {} alpha {} option_alpha {} frame_idx {}'.format(
                    run, episode_cnt, step,
                    episode_reward, avgduration,
                    option_switches,
                    self.alphas, self.option_alpha, frame_idx))
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
        if len(log_buffer) != 0:
            self.log.writelines("%s" % item for item in log_buffer)
            log_buffer.clear()
            self.log.flush()
        # self.log.close()
        self.env.close()

        return episode_cnt


def train_helper(idx, extra=None):
    sac = SoftActorCritic(args)
    sac.train(idx)


def multiprocessing(func, args, workers):
    import time
    begin_time = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        res = executor.map(func, args, [begin_time for i in range(len(args))])
    return list(res)
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
    parser.add_argument('--beta_lr', help='Discount factor', type=float, default=3e-3)
    parser.add_argument('--option_policy_lr', help='Discount factor', type=float, default=3e-4)#* #3e-4
    parser.add_argument('--option_alpha_lr', help='Discount factor', type=float, default=3e-4)
    parser.add_argument('--alpha_lr', help='Discount factor', type=float, default=1e-3)
    parser.add_argument('--env_name', help='Discount factor', type=str, default="HopperBulletEnv-v0")
    parser.add_argument('--log_dir', help='Log directory', type=str, default="log_dir")
    parser.add_argument('--model_dir', help='Model directory', type=str, default="model/")
    parser.add_argument('--plot_dir', help='Model directory', type=str, default="plots/")
    parser.add_argument('--evaluate_iter', help='num of episodes before evaluation', type=int, default=4000)

    parser.add_argument('--hidden_dim', help='Hidden dimension', type=int, default=256)
    parser.add_argument('--num_actions', help='Action dimension', type=int, default=2)
    parser.add_argument('--options_cnt', help='Option count', type=int, default=4)
    parser.add_argument('--runs', help='Runs', type=int, default=5)

    parser.add_argument('--is_continuous', help='is_continuous', type=bool, default=True)
    parser.add_argument('--temp', help='temp', type=float, default=1.0)
    parser.add_argument('--replay_buffer_size', help='Replay buffer size', type=float, default=2000000)
    parser.add_argument('--max_frames', help='Maximum no of frames', type=int, default=1500000)
    parser.add_argument('--max_steps', help='Maximum no of steps', type=int, default=1500000)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--discriminator_lr', help='Discriminator lr', type=float, default=0.000)
    parser.add_argument('--discriminator_init', help='Discriminator init', type=float, default=0.00)
    parser.add_argument('--update_discriminator', help='Update discriminator after this many episodes', type=int,
                        default=1000)
    parser.add_argument('--decay', help='Decay', type=float, default=0.0)
    parser.add_argument('--target_entropy', help='target_entropy', type=int, default=-1)
    parser.add_argument('--target_entropy_', help='target_entropy_', type=int, default=0)
    parser.add_argument('--trial', help='trial', type=int, default=0)
    parser.add_argument('--soft_tau', help='soft_tau', type=float, default=0.01)
    parser.add_argument('--tau', help='tau', type=float, default=0.5)
    parser.add_argument('--le', help='le', type=float, default=0.0)
    parser.add_argument('--lb', help='lb', type=float, default=0) #0.01 #1 #10 in terminal

    parser.add_argument('--lv', help='lv', type=float, default=0)
    parser.add_argument('--mi_penalty', help='mi_penalty', type=float, default=10)  # 10

    option_policy_lrs = [3e-4, 3e-3, 1e-2]
    policy_lrs = [3e-4, 3e-3, 3e-5]
    options_cnts = [2, 3, 4]
    discriminator_lrs = [0.01, 0.001, 0.0001, 0.0]
    temps = [1,0,0.1, 0.01]
    option_alpha_lrs = [1e-3, 1e-2]
    option_policy_lrs = [3e-3, 3e-4, 3e-5]
    les = [10, 1, 0.1]
    lbs = [10,1,0.1, 0.01]
    lvs = [10,1, 0.1]
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
    if args.options_cnt==2:
        args.lb=10
    rewards = 0.0
    reward_log=[]

    args = parser.parse_args()
    for idx in range(5):
        sac = SoftActorCritic(args)
        sac.train(1)
        del sac





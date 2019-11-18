
import gym
try:
    import roboschool
except:
    pass
import numpy as np
import argparse
from running_state  import ZFilter
from tensorboardX import SummaryWriter

import torch.multiprocessing

from datetime import datetime

from utils import *
from modelv2 import *
from replay_buffer import *

from scipy.stats import multivariate_normal


def softmax(values, temp=0.3): #0.3
    values = values / temp
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

rng = np.random.RandomState(1234)

import torch

torch.set_num_threads(8)
print(torch.get_num_threads())
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


use_cuda = torch.cuda.is_available()
device = "cpu" #torch.device("cuda" if use_cuda else "cpu")
import numpy as np

colours = ['y', 'r', 'g', 'm', 'b', 'k', 'w', 'c']

import pickle



import time

import torch
from matplotlib import pyplot as plt

from pykeops.torch import LazyTensor

class SoftActorCritic():
    def __init__(self, args):
        self.reset()

    def reset(self):
        self.env = gym.make(args.env_name)
        self.test_env = gym.make(args.env_name)
        print(args)
        self.env_name = args.env_name
        self.seed=np.random.randint(1,10000)
        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filename = args.log_dir + "/" + args.env_name + "_" + self.timestamp + str(
            args.options_cnt) + str(self.seed)
        self.soft_tau = args.soft_tau
        self.model_dir = args.model_dir
        self.plot_dir = args.plot_dir


        self.le = args.le
        self.lb = args.lb
        self.lv = args.lv
        self.mi_penality = args.mi_penalty
        #self.writer = SummaryWriter(
        #    logdir='runs/{}_SAC_{}_{}'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,self.seed
        #                                         ))

        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.log = open(self.filename, 'w+')
        self.log.write(str(args) + "\n")
        self.log.write(self.filename+"\n")
        print(self.filename)
        self.trial = args.trial
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.hidden_dim = args.hidden_dim
        self.options_cnt = args.options_cnt
        self.temp = 1.0
        self.options_tensor = tensor(np.arange(self.options_cnt)).unsqueeze(-1)
        self.tau = 1.0 / self.options_cnt  # args.tau
        self.replay_buffer_size = args.replay_buffer_size
        self.total_steps = 0.0
        self.max_frames = args.max_frames
        self.max_steps = args.max_steps
        self.frame_idx = 0
        self.rewards = []
        self.entropy_lr = 1e-2
        self.bl=args.bl
        self.batch_size = args.batch_size
        self.running_state = ZFilter((self.state_dim,), clip=5)

        self.decay = 1.0
        self.target_entropy = args.target_entropy or None
        self.target_entropy_ = args.target_entropy_ or None
        self.log.write("qlearn")
        print("qlearn")
        #self.log.write("separate policies " + str(self.separate_policies) + " sample_options "+ str(self.sample_option_type)+ " \n")
        print(self.target_entropy_)
        self.option_log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = None
        self.option_alpha = self.option_log_alpha.exp().detach()
        self.log_alphas = [torch.zeros(1, requires_grad=True) for x in range(self.options_cnt)]
        self.alpha_optimizers = [None for x in range(self.options_cnt)]
        self.evaluate_iter = args.evaluate_iter
        self.update_option_iter = args.update_option_iter
        self.update_policy_iter = args.update_policy_iter

        self.num_updates = args.num_updates



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

        self.value_lr = args.value_lr
        self.soft_q_lr = args.soft_q_lr
        self.policy_lr = args.policy_lr
        self.beta_lr = args.beta_lr
        self.option_policy_lr = args.option_policy_lr
        self.option_alpha_lr = args.option_alpha_lr
        self.alpha_lr = args.alpha_lr


        self.value_net = OptionsValueNetwork(self.state_dim ,self.options_cnt, self.hidden_dim).to(device)
        self.target_value_net = OptionsValueNetwork(self.state_dim , self.options_cnt, self.hidden_dim).to(device)

        self.option_policy_net = DiscretePolicyNetwork(self.state_dim, self.options_cnt, hidden_size=self.hidden_dim).to(device)
        self.soft_q_net1 = OptionsSoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(
                device)

        self.soft_q_net2 = OptionsSoftQNetwork(self.state_dim,self.action_dim, self.hidden_dim).to(
                device)

        self.beta_net = BetaBody(self.state_dim, self.options_cnt, self.hidden_dim).to(device)

        self.actor_net_list = []
        self.target_actor_net_list = []
        for option_idx in range(self.options_cnt):
            self.actor_net_list.append(OptionsPolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(
                device))
            self.target_actor_net_list.append(
                OptionsPolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(
                    device))

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_criterion = nn.MSELoss()
        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        self.action_noise = args.action_noise

        self.replay_buffer = OptionsReplayBuffer(self.replay_buffer_size)
        self.replay_buffer_onpolicy = OptionsReplayBuffer(self.replay_buffer_size)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=self.soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=self.soft_q_lr)
        self.beta_optimizer = optim.Adam(self.beta_net.parameters(), lr=self.beta_lr)
        self.option_policy_optimizer = optim.Adam(self.option_policy_net.parameters(), lr=self.option_policy_lr)
        self.actor_optimizer_list = []
        for option_idx in range(self.options_cnt):
            self.actor_optimizer_list.append(optim.Adam(self.actor_net_list[option_idx].parameters(), lr=self.policy_lr))

        self.embeddings = torch.eye(self.options_cnt)

        self.option_indices = tensor(np.ones((self.batch_size, self.options_cnt)) * np.arange(self.options_cnt))
        self.option_indices = self.option_indices.unsqueeze(-1)

        self.batch_iter = np.arange(self.batch_size)  # trying to cache as much as possible
        self.batch_iter_t = tensor(self.batch_iter).long()

        self.mini_batch_size = 256

        self.mini_batch_size = 256
        self.mini_batch_options_tensor = self.encode(self.options_tensor.repeat(self.mini_batch_size, 1))
        self.mini_batch_tensor = self.options_tensor.repeat(self.mini_batch_size, 1)
        self.batch_tensor = self.options_tensor.repeat(self.batch_size, 1)






    def save_weights(self):
        print('saving weights')
        torch.save(self.value_net.state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-value_net")
        torch.save(self.beta_net.state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-beta_net")
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
        assert(options.shape[-1]==1)
        return self.embeddings[options.flatten().long()]

    def decode(self, options):
        assert(options.shape[-1]>1)
        return torch.argmax(options, -1).reshape(-1,1)

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

    def get_option_vals(self, states):
        return self.value_net(states)

    def get_target_vals(self, states):
        return self.target_value_net(states)

    def get_beta_vals(self, state):
        beta_vals = self.beta_net(state)
        return beta_vals




    def get_option_values(self, states, options, target=False):
        option_q_values = None

        if target==False:
            option_q_values = self.value_net(states)[np.arange(options.shape[0]),options.flatten()].reshape(-1,1)
        else:
            option_q_values = self.target_value_net(states)[np.arange(options.shape[0]),options.flatten()].reshape(-1,1)
        return option_q_values

    def get_single_option_values(self,state,option_idx,target=False):
        option_q_values = self.value_net(state)[np.arange(state.shape[0]), option_idx].reshape(-1, 1)
        return option_q_values




    def get_log_prob(self, actions, log_probs):
        assert(actions.shape[-1]>1)
        expected_log_probs = log_probs[np.arange(self.batch_size), torch.argmax(actions, -1)]
        return expected_log_probs.reshape(-1,1)



    def get_mutual_information_penalty(self, actions_obs):
        penalty = 0.0
        mutual_info = []

        for idx in range(self.action_dim):
            covariance = cov(actions_obs[idx, ::])
            # print(covariance)
            variance = torch.sum(covariance * torch.eye(self.options_cnt), -1).reshape(-1, 1)
            sigma_ij = torch.sqrt(torch.mm(variance, variance.t())+1e-20)
            corr = covariance / (sigma_ij+1e-20)
            mutual_information = (-0.5 * torch.log(1 - corr ** 2 + 1e-20))
            mutual_information = mutual_information - mutual_information* torch.eye(self.options_cnt)

            mutual_info.append(torch.mean(mutual_information))

        return torch.mean(torch.stack(mutual_info))

    def get_actual_option(self,states,actions,epsilon=1e-6):
        actions = actions.unsqueeze(2).repeat(1, 1, self.options_cnt).transpose(1, 2).reshape(
            self.batch_size * self.options_cnt, -1)


        new_actions = []
        new_log_probs = []
        new_means = []
        new_stds = []
        for idx in range(self.options_cnt):
            new_action, new_log_prob, z, mean, log_std = self.actor_net_list[idx].evaluate(states)
            new_actions.append(new_action)
            new_log_probs.append(new_log_prob)
            new_means.append(mean)
            new_stds.append(log_std)

        new_actions = torch.stack(new_actions,-1).transpose(1,2)
        new_log_probs = torch.stack(new_log_probs,-1).transpose(1,2)
        new_means = torch.stack(new_means,-1).transpose(1,2)
        new_stds = torch.stack(new_stds,-1).transpose(1,2)


        actions = actions.reshape(-1,self.action_dim)



        actions = actions.reshape(-1,self.action_dim)
        mean = new_means.reshape(-1,self.action_dim)
        log_std = new_stds.reshape(-1,self.action_dim)

        std = log_std.exp()
        assert(torch.sum(torch.isnan(inverse_tanh(actions)))==0)
        log_probs = Normal(mean, std).log_prob(inverse_tanh(actions)) - torch.log(1 - actions.pow(2) + epsilon)
        log_probs = torch.sum(log_probs, dim=-1, keepdim=True)
        log_probs = log_probs.reshape(-1,self.options_cnt)
        options = torch.argmax(log_probs, -1).reshape(-1, 1).detach()


        return options



    def get_action(self,state,option):


        action, log_prob, z, mean, log_std = self.actor_net_list[option].evaluate(state)
        return action.reshape(-1,self.action_dim), log_prob.reshape(-1,1), z, mean.reshape(-1,self.action_dim), log_std.reshape(-1,self.action_dim)


    def get_action_list(self,state,option):
        actions = []
        new_log_probs = []
        new_actions = []

        for option_idx in range(self.options_cnt):
                new_action, new_log_prob, z, mean, log_std = self.actor_net_list[idx].evaluate(state)
                new_actions.append(new_action)
                new_log_probs.append(new_log_prob)

        new_actions = torch.stack(new_actions, -1).transpose(1, 2)
        new_actions = new_actions[np.arange(state.shape[0]), option.flatten(), :]

        new_log_probs = torch.stack(new_log_probs, -1).transpose(1, 2)
        new_log_probs = new_log_probs[np.arange(state.shape[0]), option.flatten(), :]




        return new_actions.reshape(-1,self.action_dim), new_log_probs.reshape(-1,1)

    def get_q_values(self,state,action):
        soft_q_value1 = self.soft_q_net1(state,action).reshape(-1,1)
        soft_q_value2 = self.soft_q_net2(state,action).reshape(-1,1)

        return soft_q_value1, soft_q_value2




    def update_option(self,num_updates,batch_size):


        for update_id in range(num_updates):

            state, action, reward, next_state, done, p = self.replay_buffer_onpolicy.sample(
            batch_size)





            state = tensor(state).to(device)
            next_state = tensor(next_state).to(device)




            new_option, option_prob_ = softmax(self.get_option_vals(state))
            option_log_prob_ = torch.log(option_prob_+1e-20)

            action = tensor(action).to(device)
            option=None

            option = self.get_actual_option(state,action)

            reward = tensor(reward).unsqueeze(1).to(device)
            done = tensor(np.float32(done)).unsqueeze(1).to(device)
            max_indx = option.flatten().long()
            for o in range(self.options_cnt):
                indx_o = (max_indx == o)
                if (torch.sum(indx_o) == 0):
                    continue
                s_batch_o = state[indx_o, :]
                o_batch = option[indx_o, :]

                alphas_option = self.alphas[o]
                new_action, new_log_prob, new_epsilon, new_mean, new_log_std = self.get_action(
                    s_batch_o, o)

                predicted_q_value1, predicted_q_value2 = self.get_q_values(s_batch_o, new_action)

                predicted_q_value2 = predicted_q_value2.reshape(-1, 1)
                predicted_q_value1 = predicted_q_value1.reshape(-1, 1)
                predicted_new_q_value = torch.min(predicted_q_value1,
                                                  predicted_q_value2)


                alphas_option = self.alphas[o]
                predicted_value = self.get_single_option_values(s_batch_o,o)


                target_value_func = predicted_new_q_value - alphas_option.view(-1,
                                                                           1) * new_log_prob  # target_q_value - log_prob #

                value_loss = self.value_criterion(predicted_value, target_value_func.detach())

                self.value_optimizer.zero_grad()

                value_loss.backward()
                # for param in self.value_net.parameters():
                # if torch.sum(torch.isnan(param.grad.data)) > 0:
                #     import ipdb;
                #     ipdb.set_trace()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 5)
                self.value_optimizer.step()
        print("updated options")

    def update_option(self,num_updates,batch_size):
        for update_id in range(num_updates):

            state, action, reward, next_state, done, p = self.replay_buffer_onpolicy.sample(
            batch_size)




            state = tensor(state).to(device)
            next_state = tensor(next_state).to(device)




            new_option, option_prob_ = softmax(self.get_option_vals(state))
            option_log_prob_ = torch.log(option_prob_+1e-20)

            action = tensor(action).to(device)
            option=None

            option = self.get_actual_option(state,action)

            reward = tensor(reward).unsqueeze(1).to(device)
            done = tensor(np.float32(done)).unsqueeze(1).to(device)
            max_indx = option.flatten().long()
            for o in range(self.options_cnt):
                indx_o = (max_indx == o)
                if (torch.sum(indx_o) == 0):
                    continue
                s_batch_o = state[indx_o, :]
                o_batch = option[indx_o, :]

                alphas_option = self.alphas[o]
                new_action, new_log_prob, new_epsilon, new_mean, new_log_std = self.get_action(
                    s_batch_o, o)

                predicted_q_value1, predicted_q_value2 = self.get_q_values(s_batch_o, new_action)

                predicted_q_value2 = predicted_q_value2.reshape(-1, 1)
                predicted_q_value1 = predicted_q_value1.reshape(-1, 1)
                predicted_new_q_value = torch.min(predicted_q_value1,
                                                  predicted_q_value2)


                alphas_option = self.alphas[o]
                predicted_value = self.get_single_option_values(s_batch_o,o)


                target_value_func = predicted_new_q_value - alphas_option.view(-1,
                                                                           1) * new_log_prob  # target_q_value - log_prob #

                value_loss = self.value_criterion(predicted_value, target_value_func.detach())

                self.value_optimizer.zero_grad()

                value_loss.backward()
                # for param in self.value_net.parameters():
                # if torch.sum(torch.isnan(param.grad.data)) > 0:
                #     import ipdb;
                #     ipdb.set_trace()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 5)
                self.value_optimizer.step()
        print("updated options")




    def update_actor(self, batch_size, gamma=0.99, soft_tau=0.01, frame_idx=0, options_prob_episode=None, flag=False,
               actions_obs=None,num_updates=100):
        for update_id in range(num_updates):
            #print(update_id,batch_size)
            state, action, reward, next_state, done, p = self.replay_buffer.sample(
            batch_size)


            state = tensor(state).to(device)
            next_state = tensor(next_state).to(device)

        #


            new_option, option_prob_ = softmax(self.get_option_vals(state))
            option_log_prob_ = torch.log(option_prob_+1e-20)

            action = tensor(action).to(device)
            option=None

            option = self.get_actual_option(state,action)

            reward = tensor(reward).unsqueeze(1).to(device)
            done = tensor(np.float32(done)).unsqueeze(1).to(device)
            p = tensor(np.float32(p)).unsqueeze(1).to(device)





            # predicted_q_value1, predicted_q_value2 = self.get_q_values(state,action)
            #


            predicted_next_val = self.get_option_values(next_state,option).reshape(-1,1)



            # new_action, log_prob, epsilon, mean, log_std = self.get_action(state,option)

            next_action, next_log_prob = self.get_action_list(next_state, option)


            alphas_option = tensor([self.alphas[option_idx[0].long()] for option_idx in option])
            next_q1 , next_q2 = self.get_q_values(next_state, next_action)

            predicted_next_value = torch.min(next_q1,next_q2) - alphas_option.view(-1,
                                                                           1) * next_log_prob

            option_idx = option.detach().long().flatten().cpu().numpy()

            beta = self.get_beta_vals(state)

            next_option_vals = self.get_option_vals(next_state)
            next_option , next_option_prob = softmax(next_option_vals)
            average_next_option_val = torch.sum(next_option_prob* next_option_vals,-1)

            next_new_action, next_new_log_prob = self.get_action_list(next_state, next_option)
            next_alphas_option = tensor([self.alphas[option_idx[0].long()] for option_idx in next_option])

            next_new_q1, next_new_q2 = self.get_q_values(next_state, next_new_action)
            predicted_next_new_option_q_value = torch.min(next_new_q1,next_new_q2) - next_alphas_option.view(-1,
                                                                           1) * next_new_log_prob







            beta_next = self.get_beta_vals(next_state)



            # target_q_value = reward + (1 - done) * gamma * (((1 - beta_next[
            #     self.batch_iter, option_idx]).reshape(-1, 1) * (predicted_next_value)) + beta_next[
            #                                                 self.batch_iter, option_idx].reshape(-1, 1) * (
            #                                                     predicted_next_new_option_q_value))

            target_q_value = reward + (1 - done) * gamma *predicted_next_new_option_q_value


            predicted_q_value1, predicted_q_value2 = self.get_q_values(state, action)

            predicted_q_value2 = predicted_q_value2.reshape(-1, 1)
            predicted_q_value1 = predicted_q_value1.reshape(-1, 1)

            q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())
            q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())

            self.soft_q_optimizer1.zero_grad()
            q_value_loss1.backward()
            torch.nn.utils.clip_grad_norm_(self.soft_q_net1.parameters(), 10)
            for param in self.soft_q_net1.parameters():
                if torch.sum(torch.isnan(param.grad.data)) > 0:
                    import ipdb; ipdb.set_trace()
            self.soft_q_optimizer1.step()

            self.soft_q_optimizer2.zero_grad()
            q_value_loss2.backward()
            for param in self.soft_q_net2.parameters():
                if torch.sum(torch.isnan(param.grad.data)) > 0:
                    import ipdb; ipdb.set_trace()
                #print(torch.sum(param.data))
            torch.nn.utils.clip_grad_norm_(self.soft_q_net2.parameters(), 10)
            self.soft_q_optimizer2.step()

            advantage = (predicted_next_val - (
            average_next_option_val.reshape(-1, 1)))



            beta_loss = (beta_next[self.batch_iter, option_idx].reshape(-1, 1) * (
                advantage.detach())).mean()

            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            for param in self.beta_net.parameters():
                if torch.sum(torch.isnan(param.grad.data)) > 0:
                    import ipdb;
                    ipdb.set_trace()
            torch.nn.utils.clip_grad_norm_(self.beta_net.parameters(), 5)
            self.beta_optimizer.step()


            max_indx = option.flatten().long()
            for o in range(self.options_cnt):
                indx_o = (max_indx==o)
                if(torch.sum(indx_o)==0):
                    continue
                s_batch_o = state[indx_o, :]
                o_batch = option[indx_o,:]

                alphas_option = self.alphas[o]
                new_action, new_log_prob, new_epsilon, new_mean, new_log_std = self.get_action(
                    s_batch_o,o)

                predicted_q_value1, predicted_q_value2 = self.get_q_values(s_batch_o, new_action)

                predicted_q_value2 = predicted_q_value2.reshape(-1, 1)
                predicted_q_value1 = predicted_q_value1.reshape(-1, 1)
                predicted_new_q_value = torch.min(predicted_q_value1,
                                                  predicted_q_value2)
                self.actor_optimizer_list[o].zero_grad()
                policy_loss = (
                        alphas_option * new_log_prob - predicted_new_q_value).mean()

                # print("policy_loss",policy_loss)




                policy_loss.backward()



                for param in self.actor_net_list[o].parameters():
                    if torch.sum(torch.isnan(param.grad.data)) > 0:
                        import ipdb;
                        ipdb.set_trace()
                    #print("loss")
                    #print(torch.sum(param.data))
                    break

                self.actor_optimizer_list[o].step()
                if self.target_entropy is not None:
                    log_alphas_option = self.log_alphas[o]

                    alpha_loss = -(log_alphas_option.exp() * (new_log_prob + self.target_entropy).detach()).mean()

                    self.alpha_optimizers[o].zero_grad()

                    alpha_loss.backward()

                    self.alpha_optimizers[o].step()

                    self.alphas[o] = self.log_alphas[o].detach().exp()

                    torch.nn.utils.clip_grad_norm_(self.actor_net_list[o].parameters(), 5)






            # option_policy_loss = None


            # action_probabilities, __, ___, ____, _ = self.get_action(state[
            #                                                                          :self.mini_batch_size].unsqueeze(
            #     2).repeat(1, 1, self.options_cnt).transpose(1,2).reshape(self.mini_batch_size*self.options_cnt,-1),
            #                                                                          self.mini_batch_tensor)
            #
            # action_probabilities = action_probabilities.reshape(self.mini_batch_size, self.options_cnt, self.action_dim)
            # action_probabilities = action_probabilities.transpose(0, 2).transpose(1, 2)
            #
            # mutual_info_penalty= self.mi_penality * self.get_mutual_information_penalty(action_probabilities)
            # #mutual_info_penalty=None
            # mutual_info_penalty = torch.clamp(mutual_info_penalty,-5,5)
            #
            # if self.mi_penality > 0.0 and frame_idx< 50000:
            #     policy_loss = policy_loss + mutual_info_penalty
            # self.policy_optimizer.zero_grad()
            # if flag:
            #     print(mutual_info_penalty)
            #
            #



        #
        # if self.lv >0.0:
        #
        #     value_loss = value_loss  - self.lv * torch.var(option_prob_,-1).mean()
        #     if flag:
        #         value_loss = value_loss - self.le * torch.mean(torch.var(options_prob_episode, 0)) + self.lb * torch.mean(
        #     torch.pow(torch.mean(options_prob_episode, -1) - self.tau,2))
        #




            # value_loss = value_loss
            # - self.le * torch.mean(torch.var(options_prob_episode, 0))

            # - self.lv * torch.mean(
            #     torch.var(options_prob_episode, -1))

            # option_policy_loss = option_policy_loss + self.lb * torch.mean(
            #     torch.pow(torch.mean(options_prob_episode, -1) - self.tau))
            # print(self.lb * torch.mean(
            #     torch.pow(torch.mean(options_prob_episode, -1) - self.tau)))
            # print(- self.lv * torch.mean(
            #     torch.var(options_prob_episode, -1)))
            # print(- self.le * torch.mean(torch.var(options_prob_episode, 0)) )

        option_alpha_loss = None


        if frame_idx % 1 == 0:
            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )
        return policy_loss, beta_loss, None, None, q_value_loss2, q_value_loss1



    def test(self, max_episode_len=1000):
        env = self.test_env
        env.reset()
        frame_idx = 0
        run = self.trial
        options_used = ""
        total_rewards = []
        total_steps = 0
        episode_cnt = 0

        #self.running_state = ZFilter((self.state_dim,), clip=5)
        mean_activation = []

        while episode_cnt < 10:
            option_switches = 1
            avgduration = 0
            state = np.array(env.reset())
            #state =  self.running_state(state)
            state = tensor(state)
            episode_reward = 0
            options_probs = torch.zeros(self.options_cnt, max_episode_len)
            option, options_prob = softmax(self.get_option_vals(state.unsqueeze(0)))
            option = tensor(option[0])



            option = tensor(option)
            total_steps=0
            options_used = ""
            duration = 1
            option_switches = 1
            avgduration = 0.

            for step in range(self.max_steps):
                if episode_cnt==1 and self.options_cnt>=2:
                    option[0]=1
                if episode_cnt==2 and self.options_cnt>=3:
                    option[0]=2
                if episode_cnt==3 and self.options_cnt>=4:
                    option[0]=3
                if episode_cnt==0:
                    option[0]=0
                option_idx = int(option.cpu().numpy()[0])
                #if True:
                options_probs[:, step] = options_prob[0].exp()
                total_steps+=1
                options_used+=str(option_idx)

                action, log_prob, z, mean, log_std = self.get_action(state, option_idx)
                action = action[0].detach()




                next_state, reward, done, _ = env.step(action.cpu().numpy())
                #next_state = self.running_state(next_state)

                next_state = tensor(next_state)

                beta_next = self.get_beta_vals(next_state.unsqueeze(0))
                beta_next_np = beta_next.detach().cpu().numpy()[0]
                option_terminations = sample_sigmoid(beta_next_np[option_idx])
                if True:
                #if option_terminations == 1:
                    new_option, options_prob = softmax(self.get_option_vals(next_state.unsqueeze(0)))
                    new_option = tensor(new_option[0])
                    length = 0

                    if new_option != option:
                        option_switches += 1
                        avgduration += (1. / option_switches) * (duration - avgduration)
                        duration = 1
                    option = new_option

                state = next_state

                episode_reward += reward
                #if True:
                if done:
                    # print(options_probs)
                    if episode_cnt >= self.options_cnt:
                        total_rewards.append(episode_reward)
                        mean_activation.append(torch.mean(options_probs[:, :total_steps], -1))
                    print(episode_reward)
                    print(options_used)




                    break

            episode_cnt += 1
        #env.close()
        #del env
        return np.mean(total_rewards), options_used, torch.mean(torch.stack(mean_activation),0).detach().cpu().numpy().tolist()




    def train(self, run, max_episode_len=1000):


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
            #state =  self.running_state(state)
            state = tensor(state)
            episode_reward = 0
            options_probs = torch.zeros(self.options_cnt, max_episode_len)
            actions_obs = torch.zeros(self.action_dim, max_episode_len, self.options_cnt)

            option, options_prob = softmax(self.get_option_vals(state.unsqueeze(0)))
            option = tensor(option[0])

            duration = 1
            option_switches = 1
            avgduration = 0.
            new_option = option
            length=0
            done = False
            for step in range(self.max_steps):
                option_idx = int(option.cpu().numpy()[0])
                options_probs[:, step] = options_prob[0] # beta_next[0][option_idx]

                if frame_idx >=1e3:

                    action, log_prob, z, mean, log_std = self.get_action(state, option_idx)
                    action = action[0].detach()


                else:
                    action = self.env.action_space.sample()
                    action = tensor(action)

                next_state, reward, done, _ = self.env.step(action.cpu().numpy())


                #next_state = self.running_state(next_state)
                next_state = tensor(next_state)

                beta_next = self.get_beta_vals(next_state.unsqueeze(0))
                beta_next_np = beta_next.detach().cpu().numpy()[0]
                new_option=option
                noise = Normal(0, self.action_noise).sample(self.env.action_space.shape)
                p_noise = multivariate_normal.pdf(noise, np.zeros(shape=self.env.action_space.shape[0]),
                                                  self.action_noise * self.action_noise * torch.eye(noise.shape[0]))

                Q_predict = self.get_option_vals(state.unsqueeze(0))
                p = tensor(p_noise) * softmax(Q_predict.detach())[1][0][option.long()]

                option_terminations = sample_sigmoid(beta_next_np[option_idx])
                length+=1
                if True:
                #if option_terminations ==1:
                    new_option, options_prob = softmax(self.get_option_vals(next_state.unsqueeze(0)))
                    new_option = tensor(new_option[0])
                    length=0

                    if new_option != option:
                        option_switches += 1
                        avgduration += (1. / option_switches) * (duration - avgduration)
                        duration = 1
                option = new_option

                self.replay_buffer.push(state.detach().cpu().numpy(),
                                         action.cpu().numpy(),
                                        reward, next_state.cpu().numpy(), done,p)
                self.replay_buffer_onpolicy.push(state.detach().cpu().numpy(),
                                        action.cpu().numpy(),
                                        reward, next_state.cpu().numpy(), done,p)



                # env.render()

                state = next_state

                episode_reward += reward

                if len(self.replay_buffer) > self.batch_size:
                    if frame_idx % self.update_policy_iter == 0:
                        pass
                        # policy_loss, beta_loss, value_loss, option_alpha_loss, q_value_loss2, q_value_loss1 = self.update_actor(
                        #     self.batch_size, flag=done, frame_idx=frame_idx,
                        #     options_prob_episode=options_probs[:, :step],
                        #     soft_tau=self.soft_tau, actions_obs=None)
                        # log_buffer.append(
                        #     " policy_loss " + str(policy_loss) + " beta_loss " + str(beta_loss) + " value_loss " + str(
                        #         value_loss) +
                        #     " option_alpha_loss " + str(option_alpha_loss) + " q_value_loss " + str(
                        #         q_value_loss1) + " q_value_loss_2 " + str(q_value_loss2) + "\n")
                        # options_probs = options_probs * 0.0
                        # print(
                        #     " policy_loss " + str(policy_loss) + " beta_loss " + str(beta_loss) + " value_loss " + str(
                        #         value_loss) +
                        #     " option_alpha_loss " + str(option_alpha_loss) + " q_value_loss " + str(
                        #         q_value_loss1) + " q_value_loss_2 " + str(q_value_loss2) + "\n")




                frame_idx += 1
                if (frame_idx % self.evaluate_iter == 0):
                    test_rewards, options_used, test_option_probs = self.test()
                    #self.check_options()
                    print("Test rewards: " + str(test_rewards) + " " + str(test_option_probs))
                    print(options_used)
                    self.save_weights()
                    log_buffer.append(
                        "Test rewards for frame_idx " + str(frame_idx) + " : " + str(test_rewards) + " " + str(
                            options_used) + " " + str(test_option_probs) + "\n")



                if(frame_idx% self.update_option_iter==0):
                    policy_loss, beta_loss, value_loss, option_alpha_loss, q_value_loss2, q_value_loss1 = self.update_actor(
                        self.batch_size, flag=done, frame_idx=frame_idx,
                        options_prob_episode=options_probs[:, :step],
                        soft_tau=self.soft_tau, actions_obs=None,num_updates=self.num_updates)
                    self.update_option(self.num_updates,self.batch_size)
                    self.replay_buffer_onpolicy.clear()
                    #import ipdb; ipdb.set_trace()

                if done:
                    # self.plot_temporal_activations(options_probs,step)
                    break
                # if frame_idx> 5e5:
                #     self.mi_penality=0.0


            episode_cnt += 1
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
        #self.log.close()
        self.env.close()
        return episode_cnt




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--value_lr', help='Discount factor', type=float, default=3e-4)
    parser.add_argument('--soft_q_lr', help='Discount factor', type=float, default=3e-4)
    parser.add_argument('--policy_lr', help='Discount factor', type=float, default=3e-4)  # 3e-5
    parser.add_argument('--beta_lr', help='Discount factor', type=float, default=3e-4)
    parser.add_argument('--option_policy_lr', help='Discount factor', type=float, default=1e-2)#* #3e-4
    parser.add_argument('--option_alpha_lr', help='Discount factor', type=float, default=1e-2)
    parser.add_argument('--alpha_lr', help='Discount factor', type=float, default=1e-2)
    parser.add_argument('--env_name', help='Discount factor', type=str, default="HopperBulletEnv-v0")
    parser.add_argument('--log_dir', help='Log directory', type=str, default="log_dir")
    parser.add_argument('--model_dir', help='Model directory', type=str, default="model/")
    parser.add_argument('--plot_dir', help='Model directory', type=str, default="plots/")
    parser.add_argument('--evaluate_iter', help='num of episodes before evaluation', type=int, default=250)
    parser.add_argument('--update_policy_iter', help='num of episodes before evaluation', type=int, default=250)

    parser.add_argument('--update_option_iter', help='num of episodes before evaluation', type=int, default=2000)
    parser.add_argument('--num_updates', help='num of episodes before evaluation', type=int, default=250)
    parser.add_argument('--action-noise', help='parameter of the noise for exploration', default=0.2)





    parser.add_argument('--hidden_dim', help='Hidden dimension', type=int, default=256)
    parser.add_argument('--options_cnt', help='Option count', type=int, default=4)
    parser.add_argument('--runs', help='Runs', type=int, default=5)
    parser.add_argument('--separate_q_values', help='separate_q_values', type=int, default=0)
    parser.add_argument('--separate_policies', help='separate_policies', type=int, default=1)
    parser.add_argument('--sample_option_type', help='sample_option_type', type=int, default=2)
    parser.add_argument('--temp', help='temp', type=float, default=1.0)
    parser.add_argument('--replay_buffer_size', help='Replay buffer size', type=float, default=1000000)
    parser.add_argument('--max_frames', help='Maximum no of frames', type=int, default=1500000)
    parser.add_argument('--max_steps', help='Maximum no of steps', type=int, default=1500000)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=256)
    parser.add_argument('--discriminator_lr', help='Discriminator lr', type=float, default=0.001)
    parser.add_argument('--discriminator_init', help='Discriminator init', type=float, default=0.00)
    parser.add_argument('--update_discriminator', help='Update discriminator after this many episodes', type=int,
                        default=1000)
    parser.add_argument('--decay', help='Decay', type=float, default=0.0)
    parser.add_argument('--target_entropy', help='target_entropy', type=int, default=-1)
    parser.add_argument('--target_entropy_', help='target_entropy_', type=int, default=0)
    parser.add_argument('--trial', help='trial', type=int, default=0)
    parser.add_argument('--soft_tau', help='soft_tau', type=float, default=0.01)
    parser.add_argument('--tau', help='tau', type=float, default=0.5)
    parser.add_argument('--le', help='le', type=float, default=1)
    parser.add_argument('--lb', help='lb', type=float, default=1.0) #0.01 #1 #10 in terminal
    parser.add_argument('--bl', help='bl', type=float, default=0.000) #0.01 #1 #10 in terminal

    parser.add_argument('--lv', help='lv', type=float, default=1)
    parser.add_argument('--mi_penalty', help='mi_penalty', type=float, default=1)  # 10

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

        #args.env_name="HopperBulletEnv-v0"
        sac = SoftActorCritic(args)
        sac.reset()
        #sac.save_weights()
        #sac.load_weights()
        #sac.load_weights("HopperBulletEnv-v0_2019-08-18-21-52-57")
        #sac.test()
        sac.train(1)
        sac.log.close()

        sac.test()
        del sac



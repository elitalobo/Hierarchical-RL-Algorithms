import gym
try:
    import roboschool
except:
    pass
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch.multiprocessing

from datetime import datetime

from utils import *
from models import *

from torch.distributions import Normal
import torch

def weighted_entropy(prob, w_norm):
    return torch.sum(w_norm*prob*torch.log(prob+1e-8))

def weighted_mean(prob,w_norm):
    p_weighted = w_norm*prob
    return torch.mean(p_weighted,0)


def weighted_mse(q_target,q_pred,w_norm):
    error = torch.mean(w_norm*torch.pow(q_target-q_pred,2))
    return error

def softmax(values, temp=0.3):
    values = values / temp
    values = values - torch.max(values,dim=-1)[0].reshape(-1,1)
    values_exp = torch.exp(values)

    probs = values - torch.log(torch.sum(values_exp,dim=-1).reshape(-1,1))
    probs = torch.exp(probs)
    return probs

def weighted_mean_array(x,weights):
    weighted_mean = torch.mean(weights,-1)
    x_weighted = x * weights
    mean_weighted = torch.mean(x_weighted,-1)/weighted_mean
    return mean_weighted.reshape(-1,1)

def p_sample(probs,temp=0.3):

    selected = torch.multinomial(probs, num_samples=1)
    return selected

def entropy(p):
    return torch.sum(p* torch.log(p+1e-8))

def add_normal(x_input, outshape, at_eps):
    epsilon = torch.distributions.Normal(0,1).sample(outshape)
    x_out = x_input + at_eps * (epsilon* torch.abs(x_input))

    return x_out


def kl(p,q):
    res= torch.sum(p* torch.log(p+1e-8)/(q+1e-8))
    return res



class adInfoHRLTD3(object):
    def __init__(self,args, env ,state_dim, action_dim, action_bound, batch_size=64,tau=0.001,option_num=5,actor_lr=1e-4,critic_lr=1e-3,option_lr=1e-3,gamma=0.99, hidden_dim=(400,300),entropy_coeff = 0.1,c_reg=1.0, vat_noise=0.005,c_ent=4):

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = tensor(action_bound).unsqueeze(0)
        self.batch_size = batch_size

        self.soft_tau = tau
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.option_num = option_num
        self.entropy_coeff = entropy_coeff
        self.c_reg = c_reg
        self.vat_noise = vat_noise
        self.c_ent = c_ent
        self.option_lr = option_lr
        self.reset(args)

    def reset(self,args):
        self.env_name = args["env_name"]
        self.timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.filename = args["log_dir"] + "/" + args["env_name"] + "_" + self.timestamp + str(
            args["options_cnt"])
        self.soft_tau = args["tau"]
        self.model_dir = args["model_dir"]
        self.plot_dir = args["plot_dir"]
        self.vat_noise = args["vat_noise"]


        self.log = open(self.filename, 'w+')
        self.log.write(self.filename + "\n")
        self.log.write(str(args)+"\n")
        self.trial = args["trial_num"]
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]

        self.num_actions = 2
        self.options_cnt = args["options_cnt"]
        self.temp = 1.0

        self.replay_buffer_size = 1000000
        self.total_steps = 0.0
        self.max_frames = args["max_frames"]
        self.max_steps = args["max_steps"]
        self.frame_idx = 0
        self.rewards = []

        self.gamma = args["gamma"]
        self.entropy_coeff = args["entropy_coeff"]
        self.c_reg = args["c_reg"]
        self.c_ent = args["c_ent"]
        self.test_num = args["test_num"]
        self.max_episode_len = args["max_episode_len"]




        self.runs = args["runs"]

        self.critic_lr = args["critic_lr"]  # 3e-5
        self.actor_lr = args["actor_lr"]  # 3e-5
        self.option_lr = args["option_lr"]  # 3e-4
        self.beta_lr = args["beta_lr"]



        self.option_net = OptionNetwork(self.state_dim, self.action_dim,self.options_cnt, self.hidden_dim,self.vat_noise).to(device)

        self.beta_net = BetaNetwork(self.state_dim,self.options_cnt).to(device)


        self.critic_net = CriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)

        self.actor_net_list = []
        self.target_actor_net_list = []
        for option_idx in range(self.options_cnt):

            self.actor_net_list.append(StochasticActorNetwork(self.state_dim, self.action_dim,self.hidden_dim).to(
                device))
            self.target_actor_net_list.append(StochasticActorNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(
            device))


        for option_idx in range(self.options_cnt):

            for target_param, param in zip(self.target_actor_net_list[option_idx].parameters(), self.actor_net_list[option_idx].parameters()):
                target_param.data.copy_(param.data)


        self.target_critic_net = CriticNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)

        self.critic_criterion = nn.MSELoss()
        self.actor_criterion = nn.MSELoss()
        self.option_criterion = nn.L1Loss()



        self.actor_optimizer_list = []

        for option_idx in range(self.options_cnt):

            self.actor_optimizer_list.append(optim.Adam(self.actor_net_list[option_idx].parameters(), lr=self.actor_lr))
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)
        self.option_optimizer = optim.Adam(self.option_net.parameters(), lr=self.option_lr)
        self.beta_optimizer = optim.Adam(self.beta_net.parameters(),lr=self.beta_lr)


    def update_targets(self):
        #print("updating target")
        for option_idx in range(self.options_cnt):
            for target_param, param in zip(self.target_actor_net_list[option_idx].parameters(), self.actor_net_list[option_idx].parameters()):
                target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(
                    target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
        )


    def train_critic(self, state, action, target_q_value,predicted_v_value,sampling_prob):
        #print("train critic")
        #print("updating critic")
        critic_out_Q1, critic_out_Q2 = self.critic_net(state, action)

        critic_loss = self.critic_criterion(target_q_value, critic_out_Q1) \
                           + self.critic_criterion(target_q_value, critic_out_Q2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 10)
        self.critic_optimizer.step()
        # for param in self.critic_net.parameters():
        #     print(torch.sum(param.data))
        # print("updated critic")
        # print("critic weights")


    def train_beta(self,beta_next,next_q_value,next_v_value):
        self.beta_optimizer.zero_grad()
        beta_loss = torch.mean(beta_next *
           (next_q_value-next_v_value).detach())
        assert((beta_next *
           (next_q_value-next_v_value).detach()).shape[1]==1)
        beta_loss.backward()
        # for param in self.beta_net.parameters():
        #     print(torch.sum(param.data))

        self.beta_optimizer.step()


    def train_option(self,input,action,target_q_value,predicted_v_value,sampling_prob):


        enc_output, option_out, output_option_noise, dec_output, option_input_concat = self.option_net(input,action)

        Advantage = (target_q_value - predicted_v_value).detach()
        Weight = torch.exp(Advantage - torch.max(Advantage))/sampling_prob.reshape(-1,1)
        W_norm = Weight / torch.mean(Weight)

        critic_conditional_entropy = weighted_entropy(option_out,W_norm.detach())
        p_weighted_ave = weighted_mean(option_out, W_norm.detach())
        critic_entropy = critic_conditional_entropy - self.c_ent * entropy(p_weighted_ave)


        vat_loss = kl(option_out, output_option_noise)
        reg_loss = self.option_criterion(option_input_concat, dec_output)
        option_loss = reg_loss + self.entropy_coeff * (critic_entropy) + self.c_reg * vat_loss

        #print(option_loss)
        self.option_optimizer.zero_grad()
        option_loss.backward()

        #torch.nn.utils.clip_grad_norm_(self.option_net.parameters(), 10)
        self.option_optimizer.step()



    def train_actor_option(self,option_q_value,option):
        # print("Train actor")
        # print("updating actor")
        # for param in self.actor_net_list[option].parameters():
        #     print(torch.sum(param.data))
        #     break
        option_loss = -option_q_value
        option_loss = torch.mean(option_loss)

        self.actor_optimizer_list[option].zero_grad()
        option_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_net_list[option].parameters(), 10)
        self.actor_optimizer_list[option].step()

        # for param in self.actor_net_list[option].parameters():
        #     print(torch.sum(param.data))
        #     break
        # print("updated actor")


    def predict_actor_option(self,inputs,option):
        action, log_prob, z, mean, log_std =  self.actor_net_list[option](inputs)
        return action, log_prob

    def predict_actor_option_target(self,inputs,option):
        action, log_prob, z, mean, log_std = self.target_actor_net_list[option](inputs)
        return action, log_prob



    def predict_actor(self,inputs,options,target=False):
        action_list = []
        prob_list = []
        for option_idx in range(self.options_cnt):
            action_o = None
            if target==False:
                action_o, log_prob = self.predict_actor_option(inputs,option_idx)
            else:
                action_o, log_prob = self.predict_actor_option_target(inputs,option_idx)
            action_list.append(action_o)
            prob_list.append(log_prob)
        n = inputs.shape[0]
        action = 0

        if n==1 or isinstance(options,int):

            action = action_list[options]
            prob = prob_list[options]
        else:

            selected_actions=[]
            selected_options=[]

            for idx in range(n):
                    #import ipdb; ipdb.set_trace()
                    selected_actions.append(action_list[int(options[idx])][idx,:])
                    selected_options.append(prob_list[int(options[idx])][idx, :])

            action = torch.stack(selected_actions)
            prob = torch.stack(selected_options)
            #action = torch.stack(action_list,0).transpose(0,1)[np.arange(n),options.flatten(),:]
        return action, prob






    def predict_critic(self,inputs,actions):
        return self.critic_net(inputs,actions)

    def predict_critic_target(self,inputs,actions):
        q1,q2= self.target_critic_net(inputs,actions)
        return q1, q2

    def predict_option(self,inputs,actions):
        _,option_out,__,___,____ = self.option_net(inputs,actions)
        return option_out

    def value_func(self,inputs):
        Q_predict = []
        n = inputs.shape[0]

        for o in range(self.options_cnt):
            import time
            start = time.time()
            action_i, log_prob_i = self.predict_actor_option_target(inputs,o)



            Q_predict_1, Q_predict_2 = self.predict_critic_target(inputs,action_i)
            end = time.time()
            #print("time value func")
            #print(end - start)

            Q_predict_i = torch.min(Q_predict_1,Q_predict_2) - 0.01 *log_prob_i
            Q_predict.append(Q_predict_i.reshape(-1,1))

        Q_predict = torch.stack(Q_predict)
        Q_predict = Q_predict.squeeze(dim=-1).t()
        po = softmax(Q_predict)
        state_values = weighted_mean_array(Q_predict,po)
        return state_values



    def softmax_option_target(self,inputs):
        Q_predict = []
        n = inputs.shape[0]
        for o in range(self.options_cnt):
            action_i, log_prob_i = self.predict_actor(inputs,o,True)


            Q_predict_i, _ = self.predict_critic_target(inputs,action_i)

            Q_predict.append(Q_predict_i)


        Q_predict = torch.stack(Q_predict)
        Q_predict = Q_predict.squeeze(dim=-1).t()
        p = softmax(Q_predict)
        o_softmax = p_sample(p)
        n = Q_predict.shape[0]

        Q_softmax = Q_predict[np.arange(n),o_softmax.flatten()]
        return o_softmax, Q_softmax.reshape(n,1), Q_predict




    def max_option(self,inputs):
        Q_predict = []
        n = inputs.shape[0]
        for o in range(self.options_cnt):
            action_i, log_prob_i = self.predict_actor(inputs,o,True)
            Q_predict_i, _ = self.predict_critic_target(inputs,action_i)

            Q_predict.append(Q_predict_i.reshape(-1,1))
        Q_predict = torch.stack(Q_predict,-1)

        o_max = torch.argmax(Q_predict,-1)
        Q_max = torch.max(Q_predict,-1)

        return o_max, Q_max, Q_predict

    def _add_normal(self,args):
        xx = args
        return add_normal(xx,xx.shape,self.vat_noise)

    def save_weights(self):
        for option_idx in range(self.options_cnt):
            torch.save(self.actor_net_list[option_idx].state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-actor_net_" + str(option_idx))
            torch.save(self.target_actor_net_list[option_idx].state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-target_actor_net_" + str(option_idx))



        torch.save(self.critic_net.state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-critic_net")
        torch.save(self.target_critic_net.state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-target_critic_net")

        torch.save(self.option_net.state_dict(), self.model_dir + self.env_name + "_" + self.timestamp + "-option_net")


    def encode(self, options):
        return self.embeddings[options.flatten().long()]

    def decode(self, options):
        return torch.argmax(options, -1).reshape(-1,1)

    def load_weights(self, prefix=None):
        if prefix==None:
            prefix = self.env_name + "_" + self.timestamp
        for option_idx in range(self.options_cnt):

            self.actor_net_list[option_idx].load_state_dict(torch.load(self.model_dir + prefix + "-actor_net_" +str(option_idx)))
            self.target_actor_net_list[option_idx].load_state_dict(torch.load(self.model_dir + prefix + "-target_actor_net_" +str(option_idx)))

        self.critic_net.load_state_dict(torch.load(self.model_dir  + prefix + "-critic_net"))
        self.target_critic_net.load_state_dict(torch.load(self.model_dir  + prefix + "-target_critic_net"))

        self.option_net.load_state_dict(torch.load(self.model_dir + prefix + "-option_net"))




import torch
import gym
import argparse

from replay_buffer import  *
from adInfoHRL_agent import *
from scipy.stats import multivariate_normal
#import roboschool
import argparse
import pprint as pp
import pybullet_envs
from utils import *
from gym import wrappers

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x.to(device).float()
    x = torch.tensor(x, device=device)
    return x.to(device).float()


def update_option(env, args, agent, replay_buffer_onpolicy, action_noise, update_num):
    for ite in range(update_num):
        state, action, reward, next_state, done, p_batch = replay_buffer_onpolicy.sample(
        args["option_minibatch_size"])


        state = tensor(state).to(device)
        action = tensor(action).to(device)
        next_state = tensor(next_state).to(device)

        reward = tensor(reward).unsqueeze(1).to(device)
        done = tensor(np.float32(done)).unsqueeze(1).to(device)
        p_batch = tensor(p_batch).unsqueeze(1).to(device)


        import time


        start = time.time()
        next_option , _ , Q_predict = agent.softmax_option_target(next_state)


        noise_clip = 0.5
        noise_clip = 0.5
        noise = torch.clamp(Normal(0,action_noise).sample((next_option.shape[0] , agent.action_dim)) + action_noise,-noise_clip,noise_clip)
        next_action, log_prob = agent.predict_actor(next_state,next_option,target=True)
        next_action = next_action + noise

        next_action = next_action.detach()


        next_action = torch.max(torch.min(next_action, agent.action_bound), -agent.action_bound)

        target_Q1, target_Q2 = agent.predict_critic_target(next_state, next_action)

        target_q = torch.min(target_Q1, target_Q2) - 0.01* log_prob
        done = done.reshape(-1,1)

        y_i = reward + agent.gamma * (1-done) * target_q.reshape(-1,1)
        predicted_v_i = agent.value_func(state)

        assert(y_i.shape[0]==state.shape[0] and y_i.shape[1]==1)
        assert (predicted_v_i.shape[0] == state.shape[0] and predicted_v_i.shape[1] == 1)



        for option_idx in range(args["option_ite"]):

            agent.train_option(state,action,y_i,predicted_v_i,p_batch)

        end = time.time()
        #print("softmax option target ")
        #print(end - start)








def update_policy(env, args, agent, replay_buffer, action_noise, update_num):
    print("update num")
    print(update_num)

    for ite in range(update_num):
        state, action, reward, next_state, done, p_batch = replay_buffer.sample(
            args["minibatch_size"])

        state = tensor(state).to(device)
        action = tensor(action).to(device)
        next_state = tensor(next_state).to(device)

        reward = tensor(reward).unsqueeze(1).to(device)
        done = tensor(np.float32(done)).unsqueeze(1).to(device)
        p_batch = tensor(p_batch).unsqueeze(1).to(device)

        next_option, _, Q_predict = agent.softmax_option_target(next_state)

        cur_option = torch.argmax(agent.predict_option(state,action),-1).detach()
        beta_vals = agent.beta_net(next_state)
        beta_vals = beta_vals[np.arange(next_state.shape[0]), cur_option]
        beta_new = beta_vals.reshape(-1,1)


        #beta_next = beta_next.reshape(-1,1)
        cur_option_next_action, cur_option_next_log_prob = agent.predict_actor(next_state, cur_option)
        cur_option_target_Q1, cur_option_target_Q2 = agent.predict_critic_target(next_state, cur_option_next_action)

        cur_option_target_q = torch.min(cur_option_target_Q1,cur_option_target_Q2).detach() - 0.01* cur_option_next_log_prob.reshape(-1,1)


        noise_clip = 0.5
        noise = torch.clamp(Normal(0, action_noise).sample((action.shape[0], agent.action_dim)), -noise_clip,
                            noise_clip)
        next_option_action, next_log_prob = agent.predict_actor(next_state, next_option)
        next_option_action = next_option_action + noise

        next_option_action = torch.max(torch.min(next_option_action, agent.action_bound), -agent.action_bound)

        target_Q1, target_Q2 = agent.predict_critic_target(next_state, next_option_action)


        target_q = (torch.min(target_Q1, target_Q2).detach() -0.01* next_log_prob).reshape(-1, 1)


        #y_i = reward + agent.gamma * (1-done)*((1-beta_new.detach())*cur_option_target_q + beta_new.detach()*target_q)
        y_i = reward + agent.gamma * (1-done)*(target_q)

        assert(y_i.shape[1]==1)

        predicted_v_i = agent.value_func(state)
        predicted_next_v_value = agent.value_func(next_state)


        agent.train_critic(state, action,y_i,
                           predicted_v_i,
                           p_batch)



        agent.train_beta(beta_new,cur_option_target_q,predicted_next_v_value)


        if ite % int(args["policy_freq"]) == 0:
            state, action, reward, next_state, done, p_batch = replay_buffer.sample(
                args["policy_minibatch_size"])
            state = tensor(state).to(device)
            action = tensor(action).to(device)
            next_state = tensor(next_state).to(device)

            reward = tensor(reward).unsqueeze(1).to(device)
            done = tensor(np.float32(done)).unsqueeze(1).to(device)
            p_batch = tensor(p_batch).unsqueeze(1).to(device)

            option_estimated = agent.predict_option(state, action)
            option_estimated = option_estimated.reshape(args["policy_minibatch_size"],agent.options_cnt)
            max_indx = torch.argmax(option_estimated,-1)

            for o in range(agent.options_cnt):
                indx_o = (max_indx==o)
                s_batch_o = state[indx_o, :]
                # print('s_batch_o.shape', s_batch_o.shape)
                a_outs, log_prob = agent.predict_actor_option(s_batch_o, o)

                #grads = agent.action_gradients(s_batch_o, a_outs)
                if a_outs.shape[0]!=0:
                    critic_out_Q1, critic_out_Q2 = agent.critic_net(s_batch_o,a_outs)
                    agent.train_actor_option(critic_out_Q1 - 0.01*log_prob, o)

            agent.update_targets()


def evaluate_deterministic_policy(agent,args,return_test,test_iter):
    print(agent.env_name)
    env_test = gym.make(agent.env_name)
    for nn in range(int(agent.test_num)):

        state_test = env_test.reset()
        state_test = tensor(state_test).unsqueeze(0)
        return_epi_test = 0
        option_test = None
        terminate=0
        switches =0
        prev_option = option_test
        for t_test in range(int(agent.max_episode_len)):

            if True or terminate == 1 or option_test is None:

                if option_test is not None and  prev_option is not None and prev_option != option_test:
                    switches+=1

                prev_option = option_test
                option_test, _, Q_predict = agent.softmax_option_target(state_test)


            action_test, log_prob_test = agent.predict_actor_option(state_test, option_test[0])



            action_test = torch.max(torch.min(action_test, tensor(env_test.action_space.high)), tensor(env_test.action_space.low))
            state_test2, reward_test, terminal_test, info_test = env_test.step(action_test[0].detach().cpu().numpy())
            state_test2 = tensor(state_test2).unsqueeze(0)



            state_test = state_test2
            terminate_prob = agent.beta_net(state_test).detach().cpu().numpy()
            terminate = sample_sigmoid(terminate_prob[0][option_test])
            return_epi_test = return_epi_test + reward_test


            if terminal_test:
                break


        print('test_iter:{:d}, nn:{:d}, return_epi_test: {:d}'.format(int(test_iter), int(nn), int(return_epi_test)))
        print("No of switches " + str(switches))
        return_test[test_iter] = return_test[test_iter] + return_epi_test / float(agent.test_num)



def train(args, agent):

    episode_R = []

    agent.update_targets()

    replay_buffer = ReplayBufferWeighted(args["buffer_size"])
    replay_buffer_onpolicy = ReplayBufferWeighted(args["buffer_size"])

    return_test = torch.zeros(int(np.ceil(args["total_step_num"] / args["sample_step_num"])+1000))


    result_name = 'adInfoHRLTD3VAT_' + args["env_name"] \
                + '_lambda_' + str(args["entropy_coeff"]) \
                + '_c_reg_' + str(args["c_reg"]) \
                + '_vat_noise_' + str(args["vat_noise"]) \
                + '_c_ent_' + str(args["c_ent"]) \
                + '_option_' + str(args["option_num"]) \
                + '_temporal_' + str(args["temporal_num"]) \
                + '_trial_idx_' + str(args["trial_idx"])

    action_noise = float(args['action_noise'])

    total_step_cnt = 0
    test_iter = 0
    epi_cnt = 0
    trained_times_steps = 0
    save_cnt = 1
    policy_ite=0
    option_ite = 0
    env = gym.make(agent.env_name)

    while total_step_cnt in range(args["total_step_num"]):

        state = env.reset()
        #state =tensor(state).unsqueeze(0)
        state = tensor(state)

        ep_reward = 0
        ep_ave_max_q = 0
        T_end = False

        terminate = 0
        option = None
        for j in range(args["max_episode_len"]):


            if args["render_env"]:
                    env.render()

            if total_step_cnt < 1e4:
                action = env.action_space.sample()
                action = action.reshape(1,-1)
                p = 1
            else:
                #if j % args["temporal_num"] == 0 or not np.isscalar(option):

                if True or terminate==1 or option is None:

                    option, _, Q_predict = agent.softmax_option_target(state.unsqueeze(0))
                    option = option[0,0]
                    #print(option)



                action, log_prob = agent.predict_actor_option(state.unsqueeze(0), option)

                noise = Normal(0, action_noise).sample(env.action_space.shape)
                p_noise = multivariate_normal.pdf(noise, np.zeros(shape=env.action_space.shape[0]), action_noise*action_noise*torch.eye(noise.shape[0]))
                action = torch.max(torch.min(action, tensor(env.action_space.high)), tensor(env.action_space.low))

                p = tensor(p_noise) * softmax(Q_predict.detach())[1][0][option]

                action = action.detach().cpu().numpy()
            state2, reward, terminal, info = env.step(action[0])
            #state2 = tensor(state2).unsqueeze(0)
            state2 = tensor(state2)

            replay_buffer.push(state, action.squeeze(0), reward,
                            state2, terminal, p)

            replay_buffer_onpolicy.push(state, action.squeeze(0), reward,
                              state2, terminal, p)

            if j == int(args['max_episode_len']) - 1:
                T_end = True

            state = state2
            if option is not None:
                terminate_prob = agent.beta_net(state.unsqueeze(0)).detach().cpu().numpy()
                terminate = sample_sigmoid(terminate_prob[0][option])

            ep_reward += reward

            total_step_cnt += 1

            if total_step_cnt >= test_iter * int(args['sample_step_num']) or total_step_cnt == 1:
                print('total_step_cnt', total_step_cnt)
                print('evaluating the deterministic policy...')
                evaluate_deterministic_policy(agent, args, return_test, test_iter)

                print('return_test[{:d}] {:d}'.format(int(test_iter), int(return_test[test_iter])))
                test_iter += 1

            if total_step_cnt >= int(args['save_model_num']) * save_cnt:
                model_path = "./Model/adInfoHRL/" + args['env'] + '/'
                try:
                    import pathlib
                    pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
                except:
                    print("A model directory does not exist and cannot be created. The policy models are not saved")


                #agent.save_model(iteration=test_iter, expname=result_name, model_path=model_path)
                agent.model_dir = model_path
                agent.save_weights()
                print('Models saved.')
                save_cnt += 1


            if terminal or T_end:
                epi_cnt += 1
                print('| Reward: {:d} | Episode: {:d} | Total step num: {:d} |'.format(int(ep_reward), epi_cnt, total_step_cnt ))
                # episode_R.append(ep_reward)
                break

        if total_step_cnt != args["total_step_num"] and total_step_cnt > 1e3 \
                and total_step_cnt >= option_ite * args["option_batch_size"]:
            update_num = args["option_update_num"]
            print('update option', update_num)
            update_option(env, args, agent, replay_buffer_onpolicy, action_noise, update_num)
            option_ite = option_ite + 1
            replay_buffer_onpolicy.clear()

        if total_step_cnt != int(args['total_step_num']) and total_step_cnt > 1e3:
            update_num = total_step_cnt - trained_times_steps
            trained_times_steps = total_step_cnt
            print('update_num', update_num)
            update_policy(env, args, agent, replay_buffer, action_noise, update_num)


    return return_test


def main(args):
    for trial in range(args["trial_num"]):
        print('Trial Number:', trial)



        if args["change_seed"]:
            rand_seed = 10 * trial
        else:
            rand_seed = 0
        env = gym.make(args["env_name"])
        env.seed(args["random_seed"] + int(rand_seed))


        np.random.seed(int(args['random_seed']) + int(rand_seed))
        env.seed(int(args['random_seed']) + int(rand_seed))

        env_test = gym.make(args['env'])
        env_test.seed(int(args['random_seed']) + int(rand_seed))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print('action_space.shape', env.action_space.shape)
        print('observation_space.shape', env.observation_space.shape)
        action_bound = tensor(env.action_space.high)

        assert (env.action_space.high[0] == -env.action_space.low[0])

        agent = adInfoHRLTD3(args,env,state_dim,action_dim,action_bound,tau=float(args['tau']),
                         actor_lr=float(args['actor_lr']),
                         critic_lr=float(args['critic_lr']),
                         option_lr=float(args['option_lr']),
                         gamma=float(args['gamma']),
                         hidden_dim=np.asarray(args['hidden_dim']),
                         entropy_coeff=float(args['entropy_coeff']),
                         c_reg=float(args['c_reg']),
                         option_num=int(args['option_num']),
                         vat_noise= float(args['vat_noise']),
                         c_ent=float(args['c_ent']))

        if args["use_gym_monitor"]:
            if not args["render_env"]:
                env = wrappers.Monitor(
                        env, args["monitor_dir"], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args["monitor_dir"], video_callable=lambda episode_id: episode_id%50==0, force=True)

        step_R_i = train(args, agent)

        result_path = "./results/trials/separate/"
        try:
            import pathlib
            pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)
        except:
            print("A result directory does not exist and cannot be created. The trial results are not saved")

        result_filename = args['result_file'] + '_' + args['env'] \
                              + '_lambda_' + str(float(args['lambda'])) \
                              + '_c_reg_' + str(float(args['c_reg'])) \
                              + '_vat_noise_' + str(float(args['vat_noise'])) \
                              + '_c_ent_' + str(float(args['c_ent'])) \
                              + '_option_' + str(float(args['option_num'])) \
                              + '_temporal_' + str(float(args['temporal_num'])) \
                              + '_trial_idx_' + str(int(args['trial_idx'])) \
                              + '.txt'

        if args.overwrite_result and trial == 0:
            np.savetxt(result_filename, np.asarray(step_R_i))
        else:
            data = np.loadtxt(result_filename, dtype=float)
            data_new = np.vstack((data, np.asarray(step_R_i)))
            np.savetxt( result_filename, data_new)

        if args['use_gym_monitor']:
            env.monitor.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for AdInfoHRLTD3 agent')

    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--option-lr', help='option network learning rate', default=0.001)
    parser.add_argument('--beta-lr', help='beta network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.005)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--hidden-dim', help='number of units in the hidden layers', default=(400, 300))
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=100)
    parser.add_argument('--policy-minibatch-size', help='batch for updating policy', default=400)

    parser.add_argument('--option-batch-size', help='batch size for updating option', default=5000)
    parser.add_argument('--option-update-num', help='iteration for updating option', default=4000)
    parser.add_argument('--option-minibatch-size', help='size of minibatch for minibatch-SGD', default=50)

    parser.add_argument('--option-ite', help='batch size for updating policy', default=1)

    parser.add_argument('--total-step-num', help='total number of time steps', default=1000000)
    parser.add_argument('--sample-step-num', help='number of time steps for recording the return', default=5000)
    parser.add_argument('--test-num', help='number of episode for recording the return', default=10)
    parser.add_argument('--action-noise', help='parameter of the noise for exploration', default=0.2)
    parser.add_argument('--policy-freq', help='frequency of updating the policy', default=2)

    parser.add_argument('--temporal-num', help='frequency of the gating policy selection', default=3)
    parser.add_argument('--hard-sample-assignment', help='False means soft assignment', default=True)
    parser.add_argument('--option-num', help='number of options', default=4)

    parser.add_argument('--entropy_coeff', help='cofficient for the mutual information term', default=0.1)
    parser.add_argument('--c-reg', help='cofficient for regularization term', default=1.0)
    parser.add_argument('--c-ent', help='cofficient for regularization term', default=4.0)
    parser.add_argument('--vat-noise', help='noise for vat in clustering', default=0.04)


    # run parameters
    # parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default="HopperBulletEnv-v0")
    parser.add_argument('--env-id', type=int, default=6, help='choose the gym env- tested on {Pendulum-v0}')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=1001)  # 50000
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_adInfoHRL')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info',
                        default='./results/tf_adInfoHRL')
    parser.add_argument('--result-file', help='file name for storing results from multiple trials',
                        default='./results/trials/trials_AdInfoHRLAlt')
    parser.add_argument('--overwrite-result', help='flag for overwriting the trial file', default=True)
    parser.add_argument('--trial-num', help='number of trials', default=1)
    parser.add_argument('--trial-idx', help='index of trials', default=0)
    parser.add_argument('--change-seed', help='change the random seed to obtain different results', default=False)
    parser.add_argument('--save_model-num', help='number of time steps for saving the network models', default=500000)


    parser.add_argument('--hidden-dim_0', help='number of units in the hidden layers', type=int,default=400)
    parser.add_argument('--hidden-dim_1', help='number of units in the hidden layers', type=int,default=300)




    parser.add_argument('--max_frames', help='Maximum no of frames', type=int, default=1500000)
    parser.add_argument('--max_steps', help='Maximum no of steps', type=int, default=1500000)



    parser.add_argument('--trial_num', help='Trial num', type=int,default=1)

    parser.add_argument('--log_dir', help='Log directory', type=str, default="log_dir")
    parser.add_argument('--model_dir', help='Model directory', type=str, default="model/")
    parser.add_argument('--plot_dir', help='Model directory', type=str, default="plots/")

    parser.add_argument('--options_cnt', help='Option count', type=int, default=4)
    parser.add_argument('--runs', help='Runs', type=int, default=5)







    parser.add_argument('--env_name', help='choose the gym env- tested on {Pendulum-v0}',type=str,default="HopperBulletEnv-v0")

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)

    parser.set_defaults(change_seed=True)
    parser.set_defaults(overwrite_result=False)

    args_tmp = parser.parse_args()


    args = vars(args_tmp)

    pp.pprint(args)

    main(args)



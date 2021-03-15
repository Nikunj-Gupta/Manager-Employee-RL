import argparse, pprint
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from maddpg.utils.buffer import ReplayBuffer
from maddpg.algorithms.maddpg import MADDPG
from pettingzoo.mpe import simple_spread_v2


def run(config):
    torch.manual_seed(config.seed)

    np.random.seed(config.seed)
    env = simple_spread_v2.parallel_env(local_ratio=0.5)
    env.seed(config.seed)
    env.reset() 
    logger = SummaryWriter(str(config.log_dir)+config.expname) 

    agent_alg=config.agent_alg 
    adversary_alg=config.adversary_alg 
    tau=config.tau 
    gamma=0.95 
    lr=config.lr 
    hidden_dim=config.hidden_dim 

    """ 
    Local Agents 
    """ 

    agent_init_params = []
    alg_types = [agent_alg for agent in env.agents] 
    for acsp, obsp, algtype in zip(env.action_spaces.values(), env.observation_spaces.values(), alg_types): 
            num_in_pol = obsp.shape[0] 
            discrete_action = True
            get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
    init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                    'hidden_dim': hidden_dim,
                    'alg_types': alg_types,
                    'agent_init_params': agent_init_params,
                    'discrete_action': discrete_action} 

    local_agents = MADDPG(**init_dict) 
    local_agents.init_dict = init_dict 

    local_replay_buffer = ReplayBuffer(
            config.buffer_length, 
            local_agents.nagents, 
            [obsp.shape[0] for obsp in env.observation_spaces.values()], 
            [acsp.shape[0] if isinstance(acsp, Box) else acsp.n for acsp in env.action_spaces.values()]
        ) 

    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads): 
        obs = env.reset()

        local_agents.prep_rollouts(device='cpu') 

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps

        local_agents.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining) 
        local_agents.reset_noise() 
        
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable 
            torch_obs = [Variable(torch.Tensor(list(obs.values())[i]).unsqueeze(dim=0),
                                  requires_grad=False)
                         for i in range(local_agents.nagents)] 
            
            # get actions as torch Variables
            torch_agent_actions = local_agents.step(torch_obs, explore=True) 

            # convert actions to numpy arrays
            agent_actions = [ac.data.numpy() for ac in torch_agent_actions]
            # rearrange actions to be as per environment
            actions = {agent : np.argmax(ac[0]) for ac, agent in zip(agent_actions, env.agents)}
            
            # get next_obs, rewards and dones for local agents  
            next_obs, rewards, dones, infos = env.step(actions) 
            
            local_replay_buffer.push(
                np.array([[obs[i] for i in obs]]), 
                agent_actions, 
                np.array(list(rewards.values())).reshape(1, -1), 
                np.array([[next_obs[i] for i in next_obs]]), 
                np.array(list(dones.values())).reshape(1, -1)
            ) 

            obs = next_obs 

            t += config.n_rollout_threads
            if (len(local_replay_buffer) >= config.batch_size and (t % config.steps_per_update) < config.n_rollout_threads): 

                local_agents.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(local_agents.nagents):
                        sample = local_replay_buffer.sample(config.batch_size, to_gpu=False) 
                        local_agents.update(sample, a_i) 
                    local_agents.update_all_targets()
                local_agents.prep_rollouts(device='cpu') 
        ep_rews = local_replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads) 
        logger.add_scalar('mean_episode_reward', np.mean(ep_rews), ep_i) 
        
        if ep_i % config.save_interval < config.n_rollout_threads: 
            local_agents.save(os.path.join(config.log_dir, 'local_model.pt')) 

        print("Episodes %i-%i of %i --> Team Reward: %i " % (ep_i + 1, ep_i + 1 + config.n_rollout_threads, config.n_episodes, ep_rews[0])) 



    env.close()
    local_agents.save(os.path.join(config.log_dir, 'local_model.pt'))
    logger.export_scalars_to_json(str(config.log_dir / 'summary.json'))
    logger.close()






    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1, type=int, help="Random seed") 
    parser.add_argument("--expname", default="independent_learners", type=str) 
    parser.add_argument("--log_dir", default="test/logs/", type=str) 

    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=50_000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="DDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="DDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--continuous_action",
                        action='store_false')

    config = parser.parse_args()

    run(config)

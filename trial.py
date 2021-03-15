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

env = simple_spread_v2.parallel_env(local_ratio=0.5)
env.reset() 


agent_alg="DDPG" 
tau=0.01
gamma=0.95 
lr=0.01
hidden_dim=64

agent_init_params = []
alg_types = [agent_alg] 
obsp = list(env.observation_spaces.values())[0].shape[0] * len(env.agents) 
num_in_pol = obsp 
discrete_action = False 
num_out_pol = len(env.agents) 
num_in_critic = obsp + len(env.agents) 
agent_init_params.append({'num_in_pol': num_in_pol,
                            'num_out_pol': num_out_pol,
                            'num_in_critic': num_in_critic})
init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                'hidden_dim': hidden_dim,
                'alg_types': alg_types,
                'agent_init_params': agent_init_params,
                'discrete_action': discrete_action} 

manager = MADDPG(**init_dict) 
manager.init_dict = init_dict 

manager_replay_buffer = ReplayBuffer(
        int(1e6), 
        manager.nagents, 
        [obsp], 
        [len(env.agents)]
    ) 
print(manager) 
t = 0

for ep_i in range(0, 50_000, 1): 
    obs = env.reset() 
    obs = np.array(list(obs.values())).reshape(-1)

    manager.prep_rollouts(device='cpu') 

    explr_pct_remaining = max(0, 25000 - ep_i) / 25000 

    manager.scale_noise(0.0 + (0.3 - 0.0) * explr_pct_remaining) 
    manager.reset_noise() 
    
    for et_i in range(25):
        torch_obs = [Variable(torch.Tensor(obs).unsqueeze(dim=0), requires_grad=False) for i in range(manager.nagents)] 
        
        torch_agent_actions = manager.step(torch_obs, explore=True) 
        
        agent_actions = [np.random.randint(0,5) for ac in env.agents]
        actions = {agent : ac for ac, agent in zip(agent_actions, env.agents)}
        next_obs, rewards, dones, infos = env.step(actions) 
        next_obs = np.array(list(next_obs.values())).reshape(-1)
    
        obs = next_obs     

    print("Episodes %i of %i --> Team Reward: %i " % (ep_i + 1, 50_000, rewards['agent_0'])), 
    print(torch_agent_actions) 


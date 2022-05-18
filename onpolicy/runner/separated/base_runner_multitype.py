    
import time
import wandb
import os
import numpy as np
from itertools import chain
from collections import Counter
import torch
from tensorboardX import SummaryWriter

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.util import update_linear_schedule

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.unit_type_bits = config['unit_type_bits']
        # agents = [self.envs.get_unit_by_id(agent) for agent in range(self.num_agents)]
        # unit_types = [self.get_unit_type_id(agent, True) for agent in agents]
        unit_types = [0, 0, 0, 1, 1, 1, 1, 1]

        unit_types = {'2s3z':           [0, 0, 1, 1, 1],
                      '3s5z':           [0, 0, 0, 1, 1, 1, 1, 1],
                      '3s5z_vs_3s6z':   [0, 0, 0, 1, 1, 1, 1, 1],
                      'MMM':            [0, 1, 1, 2, 2, 2, 2, 2, 2, 2],
                      'MMM2':           [0, 1, 1, 2, 2, 2, 2, 2, 2, 2],
                      '1c3s5z':         [0, 1, 1, 1, 2, 2, 2, 2, 2],
                      'bane_vs_bane':   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                         1, 1, 1, 1]}
        self.type_count = list(Counter(unit_types[self.all_args.map_name]).values())

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)


        from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy


        self.policy = []
        for _ in range(self.unit_type_bits):
            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[0],
                        share_observation_space,
                        self.envs.action_space[0],
                        device = self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for unit_type in range(self.unit_type_bits):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[unit_type], device = self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
            bu = SharedReplayBuffer(self.all_args,
                                    self.type_count[unit_type],
                                    self.envs.observation_space[0],
                                    share_observation_space,
                                    self.envs.action_space[0])
            self.buffer.append(bu)
            self.trainer.append(tr)
            
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for unit_type in range(self.unit_type_bits):
            self.trainer[unit_type].prep_rollout()
            next_value = self.trainer[unit_type].policy.get_values(np.concatenate(self.buffer[unit_type].share_obs[-1]),
                                                                   np.concatenate(self.buffer[unit_type].rnn_states_critic[-1]),
                                                                   np.concatenate(self.buffer[unit_type].masks[-1]))
            next_value = np.array(np.split(_t2n(next_value), self.n_rollout_threads))
            self.buffer[unit_type].compute_returns(next_value, self.trainer[unit_type].value_normalizer)

    def train(self):
        train_infos = []
        for unit_type in range(self.unit_type_bits):
            self.trainer[unit_type].prep_training()
            train_info = self.trainer[unit_type].train(self.buffer[unit_type])
            train_infos.append(train_info)       
            self.buffer[unit_type].after_update()

        return train_infos

    def save(self):
        for unit_type in range(self.unit_type_bits):
            policy_actor = self.trainer[unit_type].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(unit_type) + ".pt")
            policy_critic = self.trainer[unit_type].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(unit_type) + ".pt")

    def restore(self):
        for unit_type in range(self.unit_type_bits):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(unit_type) + '.pt')
            self.policy[unit_type].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(unit_type) + '.pt')
            self.policy[unit_type].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for unit_type in range(self.unit_type_bits):
            for k, v in train_infos[unit_type].items():
                agent_k = "agent%i/" % unit_type + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
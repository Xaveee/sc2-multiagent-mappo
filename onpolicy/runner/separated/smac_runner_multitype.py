import time
import wandb
import numpy as np
from functools import reduce
from itertools import chain
import torch
from onpolicy.runner.separated.base_runner_multitype import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(
            self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            for unit_type in range(self.unit_type_bits):
                if self.use_linear_lr_decay:
                    self.trainer[unit_type].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(
                    step)

                # Obser reward and next obs
                # print(actions.shape)
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(
                    actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * \
                self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.map_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_draw = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_draw = []
                    incre_battles_game = []

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(
                                info[0]['battles_won']-last_battles_won[i])
                        if 'battles_draw' in info[0].keys():
                            battles_draw.append(info[0]['battles_draw'])
                            incre_battles_draw.append(
                                info[0]['battles_won']-last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(
                                info[0]['battles_game']-last_battles_game[i])

                    incre_win_rate = np.sum(
                        incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game) > 0 else 0.0
                    incre_draw_rate = np.sum(
                        incre_battles_draw)/np.sum(incre_battles_game) if np.sum(incre_battles_game) > 0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate},
                                  step=total_num_steps)
                        # wandb.log({"incre_draw_rate": incre_win_rate},
                        #           step=total_num_steps)
                        wandb.log({"average_step_rewards": np.mean(self.buffer[0].rewards)}, step=total_num_steps)

                    else:
                        self.writter.add_scalars(
                            "incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)
                        self.writter.add_scalars(
                            "incre_draw_rate", {"incre_draw_rate": incre_draw_rate}, total_num_steps)

                    last_battles_game = battles_game
                    last_battles_won = battles_won

                for unit_type in range(self.unit_type_bits):
                    train_infos[unit_type].update(
                        {'dead_ratio': 1 - self.buffer[unit_type].active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer[unit_type].active_masks.shape))})
                    # train_infos[unit_type].update({'average_step_rewards': np.mean(self.buffer[unit_type].rewards)})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
        # print("saving")
        # self.envs.envs[0].save_replay()
        # print("saved")

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # _obs = obs.copy()
        # _share_obs = share_obs.copy()
        # _available_action = available_actions.copy()

        if not self.use_centralized_V:
            share_obs = obs
            
        
        bit = 0
        for count in self.type_count:
            self.buffer[bit].share_obs[0] = share_obs[:, :count]
            self.buffer[bit].obs[0] = obs[:, :count]
            self.buffer[bit].available_actions[0] = available_actions[:, :count]
            bit += 1

            obs = obs[:, count:]
            share_obs = share_obs[:, count:]
            available_actions = available_actions[:, count:]

    @ torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critics = []
        
        for unit_type in range(self.unit_type_bits):
            self.trainer[unit_type].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_states_critic \
                = self.trainer[unit_type].policy.get_actions(np.concatenate(self.buffer[unit_type].share_obs[step]),
                                                            np.concatenate(self.buffer[unit_type].obs[step]),
                                                            np.concatenate(self.buffer[unit_type].rnn_states[step]),
                                                            np.concatenate(self.buffer[unit_type].rnn_states_critic[step]),
                                                            np.concatenate(self.buffer[unit_type].masks[step]),
                                                            np.concatenate(self.buffer[unit_type].available_actions[step]))
            value = np.array(np.split(_t2n(value), self.n_rollout_threads))
            action = np.array(np.split(_t2n(action), self.n_rollout_threads))
            action_log_prob = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            rnn_state = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
            
            values.append(value)
            actions.append(action)
            action_log_probs.append(action_log_prob)
            rnn_states.append(rnn_state)
            rnn_states_critics.append(rnn_states_critic)
        # [self.envs, agents, dim]
        values = np.concatenate(values,1)
        actions = np.concatenate(actions,1)
        action_log_probs = np.concatenate(action_log_probs,1)
        rnn_states = np.concatenate(rnn_states,1)
        rnn_states_critics = np.concatenate(rnn_states_critics,1)
        
        # values = np.array(np.split(values, self.n_rollout_threads))
        # actions = np.array(np.split(actions, self.n_rollout_threads))
        # action_log_probs = np.array(np.split(action_log_probs, self.n_rollout_threads))
        # rnn_states = np.array(np.split(rnn_states, self.n_rollout_threads))
        # rnn_states_critics = np.array(np.split(rnn_states_critics, self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critics

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        # rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, *self.buffer[0].rnn_states_critic.shape[3:]), dtype=np.float32)


        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [
                             1.0] for agent_id in range(self.num_agents)] for info in infos])

        # _obs = obs.copy()
        # _share_obs = share_obs.copy()
        # _rnn_states = rnn_states.copy()
        # _rnn_states_critic = rnn_states_critic.copy()
        # _actions = actions.copy()
        # _action_log_probs = action_log_probs.copy()
        # _values = values.copy()
        # _rewards = rewards.copy()
        # _masks = masks.copy()
        # _bad_masks = bad_masks.copy()
        # _active_masks = active_masks.copy()
        # _available_actions = available_actions.copy()

        if not self.use_centralized_V:
            share_obs = obs
            
        bit = 0
        for count in self.type_count:
            self.buffer[bit].insert(share_obs[:, :count], 
                                    obs[:, :count], 
                                    rnn_states[:, :count], 
                                    rnn_states_critic[:, :count],
                                    actions[:, :count], 
                                    action_log_probs[:, :count],
                                    values[:, :count],
                                    rewards[:, :count],
                                    masks[:, :count],
                                    bad_masks[:, :count],
                                    active_masks[:, :count],
                                    available_actions[:, :count])

            share_obs = share_obs[:, count:]
            obs = obs[:, count:]
            rnn_states = rnn_states[:, count:]
            rnn_states_critic = rnn_states_critic[:, count:]
            actions = actions[:, count:]
            action_log_probs = action_log_probs[:, count:]
            values = values[:, count:]
            rewards = rewards[:, count:]
            masks = masks[:, count:]
            bad_masks = bad_masks[:, count:]
            active_masks = active_masks[:, count:]
            available_actions = available_actions[:, count:]

            bit += 1

    # def log_train(self, train_infos, total_num_steps):
    #     train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
    #     for k, v in train_infos.items():
    #         if self.use_wandb:
    #             wandb.log({k: v}, step=total_num_steps)
    #         else:
    #             self.writter.add_scalars(k, {k: v}, total_num_steps)

    @ torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents,
                                    self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads,
                              self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions = []
            _eval_rnn_states = []
            for unit_type in range(self.unit_type_bits):
                self.trainer[unit_type].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[unit_type].policy.act(eval_obs[:, :self.type_count[unit_type]],
                                                                                  eval_rnn_states[:, :self.type_count[unit_type]],
                                                                                  eval_masks[:, :self.type_count[unit_type]],
                                                                                  eval_available_actions[:, :self.type_count[unit_type]],
                                                                                  deterministic=True)
                eval_actions.append(_t2n(eval_action))
                _eval_rnn_states.append(_t2n(eval_rnn_state))
                
                eval_obs = eval_obs[:, self.type_count[unit_type]:]
                eval_rnn_states = eval_rnn_states[:, self.type_count[unit_type]:]
                eval_masks = eval_masks[:, self.type_count[unit_type]:]
                eval_available_actions = eval_available_actions[:, self.type_count[unit_type]:]

            eval_actions = np.concatenate(eval_actions,axis=1)
            eval_rnn_states = np.concatenate(_eval_rnn_states,axis=1)
            
            
            # eval_actions = np.array(
            #     np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            # eval_rnn_states = np.array(
            #     np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones(
                (self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(
                        np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {
                    'eval_average_episode_rewards': eval_episode_rewards}
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate},
                              step=total_num_steps)
                else:
                    self.writter.add_scalars(
                        "eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
        print('\nSaving replay')
        self.eval_envs.envs[0].save_replay()    
        print('Replay saved')
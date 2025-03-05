# Created by xionghuichen at 2022/9/21
# Email: chenxh@lamda.nju.edu.cn
import torch
import torch.optim as optim
from itertools import chain

from algo.base_agent import BaseAgent
from utils import soft_update
import torch.nn.functional as F
from utils import preprocess_traj
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from model.actor_multi_task import AttentionGaussianMultiTaskActor
from model.critic_mujoco import AttenCritic
from torch.distributions.normal import Normal

class MT_GoalPolicySACAgent(BaseAgent):
    """
    Soft Actor Critic
    """

    def __init__(self, configs: dict):
        super(MT_GoalPolicySACAgent, self).__init__(configs)

    def get_samples(self, task_id, trans_buffer=None, traj_buffer=None, demo_num=-1):
        if trans_buffer is None:
            trans_buffer = self.trans_buffer
        if traj_buffer is None:
            traj_buffer = self.traj_buffer

        states, actions, next_states, rewards, masks = trans_buffer.random_sample(
            task_id, self.batch_size
        )
        idx, traj = traj_buffer.random_sample(task_id)
        if isinstance(traj, list):
            traj, _, _, _  = traj
        # add demo to batch
        if self.batch_with_demo:
            (
                demo_states,
                demo_actions,
                demo_next_states,
                demo_rewards,
                demo_masks,
            ) = traj_buffer.get_trans(task_id, idx)
            if demo_num > 0:
                sample_idx = torch.randint(demo_states.shape[0], (demo_num,))
                demo_states = demo_states[sample_idx]
                demo_actions = demo_actions[sample_idx]
                demo_next_states = demo_next_states[sample_idx]
                demo_rewards = demo_rewards[sample_idx]
                demo_masks = demo_masks[sample_idx]
            states = torch.cat((states, demo_states), dim=0)
            actions = torch.cat((actions, demo_actions), dim=0)
            next_states = torch.cat((next_states, demo_next_states), dim=0)
            rewards = torch.cat((rewards, demo_rewards), dim=0)
            masks = torch.cat((masks, demo_masks), dim=0)

        traj = preprocess_traj(traj, self.device)
        return states, actions, next_states, rewards, masks, traj

    def critic_evaluate(self, states, actions, traj):

        states = (
            torch.FloatTensor(states.reshape(1, -1)).to(self.device)
            if type(states) == np.ndarray
            else states
        )

        actions = (
            torch.FloatTensor(actions.reshape(1, -1)).to(self.device)
            if type(actions) == np.ndarray
            else actions
        )
        if len(traj.shape) == 2:
            traj = traj.reshape((1, *traj.shape))
        demo_states = traj[0, :, :self.state_dim]
        demo_acs = traj[0, :, self.state_dim:]
        demo_next_actions, demo_next_log_pis = self(
            demo_states, traj, training=True, calcu_log_prob=True
        )
        demo_states = torch.FloatTensor(demo_states).to(self.device)
        demo_acs = torch.FloatTensor(demo_acs).to(self.device)

        if len(demo_next_actions.shape) > len(demo_states.shape):
            demo_next_actions = torch.squeeze(demo_next_actions)

        if self.use_rnn_critic:
            critic_traj = torch.cat((demo_states, demo_next_actions), dim=-1).unsqueeze(axis=0).transpose(1, 2)
        else:
            critic_traj = torch.cat((demo_states, demo_next_actions, demo_acs), dim=-1).unsqueeze(axis=0).transpose(1,2)

        # states[:] = 0
        # critic_traj[:] = 0
        # actions[:] = 0

        current_Q1, current_Q2 = (
            self.critic_1(states, critic_traj, actions),
            self.critic_2(states, critic_traj, actions),
        )
        return current_Q1, current_Q2

    def get_multi_map_samples(self, recent_buf_list):
        res_dict = {}
        for k in ['map_info', 'states', 'actions', 'next_states', 'rewards', 'masks', 'traj', 'goal']:
            res_dict[k] = []
        do_train = False
        max_len = 0
        for item in recent_buf_list:
            trans_buffer, traj_buffer, task_id, map_id = item
            if (
                    trans_buffer.size[0] < self.start_timesteps or
                    trans_buffer.size[0] < self.batch_size or
                    (trans_buffer.size[0] - self.start_timesteps) % self.updates_per_step != 0):
                continue
            do_train = True
            map_info = torch.tensor(map_id, dtype=torch.int32).to(self.device).long()
            states, actions, next_states, rewards, masks, traj = self.get_samples(0, trans_buffer, traj_buffer, int(self.batch_size/5))
            res_dict['map_info'].append(map_info)
            res_dict['states'].append(states)
            res_dict['actions'].append(actions)
            res_dict['next_states'].append(next_states)
            res_dict['rewards'].append(rewards)
            res_dict['masks'].append(masks)
            res_dict['traj'].append(traj.transpose(1, 2)[0])
            res_dict['goal'].append(traj.transpose(1, 2)[0, -1])
            max_len = np.maximum(max_len, traj[0].shape[0])

        if do_train:
            for k, v in res_dict.items():
                if k in ['states', 'actions', 'next_states', 'rewards', 'masks']:
                    res_dict[k] = torch.stack(v, dim=1)
                elif k in ['traj']:
                    res_dict[k] = pad_sequence(res_dict['traj'], batch_first=True).transpose(1, 2)
                    pass
                else:
                    res_dict[k] = torch.stack(v, dim=0)
        return res_dict, do_train

    def update(self, state, action, next_state, reward, done, task_id, map_id, recent_buf_list):
        # updata param
        self.info = dict()
        for _ in range(self.updates_per_step):
            res_dict, do_train = self.get_multi_map_samples(list(recent_buf_list) + [[self.trans_buffer, self.traj_buffer, task_id, map_id]])
            if not do_train:
                return None
            states = res_dict['states']
            actions = res_dict['actions']
            next_states = res_dict['next_states']
            rewards = res_dict['rewards']
            masks = res_dict['masks']
            traj = res_dict['traj']
            map_info = res_dict['map_info']
            goal_info = res_dict['goal']
            # states, actions, next_states, rewards, masks, traj = self.get_samples(task_id)
            # critic traj
            # if len(traj) == 2:
            #     print('debug')
            with torch.no_grad():
                demo_traj_perm = traj.permute([2, 0, 1])
                demo_states = demo_traj_perm[..., :self.state_dim]
                demo_acs = demo_traj_perm[..., self.state_dim:]

                demo_next_actions, demo_next_log_pis = self(
                    demo_states, traj, training=True, calcu_log_prob=True, squeeze = False, goal_info = goal_info
                )

                # print(demo_states.shape, demo_next_actions.shape)
                if self.use_rnn_critic:
                    # critic_traj = torch.cat((demo_states, demo_next_actions), dim=-1).unsqueeze(axis=0).transpose(1, 2)
                    critic_traj = torch.cat((demo_states, demo_next_actions), dim=-1).permute([1, 2, 0])
                else:
                    # critic_traj = torch.cat((demo_states, demo_next_actions, demo_acs), dim=-1).transpose(1, 2)
                    critic_traj = torch.cat((demo_states, demo_next_actions, demo_acs), dim=-1).permute([1, 2, 0])

            # calculate target q valued
            with torch.no_grad():
                next_actions, next_log_pis = self(next_states, traj, training=True, calcu_log_prob=True, squeeze = False, goal_info = goal_info)

                target_Q1, target_Q2 = (
                    self.critic_1_target(next_states, critic_traj, next_actions, map_info, squeeze = False),
                    self.critic_2_target(next_states, critic_traj, next_actions, map_info, squeeze = False),
                )
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pis
                target_Q = rewards + masks * self.gamma * target_Q
                target_Q = torch.clip(target_Q, -200, 200)  # hyper-param for clipping Q

            # update critic
            current_Q1, current_Q2 = (
                self.critic_1(states, critic_traj, actions, map_info, squeeze = False),
                self.critic_2(states, critic_traj, actions, map_info, squeeze = False),
            )
            # print('current_Q1: {}'.format(current_Q1), 'current_Q2: {}'.format(current_Q2))
            self.info.update(
                {
                    "Q1": current_Q1.mean().item(),
                    "Q2": current_Q2.mean().item(),
                    "Q_target": target_Q.mean().item(),
                }
            )

            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            pred_actions, pred_log_pis = self(
                states, traj, training=True, calcu_log_prob=True, squeeze = False, goal_info = goal_info
            )
            if len(pred_actions.shape) == 2:
                pred_actions = pred_actions.unsqueeze(axis=1)
            current_Q1, current_Q2 = (
                self.critic_1(states, critic_traj, pred_actions, map_info, squeeze = False),
                self.critic_2(states, critic_traj, pred_actions, map_info, squeeze = False),
            )
            actor_loss = (
                    self.alpha * pred_log_pis - torch.min(current_Q1, current_Q2)
            ).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # update alpha
            if self.learn_alpha:
                pred_log_pis += self.entropy_target
                alpha_loss = -(self.log_alpha * pred_log_pis.detach()).mean()

                # print('current_Q1: {}'.format(current_Q1.reshape(-1).detach().cpu().numpy()[0:10]))
                # print('pred_log_pis',      pred_log_pis.reshape(-1).detach().cpu().numpy()[0:10])
                # print('pred_log_pis mean', pred_log_pis.mean().item())
                # print('log_alpha', self.log_alpha.item())
                # print('alpha_loss', alpha_loss.item())

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                # update log_alpha in self.models
                self.models["log_alpha"].data = self.log_alpha.data
                self.alpha = self.log_alpha.clone().detach().exp().item()

                self.info.update({"alpha": self.alpha})

            # update target critic
            soft_update(self.rho, self.critic_1, self.critic_1_target)
            soft_update(self.rho, self.critic_2, self.critic_2_target)

            self.info.update(
                {
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "alpha_loss": alpha_loss.item() if self.learn_alpha else 0.0,
                }
            )

        return self.info


    def init_critic(self):
        # Q1
        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "goal_embed_dim": self.critic_goal_embed_dim,
            "embed_goal": self.critic_embed_goal,
            "output_dim": 1,
            "hidden_size": self.critic_hidden_size,
            "seperate_encode": False,
            "use_map_id": False,
            "map_num": self.map_num,
        }


        kwargs['pos_encode'] = False
        critic_module = AttenCritic
        print("\n use DAAC attention critic!!!!!!!!!!\n")

        self.critic_1 = critic_module(**kwargs).to(self.device)
        self.critic_1_target = critic_module(**kwargs).to(self.device)
        soft_update(1.0, self.critic_1, self.critic_1_target)

        # Q2
        self.critic_2 = critic_module(**kwargs).to(self.device)
        self.critic_2_target = critic_module(**kwargs).to(self.device)
        soft_update(1.0, self.critic_2, self.critic_2_target)
        self.critic_optim = optim.RMSprop(
            chain(self.critic_1.parameters(), self.critic_2.parameters()),
            lr=self.critic_lr,
        )

    def init_policy(self):
        # policy
        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "embed_dim": self.actor_embed_dim,
            "num_heads": self.actor_num_heads,
            "num_encoder_layers": self.actor_num_encoder_layers,
            "num_decoder_layers": self.actor_num_decoder_layers,
            "dim_feedforward": self.actor_dim_feedforward,
            "dropout": self.actor_dropout,
            "pos_encode": self.actor_pos_encode,
            "state_std_independent": False,
            "share_state_encoder": self.share_state_encoder,
            "seperate_encode": False,
            "use_rnn_actor": self.use_rnn_actor
        }


        self.actor = AttentionGaussianMultiTaskActor(**kwargs).to(self.device)
        print("\nuse DAAC!!!\n")

        # if self.use_only_decoder:
        #     self.actor = AttentionDecoderMultiTaskActor(**kwargs).to(self.device)
        #     print("\nuse only Decoder!!!\n")
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=self.actor_lr)
        self.BC_optim=optim.Adam(self.actor.parameters(),lr=self.actor_lr,weight_decay=1e-6) # 1e-6
        

    def forward(self, state, traj, training, calcu_log_prob=False, squeeze = True, goal_info = None, return_atten_wei_lst = False):
        state = (
            torch.FloatTensor(state).to(self.device)
            if type(state) == np.ndarray
            else state
        )
        if len(state.shape) < 2:
            state = torch.unsqueeze(state, dim=0)
        # pre-process traj
        traj = preprocess_traj(traj, self.device) if type(traj) == np.ndarray else traj
        forward_result = self.actor(state, traj,
                                           squeeze = squeeze,
                                           goal = goal_info,
                                           return_atten_wei_lst = return_atten_wei_lst)

        if return_atten_wei_lst:
            action_mu, action_std, atten_wei_lst = forward_result
        else:
            action_mu, action_std = forward_result

        pi_dist = Normal(action_mu, action_std)
        # print('action_mu', action_mu)
        # get action
        if training:
            action = pi_dist.rsample()
        else:
            action = action_mu

        # get log_prob
        if calcu_log_prob:
            log_prob = torch.sum(pi_dist.log_prob(action), axis=-1, keepdims=True)
            log_prob -= torch.sum(
                2 * (np.log(2) - action - F.softplus(-2 * action)),
                axis=-1,
                keepdims=True,
            )  # equivalent to Eq 26 in SAC paper, but more numerically stable

            log_prob -= torch.sum(
                np.log(1.0 / self.action_high) * torch.ones_like(action),
                axis=-1,
                keepdim=True,
            )  # for action reshaping from [-1, 1] into [action_low, action_high]
        else:
            log_prob = None

        action = self.squash_action(action)

        self.info.update(
            {
                "action_std": action_std.mean().item(),
                # "action_mu": action_mu.mean().item(),
            }
        )
        if return_atten_wei_lst:
            return action, log_prob, atten_wei_lst
        else:
            return action, log_prob

    @torch.no_grad()
    def select_action(self, state, traj, training, return_atten_wei_lst = False):
        if return_atten_wei_lst:
            action, _, atten_weights = self(state, traj, training, return_atten_wei_lst = return_atten_wei_lst)
            return action.cpu().data.numpy().flatten(), atten_weights
        else:
            action, _ = self(state, traj, training, return_atten_wei_lst = return_atten_wei_lst)
            return action.cpu().data.numpy().flatten()

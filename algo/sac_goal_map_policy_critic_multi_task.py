# Created by xionghuichen at 2022/9/21
# Email: chenxh@lamda.nju.edu.cn
import sys
import os
from itertools import chain
import numpy as np
import torch
import torch.optim as optim
from torch.distributions.normal import Normal

from torch.nn.utils.rnn import pad_sequence
from algo.sac_goal_map_policy_critic import MT_GoalMapPolicySACAgent
from model.actor_multi_task import (
    TransformerGaussianMultiTaskActor,
    AttentionGaussianMultiTaskActor,
    AttentionDecoderMultiTaskActor,
    TransformerRNNMultiTaskActor,
)
from model.critic_multi_task import GoalMapMLPMultiTaskCritic, AttenMultiTaskCritic
import torch.nn.functional as F
from torch_utils import soft_update, preprocess_traj
from CONST import MapType

# from consts import *


# used
class MT_GoalMapPolicyMultiTaskSACAgent(MT_GoalMapPolicySACAgent):
    """
    Soft Actor Critic
    """

    def __init__(self, configs: dict):
        super(MT_GoalMapPolicyMultiTaskSACAgent, self).__init__(configs)
        self.use_transformer = configs["use_transformer"]
        self.use_rnn_critic = configs["use_rnn_critic"]
        self.use_only_decoder = configs["use_only_decoder"]
        self.use_rnn_actor = configs["use_rnn_actor"]
        self.configs = configs

    def init_critic(self):
        # Q1
        print("\ninit critic!!!!!!!!!!!!!!!\n")
        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "goal_embed_dim": self.critic_goal_embed_dim,
            "embed_goal": self.critic_embed_goal,
            "output_dim": 1,
            "hidden_size": self.critic_hidden_size,
            "map_num": self.map_num,
            "map_type": self.map_type,
            "map_shape": self.map_shape,
        }
        if self.use_rnn_critic:
            critic_module = GoalMapMLPMultiTaskCritic
            print("\n use rnn critic!!!!!!!!!!\n")
        else:
            kwargs["num_encoder_layers"] = self.critic_num_encoder_layers  # 4
            kwargs["num_decoder_layers"] = self.critic_num_decoder_layers  # 4
            kwargs["dim_feedforward"] = self.critic_dim_feedforward  # 256
            kwargs["atten_emb_dim"] = self.critic_embed_dim  # 128
            kwargs["num_heads"] = self.critic_num_heads  # 16
            kwargs["dropout"] = self.critic_dropout  # 0.1
            critic_module = AttenMultiTaskCritic
            print("\n use DDT attention critic!!!!!!!!!!\n")
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
        print("\ninit policy!!!!!!!!!!!!!!!\n")
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
            "no_coordinate": self.configs["no_coordinate"],
        }
        if self.use_transformer:
            self.actor = TransformerGaussianMultiTaskActor(**kwargs).to(self.device)
            print("\nuse transformer!!!\n")
        else:
            self.actor = AttentionGaussianMultiTaskActor(**kwargs).to(self.device)
            print("\nuse DDT!!!\n")

        if self.use_rnn_actor:
            self.actor = TransformerRNNMultiTaskActor(**kwargs).to(self.device)
            print("\nuse RNN!!!\n")

        if self.use_only_decoder:
            self.actor = AttentionDecoderMultiTaskActor(**kwargs).to(self.device)
            print("\nuse only Decoder!!!\n")

        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=self.actor_lr)
        self.BC_optim = optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, weight_decay=1e-4
        )

    def forward(self, state, traj, training, goal_info=None, calcu_log_prob=False):
        state = (
            torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if type(state) == np.ndarray
            else state
        )
        # pre-process traj
        traj = preprocess_traj(traj, self.device) if type(traj) == np.ndarray else traj
        action_mu, action_std = self.actor(state, traj, goal_info)
        pi_dist = Normal(action_mu, action_std)

        # get action
        if "rot_maze_tp" in sys.argv[0] and not self.configs["rot_sac"]:
            action = action_mu
        else:
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

        if "rot_maze_tp" in sys.argv[0] and not self.configs["rot_sac"]:  # td3
            action = self.squash_action(action)
            if training:
                pi_dist = Normal(action, action_std)
                action = pi_dist.rsample()
        else:
            action = self.squash_action(action)  # action std should be add here

        self.info.update(
            {
                "action_std": action_std.mean().item(),
                # "action_mu": action_mu.mean().item(),
            }
        )

        return action, log_prob

    def get_multi_map_samples(self, recent_buf_list):
        res_dict = {}
        for k in [
            "map_info",
            "states",
            "actions",
            "next_states",
            "rewards",
            "masks",
            "traj",
            "goal",
        ]:
            res_dict[k] = []
        do_train = False
        replace_sample = True
        max_len = 0
        for item in recent_buf_list:
            trans_buffer, traj_buffer, task_id, task_config = item
            map_id = task_config["map_id"]
            if replace_sample:
                if (
                    trans_buffer.stored_eps_num < self.start_epi
                    or trans_buffer.size[0] < 2.0
                ):
                    continue
            else:
                if (
                    trans_buffer.stored_eps_num < self.start_epi
                    or trans_buffer.size[0] < self.batch_size
                ):
                    continue

            do_train = True
            if self.map_type == MapType.ID:
                map_info = (
                    torch.tensor(map_id, dtype=torch.int32).to(self.device).long()
                )
            elif self.map_type == MapType.FIG:
                map_info = torch.from_numpy(
                    (self.map_fig_dict[map_id] / 255).astype(np.float32)
                ).to(self.device)
            else:
                raise RuntimeError
            states, actions, next_states, rewards, masks, traj = self.get_samples(
                0,
                trans_buffer,
                traj_buffer,
                int(self.batch_size / len(recent_buf_list)),
            )
            res_dict["map_info"].append(map_info)
            res_dict["states"].append(states)
            res_dict["actions"].append(actions)
            res_dict["next_states"].append(next_states)
            res_dict["rewards"].append(rewards)
            res_dict["masks"].append(masks)
            res_dict["traj"].append(traj.transpose(1, 2)[0])
            res_dict["goal"].append(traj.transpose(1, 2)[0, -1])
            max_len = np.maximum(max_len, traj[0].shape[0])
        if do_train:
            for k, v in res_dict.items():
                if k in ["states", "actions", "next_states", "rewards", "masks"]:
                    res_dict[k] = torch.stack(v, dim=1)
                elif k in ["traj"]:
                    res_dict[k] = pad_sequence(
                        res_dict["traj"], batch_first=True
                    ).transpose(1, 2)
                    pass
                else:
                    res_dict[k] = torch.stack(v, dim=0)
        return res_dict, do_train

    def get_multi_map_samplesROT(self, recent_buf_list):
        res_dict = {}
        for k in [
            "map_info",
            "states",
            "actions",
            "next_states",
            "rewards",
            "masks",
            "traj",
            "goal",
            "demo_states",
            "demo_actions",
        ]:
            res_dict[k] = []
        do_train = False
        max_len = 0
        for item in recent_buf_list:
            trans_buffer, traj_buffer, task_id, map_id = item
            if (
                trans_buffer.stored_eps_num < self.start_epi
                or trans_buffer.size[0] < self.batch_size
            ):
                continue
            do_train = True
            if self.map_type == MapType.ID:
                map_info = (
                    torch.tensor(map_id, dtype=torch.int32).to(self.device).long()
                )
            elif self.map_type == MapType.FIG:
                map_info = torch.from_numpy(
                    (self.map_fig_dict[map_id] / 255).astype(np.float32)
                ).to(self.device)
            else:
                raise RuntimeError
            states, actions, next_states, rewards, masks, traj = self.get_samples(
                0, trans_buffer, traj_buffer, 0
            )  # no expert data will be used for td learning
            demo_states, demo_actions, traj = self.get_demo_trans(
                0, traj_buffer, int(self.batch_size / len(recent_buf_list))
            )
            res_dict["map_info"].append(map_info)
            res_dict["states"].append(states)
            res_dict["actions"].append(actions)
            res_dict["next_states"].append(next_states)
            res_dict["rewards"].append(rewards)
            res_dict["masks"].append(masks)
            res_dict["traj"].append(traj.transpose(1, 2)[0])
            res_dict["goal"].append(traj.transpose(1, 2)[0, -1])
            res_dict["demo_states"].append(demo_states)
            res_dict["demo_actions"].append(demo_actions)
            max_len = np.maximum(max_len, traj[0].shape[0])
        if do_train:
            for k, v in res_dict.items():
                if k in [
                    "states",
                    "actions",
                    "next_states",
                    "rewards",
                    "masks",
                    "demo_states",
                    "demo_actions",
                ]:
                    res_dict[k] = torch.stack(v, dim=1)
                elif k in ["traj"]:
                    res_dict[k] = pad_sequence(
                        res_dict["traj"], batch_first=True
                    ).transpose(1, 2)
                else:
                    res_dict[k] = torch.stack(v, dim=0)
        return res_dict, do_train

    def update(
        self, state, action, next_state, reward, done, task_id, map_id, recent_buf_list
    ):
        # updata param
        self.info = dict()
        for _ in range(self.updates_per_step):
            res_dict, do_train = self.get_multi_map_samples(
                list(recent_buf_list)
                + [[self.trans_buffer, self.traj_buffer, task_id, map_id]]
            )
            if not do_train:
                return None
            states = res_dict["states"]
            actions = res_dict["actions"]
            next_states = res_dict["next_states"]
            rewards = res_dict["rewards"]
            masks = res_dict["masks"]
            traj = res_dict["traj"]
            map_info = res_dict["map_info"]
            goal_info = res_dict["goal"]
            # critic traj

            with torch.no_grad():
                demo_traj_perm = traj.permute([2, 0, 1])
                demo_states = demo_traj_perm[..., : self.state_dim]
                demo_acs = demo_traj_perm[..., self.state_dim :]
                demo_next_actions, demo_next_log_pis = self(
                    demo_states,
                    traj,
                    goal_info=goal_info,
                    training=True,
                    calcu_log_prob=True,
                )
                critic_traj = torch.cat(
                    (demo_states, demo_next_actions, demo_acs), dim=-1
                ).permute([1, 2, 0])

            # calculate target q valued
            with torch.no_grad():
                next_actions, next_log_pis = self(
                    next_states,
                    traj,
                    goal_info=goal_info,
                    training=True,
                    calcu_log_prob=True,
                )

                target_Q1, target_Q2 = (
                    self.critic_1_target(
                        next_states, critic_traj, next_actions, map_info, goal_info
                    ),
                    self.critic_2_target(
                        next_states, critic_traj, next_actions, map_info, goal_info
                    ),
                )
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pis
                target_Q = rewards + masks * self.gamma * target_Q
                target_Q = torch.clip(target_Q, -200, 200)  # hyper-param for clipping Q

            # update critic
            current_Q1, current_Q2 = (
                self.critic_1(states, critic_traj, actions, map_info, goal_info),
                self.critic_2(states, critic_traj, actions, map_info, goal_info),
            )

            self.info.update(
                {
                    "Q1": current_Q1.mean().item(),
                    "Q2": current_Q2.mean().item(),
                    "Q_target": target_Q.mean().item(),
                }
            )

            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # update actor
            # self.critic_1.eval(), self.critic_2.eval()  # Freeze Q-networks to save computational effort

            pred_actions, pred_log_pis = self(
                states, traj, goal_info=goal_info, training=True, calcu_log_prob=True
            )
            current_Q1, current_Q2 = (
                self.critic_1(states, critic_traj, pred_actions, map_info, goal_info),
                self.critic_2(states, critic_traj, pred_actions, map_info, goal_info),
            )
            actor_loss = (
                self.alpha * pred_log_pis - torch.min(current_Q1, current_Q2)
            ).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # self.critic_1.train(), self.critic_2.train()

            # update alpha
            if self.learn_alpha:
                pred_log_pis += self.entropy_target
                alpha_loss = -(self.log_alpha * pred_log_pis.detach()).mean()
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

    def BCupdate(
        self, state, action, next_state, reward, done, task_id, map_id, recent_buf_list
    ):
        # updata param
        self.info = dict()
        for _ in range(self.updates_per_step):
            res_dict, do_train = self.get_multi_map_samples_BC(
                list(recent_buf_list)
                + [[self.trans_buffer, self.traj_buffer, task_id, map_id]]
            )
            if not do_train:
                return None
            states = res_dict["states"]
            actions = res_dict["actions"]
            next_states = res_dict["next_states"]
            rewards = res_dict["rewards"]
            masks = res_dict["masks"]
            traj = res_dict["traj"]
            map_info = res_dict["map_info"]
            goal_info = res_dict["goal"]

            # update actor
            pred_actions, pred_log_pis = self(
                states, traj, goal_info=goal_info, training=True, calcu_log_prob=True
            )
            actor_bc_loss = ((pred_actions - actions) ** 2).mean()

            self.BC_optim.zero_grad()
            actor_bc_loss.backward()
            self.BC_optim.step()

            # update alpha
            if self.learn_alpha:
                pred_log_pis += self.entropy_target
                alpha_loss = -(self.log_alpha * pred_log_pis.detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                # update log_alpha in self.models
                self.models["log_alpha"].data = self.log_alpha.data
                self.alpha = self.log_alpha.clone().detach().exp().item()

                self.info.update({"alpha": self.alpha})

            self.info.update(
                {
                    "BC_loss": actor_bc_loss.item(),
                    "alpha_loss": alpha_loss.item() if self.learn_alpha else 0.0,
                }
            )

        return self.info

    def get_multi_map_samples_BC(self, recent_buf_list):
        res_dict = {}
        for k in [
            "map_info",
            "states",
            "actions",
            "next_states",
            "rewards",
            "masks",
            "traj",
            "goal",
        ]:
            res_dict[k] = []
        do_train = False
        max_len = 0
        for item in recent_buf_list:
            trans_buffer, traj_buffer, task_id, map_id = item
            # if (
            #         trans_buffer.stored_eps_num < self.start_epi or
            #         trans_buffer.size[0] < self.batch_size):
            #     continue
            do_train = True
            if self.map_type == MapType.ID:
                map_info = (
                    torch.tensor(map_id, dtype=torch.int32).to(self.device).long()
                )
            elif self.map_type == MapType.FIG:
                map_info = torch.from_numpy(
                    (self.map_fig_dict[map_id] / 255).astype(np.float32)
                ).to(self.device)
            else:
                raise RuntimeError
            states, actions, next_states, rewards, masks, traj = self.get_samples_BC(
                0, None, traj_buffer, int(self.batch_size / 5)
            )
            res_dict["map_info"].append(map_info)
            res_dict["states"].append(states)
            res_dict["actions"].append(actions)
            res_dict["next_states"].append(next_states)
            res_dict["rewards"].append(rewards)
            res_dict["masks"].append(masks)
            res_dict["traj"].append(traj.transpose(1, 2)[0])
            res_dict["goal"].append(traj.transpose(1, 2)[0, -1])
            max_len = np.maximum(max_len, traj[0].shape[0])
        if do_train:
            for k, v in res_dict.items():
                if k in ["states", "actions", "next_states", "rewards", "masks"]:
                    res_dict[k] = torch.stack(v, dim=1)
                elif k in ["traj"]:
                    res_dict[k] = pad_sequence(
                        res_dict["traj"], batch_first=True
                    ).transpose(1, 2)
                    pass
                else:
                    res_dict[k] = torch.stack(v, dim=0)
        return res_dict, do_train

    def get_samples_BC(self, task_id, trans_buffer=None, traj_buffer=None, demo_num=-1):
        if traj_buffer is None:
            traj_buffer = self.traj_buffer

        idx, traj = traj_buffer.random_sample(task_id)

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
            states = demo_states
            actions = demo_actions
            next_states = demo_next_states
            rewards = demo_rewards
            masks = demo_masks

        traj = preprocess_traj(traj, self.device)
        return states, actions, next_states, rewards, masks, traj

    def get_samples_pretrain_critic(
        self, task_id, trans_buffer=None, traj_buffer=None, demo_num=-1
    ):
        if trans_buffer is None:
            trans_buffer = self.trans_buffer
        if traj_buffer is None:
            traj_buffer = self.traj_buffer

        states, actions, next_states, rewards, masks = trans_buffer.random_sample(
            task_id, self.batch_size
        )
        idx, traj = traj_buffer.random_sample(task_id)

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

    def get_mutli_map_samples_pretrain_critic(self, recent_buf_list):
        res_dict = {}
        for k in [
            "map_info",
            "states",
            "actions",
            "next_states",
            "rewards",
            "masks",
            "traj",
            "goal",
        ]:
            res_dict[k] = []
        do_train = False
        max_len = 0
        for item in recent_buf_list:
            trans_buffer, traj_buffer, task_id, map_id = item
            if (
                trans_buffer.stored_eps_num < self.start_epi
                or trans_buffer.size[0] < self.batch_size
            ):
                continue
            do_train = True
            if self.map_type == MapType.ID:
                map_info = (
                    torch.tensor(map_id, dtype=torch.int32).to(self.device).long()
                )
            elif self.map_type == MapType.FIG:
                map_info = torch.from_numpy(
                    (self.map_fig_dict[map_id] / 255).astype(np.float32)
                ).to(self.device)
            else:
                raise RuntimeError
            states, actions, next_states, rewards, masks, traj = (
                self.get_samples_pretrain_critic(
                    0, trans_buffer, traj_buffer, int(self.batch_size / 5)
                )
            )
            res_dict["map_info"].append(map_info)
            res_dict["states"].append(states)
            res_dict["actions"].append(actions)
            res_dict["next_states"].append(next_states)
            res_dict["rewards"].append(rewards)
            res_dict["masks"].append(masks)
            res_dict["traj"].append(traj.transpose(1, 2)[0])
            res_dict["goal"].append(traj.transpose(1, 2)[0, -1])
            max_len = np.maximum(max_len, traj[0].shape[0])
        if do_train:
            for k, v in res_dict.items():
                if k in ["states", "actions", "next_states", "rewards", "masks"]:
                    res_dict[k] = torch.stack(v, dim=1)
                elif k in ["traj"]:
                    res_dict[k] = pad_sequence(
                        res_dict["traj"], batch_first=True
                    ).transpose(1, 2)
                    pass
                else:
                    res_dict[k] = torch.stack(v, dim=0)
        return res_dict, do_train

    def pretrain_update(
        self, state, action, next_state, reward, done, task_id, map_id, recent_buf_list
    ):
        # current run samples for Q learning has no fresh samples
        info = self.BCupdate(
            state, action, next_state, reward, done, task_id, map_id, recent_buf_list
        )
        info_q = self.PretrainCritic_update(
            state, action, next_state, reward, done, task_id, map_id, recent_buf_list
        )
        if info_q is not None:
            info.update(info_q)
        return info

    def PretrainCritic_update(
        self, state, action, next_state, reward, done, task_id, map_id, recent_buf_list
    ):
        # foreward param training set must be set true for we use policy with exploration cap. to collect data
        # updata param
        self.info = dict()
        for _ in range(self.updates_per_step):
            res_dict, do_train = self.get_mutli_map_samples_pretrain_critic(
                list(recent_buf_list)
                + [[self.trans_buffer, self.traj_buffer, task_id, map_id]]
            )
            if not do_train:
                return None
            states = res_dict["states"]
            actions = res_dict["actions"]
            next_states = res_dict["next_states"]
            rewards = res_dict["rewards"]
            masks = res_dict["masks"]
            traj = res_dict["traj"]
            map_info = res_dict["map_info"]
            goal_info = res_dict["goal"]
            # states, actions, next_states, rewards, masks, traj = self.get_samples(task_id)
            # critic traj
            with torch.no_grad():
                demo_traj_perm = traj.permute([2, 0, 1])
                demo_states = demo_traj_perm[..., : self.state_dim]
                demo_acs = demo_traj_perm[..., self.state_dim :]
                demo_next_actions, demo_next_log_pis = self(
                    demo_states,
                    traj,
                    goal_info=goal_info,
                    training=True,
                    calcu_log_prob=True,
                )  # training set must be set true for we use policy with exploration cap. to collect data
                critic_traj = torch.cat(
                    (demo_states, demo_next_actions, demo_acs), dim=-1
                ).permute([1, 2, 0])

            # calculate target q valued
            with torch.no_grad():
                next_actions, next_log_pis = self(
                    next_states,
                    traj,
                    goal_info=goal_info,
                    training=True,
                    calcu_log_prob=True,
                )

                target_Q1, target_Q2 = (
                    self.critic_1_target(
                        next_states, critic_traj, next_actions, map_info, goal_info
                    ),
                    self.critic_2_target(
                        next_states, critic_traj, next_actions, map_info, goal_info
                    ),
                )
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pis
                target_Q = rewards + masks * self.gamma * target_Q
                target_Q = torch.clip(target_Q, -200, 200)  # hyper-param for clipping Q

            # update critic
            current_Q1, current_Q2 = (
                self.critic_1(states, critic_traj, actions, map_info, goal_info),
                self.critic_2(states, critic_traj, actions, map_info, goal_info),
            )

            self.info.update(
                {
                    "Q1": current_Q1.mean().item(),
                    "Q2": current_Q2.mean().item(),
                    "Q_target": target_Q.mean().item(),
                }
            )

            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # update target critic
            soft_update(self.rho, self.critic_1, self.critic_1_target)
            soft_update(self.rho, self.critic_2, self.critic_2_target)

            self.info.update(
                {
                    "critic_loss": critic_loss.item(),
                }
            )
        return self.info

    def load_actor(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found: {}".format(model_path))
        else:
            state_dicts = torch.load(model_path)
            for model in self.models:
                if model == "actor":
                    if isinstance(
                        self.models[model], torch.Tensor
                    ):  # especially for sac, which has log_alpha to be loaded
                        self.models[model] = state_dicts[model][model]
                    else:
                        self.models[model].load_state_dict(state_dicts[model])
        # self.log_alpha.data = self.models["log_alpha"].data
        # self.alpha = self.log_alpha.clone().detach().exp().item()

        print(f"Successfully load model from {model_path}!")

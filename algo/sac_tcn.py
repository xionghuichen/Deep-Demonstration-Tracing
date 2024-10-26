import os
import copy
from itertools import chain
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rl.model.actor import TCNGaussianActor
from rl.model.critic import TCNCritic
from rl.algo.sac_mlp import SACAgent
from rl.utils.buffer import MT_TransitionBuffer, TrajectoryBuffer
from rl.utils.net import soft_update
from rl.utils.exp import preprocess_traj
from rl.utils.pretrain import get_demo


class MT_TCNSACAgent(SACAgent):
    """
    Soft Actor Critic
    """

    def __init__(self, configs: dict):
        super().__init__(configs)
        # hyper param
        self.task_nums = configs["task_nums"]
        ## actor
        self.actor_embed_dim = configs["actor_embed_dim"]
        self.actor_kernel_size = configs["actor_kernel_size"]
        self.actor_dropout = configs["actor_dropout"]
        self.actor_num_channels = [configs["actor_n_hidden"]] * configs["actor_levels"]

        ## critic
        self.critic_embed_dim = configs["critic_embed_dim"]
        self.critic_kernel_size = configs["critic_kernel_size"]
        self.critic_dropout = configs["critic_dropout"]
        self.critic_num_channels = [configs["critic_n_hidden"]] * configs[
            "critic_levels"
        ]

    def init_component(self):
        # replay buffer
        self.trans_buffer = MT_TransitionBuffer(
            self.state_dim,
            self.action_dim,
            self.buffer_size,
            self.device,
            self.task_nums,
        )
        self.traj_buffer = TrajectoryBuffer(
            self.task_nums, self.env_name, self.device, self.with_local_view
        )

        # alpha, the entropy coefficient
        self.log_alpha = torch.zeros(
            1, device=self.device, requires_grad=True
        )  # We optimize log(alpha) because alpha should always be bigger than 0.
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        with torch.no_grad():
            if self.learn_alpha:
                self.alpha = self.log_alpha.exp().item()
            else:
                self.alpha = self.alpha

        # policy
        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_size": self.actor_hidden_size,
            "dropout": self.actor_dropout,
            "kernel_size": self.actor_kernel_size,
            "embed_dim": self.actor_embed_dim,
            "num_channels": self.actor_num_channels,
            "activation_fn": nn.ReLU,
            "state_std_independent": False,
        }
        self.actor = TCNGaussianActor(**kwargs).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Q1
        kwargs = {
            "state_dim": self.state_dim,
            "hidden_size": self.critic_hidden_size,
            "num_channels": self.critic_num_channels,
            "kernel_size": self.critic_kernel_size,
            "dropout": self.critic_dropout,
            "embed_dim": self.critic_embed_dim,
            "action_dim": self.action_dim,
            "activation_fn": nn.ReLU,
            "output_dim": 1,
        }
        self.critic_1 = TCNCritic(**kwargs).to(self.device)
        # self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_target = TCNCritic(**kwargs).to(self.device)
        soft_update(1.0, self.critic_1, self.critic_1_target)
        # Q2
        self.critic_2 = TCNCritic(**kwargs).to(self.device)
        # self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_target = TCNCritic(**kwargs).to(self.device)
        soft_update(1.0, self.critic_2, self.critic_2_target)

        self.critic_optim = optim.Adam(
            chain(self.critic_1.parameters(), self.critic_2.parameters()),
            lr=self.critic_lr,
        )

        self.models = {
            "actor": self.actor,
            "actor_optim": self.actor_optim,
            "critic_1": self.critic_1,
            "critic_1_target": self.critic_1_target,
            "critic_2": self.critic_2,
            "critic_2_target": self.critic_2_target,
            "critic_optim": self.critic_optim,
            "log_alpha": self.log_alpha,
            "alpha_optim": self.alpha_optim,
        }

    def forward(self, state, traj, training, calcu_log_prob=False):
        state = (
            torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if type(state) == np.ndarray
            else state
        )
        # pre-process traj
        traj = (
            preprocess_traj(traj, state.shape[0], self.device)
            if type(traj) == np.ndarray
            else traj
        )

        action_mu, action_std = self.actor(state, traj)
        pi_dist = Normal(action_mu, action_std)

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

        return action, log_prob

    @torch.no_grad()
    def select_action(self, state, traj, training):
        action, _ = self(state, traj, training)
        return action.cpu().data.numpy().flatten()

    def update(self, state, action, next_state, reward, done, task_id):
        self.trans_buffer.insert(state, action, next_state, reward, done, task_id)
        if (
            self.trans_buffer.size[task_id] < self.start_timesteps
            or self.trans_buffer.size[task_id] < self.batch_size
            or (self.trans_buffer.size[task_id] - self.start_timesteps)
            % self.updates_per_step
            != 0
        ):
            return None

        # updata param
        self.info = dict()
        for _ in range(self.updates_per_step):
            states, actions, next_states, rewards, masks = self.trans_buffer.random_sample(
                task_id, self.batch_size
            )
            traj = self.traj_buffer.random_sample(task_id)
            traj = preprocess_traj(traj, self.batch_size, self.device)
            # calculate target q value
            with torch.no_grad():
                next_actions, next_log_pis = self(
                    next_states, traj, training=True, calcu_log_prob=True
                )
                target_Q1, target_Q2 = (
                    self.critic_1_target(next_states, traj, next_actions),
                    self.critic_2_target(next_states, traj, next_actions),
                )
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pis
                target_Q = rewards + masks * self.gamma * target_Q

                target_Q = torch.clip(target_Q, -200, 200)  # hyper-param for clipping Q

            # update critic
            current_Q1, current_Q2 = (
                self.critic_1(states, traj, actions),
                self.critic_2(states, traj, actions),
            )
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # update actor
            self.critic_1.eval(), self.critic_2.eval()  # Freeze Q-networks to save computational effort

            pred_actions, pred_log_pis = self(
                states, traj, training=True, calcu_log_prob=True
            )
            current_Q1, current_Q2 = (
                self.critic_1(states, traj, pred_actions),
                self.critic_2(states, traj, pred_actions),
            )
            actor_loss = (
                self.alpha * pred_log_pis - torch.min(current_Q1, current_Q2)
            ).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            self.critic_1.train(), self.critic_2.train()

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

    def preprocess_data(self, task_nums):
        """
        Preprocess demos into buffer
        """
        for i in range(task_nums):
            demo_path = os.path.join("sac_maze", "demo", "demo_" + str(i) + ".csv")
            demo_traj = get_demo(demo_path)
            demo_traj = demo_traj.reshape(
                demo_traj.shape[0], -1
            )  # [traj_len, state_dim + action_dim]
            self.traj_buffer.insert(i, demo_traj)

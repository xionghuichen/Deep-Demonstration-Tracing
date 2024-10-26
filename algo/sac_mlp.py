import os
import copy
from itertools import chain
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from model.actor import MLPGaussianActor
from model.critic import Critic
from buffer import TransitionBuffer
from torch_utils import soft_update

# used


class SACAgent(nn.Module):
    """
    Soft Actor Critic
    """

    def __init__(self, configs: dict):
        super().__init__()
        # init hyper param
        ## env
        self.state_dim = configs["state_dim"]
        self.action_dim = configs["action_dim"]
        self.action_high = configs["action_high"]
        self.action_low = configs["action_low"]
        self.device = configs["device"]
        ## algo
        self.gamma = configs["gamma"]
        self.learn_alpha = configs["learn_alpha"]
        self.start_timesteps = configs["start_timesteps"]
        self.start_epi = configs["start_epi"]
        self.updates_per_step = configs["updates_per_step"]
        self.batch_size = configs["batch_size"]
        self.entropy_target = -self.action_dim
        self.rho = configs["rho"]
        self.buffer_size = configs["buffer_size"]
        ## alpha
        self.alpha_lr = configs["alpha_lr"]
        self.alpha = configs["alpha"]
        ## actor
        self.actor_hidden_size = configs["actor_hidden_size"]
        self.actor_lr = configs["actor_lr"]
        ## critic
        self.critic_hidden_size = configs["critic_hidden_size"]
        self.critic_lr = configs["critic_lr"]
        ## for tensorboard
        self.info = dict()

        ## for reward
        self.action_weight = configs["action_weight"]

        ## for pretrain
        self.batch_with_demo = configs["batch_with_demo"]
        self.env_name = configs["env_name"]
        self.with_local_view = True
        self.scale = configs["scale"]
        self.add_bc_reward = True
        self.do_scale = True

        # for net
        self.share_state_encoder = configs["share_state_encoder"]

        # imitation
        self.with_distance_reward = configs["with_distance_reward"]
        self.distance_weight = configs["distance_weight"]

        self.no_coordinate = configs["no_coordinate"]

    def init_component(self):
        """
        Initialize essential components
        """
        # replay buffer
        self.trans_buffer = TransitionBuffer(
            self.state_dim, self.action_dim, self.buffer_size, self.device
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
        self.actor = MLPGaussianActor(
            self.state_dim, self.actor_hidden_size, self.action_dim, nn.ReLU
        ).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Q1
        self.critic_1 = Critic(
            self.state_dim, self.critic_hidden_size, self.action_dim, nn.ReLU
        ).to(self.device)
        self.critic_1_target = copy.deepcopy(self.critic_1)
        # Q2
        self.critic_2 = Critic(
            self.state_dim, self.critic_hidden_size, self.action_dim, nn.ReLU
        ).to(self.device)
        self.critic_2_target = copy.deepcopy(self.critic_2)
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

    def forward(self, state, training, calcu_log_prob=False):
        state = (
            torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            if type(state) == np.ndarray
            else state
        )
        action_mu, action_std = self.actor(state)
        pi_dist = Normal(action_mu, action_std)
        self.info.update(
            {
                "action_std": action_std.mean().item(),
                "action_mu": action_mu.mean().item(),
            }
        )

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

            # log_prob -= torch.sum(
            #     np.log(2.0 / (self.action_high - self.action_low))
            #     * torch.ones_like(action),
            #     axis=-1,
            #     keepdim=True,
            # )  # for action reshaping from [-1, 1] into [action_low, action_high]

            log_prob -= torch.sum(
                np.log(1.0 / self.action_high) * torch.ones_like(action),
                axis=-1,
                keepdim=True,
            )  # for action reshaped from [-1, 1] to [action_low, action_high]

        else:
            log_prob = None

        action = self.squash_action(action)
        return action, log_prob

    @torch.no_grad()
    def select_action(self, state, training):
        action, _ = self(state, training)
        return action.cpu().data.numpy().flatten()

    def squash_action(self, action):
        action = self.action_high * torch.tanh(
            action
        )  # [-inf, +inf] -> [-1.0, 1.0] -> [-1.0-eps, +1.0+eps]
        # action = (self.action_high - self.action_low) / 2.0 * action + (
        #     self.action_high + self.action_low
        # ) / 2.0
        return action

    def update(self, state, action, next_state, reward, done):
        self.trans_buffer.insert(state, action, next_state, reward, done)
        if (
            self.trans_buffer.size < self.start_timesteps
            or self.trans_buffer.size < self.batch_size
            or (self.trans_buffer.size - self.start_timesteps) % self.updates_per_step
            != 0
        ):
            return None

        # updata param
        self.info = dict()
        for _ in range(self.updates_per_step):
            states, actions, next_states, rewards, masks = (
                self.trans_buffer.random_sample(self.batch_size)
            )
            # calculate target q value
            with torch.no_grad():
                next_actions, next_log_pis = self(
                    next_states, training=True, calcu_log_prob=True
                )
                target_Q1, target_Q2 = (
                    self.critic_1_target(next_states, next_actions),
                    self.critic_2_target(next_states, next_actions),
                )
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pis
                target_Q = rewards + masks * self.gamma * target_Q

            # update critic
            current_Q1, current_Q2 = (
                self.critic_1(states, actions),
                self.critic_2(states, actions),
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
                states, training=True, calcu_log_prob=True
            )
            current_Q1, current_Q2 = (
                self.critic_1(states, pred_actions),
                self.critic_2(states, pred_actions),
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

    def save_model(self, model_path):
        if not self.models:
            raise ValueError("Models to be saved is None!")
        state_dicts = {}
        for model in self.models:
            if isinstance(self.models[model], torch.Tensor):
                state_dicts[model] = {model: self.models[model]}
            else:
                state_dicts[model] = self.models[model].state_dict()
        torch.save(state_dicts, model_path)
        return state_dicts

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found: {}".format(model_path))
        else:
            state_dicts = torch.load(model_path)
            for model in self.models:
                if isinstance(
                    self.models[model], torch.Tensor
                ):  # especially for sac, which has log_alpha to be loaded
                    self.models[model] = state_dicts[model][model]
                else:
                    self.models[model].load_state_dict(state_dicts[model])
        self.log_alpha.data = self.models["log_alpha"].data
        self.alpha = self.log_alpha.clone().detach().exp().item()

        print(f"Successfully load model from {model_path}!")

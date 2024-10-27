import os
import copy
from itertools import chain
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from model.actor import AttnGaussianActor
from model.critic import AttnCritic
from algo.sac_mlp import SACAgent
from buffer import MT_TransitionBuffer, TrajectoryBuffer, save_buffer
from torch_utils import soft_update, preprocess_traj

# from rl.utils.pretrain import get_demo
from RLA import logger


# used
class MT_AttnSACAgent(SACAgent):
    """
    Soft Actor Critic
    """

    def __init__(self, configs: dict):
        super(MT_AttnSACAgent, self).__init__(configs)
        # hyper param
        self.task_nums = configs["task_nums"]
        ## actor
        self.actor_embed_dim = configs["actor_embed_dim"]
        self.actor_num_heads = configs["actor_num_heads"]
        self.actor_dropout = configs["actor_dropout"]
        self.actor_pos_encode = configs["actor_pos_encode"]

        ## critic
        self.critic_embed_dim = configs["critic_embed_dim"]
        self.critic_num_heads = configs["critic_num_heads"]
        self.critic_dropout = configs["critic_dropout"]
        self.critic_pos_encode = configs["critic_pos_encode"]
        self.alpha_init = configs["alpha"]

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
            self.task_nums,
            self.env_name,
            self.device,
            self.action_weight,
            self.scale,
            self.add_bc_reward,
            self.do_scale,
            env_creator=self.env_creator,
        )

        # alpha, the entropy coefficient
        self.log_alpha = torch.log(
            torch.tensor(self.alpha_init, device=self.device, requires_grad=True)
        )
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
            "embed_dim": self.actor_embed_dim,
            "num_heads": self.actor_num_heads,
            "dropout": self.actor_dropout,
            "pos_encode": self.actor_pos_encode,
            "state_std_independent": False,
        }
        self.actor = AttnGaussianActor(**kwargs).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Q1
        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "embed_dim": self.critic_embed_dim,
            "num_heads": self.critic_num_heads,
            "dropout": self.critic_dropout,
            "pos_encode": self.critic_pos_encode,
            "output_dim": 1,
        }
        self.critic_1 = AttnCritic(**kwargs).to(self.device)
        # self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_target = AttnCritic(**kwargs).to(self.device)
        soft_update(1.0, self.critic_1, self.critic_1_target)
        # Q2
        self.critic_2 = AttnCritic(**kwargs).to(self.device)
        # self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_target = AttnCritic(**kwargs).to(self.device)
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
        traj = preprocess_traj(traj, self.device) if type(traj) == np.ndarray else traj
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
            states, actions, next_states, rewards, masks = (
                self.trans_buffer.random_sample(task_id, self.batch_size)
            )
            idx, traj = self.traj_buffer.random_sample(task_id)

            # add demo to batch
            if self.batch_with_demo:
                (
                    demo_states,
                    demo_actions,
                    demo_next_states,
                    demo_rewards,
                    demo_masks,
                ) = self.traj_buffer.get_trans(task_id, idx)
                states = torch.cat((states, demo_states), dim=0)
                actions = torch.cat((actions, demo_actions), dim=0)
                next_states = torch.cat((next_states, demo_next_states), dim=0)
                rewards = torch.cat((rewards, demo_rewards), dim=0)
                masks = torch.cat((masks, demo_masks), dim=0)

            traj = preprocess_traj(traj, self.device)

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


    def load_traj(self, task_ids, demo_trajs, buffer_dir, reuse_existing_buf):
        # not support multi map
        raise NotImplementedError
        gen_traj_buffer = []
        skip_traj_buffer = []
        for task_id in task_ids:
            buf_path = os.path.join(buffer_dir, str(task_id) + "_traj")
            if os.path.exists(buf_path) and reuse_existing_buf:
                skip_traj_buffer.append(task_id)
                continue
            else:
                self.traj_buffer.insert(0, demo_trajs[task_id])
                save_buffer(self.traj_buffer, buf_path)
                self.traj_buffer.clear()
                gen_traj_buffer.append(task_id)
        logger.info(f"gen traj buffer {len(gen_traj_buffer)}", gen_traj_buffer)
        logger.info(f"reuse traj buffer {len(skip_traj_buffer)}", skip_traj_buffer)

# Created by xionghuichen at 2022/9/21
# Email: chenxh@lamda.nju.edu.cn
import sys
from itertools import chain
import numpy as np
import torch
import torch.optim as optim
from torch.distributions.normal import Normal

from algo.base_agent import BaseAgent
from model.actor_multi_task import (
    TransformerGaussianMultiTaskActor,
    AttentionGaussianMultiTaskActor,
    AttentionDecoderMultiTaskActor,
    TransformerRNNMultiTaskActor,
)
from model.critic_multi_task import GoalMapMLPMultiTaskCritic, AttenMultiTaskCritic
import torch.nn.functional as F
from torch_utils import soft_update, preprocess_traj

# used
class MT_GoalMapPolicyMultiTaskSACAgent(BaseAgent):
    """
    Soft Actor Critic
    """

    def __init__(self, configs: dict):
        super(MT_GoalMapPolicyMultiTaskSACAgent, self).__init__(configs)

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
            }
        )

        return action, log_prob

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

# Created by xionghuichen at 2022/9/21
# Email: chenxh@lamda.nju.edu.cn
from itertools import chain
import numpy as np
import torch
import torch.optim as optim

from model.critic import GoalMapMLPCritic
from algo.sac_goalpolicycritic import MT_GoalPolicySACAgent
from torch_utils import soft_update
import torch.nn.functional as F


# used
class MT_GoalMapPolicySACAgent(MT_GoalPolicySACAgent):
    """
    Soft Actor Critic
    """

    def __init__(self, configs: dict):
        super(MT_GoalMapPolicySACAgent, self).__init__(configs)
        self.map_fig_dict = configs["map_fig_dict"]
        self.map_type = configs["map_type"]
        self.map_shape = configs["map_shape"]

    def init_critic(self):
        # Q1
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

        self.critic_1 = GoalMapMLPCritic(**kwargs).to(self.device)
        self.critic_1_target = GoalMapMLPCritic(**kwargs).to(self.device)
        soft_update(1.0, self.critic_1, self.critic_1_target)
        # Q2
        self.critic_2 = GoalMapMLPCritic(**kwargs).to(self.device)
        self.critic_2_target = GoalMapMLPCritic(**kwargs).to(self.device)
        soft_update(1.0, self.critic_2, self.critic_2_target)

        self.critic_optim = optim.RMSprop(
            chain(self.critic_1.parameters(), self.critic_2.parameters()),
            lr=self.critic_lr,
        )

    def update(self, state, action, next_state, reward, done, task_id, map_id):
        if (
            self.trans_buffer.size[task_id] < self.start_timesteps
            or self.trans_buffer.size[task_id] < self.batch_size
            or (self.trans_buffer.size[task_id] - self.start_timesteps)
            % self.updates_per_step
            != 0
        ):
            return None

        if self.map_type == MapType.ID:
            map_info = torch.tensor(map_id, dtype=torch.int32).to(self.device).long()
        elif self.map_type == MapType.FIG:
            map_info = torch.from_numpy(
                (self.map_fig_dict[map_id] / 255).astype(np.float32)
            ).to(self.device)
        else:
            raise RuntimeError

        # updata param
        self.info = dict()
        for _ in range(self.updates_per_step):
            states, actions, next_states, rewards, masks, traj = self.get_samples(
                task_id
            )
            # critic traj
            with torch.no_grad():
                demo_states = traj.transpose(1, 2)[0, :, : self.state_dim]
                demo_next_actions, demo_next_log_pis = self(
                    demo_states, traj, training=True, calcu_log_prob=True
                )
                critic_traj = (
                    torch.cat((demo_states, demo_next_actions), dim=-1)
                    .unsqueeze(axis=0)
                    .transpose(1, 2)
                )

            # calculate target q valued
            with torch.no_grad():
                next_actions, next_log_pis = self(
                    next_states, traj, training=True, calcu_log_prob=True
                )

                target_Q1, target_Q2 = (
                    self.critic_1_target(
                        next_states, critic_traj, next_actions, map_info
                    ),
                    self.critic_2_target(
                        next_states, critic_traj, next_actions, map_info
                    ),
                )
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pis
                target_Q = rewards + masks * self.gamma * target_Q
                target_Q = torch.clip(target_Q, -200, 200)  # hyper-param for clipping Q

            # update critic
            current_Q1, current_Q2 = (
                self.critic_1(states, critic_traj, actions, map_info),
                self.critic_2(states, critic_traj, actions, map_info),
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
                self.critic_1(states, critic_traj, pred_actions, map_info),
                self.critic_2(states, critic_traj, pred_actions, map_info),
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

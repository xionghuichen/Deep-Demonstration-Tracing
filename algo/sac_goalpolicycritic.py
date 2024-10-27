# Created by xionghuichen at 2022/9/21
# Email: chenxh@lamda.nju.edu.cn
from itertools import chain

import torch
import torch.optim as optim

from model.actor import TransformerGaussianActor
from model.critic import GoalMLPCritic
from algo.sac_goalcritic import MT_GoalSACAgent
from torch_utils import soft_update
import torch.nn.functional as F
from torch_utils import preprocess_traj


# used
class MT_GoalPolicySACAgent(MT_GoalSACAgent):
    """
    Soft Actor Critic
    """

    def __init__(self, configs: dict):
        super(MT_GoalPolicySACAgent, self).__init__(configs)
        self.replace_sample = True

    def get_demo_trans(self, task_id, traj_buffer, demo_num=-1):
        if traj_buffer is None:
            traj_buffer = self.traj_buffer
        idx, traj = traj_buffer.random_sample(task_id)
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
        else:
            raise NotImplementedError

        traj = preprocess_traj(traj, self.device)
        return demo_states, demo_actions, traj

    def get_samples(self, task_id, trans_buffer=None, traj_buffer=None, demo_num=-1):
        if trans_buffer is None:
            trans_buffer = self.trans_buffer
        if traj_buffer is None:
            traj_buffer = self.traj_buffer

        states, actions, next_states, rewards, masks = trans_buffer.random_sample(
            task_id, self.batch_size, replace_sample=self.replace_sample
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
            if demo_num >= 0:
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

    def update(self, state, action, next_state, reward, done, task_id, map_id):
        if (
            self.trans_buffer.size[task_id] < self.start_timesteps
            or self.trans_buffer.size[task_id] < self.batch_size
            or (self.trans_buffer.size[task_id] - self.start_timesteps)
            % self.updates_per_step
            != 0
        ):
            return None
        map_id = torch.tensor(map_id, dtype=torch.int32).to(self.device).long()
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
                        next_states, critic_traj, next_actions, map_id
                    ),
                    self.critic_2_target(
                        next_states, critic_traj, next_actions, map_id
                    ),
                )
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pis
                target_Q = rewards + masks * self.gamma * target_Q
                target_Q = torch.clip(target_Q, -200, 200)  # hyper-param for clipping Q

            # update critic
            current_Q1, current_Q2 = (
                self.critic_1(states, critic_traj, actions, map_id),
                self.critic_2(states, critic_traj, actions, map_id),
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
                self.critic_1(states, critic_traj, pred_actions, map_id),
                self.critic_2(states, critic_traj, pred_actions, map_id),
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

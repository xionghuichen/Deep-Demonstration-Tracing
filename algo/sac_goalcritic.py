from itertools import chain

import torch
import torch.optim as optim

from model.actor import TransformerGaussianActor
from model.critic import GoalMLPCritic
from algo.sac_attn import MT_AttnSACAgent
from buffer import MT_TransitionBuffer, TrajectoryBuffer
from torch_utils import soft_update

# used


class MT_GoalSACAgent(MT_AttnSACAgent):
    """
    Soft Actor Critic
    """

    def __init__(self, configs: dict):
        super(MT_GoalSACAgent, self).__init__(configs)
        ## actor
        self.actor_num_encoder_layers = configs["actor_num_encoder_layers"]
        self.actor_num_decoder_layers = configs["actor_num_decoder_layers"]
        self.actor_dim_feedforward = configs["actor_dim_feedforward"]

        ## critic
        self.critic_goal_embed_dim = configs["critic_goal_embed_dim"]
        self.critic_embed_goal = configs["critic_embed_goal"]
        self.alpha_init = configs["alpha"]

        self.critic_num_encoder_layers = configs["critic_num_encoder_layers"]
        self.critic_num_decoder_layers = configs["critic_num_decoder_layers"]
        self.critic_dim_feedforward = configs["critic_dim_feedforward"]

        self.distance_weight = configs["distance_weight"]
        self.reward_fun_type = configs["reward_fun_type"]
        self.max_space_dist = configs["max_space_dist"]
        # self.env_creator = configs["env_creator"]
        self.map_num = configs["map_num"]
        self.no_itor = configs["no_itor"]
        self.actor_cls = TransformerGaussianActor

    def init_component(self, env_handler, demo_collect_env):
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
            self.with_local_view,
            self.scale,
            self.add_bc_reward,
            self.do_scale,
            self.with_distance_reward,
            self.distance_weight,
            env_handler=env_handler,
            demo_collect_env=demo_collect_env,
            max_space_dist=self.max_space_dist,
            reward_fun_type=self.reward_fun_type,
            no_itor=self.no_itor,
        )

        # alpha, the entropy coefficient
        print(self.alpha_init)
        self.log_alpha = torch.log(
            torch.ones(1, device=self.device) * self.alpha_init
        ).requires_grad_(True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        with torch.no_grad():
            if self.learn_alpha:
                self.alpha = self.log_alpha.exp().item()
            else:
                self.alpha = self.alpha
        self.init_critic()
        self.init_policy()

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

    def init_critic(self):
        # Q1
        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "goal_embed_dim": self.critic_goal_embed_dim,
            "embed_goal": self.critic_embed_goal,
            "output_dim": 1,
            "hidden_size": self.critic_hidden_size,
            # "map_num": self.map_num,
        }
        self.critic_1 = GoalMLPCritic(**kwargs).to(self.device)
        self.critic_1_target = GoalMLPCritic(**kwargs).to(self.device)
        soft_update(1.0, self.critic_1, self.critic_1_target)
        # Q2
        self.critic_2 = GoalMLPCritic(**kwargs).to(self.device)
        self.critic_2_target = GoalMLPCritic(**kwargs).to(self.device)
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
            "no_coordinate": self.no_coordinate,
        }
        self.actor = TransformerGaussianActor(**kwargs).to(self.device)
        self.actor_optim = optim.RMSprop(self.actor.parameters(), lr=self.actor_lr)

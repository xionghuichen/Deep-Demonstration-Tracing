from itertools import chain

import torch
import torch.optim as optim

from rl.model.actor import TransformerGaussianActor
from rl.model.critic import TransformerCritic
from rl.algo.sac_attn import MT_AttnSACAgent
from rl.utils.buffer import MT_TransitionBuffer, TrajectoryBuffer
from rl.utils.net import soft_update


class MT_TransformerSACAgent(MT_AttnSACAgent):
    """
    Soft Actor Critic
    """

    def __init__(self, configs: dict):
        super(MT_TransformerSACAgent, self).__init__(configs)
        ## actor
        self.actor_num_encoder_layers = configs["actor_num_encoder_layers"]
        self.actor_num_decoder_layers = configs["actor_num_decoder_layers"]
        self.actor_dim_feedforward = configs["actor_dim_feedforward"]
        ## critic
        self.critic_num_encoder_layers = configs["critic_num_encoder_layers"]
        self.critic_num_decoder_layers = configs["critic_num_decoder_layers"]
        self.critic_dim_feedforward = configs["critic_dim_feedforward"]

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
            self.with_local_view,
            self.scale,
            self.add_bc_reward,
            self.do_scale,
            self.with_distance_reward,
            self.distance_weight,
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
            "embed_dim": self.actor_embed_dim,
            "num_heads": self.actor_num_heads,
            "num_encoder_layers": self.actor_num_encoder_layers,
            "num_decoder_layers": self.actor_num_decoder_layers,
            "dim_feedforward": self.actor_dim_feedforward,
            "dropout": self.actor_dropout,
            "pos_encode": self.actor_pos_encode,
            "state_std_independent": False,
            "share_state_encoder": self.share_state_encoder,
        }
        self.actor = TransformerGaussianActor(**kwargs).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        # Q1
        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "embed_dim": self.critic_embed_dim,
            "num_heads": self.critic_num_heads,
            "num_encoder_layers": self.critic_num_encoder_layers,
            "num_decoder_layers": self.critic_num_decoder_layers,
            "dim_feedforward": self.critic_dim_feedforward,
            "dropout": self.critic_dropout,
            "pos_encode": self.critic_pos_encode,
            "output_dim": 1,
            "share_state_encoder": self.share_state_encoder,
        }
        self.critic_1 = TransformerCritic(**kwargs).to(self.device)
        # self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_1_target = TransformerCritic(**kwargs).to(self.device)
        soft_update(1.0, self.critic_1, self.critic_1_target)
        # Q2
        self.critic_2 = TransformerCritic(**kwargs).to(self.device)
        # self.critic_2_target = copy.deepcopy(self.critic_2)
        self.critic_2_target = TransformerCritic(**kwargs).to(self.device)
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

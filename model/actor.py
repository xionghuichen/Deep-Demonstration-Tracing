from os import stat
import torch
import torch.nn as nn
import numpy as np
import copy
from torch_utils import build_mlp_extractor, weight_init
from model.attention import MultiheadAttention, PositionalEncoding
from abc import abstractmethod

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class BasicActor(nn.Module):
    def __init__(self, state_dim, action_dim, coor_dim=2, no_coordinate=False) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.coor_dim = coor_dim
        self.no_coordinate = no_coordinate

    def state_processor(self, input_tensor, with_acs=False):
        if self.no_coordinate:
            if with_acs:
                input_tensor[..., -self.coor_dim :] = 0
            else:
                input_tensor[..., self.state_dim - self.coor_dim : self.state_dim] = 0
        return input_tensor

    # @abstractmethod
    # def process_input(self,*args,**kwargs):
    #     pass
    #
    # @abstractmethod
    # def _forward(self,*args,**kwargs):
    #     pass
    # #
    # def forward(self,*args,**kwargs):
    #     data=self.process_input(*args,**kwargs)# tuple
    #     return self._forward(*data)


class MLPGaussianActor(BasicActor):
    def __init__(
        self,
        state_dim,
        hidden_size,
        action_dim,
        activation_fn=nn.LeakyReLU,
        state_std_independent=False,
    ):
        super().__init__(state_dim, action_dim)
        self.state_std_independent = state_std_independent

        # feature extractor
        self.feature_extractor = nn.Sequential(
            *build_mlp_extractor(state_dim, hidden_size, activation_fn)
        )

        if len(hidden_size) > 0:
            input_dim = hidden_size[-1]
        else:
            input_dim = state_dim

        # mean and log std
        self.mu = nn.Linear(input_dim, action_dim)
        if state_std_independent:
            self.log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        else:
            self.log_std = nn.Linear(input_dim, action_dim)

        self.apply(weight_init)

    def forward(self, state):
        feature = self.feature_extractor(state)
        action_mean = self.mu(feature)
        action_log_std = (
            self.log_std if self.state_std_independent else self.log_std(feature)
        )
        action_log_std = torch.clamp(action_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return action_mean, action_log_std.exp()


class AttnGaussianActor(BasicActor):
    def __init__(
        self,
        state_dim,
        action_dim,
        embed_dim,
        num_heads,
        dropout=0.0,
        pos_encode=True,
        state_std_independent=False,
    ):
        super().__init__(state_dim, action_dim)
        self.state_std_independent = state_std_independent

        # multi-head attention
        input_dim = state_dim + action_dim
        self.pos_encode = (
            PositionalEncoding(input_dim, dropout=dropout)
            if pos_encode
            else lambda x: x
        )
        self.task_mh_attn = MultiheadAttention(
            query_dim=input_dim,
            key_dim=input_dim,
            value_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm_1 = nn.LayerNorm(embed_dim)

        self.state_mh_attn = MultiheadAttention(
            query_dim=embed_dim,
            key_dim=embed_dim,
            value_dim=embed_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm_2 = nn.LayerNorm(embed_dim)

        # state encoder
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, embed_dim),
            nn.LeakyReLU(),
        )

        # mean and log std
        self.mu = nn.Linear(embed_dim, action_dim)
        if state_std_independent:
            self.log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        else:
            self.log_std = nn.Linear(embed_dim, action_dim)

    def get_task_embed(self, traj, batch_size):
        traj = traj.transpose(1, 2)  # [1, seq_len, state_dim + action_dim]
        traj = traj.repeat(
            batch_size, 1, 1
        )  # [batch_size, seq_len, state_dim + action_dim]
        traj = self.pos_encode(traj)
        task_context, _ = self.task_mh_attn(traj, traj, traj)  # (query, key, value)
        task_context = self.norm_1(task_context)
        return task_context

    def forward(self, state, traj):
        """
        state: [batch_size, state_dim]
        traj: [1, state_dim + action_dim, seq_len]
        """

        # state embedding
        state_embed = self.feature_extractor(state).unsqueeze(dim=1)

        # task embedding
        # task_context, _ = self.task_mh_attn(traj, traj, traj)  # (query, key, value)
        # task_context = self.norm_1(task_context)
        task_context = self.get_task_embed(traj, state.shape[0])

        embed, _ = self.state_mh_attn(state_embed, task_context, task_context)
        embed = self.norm_2(embed + state_embed).squeeze(dim=1)

        # get action
        action_mean = self.mu(embed)
        action_log_std = (
            self.log_std if self.state_std_independent else self.log_std(embed)
        )
        action_log_std = torch.clamp(action_log_std, LOG_STD_MIN, LOG_STD_MAX)

        return action_mean, action_log_std.exp()


class TransformerGaussianActor(BasicActor):
    def __init__(
        self,
        state_dim,
        action_dim,
        embed_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout=0.1,
        pos_encode=True,
        state_std_independent=False,
        share_state_encoder=False,
        no_coordinate=False,
    ):
        super().__init__(state_dim, action_dim, no_coordinate)
        self.state_std_independent = state_std_independent
        self.share_state_encoder = share_state_encoder
        # input_dim = state_dim + action_dim
        self.with_goal = True
        self.no_coordinate = no_coordinate
        if self.with_goal:
            input_dim = state_dim + action_dim + state_dim
        else:
            input_dim = state_dim
        # traj encoder
        if self.share_state_encoder:
            self.traj_encoder = nn.Sequential(
                nn.Linear(embed_dim + action_dim, embed_dim),
                nn.LeakyReLU(),
                nn.Linear(embed_dim, embed_dim),
                nn.LeakyReLU(),
            )
        else:
            self.traj_encoder = nn.Sequential(
                nn.Linear(input_dim, embed_dim), nn.LeakyReLU()
            )
        self.embed_dim = embed_dim
        # Transformer
        self.transformer = nn.Transformer(
            embed_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
            activation=nn.LeakyReLU(),
        )

        # position encoding
        self.pos_encode = (
            PositionalEncoding(embed_dim, dropout=dropout)
            if pos_encode
            else lambda x: x
        )

        # state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, embed_dim),
            nn.LeakyReLU(),
            # nn.Linear(embed_dim, embed_dim), nn.LeakyReLU(),
        )

        # mean and log std
        self.mu = nn.Linear(embed_dim, action_dim)
        if state_std_independent:
            self.log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        else:
            self.log_std = nn.Linear(embed_dim, action_dim)

        # tmp add for rnn actor
        # rnn_hidden = 256
        # self.embed_dim = embed_dim
        # self.rnn_layer = nn.GRU(input_size=state_dim + action_dim,  hidden_size=rnn_hidden, num_layers=3, batch_first=True)

        # self.rnn_encoder = nn.Sequential(
        #     nn.Linear(rnn_hidden, embed_dim), nn.Tanh()
        # )
        # self.mu = nn.Linear(embed_dim * 2, action_dim)
        # if state_std_independent:
        #     self.log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        # else:
        #     self.log_std = nn.Linear(embed_dim * 2, action_dim)

    def _preprocess_traj(self, traj, batch_size):
        traj = traj.transpose(1, 2)  # [1, seq_len, state_dim + action_dim]
        traj = self.state_processor(traj, with_acs=True)
        goal = traj[:, -1, :]  # [1, state_dim + action_dim]
        if self.share_state_encoder:
            # traj and state share state encoder
            states, actions = traj[:, :, : self.state_dim], traj[:, :, self.state_dim :]
            if self.with_goal:
                states = torch.cat(
                    (states, goal.repeat(states.shape[1], 1).unsqueeze(dim=0)), dim=-1
                )
            else:
                states = states
            states = self.state_encoder(states)
            # traj = torch.cat((states, actions), dim=-1)
            traj = states
        traj = traj.repeat(
            batch_size, 1, 1
        )  # [batch_size, seq_len, state_dim + action_dim]
        return traj, goal

    def get_task_embed(self, traj, batch_size):
        traj, goal = self._preprocess_traj(traj, batch_size)
        task_context = self.transformer.encoder(traj)  # (query, key, value)
        return task_context

    def forward(self, state, traj):
        """
        state: [batch_size, state_dim]
        traj: [1, state_dim + action_dim, seq_len]
        """
        state = self.state_processor(state)
        batch_size = state.shape[0]
        # preprocess traj and state

        # repeat traj version
        traj, goal = self._preprocess_traj(traj, batch_size)
        traj = traj[0:1, :, :]
        if self.with_goal:
            input_ = torch.cat([state, goal.repeat(batch_size, 1)], dim=-1)
        else:
            input_ = state
        input_ = self.state_encoder(input_).unsqueeze(dim=1)
        mem = self.transformer.encoder(traj)
        mem = mem.repeat(batch_size, 1, 1)
        embed = self.transformer.decoder(input_, mem).squeeze(dim=1)
        # get action
        action_mean = self.mu(embed)
        action_log_std = (
            self.log_std if self.state_std_independent else self.log_std(embed)
        )
        action_log_std = torch.clamp(action_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return action_mean, action_log_std.exp()


class TransformerGaussianMultiTaskActor(TransformerGaussianActor):

    def __init__(self, *args, **kwargs):
        super(TransformerGaussianMultiTaskActor, self).__init__(*args, **kwargs)

    def _mutli_task_preprocess_traj(self, traj, goal, batch_size):
        traj = traj.transpose(1, 2)  # [task_size, seq_len, state_dim + action_dim]
        traj = self.state_processor(traj, with_acs=False)
        if goal is None:
            assert traj.shape[0] == 1
            goal = traj[:, -1, :]
        # goal = traj[:, -1, :]  # [task_size, state_dim + action_dim]
        if self.share_state_encoder:
            # traj and state share state encoder
            states, actions = traj[:, :, : self.state_dim], traj[:, :, self.state_dim :]
            if self.with_goal:
                states = torch.cat(
                    (
                        states,
                        torch.repeat_interleave(
                            torch.unsqueeze(goal, dim=1), states.shape[1], 1
                        ),
                    ),
                    dim=-1,
                )
            else:
                states = states
            states = self.state_encoder(states)
            traj = states
        return traj, goal

    def forward(self, state, traj, goal=None):
        """
        state: [batch_size, task_size, state_dim]
        traj: [task_size, state_dim + action_dim, seq_len]
        """
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 0)
        state = self.state_processor(state)
        batch_size = state.shape[0]
        task_size = state.shape[1]
        # preprocess traj and state
        # repeat traj version
        traj, goal = self._mutli_task_preprocess_traj(traj, goal, batch_size)
        # traj = traj[0:1, :, :]
        if self.with_goal:
            input_ = torch.cat([state, goal.repeat(batch_size, 1, 1)], dim=-1)
        else:
            input_ = state
        input_ = self.state_encoder(input_)
        # traj_reshape = traj.reshape([task_size] + list(traj.shape[2:]))
        # unsqueeze for constructing an one-step sequence.
        input_reshape = torch.unsqueeze(
            input_.reshape([batch_size * task_size] + list(input_.shape[2:])), dim=1
        )
        mem = self.transformer.encoder(traj)
        mem = mem.repeat(batch_size, 1, 1, 1).reshape(
            [batch_size * task_size] + list(mem.shape[1:])
        )
        embed = self.transformer.decoder(input_reshape, mem).squeeze(dim=1)
        # get action
        action_mean = self.mu(embed)
        action_log_std = (
            self.log_std if self.state_std_independent else self.log_std(embed)
        )
        action_log_std = torch.clamp(action_log_std, LOG_STD_MIN, LOG_STD_MAX)
        action_mean = action_mean.reshape([batch_size, task_size, self.action_dim])
        action_log_std = action_log_std.reshape(
            [batch_size, task_size, self.action_dim]
        )
        return action_mean, action_log_std.exp()

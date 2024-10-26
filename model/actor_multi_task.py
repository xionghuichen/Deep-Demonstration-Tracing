# Created by xionghuichen at 2022/12/9
# Email: chenxh@lamda.nju.edu.cn

import sys
from os import stat
import torch
import torch.nn as nn
import numpy as np
from torch_utils import build_mlp_extractor, weight_init, ROT_feature_dim
from model.attention import PositionalEncoding
from model.actor import TransformerGaussianActor, BasicActor, LOG_STD_MIN, LOG_STD_MAX
from model.transformer import Transformer, AttenRepresentationBlock


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
        if self.no_coordinate:
            state = torch.clone(state)
            state[..., -self.coor_dim :] = 0
            traj = torch.clone(traj)
            traj[:, -(self.action_dim + self.coor_dim) : -self.action_dim, :] = 0
            assert len(traj.shape) == 3
            if goal is not None:
                assert len(goal.shape) == 2
                goal = torch.clone(goal)
                goal[:, -2:] = 0

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


class TransformerRNNMultiTaskActor(TransformerGaussianMultiTaskActor):

    def __init__(self, *args, **kwargs):
        super(TransformerRNNMultiTaskActor, self).__init__(*args, **kwargs)
        if "rot_maze_tp" in sys.argv[0]:
            raise NotImplementedError("use ROTMultiTaskActor instead")
        else:
            rnn_hidden = 256
            self.rnn_layer = nn.GRU(
                input_size=self.state_dim + self.action_dim,
                hidden_size=rnn_hidden,
                num_layers=3,
                batch_first=True,
            )

        self.rnn_encoder = nn.Sequential(
            nn.Linear(rnn_hidden, self.embed_dim), nn.Tanh()
        )
        self.mu = nn.Linear(self.embed_dim * 2, self.action_dim)
        if self.state_std_independent:
            self.log_std = nn.Parameter(
                torch.zeros(1, self.action_dim), requires_grad=True
            )
        else:
            self.log_std = nn.Linear(self.embed_dim * 2, self.action_dim)

    def return_rot_contexter(self):
        return self.rnn_layer

    def forward(self, state, traj, goal=None):
        """

        state: [batch_size, task_size, state_dim]
        traj: [task_size, state_dim + action_dim, seq_len]
        """
        if self.no_coordinate:
            state = torch.clone(state)
            state[..., -self.coor_dim :] = 0
            traj = torch.clone(traj)
            traj[:, -(self.coor_dim + self.action_dim) : -self.action_dim, :] = 0
            assert len(traj.shape) == 3
            if goal is not None:
                assert len(goal.shape) == 2
                goal = torch.clone(goal)
                goal[:, -self.coor_dim :] = 0

        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 0)
        batch_size = state.shape[0]
        task_size = state.shape[1]
        # preprocess traj and state
        # repeat traj version
        # traj, goal = self._mutli_task_preprocess_traj(traj, goal, batch_size)
        traj = traj.transpose(1, 2)
        if goal is None:
            assert traj.shape[0] == 1
            goal = traj[:, -1, :]
        # traj = traj[0:1, :, :]
        if self.with_goal:
            input_ = torch.cat([state, goal.repeat(batch_size, 1, 1)], dim=-1)
        else:
            input_ = state
        input_ = self.state_encoder(input_)
        # traj_reshape = traj.reshape([task_size] + list(traj.shape[2:]))
        # unsqueeze for constructing an one-step sequence.
        input_reshape = input_.reshape(
            [batch_size * task_size] + list(input_.shape[2:])
        )
        # mem = self.transformer.encoder(traj)
        rnn_output, h_state = self.rnn_layer(traj, None)
        lst_rnn_out = rnn_output[:, -1]
        lst_rnn_out_encoder = self.rnn_encoder(lst_rnn_out)
        mem = lst_rnn_out_encoder
        mem = mem.repeat(batch_size, 1, 1).reshape(
            [batch_size * task_size] + list(mem.shape[1:])
        )

        embed = torch.cat([input_reshape, mem], axis=-1)
        # embed = self.transformer.decoder(input_reshape, mem).squeeze(dim=1)
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


from torch.nn import MultiheadAttention
from torch import Tensor
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm


class AttentionGaussianMultiTaskActor(BasicActor):
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
        super().__init__(state_dim, action_dim)
        self.state_std_independent = state_std_independent
        self.share_state_encoder = share_state_encoder
        # input_dim = state_dim + action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.with_goal = False
        self.with_task_emb = False
        self.infer_only_by_acs = True
        self.no_coordinate = no_coordinate
        if self.no_coordinate:
            print("------no coordinate!!!------")
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

        # Transformer
        act_fn = nn.LeakyReLU()
        self.transformer = Transformer(
            embed_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
            activation=act_fn,
        )

        # state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, embed_dim),
            nn.LeakyReLU(),
            # nn.Linear(embed_dim, embed_dim), nn.LeakyReLU(),
        )
        # mean_pool

        if self.with_task_emb:
            emb_dim_after_pool = self.state_dim
            pool_target = int(embed_dim / emb_dim_after_pool)
            self.avg_p = nn.AvgPool1d(pool_target, stride=pool_target)
        else:
            emb_dim_after_pool = 0
        # qkv-encoder
        module_list = []

        for i in range(num_encoder_layers):
            if i == 0:
                module_list.append(
                    nn.Sequential(
                        nn.Linear(input_dim + emb_dim_after_pool, embed_dim),
                        nn.LeakyReLU(),
                    )
                )
            else:
                module_list.append(
                    AttenRepresentationBlock(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dropout=dropout,
                        batch_first=True,
                        activation=act_fn,
                        dim_feedforward=dim_feedforward,
                    )
                )
        self.qk_encoder = nn.ModuleList(module_list)
        module_list = []
        for i in range(num_encoder_layers):
            if i == 0:
                if self.infer_only_by_acs:
                    module_list.append(
                        nn.Sequential(
                            nn.Linear(action_dim + emb_dim_after_pool, embed_dim),
                            nn.LeakyReLU(),
                        )
                    )
                else:
                    module_list.append(
                        nn.Sequential(
                            nn.Linear(
                                input_dim + action_dim + emb_dim_after_pool, embed_dim
                            ),
                            nn.LeakyReLU(),
                        )
                    )
            else:
                module_list.append(
                    AttenRepresentationBlock(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dropout=dropout,
                        batch_first=True,
                        activation=act_fn,
                        dim_feedforward=dim_feedforward,
                    )
                )
        self.v_encoder = nn.ModuleList(module_list)

        # mean and log std
        self.mu = nn.Linear(embed_dim, action_dim)
        if state_std_independent:
            self.log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        else:
            self.log_std = nn.Linear(embed_dim, action_dim)

    def _mutli_task_preprocess_traj(
        self, traj, goal, batch_size, encoding=True, with_acs=False
    ):
        traj = traj.transpose(1, 2)  # [task_size, seq_len, state_dim + action_dim]
        if goal is None:
            assert traj.shape[0] == 1
            goal = traj[:, -1, :]
        # goal = traj[:, -1, :]  # [task_size, state_dim + action_dim]
        if self.share_state_encoder:
            # traj and state share state encoder
            states, actions = traj[:, :, : self.state_dim], traj[:, :, self.state_dim :]
            if self.with_goal:
                if with_acs:
                    states = torch.cat(
                        (
                            states,
                            actions,
                            torch.repeat_interleave(
                                torch.unsqueeze(goal, dim=1), states.shape[1], 1
                            ),
                        ),
                        dim=-1,
                    )
                else:
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
                if with_acs:
                    if self.infer_only_by_acs:
                        states = actions
                    else:
                        states = traj
                else:
                    states = states
            if encoding:
                states = self.state_encoder(states)
            # traj = torch.cat((states, actions), dim=-1)
            traj = states
        return traj, goal

    def forward(self, state, traj, goal=None):
        """

        state: [batch_size, task_size, state_dim]/[batch_size, state_dim]
        traj: [task_size, state_dim + action_dim, seq_len]
        """
        if self.no_coordinate:
            state = torch.clone(state)
            state[..., -self.coor_dim :] = 0
            traj = torch.clone(traj)
            traj[:, -(self.action_dim + self.coor_dim) : -(self.action_dim), :] = 0
            assert len(traj.shape) == 3
            if goal is not None:
                assert len(goal.shape) == 2
                goal = torch.clone(goal)
                goal[:, -self.coor_dim :] = 0

        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 0)
        batch_size = state.shape[0]
        task_size = state.shape[1]
        seq_len = traj.shape[-1]
        # preprocess traj and state
        # repeat traj version
        context, goal = self._mutli_task_preprocess_traj(
            traj, goal, batch_size, encoding=True
        )
        traj_key, goal = self._mutli_task_preprocess_traj(
            traj, goal, batch_size, encoding=False
        )
        traj_value, goal = self._mutli_task_preprocess_traj(
            traj, goal, batch_size, encoding=False, with_acs=True
        )
        # traj_key: [task_size, seq_len, state_dim + action_dim]
        # traj = traj[0:1, :, :]
        if self.with_goal:
            input_ = torch.cat([state, goal.repeat(batch_size, 1, 1)], dim=-1)
        else:
            input_ = state
        # input_ = self.state_encoder(input_)
        # traj_reshape = traj.reshape([task_size] + list(traj.shape[2:]))
        # unsqueeze for constructing an one-step sequence.
        input_reshape = torch.unsqueeze(
            input_.reshape([batch_size * task_size] + list(input_.shape[2:])), dim=1
        )

        # TODO：如果不考虑离地图太远回不去的话，我们不需要做任务的表征。他只需要寻找近似状态并做自适应决策即可
        if self.with_task_emb:
            # context: [task_size, seq_len, emb_dim]
            context_emb = self.transformer.encoder(context)
            # context_emb: [task_size, emb_dim]
            context_emb = torch.mean(context_emb, dim=-2)
            # context_emb: [task_size, emb_dim_after_avg_pool]
            context_emb = torch.squeeze(
                self.avg_p(torch.unsqueeze(context_emb, 1)), dim=-2
            )
            # context_emb_q: [batch_size * task_size, 1, emb_dim_after_avg_pool]
            context_emb_q = context_emb.repeat(batch_size, 1, 1).reshape(
                [batch_size * task_size, 1] + list(context_emb.shape[1:])
            )
            # context_emb_kv: [task_size, seq_len, emb_dim_after_avg_pool]
            context_emb_kv = context_emb.repeat(seq_len, 1, 1).transpose(0, 1)
            key_input = torch.cat([traj_key, context_emb_kv], dim=-1)
            value_input = torch.cat([traj_value, context_emb_kv], dim=-1)
            query_input = torch.cat([input_reshape, context_emb_q], dim=-1)
        else:
            key_input = traj_key  # torch.cat([traj_key, context_emb_kv], dim=-1)
            value_input = traj_value  # torch.cat([traj_value, context_emb_kv], dim=-1)
            query_input = (
                input_reshape  # torch.cat([input_reshape, context_emb_q], dim=-1)
            )
        key_encoded = key_input
        for enc in self.qk_encoder:
            key_encoded = enc(key_encoded)
        query_encoded = query_input
        for enc in self.qk_encoder:
            query_encoded = enc(query_encoded)
        value_encoded = value_input
        for enc in self.v_encoder:
            value_encoded = enc(value_encoded)
        # kv_encoded: [batch_size * task_size, seq_len, emb_dim]
        # query_encoded: [batch_size * task_size, 1, emb_dim]
        key_encoded_repeated = key_encoded.repeat(batch_size, 1, 1, 1).reshape(
            [batch_size * task_size] + list(key_encoded.shape[1:])
        )
        value_encoded_repeated = value_encoded.repeat(batch_size, 1, 1, 1).reshape(
            [batch_size * task_size] + list(value_encoded.shape[1:])
        )
        # emb: [batch_size * task_size, emb_dim]
        embed, atten_wei_lst = self.transformer.decoder(
            key=key_encoded_repeated, value=value_encoded_repeated, query=query_encoded
        )
        embed = embed.squeeze(dim=1)
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

    @torch.no_grad()
    def forward_with_atten_score(self, state, traj, goal=None):
        """

        state: [batch_size, task_size, state_dim]
        traj: [task_size, state_dim + action_dim, seq_len]
        """
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 0)
        batch_size = state.shape[0]
        task_size = state.shape[1]
        seq_len = traj.shape[-1]
        # preprocess traj and state
        # repeat traj version
        context, goal = self._mutli_task_preprocess_traj(
            traj, goal, batch_size, encoding=True
        )
        traj_key, goal = self._mutli_task_preprocess_traj(
            traj, goal, batch_size, encoding=False
        )
        traj_value, goal = self._mutli_task_preprocess_traj(
            traj, goal, batch_size, encoding=False, with_acs=True
        )
        # traj = traj[0:1, :, :]
        if self.with_goal:
            input_ = torch.cat([state, goal.repeat(batch_size, 1, 1)], dim=-1)
        else:
            input_ = state
        # input_ = self.state_encoder(input_)
        # traj_reshape = traj.reshape([task_size] + list(traj.shape[2:]))
        # unsqueeze for constructing an one-step sequence.
        input_reshape = torch.unsqueeze(
            input_.reshape([batch_size * task_size] + list(input_.shape[2:])), dim=1
        )

        # TODO：如果不考虑离地图太远回不去的话，我们不需要做任务的表征。他只需要寻找近似状态并做自适应决策即可
        if self.with_task_emb:
            # context: [task_size, seq_len, emb_dim]
            context_emb = self.transformer.encoder(context)
            # context_emb: [task_size, emb_dim]
            context_emb = torch.mean(context_emb, dim=-2)
            # context_emb: [task_size, emb_dim_after_avg_pool]
            context_emb = torch.squeeze(
                self.avg_p(torch.unsqueeze(context_emb, 1)), dim=-2
            )
            # context_emb_q: [batch_size * task_size, 1, emb_dim_after_avg_pool]
            context_emb_q = context_emb.repeat(batch_size, 1, 1).reshape(
                [batch_size * task_size, 1] + list(context_emb.shape[1:])
            )
            # context_emb_kv: [task_size, seq_len, emb_dim_after_avg_pool]
            context_emb_kv = context_emb.repeat(seq_len, 1, 1).transpose(0, 1)
            key_input = torch.cat([traj_key, context_emb_kv], dim=-1)
            value_input = torch.cat([traj_value, context_emb_kv], dim=-1)
            query_input = torch.cat([input_reshape, context_emb_q], dim=-1)
        else:
            key_input = traj_key  # torch.cat([traj_key, context_emb_kv], dim=-1)
            value_input = traj_value  # torch.cat([traj_value, context_emb_kv], dim=-1)
            query_input = (
                input_reshape  # torch.cat([input_reshape, context_emb_q], dim=-1)
            )
        key_encoded = key_input
        for enc in self.qk_encoder:
            key_encoded = enc(key_encoded)
        query_encoded = query_input
        for enc in self.qk_encoder:
            query_encoded = enc(query_encoded)
        value_encoded = value_input
        for enc in self.v_encoder:
            value_encoded = enc(value_encoded)
        # kv_encoded: [batch_size * task_size, seq_len, emb_dim]
        # query_encoded: [batch_size * task_size, 1, emb_dim]
        key_encoded_repeated = key_encoded.repeat(batch_size, 1, 1, 1).reshape(
            [batch_size * task_size] + list(key_encoded.shape[1:])
        )
        value_encoded_repeated = value_encoded.repeat(batch_size, 1, 1, 1).reshape(
            [batch_size * task_size] + list(value_encoded.shape[1:])
        )
        # emb: [batch_size * task_size, emb_dim]
        embed, atten_wei_lst = self.transformer.decoder(
            key=key_encoded_repeated, value=value_encoded_repeated, query=query_encoded
        )
        embed = embed.squeeze(dim=1)
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
        return action_mean, action_log_std.exp(), atten_wei_lst


class AttentionDecoderMultiTaskActor(BasicActor):
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
    ):
        super().__init__(state_dim, action_dim)
        self.state_std_independent = state_std_independent
        self.share_state_encoder = share_state_encoder
        # input_dim = state_dim + action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.with_goal = False
        self.with_task_emb = False
        self.infer_only_by_acs = True
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

        # Transformer
        act_fn = nn.LeakyReLU()
        self.transformer = Transformer(
            embed_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
            activation=act_fn,
        )

        # state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, embed_dim),
            nn.LeakyReLU(),
            # nn.Linear(embed_dim, embed_dim), nn.LeakyReLU(),
        )
        # mean_pool

        if self.with_task_emb:
            emb_dim_after_pool = self.state_dim
            pool_target = int(embed_dim / emb_dim_after_pool)
            self.avg_p = nn.AvgPool1d(pool_target, stride=pool_target)
        else:
            emb_dim_after_pool = 0
        # qkv-encoder
        self.qk_encoder = nn.Sequential(
            nn.Linear(input_dim + emb_dim_after_pool, embed_dim), nn.LeakyReLU()
        )

        self.v_encoder = nn.Sequential(
            nn.Linear(action_dim + emb_dim_after_pool, embed_dim), nn.LeakyReLU()
        )
        if self.infer_only_by_acs:
            self.v_encoder = nn.Sequential(
                nn.Linear(action_dim + emb_dim_after_pool, embed_dim), nn.LeakyReLU()
            )
        else:
            self.v_encoder = nn.Sequential(
                nn.Linear(input_dim + action_dim + emb_dim_after_pool, embed_dim),
                nn.LeakyReLU(),
            )

        # mean and log std
        self.mu = nn.Linear(embed_dim, action_dim)
        if state_std_independent:
            self.log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        else:
            self.log_std = nn.Linear(embed_dim, action_dim)

    def _mutli_task_preprocess_traj(
        self, traj, goal, batch_size, encoding=True, with_acs=False
    ):
        traj = traj.transpose(1, 2)  # [task_size, seq_len, state_dim + action_dim]
        if goal is None:
            assert traj.shape[0] == 1
            goal = traj[:, -1, :]
        # goal = traj[:, -1, :]  # [task_size, state_dim + action_dim]
        if self.share_state_encoder:
            # traj and state share state encoder
            states, actions = traj[:, :, : self.state_dim], traj[:, :, self.state_dim :]
            if self.with_goal:
                if with_acs:
                    states = torch.cat(
                        (
                            states,
                            actions,
                            torch.repeat_interleave(
                                torch.unsqueeze(goal, dim=1), states.shape[1], 1
                            ),
                        ),
                        dim=-1,
                    )
                else:
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
                if with_acs:
                    if self.infer_only_by_acs:
                        states = actions
                    else:
                        states = traj
                else:
                    states = states
            if encoding:
                states = self.state_encoder(states)
            # traj = torch.cat((states, actions), dim=-1)
            traj = states
        # traj = traj.repeat(batch_size, 1, 1, 1)  # [batch_size, task_size， seq_len, state_dim + action_dim]
        return traj, goal  # [task_size, seq_len, state_dim]

    def forward(self, state, traj, goal=None):
        """

        state: [batch_size, task_size, state_dim]
        traj: [task_size, state_dim + action_dim, seq_len]
        """
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 0)
        batch_size = state.shape[0]
        task_size = state.shape[1]
        seq_len = traj.shape[-1]
        # preprocess traj and state
        # repeat traj version
        context, goal = self._mutli_task_preprocess_traj(
            traj, goal, batch_size, encoding=True
        )  # [task_size, seq_len, state_dim]
        traj_key, goal = self._mutli_task_preprocess_traj(
            traj, goal, batch_size, encoding=False
        )
        traj_value, goal = self._mutli_task_preprocess_traj(
            traj, goal, batch_size, encoding=False, with_acs=True
        )
        # traj = traj[0:1, :, :]
        if self.with_goal:
            input_ = torch.cat([state, goal.repeat(batch_size, 1, 1)], dim=-1)
        else:
            input_ = state
        # input_ = self.state_encoder(input_)
        # traj_reshape = traj.reshape([task_size] + list(traj.shape[2:]))
        # unsqueeze for constructing an one-step sequence.
        input_reshape = torch.unsqueeze(
            input_.reshape([batch_size * task_size] + list(input_.shape[2:])), dim=1
        )

        # TODO：如果不考虑离地图太远回不去的话，我们不需要做任务的表征。他只需要寻找近似状态并做自适应决策即可
        if self.with_task_emb:
            # context: [task_size, seq_len, emb_dim]
            context_emb = self.transformer.encoder(context)
            # context_emb: [task_size, emb_dim]
            context_emb = torch.mean(context_emb, dim=-2)
            # context_emb: [task_size, emb_dim_after_avg_pool]
            context_emb = torch.squeeze(
                self.avg_p(torch.unsqueeze(context_emb, 1)), dim=-2
            )
            # context_emb_q: [batch_size * task_size, 1, emb_dim_after_avg_pool]
            context_emb_q = context_emb.repeat(batch_size, 1, 1).reshape(
                [batch_size * task_size, 1] + list(context_emb.shape[1:])
            )
            # context_emb_kv: [task_size, seq_len, emb_dim_after_avg_pool]
            context_emb_kv = context_emb.repeat(seq_len, 1, 1).transpose(0, 1)
            key_input = torch.cat([traj_key, context_emb_kv], dim=-1)
            value_input = torch.cat([traj_value, context_emb_kv], dim=-1)
            query_input = torch.cat([input_reshape, context_emb_q], dim=-1)
        else:
            key_input = traj_key  # torch.cat([traj_key, context_emb_kv], dim=-1)
            value_input = traj_value  # torch.cat([traj_value, context_emb_kv], dim=-1)
            query_input = (
                input_reshape  # torch.cat([input_reshape, context_emb_q], dim=-1)
            )
        key_encoded = key_input
        key_encoded = self.qk_encoder(key_encoded)
        query_encoded = query_input
        query_encoded = self.qk_encoder(query_encoded)
        value_encoded = value_input
        value_encoded = self.v_encoder(value_encoded)
        # kv_encoded: [batch_size * task_size, seq_len, emb_dim]
        # query_encoded: [batch_size * task_size, 1, emb_dim]
        key_encoded_repeated = key_encoded.repeat(batch_size, 1, 1, 1).reshape(
            [batch_size * task_size] + list(key_encoded.shape[1:])
        )
        value_encoded_repeated = value_encoded.repeat(batch_size, 1, 1, 1).reshape(
            [batch_size * task_size] + list(value_encoded.shape[1:])
        )
        # emb: [batch_size * task_size, emb_dim]
        embed = self.transformer.decoder(
            key=key_encoded_repeated, value=value_encoded_repeated, query=query_encoded
        ).squeeze(dim=1)
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


class AttentionGaussianMultiTaskActor_FullTraj(BasicActor):
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
        super().__init__(state_dim, action_dim)
        self.state_dim = state_dim
        self.state_std_independent = state_std_independent
        self.share_state_encoder = share_state_encoder
        self.no_coordinate = no_coordinate
        self.with_goal = False
        self.with_task_emb = False
        self.infer_only_by_acs = True

        if self.with_goal:
            input_dim = state_dim + action_dim + state_dim + action_dim
        else:
            input_dim = state_dim + action_dim  # s_t a_{t-1

        # position encoding
        self.pos_encode = (
            PositionalEncoding(embed_dim, dropout=dropout)
            if pos_encode
            else lambda x: x
        )

        # Transformer
        act_fn = nn.LeakyReLU()
        self.transformer = Transformer(
            embed_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
            activation=act_fn,
        )

        # state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, embed_dim),
            nn.LeakyReLU(),
            # nn.Linear(embed_dim, embed_dim), nn.LeakyReLU(),
        )

        # qkv-encoder
        module_list = []

        for i in range(num_encoder_layers):
            if i == 0:
                module_list.append(
                    nn.Sequential(nn.Linear(input_dim, embed_dim), nn.LeakyReLU())
                )
            else:
                module_list.append(
                    AttenRepresentationBlock(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dropout=dropout,
                        batch_first=True,
                        activation=act_fn,
                        dim_feedforward=dim_feedforward,
                    )
                )
        self.qk_encoder = nn.ModuleList(module_list)
        module_list = []
        for i in range(num_encoder_layers):
            if i == 0:
                if self.infer_only_by_acs:
                    module_list.append(
                        nn.Sequential(nn.Linear(action_dim, embed_dim), nn.LeakyReLU())
                    )
                else:
                    module_list.append(
                        nn.Sequential(
                            nn.Linear(input_dim + action_dim, embed_dim), nn.LeakyReLU()
                        )
                    )
            else:
                module_list.append(
                    AttenRepresentationBlock(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dropout=dropout,
                        batch_first=True,
                        activation=act_fn,
                        dim_feedforward=dim_feedforward,
                    )
                )
        self.v_encoder = nn.ModuleList(module_list)

        # mean and log std
        self.mu = nn.Linear(embed_dim, action_dim)
        if state_std_independent:
            self.log_std = nn.Parameter(torch.zeros(1, action_dim), requires_grad=True)
        else:
            self.log_std = nn.Linear(embed_dim, action_dim)

    def _preprocess_traj(self, exp_traj: torch.Tensor, encoding=True, with_acs=False):
        r"""
        Args:
            exp_traj is still (s_t,a_t) (``[T,L,S+A]`` tensor) no goal

        Return:
            input_: ``[task_size, seq_len, embed_dim]`` shape tensor, with the formation (s_t,a_{t-1})
            goal: ``[task_size, state_dim + action_dim]`` shape tensor
        """
        # exp_traj is still (s_t,a_t) [task_size,L,S+A] no goal
        goal = exp_traj[:, -1, :]  # [task_size,state_dim + action_dim]
        if self.share_state_encoder:
            # traj and state share state encoder
            states, actions = (
                exp_traj[:, :, : self.state_dim],
                exp_traj[:, :, self.state_dim :],
            )

            pre_actions = torch.zeros_like(actions)
            pre_actions[:, 1:, :] = actions.clone()[:, :-1, :]

            if self.with_goal:
                if with_acs:
                    input_ = torch.cat(
                        (
                            states,
                            pre_actions,
                            goal.unsqueeze(1).repeat(1, states.shape[1], 1),
                        ),
                        dim=-1,
                    )
                else:
                    input_ = torch.cat(
                        (states, goal.unsqueeze(1).repeat(1, states.shape[1], 1)),
                        dim=-1,
                    )
                states_encoded = self.state_encoder(input_)
            else:
                if with_acs:
                    if self.infer_only_by_acs:
                        input_ = actions  # action seq, used as Key
                    else:
                        input_ = exp_traj
                else:
                    input_ = torch.cat((states, pre_actions), dim=-1)  # (s_t, a_t-1)seq
            if encoding:
                input_ = self.state_encoder(input_)

        # traj = self.traj_encoder(traj)
        # traj = self.pos_encode(traj.transpose(0, 1)).transpose(0, 1)
        # [task_size, seq_len, embed_dim]
        return input_, goal

    def get_task_embed(self, traj: torch.Tensor, batch_size):
        traj = traj.transpose(1, 2)
        traj, goal = self._preprocess_traj(traj)
        task_context = self.transformer.encoder(traj)  # (query, key, value)
        return task_context

    def forward(
        self, full_trajs: torch.Tensor, exp_traj: torch.Tensor, atten_mask=None
    ):
        r"""Multi-task version full-traj SAC agent decision procedure. Only DDT architecture implemented! ``T`` below represented
        task_dim or said task_num, ``B``->batch size, ``S``->state dim, ``A``->action dim. Note that although data is flattened in
        some way, the task dim must be put in the first position. The tgt mask is not included for we force the inference of decoder
        is auto-regression way.

        Args:
            full_trajs: seq of (s_t,a_{t-1}) ``[T*B, max_len, S + A]``/ ``[B,S]``
            exp_traj: seq of (s_t,a_t) ``[T, max_len(exp_traj), S + A]`` no goal cat!
            atten_mask: specify the valid positions of tensor exp_traj, note that None implies all valid!
                        The shape should be ``(T,max_len)``, 1 implies not valid and 0 implies valid !

        Returns:
            action_mean: ``(T*B,L,A)``
            action_std: ``(T*B,L,A)``
        """
        if self.no_coordinate:
            full_trajs = torch.clone(full_trajs)
            full_trajs[..., self.state_dim - self.coor_dim : self.state_dim] = 0
            exp_traj = torch.clone(exp_traj)
            exp_traj[..., self.state_dim - self.coor_dim : self.state_dim] = 0

        assert full_trajs.shape[-1] == exp_traj.shape[-1]

        if len(full_trajs.shape) == 2:
            full_trajs = torch.unsqueeze(full_trajs, 0)  # add task size for decision
        if len(exp_traj.shape) == 2:
            exp_traj = torch.unsqueeze(exp_traj, 0)

        T_mul_B, T = full_trajs.shape[0], exp_traj.shape[0]
        B = T_mul_B // T

        max_len_agent, max_len_exp = full_trajs.shape[1], exp_traj.shape[1]

        if atten_mask is None:
            atten_mask = torch.full(
                (exp_traj.shape[0], max_len_exp), fill_value=float(0)
            )  # all valid!

        ### preprocess traj and state

        key_input, _ = self._preprocess_traj(
            exp_traj, encoding=False
        )  # exp_traj(s_t,a_t) or actions
        value_input, _ = self._preprocess_traj(
            exp_traj, encoding=False, with_acs=True
        )  # exp_traj (s_t,a_t-1)
        # key_input: (T, L, S+A) value_input: (T, L, A)

        query_input = full_trajs  # (T*B, L, S+A)

        key_encoded = key_input
        for enc in self.qk_encoder:
            key_encoded = enc(key_encoded)  # (T,L,E)
        query_encoded = query_input
        for enc in self.qk_encoder:
            query_encoded = enc(query_encoded)  # (T*B,L,E)
        value_encoded = value_input
        for enc in self.v_encoder:
            value_encoded = enc(value_encoded)  # (T,L,E)

        key_encoded_repeated = (
            key_encoded.unsqueeze(1)
            .repeat(1, B, 1, 1)
            .reshape([-1, max_len_exp, key_encoded.shape[-1]])
        )  # (T,L,E)->(T,1,L,E)->(T,B,L,E)->(T*B,L,E)
        value_encoded_repeated = (
            value_encoded.unsqueeze(1)
            .repeat(1, B, 1, 1)
            .reshape([-1, max_len_exp, value_encoded.shape[-1]])
        )
        tgt_mask = self.transformer.generate_square_subsequent_mask(max_len_agent).to(
            key_encoded_repeated.device
        )
        # xx_encoded (T*B,L,E)

        atten_mask = (
            atten_mask.unsqueeze(1)
            .repeat(1, B, 1)
            .reshape(T * B, -1)
            .to(key_encoded_repeated.device)
        )
        # atten_mask (T,L_exp)->(T,1,L_exp)->(T,B,L_exo)->(T*B,L_exp)

        embed = self.transformer.decoder(
            key=key_encoded_repeated,
            value=value_encoded_repeated,
            query=query_encoded,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=atten_mask,
        )

        action_mean = self.mu(embed)  # (T*B,L,A)
        action_log_std = (
            self.log_std if self.state_std_independent else self.log_std(embed)
        )  # (T*B,L,A)
        action_log_std = torch.clamp(action_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return action_mean, torch.exp(action_log_std)  # (T*B,L,A) (T*B,L,A)

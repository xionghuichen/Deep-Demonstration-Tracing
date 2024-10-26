# Created by xionghuichen at 2022/12/9
# Email: chenxh@lamda.nju.edu.cn
from turtle import forward
from typing import Union
import torch
import torch.nn as nn
from torch_utils import build_mlp_extractor, weight_init
from CONST import *
import torch.nn.functional as F
from model.map_encoder import MapEncoder
from model.critic import GoalMapMLPCritic

from model.transformer import Transformer, AttenRepresentationBlock


class GoalMapMLPMultiTaskCritic(GoalMapMLPCritic):

    def __init__(self, *args, **kwargs):
        super(GoalMapMLPMultiTaskCritic, self).__init__(*args, **kwargs)

    def forward(self, state, traj, action, map_info, goal):
        """
        state: [batch_size, task_size, state_dim]
        action: [batch_size, task_size, state_dim]
        traj: [task_size, state_dim + action_dim, seq_len]
        map_info: [task_size, map_dim]
        """
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, dim=0)
            action = torch.unsqueeze(action, dim=0)
        traj = traj.detach().clone()
        traj = traj.transpose(1, 2)  # [task_size, seq_len, state_dim + action_dim]
        # goal = traj[:, -1, :].clone()  # [task_size, state_dim + action_dim]
        batch_size = state.shape[0]
        task_size = state.shape[1]

        rnn_output, h_state = self.rnn_layer(traj, None)
        lst_rnn_out = rnn_output[:, -1].repeat(batch_size, 1, 1)
        lst_rnn_out_encoder = self.rnn_encoder(lst_rnn_out)
        goal = self.goal_encoder(goal)
        goal = goal.repeat(batch_size, 1, 1)
        goal_cat = torch.cat([goal, lst_rnn_out_encoder], dim=-1)
        goal_cat_emb = self.merge_layer(goal_cat)
        if self.map_type == MapType.ID:
            map_info = (
                F.one_hot(map_info, num_classes=self.map_num)
                .repeat(batch_size, 1, 1)
                .float()
            )
        elif self.map_type == MapType.FIG:
            map_info = self.map_encoder(map_info).repeat(batch_size, 1, 1)
        else:
            raise NotImplementedError
        map_info = self.map_feature_extractor(map_info)
        x = torch.cat([goal_cat_emb, state, action, map_info], dim=-1)
        v = x
        for enc in self.value_net:
            v = enc(v)
        return v


class AttenMultiTaskCritic(nn.Module):

    def __init__(
        self,
        state_dim,
        hidden_size,
        action_dim,
        goal_embed_dim,
        embed_goal,
        map_shape,
        map_type=MapType.ID,
        activation_fn=nn.LeakyReLU,
        map_num=1,
        output_dim=1,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=256,
        atten_emb_dim=128,
        num_heads=16,
        dropout=0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        # input_dim = state_dim + action_dim
        self.value_input_dim = state_dim + action_dim + action_dim
        self.goal_input_dim = state_dim + action_dim
        self.map_num = map_num
        self.map_type = map_type
        self.map_emb_dim = goal_embed_dim

        if embed_goal:
            self.goal_encoder = nn.Sequential(
                nn.Linear(self.goal_input_dim, goal_embed_dim), nn.LeakyReLU()
            )
        else:
            goal_embed_dim = self.goal_input_dim
            self.goal_encoder = lambda x: x

        if self.map_type == MapType.FIG:
            self.map_encoder = MapEncoder(depth=256, shape=map_shape)
            self.map_feat_dim = self.map_encoder.embed_size
            self.map_feature_extractor = nn.Sequential(
                *build_mlp_extractor(
                    self.map_feat_dim, (256, self.map_emb_dim), nn.LeakyReLU
                )
            )
        else:
            self.map_feat_dim = self.map_num
            self.map_feature_extractor = nn.Sequential(
                nn.Linear(self.map_feat_dim, self.map_emb_dim), nn.LeakyReLU()
            )
        # tmp parameters assignment
        act_fn = activation_fn()
        self.transformer = Transformer(
            atten_emb_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
            activation=act_fn,
        )

        # qkv-encoder
        module_list = []
        for i in range(num_encoder_layers):
            if i == 0:
                module_list.append(
                    nn.Sequential(
                        nn.Linear(self.state_dim, atten_emb_dim), nn.LeakyReLU()
                    )
                )
            else:
                module_list.append(
                    AttenRepresentationBlock(
                        d_model=atten_emb_dim,
                        nhead=num_heads,
                        dropout=dropout,
                        batch_first=True,
                        activation=act_fn,
                        dim_feedforward=dim_feedforward,
                    )
                )
        self.qk_encoder = nn.ModuleList(module_list)

        self.value_input_encoder = nn.Sequential(
            nn.Linear(self.value_input_dim, atten_emb_dim), nn.LeakyReLU()
        )
        module_list = []
        atten_emb_layer_num = 4
        for i in range(atten_emb_layer_num):
            if i == 0:
                module_list.append(
                    nn.Sequential(
                        nn.Linear(
                            atten_emb_dim
                            + goal_embed_dim
                            + self.map_emb_dim
                            + action_dim,
                            atten_emb_dim,
                        ),
                        nn.LeakyReLU(),
                    )
                )
            elif i == atten_emb_layer_num - 1:
                module_list.append(nn.Linear(atten_emb_dim, output_dim))
            else:
                module_list.append(
                    AttenRepresentationBlock(
                        d_model=atten_emb_dim,
                        nhead=num_heads,
                        dropout=dropout,
                        batch_first=True,
                        activation=nn.LeakyReLU(),
                        dim_feedforward=dim_feedforward,
                    )
                )
        self.value_net = nn.ModuleList(module_list)
        # model = feature_extractor + [value_head]
        #
        # self.value_net = nn.Sequential(*model)
        self.apply(weight_init)

    def _mutli_task_preprocess_traj(self, traj, goal, with_acs=False, with_goal=False):
        traj = traj.transpose(1, 2)  # [task_size, seq_len, state_dim + action_dim]
        if goal is None:
            assert traj.shape[0] == 1
            goal = traj[
                :, -1, :
            ]  # incompatible with goal encoder dim  [T,L,S+A+A] and S+A, and last element may be not legal
        # goal = traj[:, -1, :]  # [task_size, state_dim + action_dim]
        # traj and state share state encoder
        states, actions = traj[:, :, : self.state_dim], traj[:, :, self.state_dim :]
        if with_goal:
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
                pass
            else:
                traj = states
        # traj = torch.cat((states, actions), dim=-1)

        # traj = traj.repeat(batch_size, 1, 1, 1)  # [batch_size, task_size， seq_len, state_dim + action_dim]
        return traj, goal

    def forward(self, state, traj, action, map_info, goal, atten_mask=None):
        r"""
        Args:
            state: [batch_size, task_size, state_dim]
            action: [batch_size, task_size, action_dim]
            traj: [task_size, state_dim + action_dim + action_dim, seq_len]
            map_info: [task_size, map_dim]
        Return:
            q_values: [batch_size, task_size, 1]
        """
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, dim=0)
            action = torch.unsqueeze(action, dim=0)

        batch_size = state.shape[0]
        task_size = state.shape[1]
        seq_len = traj.shape[-1]
        input_reshape = torch.unsqueeze(
            state.reshape([batch_size * task_size] + list(state.shape[2:])), dim=1
        )

        traj_key, goal = self._mutli_task_preprocess_traj(traj, goal)
        traj_value, goal = self._mutli_task_preprocess_traj(traj, goal, with_acs=True)
        key_input = traj_key  # torch.cat([traj_key, context_emb_kv], dim=-1)
        value_input = traj_value  # torch.cat([traj_value, context_emb_kv], dim=-1)
        query_input = input_reshape  # torch.cat([input_reshape, context_emb_q], dim=-1)
        key_encoded = key_input
        for enc in self.qk_encoder:
            key_encoded = enc(key_encoded)
        query_encoded = query_input
        for enc in self.qk_encoder:
            query_encoded = enc(query_encoded)
        value_encoded = self.value_input_encoder(value_input)
        value_encoded = self.transformer.encoder(value_encoded)
        key_encoded_repeated = key_encoded.repeat(batch_size, 1, 1, 1).reshape(
            [batch_size * task_size] + list(key_encoded.shape[1:])
        )
        value_encoded_repeated = value_encoded.repeat(batch_size, 1, 1, 1).reshape(
            [batch_size * task_size] + list(value_encoded.shape[1:])
        )
        # emb: [batch_size * task_size, emb_dim]

        if atten_mask is None:
            atten_mask = torch.full(
                (key_encoded_repeated.shape[0], seq_len), fill_value=float(0)
            ).to(
                key_encoded_repeated.device
            )  # all valid!

        embed = self.transformer.decoder(
            key=key_encoded_repeated,
            value=value_encoded_repeated,
            query=query_encoded,
            memory_key_padding_mask=atten_mask,
        )[0].squeeze(dim=1)
        embed = embed.reshape([batch_size, task_size, embed.shape[-1]])
        # rnn_output, h_state = self.rnn_layer(traj, None)
        # lst_rnn_out = rnn_output[:, -1].repeat(batch_size, 1, 1)
        # lst_rnn_out_encoder = self.rnn_encoder(lst_rnn_out)
        goal = self.goal_encoder(goal)
        goal = goal.repeat(batch_size, 1, 1)
        # goal_cat = torch.cat([goal, lst_rnn_out_encoder], dim=-1)
        # goal_cat_emb = self.merge_layer(goal_cat)
        if self.map_type == MapType.ID:
            map_info = (
                F.one_hot(map_info, num_classes=self.map_num)
                .repeat(batch_size, 1, 1)
                .float()
            )
        elif self.map_type == MapType.FIG:
            map_info = self.map_encoder(map_info).repeat(batch_size, 1, 1)
        else:
            raise NotImplementedError
        map_info = self.map_feature_extractor(map_info)
        x = torch.cat([embed, goal, map_info, action], dim=-1)
        v = x
        for enc in self.value_net:
            v = enc(v)
        return v


class AttenMultiTaskCritic_FullTraj(AttenMultiTaskCritic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        state: torch.Tensor,
        traj: torch.Tensor,
        action: torch.Tensor,
        map_info: torch.Tensor,
        goal: torch.Tensor,
        atten_mask: Union[torch.Tensor, None] = None,
    ):
        r"""This is a wrapper function to adapt input to ``AttenMultiTaskCritic``. We currently shatter the full
        traj as trans to infer the Q values. Note that for full traj setting, we use padding operation to construct
        a regular tensor, thus there are many invalid data. These will waste GPU memory to a certain degree. So to
        avoid unexpected exit for memory explosion, pay attention to the input data size.

        Args:
            state: [T*B*L_agent, S]
            traj: [T*1,S+A+A,seq_len]
            action: [T*B*L_exp, A]
            map_info: [T,map_dim]
            goal: [T,S+A]
            atten_mask: [T,seq_len] as the input to ``memory_key_padding_mask``, refer to nn.transformer

        Return:
            q_value: [T*B*L_agent, 1]
        """
        T, S, A = traj.shape[0], state.shape[-1], action.shape[-1]
        BL = state.shape[0] // T
        assert (
            traj.dim() == 3 and state.dim() == 2 and action.dim() == 2
        )  # unify the shape before the lowest operation!

        # recover the task_dim first formation
        state = state.reshape(T, -1, S)  # [T,B*L_agent,S]
        action = action.reshape(T, -1, A)  # [T,B*L_agent,S]

        state, action = state.permute(1, 0, 2), action.permute(1, 0, 2)
        # state/action [B*L_agent,T,S/A]

        # if atten_mask is not None:
        #     atten_mask=atten_mask.unsqueeze(1).repeat(1,B*L_exp,1).reshape(-1,seq_len)

        # (B,T,1)
        q_values = super().forward(
            state, traj, action, map_info, goal, atten_mask=atten_mask
        )

        q_values = q_values.permute([1, 0, 2]).reshape(
            -1, 1
        )  # (B*L,T,1)->(T,B*L,1)->(T*B*L,1)
        return q_values

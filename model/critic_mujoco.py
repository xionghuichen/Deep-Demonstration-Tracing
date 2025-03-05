from turtle import forward
import torch
import torch.nn as nn
from model.attention import MultiheadAttention, PositionalEncoding
from torch_utils import build_mlp_extractor, weight_init, CatNet
from CONST import *
import torch.nn.functional as F
from model.transformer import Transformer, AttenRepresentationBlock
import math



class AttenCritic(nn.Module):

    def __init__(
            self,
            state_dim,
            hidden_size,
            action_dim,
            goal_embed_dim,
            embed_goal,
            # map_shape,
            map_type=MapType.ID,
            activation_fn=nn.LeakyReLU,
            pos_encode=False,
            map_num=3,
            output_dim=1,
            seperate_encode = False,
            use_map_id = False
    ):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        # input_dim = state_dim + action_dim
        self.value_input_dim = state_dim + action_dim + action_dim
        # self.value_input_dim = state_dim + action_dim
        self.goal_input_dim = state_dim + action_dim + action_dim
        self.map_num = map_num
        self.map_type = map_type
        self.map_emb_dim = 0
        self.seperate_encode = seperate_encode
        self.use_map_id = use_map_id
        if self.use_map_id:
            self.map_emb_dim = goal_embed_dim
            self.map_feat_dim = self.map_num
            self.map_feature_extractor = nn.Sequential(nn.Linear(self.map_feat_dim, self.map_emb_dim), nn.LeakyReLU())

        if embed_goal:
            self.goal_encoder = nn.Sequential(nn.Linear(self.goal_input_dim, goal_embed_dim), nn.LeakyReLU())
        else:
            goal_embed_dim = self.goal_input_dim
            self.goal_encoder = lambda x: x

        # tmp parameters assignment
        atten_emb_dim = 128
        num_heads = 16
        dropout = 0.1
        num_encoder_layers = 4
        num_decoder_layers = 4
        dim_feedforward = 256
        act_fn = nn.LeakyReLU()
        self.transformer = Transformer(
            atten_emb_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
            activation=act_fn)

        # position encoding
        self.pos_encode = (
            PositionalEncoding(atten_emb_dim, dropout=dropout)
            if pos_encode
            else lambda x: x
        )

        # qkv-encoder
        module_list = []
        for i in range(num_encoder_layers):
            if i == 0:
                if self.seperate_encode:
                    module_list.append(nn.Sequential(CatNet(atten_emb_dim), nn.LeakyReLU()))
                else:
                    module_list.append(nn.Sequential(nn.Linear(self.state_dim, atten_emb_dim), nn.LeakyReLU()))
            else:
                module_list.append(AttenRepresentationBlock(d_model=atten_emb_dim, nhead=num_heads, dropout=dropout,
                                                            batch_first=True, activation=act_fn, dim_feedforward=dim_feedforward))
        self.qk_encoder = nn.ModuleList(module_list)

        self.value_input_encoder = nn.Sequential(nn.Linear(self.value_input_dim, atten_emb_dim), nn.LeakyReLU())
        module_list = []
        atten_emb_layer_num = 4
        for i in range(atten_emb_layer_num):
            if i == 0:
                # module_list.append(nn.Sequential(nn.Linear(atten_emb_dim + goal_embed_dim + self.map_emb_dim + action_dim, atten_emb_dim), nn.LeakyReLU()))
                module_list.append(nn.Sequential(nn.Linear(atten_emb_dim + goal_embed_dim + self.map_emb_dim + action_dim, atten_emb_dim), nn.LeakyReLU()))
            elif i == atten_emb_layer_num - 1:
                module_list.append(nn.Linear(atten_emb_dim, output_dim))
            else:
                module_list.append(AttenRepresentationBlock(d_model=atten_emb_dim, nhead=num_heads, dropout=dropout,
                                                            batch_first=True, activation=nn.LeakyReLU(),
                                                            dim_feedforward=dim_feedforward))
        self.value_net = nn.ModuleList(module_list)
        # model = feature_extractor + [value_head]
        #
        # self.value_net = nn.Sequential(*model)
        self.apply(weight_init)

    def _mutli_task_preprocess_traj(self, traj, goal, with_acs=False, with_goal=False):
        traj = traj.transpose(1, 2)  # [task_size, seq_len, state_dim + action_dim]
        if goal is None:
            # assert traj.shape[0] == 1
            goal = traj[:, -1, :]
        # goal = traj[:, -1, :]  # [task_size, state_dim + action_dim]
        # traj and state share state encoder
        states, actions = traj[:, :, :self.state_dim], traj[:, :, self.state_dim:]
        if with_goal:
            if with_acs:
                states = torch.cat((states, actions, torch.repeat_interleave(torch.unsqueeze(goal, dim=1), states.shape[1], 1)), dim=-1)
            else:
                states = torch.cat((states, torch.repeat_interleave(torch.unsqueeze(goal, dim=1), states.shape[1], 1)), dim=-1)
        else:
            if with_acs:
                pass
            else:
                traj = states
        # traj = torch.cat((states, actions), dim=-1)

        # traj = traj.repeat(batch_size, 1, 1, 1)  # [batch_size, task_size， seq_len, state_dim + action_dim]
        return traj, goal

    def forward(self, state, traj, action, map_info, squeeze = True):
        """
        state: [batch_size, task_size, state_dim]
        action: [batch_size, task_size, state_dim]
        traj: [task_size, state_dim + action_dim + action_dim, seq_len]
        map_info: [task_size, map_dim]
        """

        if len(state.shape) == 2:
            state = torch.unsqueeze(state, dim=1)
        if len(action.shape) == 2:
            action = torch.unsqueeze(action, dim=1)

        goal = traj.clone().transpose(1, 2)[:, -1, :]  # [1, state_dim + action_dim]
        batch_size = state.shape[0]
        task_size = state.shape[1]
        # assert task_size == 1
        seq_len = traj.shape[-1]
        input_reshape = torch.unsqueeze(state.reshape([batch_size * task_size] + list(state.shape[2:])), dim=1)

        traj_key, goal = self._mutli_task_preprocess_traj(traj, goal)
        traj_value, goal = self._mutli_task_preprocess_traj(traj, goal, with_acs=True)
        key_input = traj_key  # torch.cat([traj_key, context_emb_kv], dim=-1)
        value_input = traj_value  # torch.cat([traj_value, context_emb_kv], dim=-1)
        query_input = input_reshape  # torch.cat([input_reshape, context_emb_q], dim=-1)
        key_encoded = key_input
        # TODO: 如果决策是非MDP的（比如图像输入），这里qkv的确定需要使用上下文信息，此时qkv_encoder都需要使用序列进行预处理
        for enc in self.qk_encoder: key_encoded = enc(key_encoded)
        query_encoded = query_input
        for enc in self.qk_encoder: query_encoded = enc(query_encoded)
        value_encoded = self.value_input_encoder(value_input)
        value_encoded = self.transformer.encoder(value_encoded)
        key_encoded_repeated = key_encoded.repeat(batch_size, 1, 1, 1).reshape([batch_size * task_size] + list(key_encoded.shape[1:]))
        value_encoded_repeated = value_encoded.repeat(batch_size, 1, 1, 1).reshape([batch_size * task_size] + list(value_encoded.shape[1:]))
        # emb: [batch_size * task_size, emb_dim]

        key_encoded_repeated = self.pos_encode(key_encoded_repeated)
        value_encoded_repeated = self.pos_encode(value_encoded_repeated)

        embed = self.transformer.decoder(key=key_encoded_repeated, value=value_encoded_repeated, query=query_encoded)[0]
        embed = embed.squeeze(dim=1)
        embed = embed.reshape([batch_size, task_size, embed.shape[-1]])

        goal = self.goal_encoder(goal)
        goal = goal.repeat(batch_size, 1, 1)

        if self.use_map_id:
            # map_info = F.one_hot(torch.tensor([map_id]).cuda().to(embed.device), num_classes=self.map_num).repeat(batch_size, 1, 1).float()
            map_info = F.one_hot(map_info, num_classes=self.map_num).repeat(batch_size, 1, 1).float()
            map_info = self.map_feature_extractor(map_info)
            x = torch.cat([embed, goal, map_info, action], dim=-1)
        else:
            x = torch.cat([embed, goal, action], dim=-1)
        v = x
        for enc in self.value_net: v = enc(v)
        if squeeze:
            v = torch.squeeze(v, dim=1)
        return v

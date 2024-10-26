from turtle import forward
import torch
import torch.nn as nn
from torch_utils import build_mlp_extractor, weight_init
from model.attention import MultiheadAttention, PositionalEncoding
import torch.nn.functional as F
from model.map_encoder import MapEncoder
from model.transformer import AttenRepresentationBlock
from CONST import *


class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_size,
        action_dim=None,
        activation_fn=nn.LeakyReLU,
        output_dim=1,
    ):
        super().__init__()
        self.action_dim = action_dim
        if action_dim != None:
            input_dim = state_dim + action_dim
        else:
            input_dim = state_dim

        feature_extractor = build_mlp_extractor(input_dim, hidden_size, activation_fn)
        value_head = nn.Linear(
            (
                hidden_size[-1]
                if (hidden_size != None and len(hidden_size) > 0)
                else input_dim
            ),
            output_dim,
        )

        # concat all the layer
        model = feature_extractor + [value_head]
        self.net = nn.Sequential(*model)

        self.apply(weight_init)

    def forward(self, state, action=None):
        if self.action_dim != None:
            assert action is not None
            x = torch.cat([state, action], dim=1)
        else:
            x = state
        return self.net(x)


class AttnCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        embed_dim,
        num_heads,
        dropout=0.0,
        pos_encode=True,
        output_dim=1,
    ):
        super().__init__()

        # multi-head attantion
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

        # state-action encoder
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, embed_dim),
            nn.LeakyReLU(),
        )

        # value head
        self.value_head = nn.Linear(embed_dim, output_dim)

    def forward(self, state, traj, action):
        """
        state: [batch_size, state_dim]
        traj: [batch_size, state_dim + action_dim, seq_len]
        """
        batch_size = state.shape[0]
        traj = traj.transpose(1, 2)  # [1, seq_len, state_dim + action_dim]
        traj = traj.repeat(
            batch_size, 1, 1
        )  # [batch_size, seq_len, state_dim + action_dim]
        traj = self.pos_encode(traj)

        # state embedding
        input_ = torch.cat([state, action], dim=-1)
        state_embed = self.feature_extractor(input_).unsqueeze(dim=1)

        # task embedding
        # input_ = state_embed.unsqueeze(dim=1)
        task_context, _ = self.task_mh_attn(traj, traj, traj)  # (query, key, value)
        task_context = self.norm_1(task_context)
        # task_embed = context.squeeze(dim=1)

        embed, _ = self.state_mh_attn(state_embed, task_context, task_context)
        embed = self.norm_2(embed + state_embed)

        return self.value_head(embed.squeeze(dim=1))


class TransformerCritic(nn.Module):
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
        output_dim=1,
        share_state_encoder=False,
    ):
        super().__init__()
        input_dim = state_dim + action_dim
        self.share_state_encoder = share_state_encoder

        # traj encoder
        self.traj_encoder = nn.Sequential(
            nn.Linear(input_dim + state_dim, embed_dim), nn.LeakyReLU()
        )
        # Transformer
        self.transformer = nn.Transformer(
            embed_dim,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
        )

        # position encoding
        self.pos_encode = (
            PositionalEncoding(embed_dim, dropout=dropout)
            if pos_encode
            else lambda x: x
        )

        # state-action encoder
        self.sag_encoder = nn.Sequential(
            nn.Linear(input_dim + state_dim, embed_dim), nn.LeakyReLU()
        )

        # value head
        self.value_head = nn.Linear(embed_dim, output_dim)

    def _preprocess_traj(self, traj, batch_size):
        traj = traj.transpose(1, 2)  # [1, seq_len, state_dim + action_dim]
        last_sa = traj[:, -1, :]
        last_state, last_action = last_sa[:, -4:-2], last_sa[:, -2:]
        with torch.no_grad():
            goal = last_state + last_action
            traj = torch.cat(
                (traj, goal.repeat(traj.shape[1], 1).unsqueeze(dim=0)), dim=-1
            )

        if self.share_state_encoder:
            # traj and state-action share state encoder
            traj = self.sag_encoder(traj)
        else:
            traj = self.traj_encoder(traj)

        traj = self.pos_encode(traj.transpose(0, 1)).transpose(0, 1)
        traj = traj.repeat(
            batch_size, 1, 1
        )  # [batch_size, seq_len, state_dim + action_dim]
        # TODO: concat, done
        return traj, goal

    def forward(self, state, traj, action):
        """
        state: [batch_size, state_dim]
        traj: [1, state_dim + action_dim, seq_len]
        """
        batch_size = state.shape[0]
        # preprocess traj and state
        traj, goal = self._preprocess_traj(traj, batch_size)
        # TODO: goal, done
        input_ = torch.cat([state, action, goal.repeat(batch_size, 1)], dim=-1)
        input_ = self.sag_encoder(input_).unsqueeze(dim=1)
        # go through transformer
        embed = self.transformer(traj, input_).squeeze(dim=1)
        return self.value_head(embed)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


class GoalMLPCritic(nn.Module):
    def __init__(
        self,
        state_dim,
        hidden_size,
        action_dim,
        goal_embed_dim,
        embed_goal,
        activation_fn=nn.LeakyReLU,
        output_dim=1,
    ):
        super().__init__()
        self.action_dim = action_dim
        input_dim = state_dim + action_dim

        if embed_goal:
            self.goal_encoder = nn.Sequential(
                nn.Linear(input_dim, goal_embed_dim), nn.LeakyReLU()
            )

        else:
            goal_embed_dim = input_dim
            self.goal_encoder = lambda x: x
        rnn_hidden = 128
        self.rnn_layer = nn.GRU(
            input_size=input_dim, hidden_size=rnn_hidden, num_layers=4, batch_first=True
        )

        self.rnn_encoder = nn.Sequential(
            nn.Linear(rnn_hidden, goal_embed_dim), nn.Tanh()
        )
        self.merge_layer = nn.Sequential(
            nn.Linear(goal_embed_dim * 2, goal_embed_dim), nn.LeakyReLU()
        )
        feature_extractor = build_mlp_extractor(
            input_dim + goal_embed_dim, hidden_size, activation_fn
        )

        value_head = nn.Linear(
            hidden_size[-1] if len(hidden_size) > 0 else input_dim + goal_embed_dim,
            output_dim,
        )

        model = feature_extractor + [value_head]
        self.value_net = nn.Sequential(*model)

        self.apply(weight_init)

    def forward(self, state, traj, action):
        traj = traj.detach().clone()
        traj = traj.transpose(1, 2)  # [1, seq_len, state_dim + action_dim]
        goal = traj[:, -1, :].clone()  # [1, state_dim + action_dim]

        # print('critic:',traj.shape,goal.shape,state.shape)
        # traj[:,:,8:10]=traj[:,:,8:10].detach().clone()*0
        # goal[:,8:10]=goal[:,8:10].detach().clone()*0
        # state[:,8:10]=state[:,8:10].detach().clone()*0

        rnn_output, h_state = self.rnn_layer(traj, None)
        lst_rnn_out = rnn_output[:, -1].repeat(state.shape[0], 1)
        lst_rnn_out_encoder = self.rnn_encoder(lst_rnn_out)
        goal = self.goal_encoder(goal)
        goal = goal.repeat(state.shape[0], 1)
        goal_cat = torch.cat([goal, lst_rnn_out_encoder], dim=1)
        goal_cat_emb = self.merge_layer(goal_cat)
        x = torch.cat([goal_cat_emb, state, action], dim=1)
        return self.value_net(x)


class GoalMapMLPCritic(nn.Module):
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
    ):
        super().__init__()
        self.action_dim = action_dim
        input_dim = state_dim + action_dim
        self.map_num = map_num
        self.map_type = map_type

        if embed_goal:
            self.goal_encoder = nn.Sequential(
                nn.Linear(input_dim, goal_embed_dim), nn.LeakyReLU()
            )
        else:
            goal_embed_dim = input_dim
            self.goal_encoder = lambda x: x

        if self.map_type == MapType.FIG:
            self.map_encoder = MapEncoder(depth=64, shape=map_shape)
            self.map_feat_dim = self.map_encoder.embed_size
            self.map_feature_extractor = nn.Sequential(
                *build_mlp_extractor(
                    self.map_feat_dim, (256, goal_embed_dim), nn.LeakyReLU
                )
            )
        else:
            self.map_feat_dim = self.map_num
            self.map_feature_extractor = nn.Sequential(
                nn.Linear(self.map_feat_dim, goal_embed_dim), nn.LeakyReLU()
            )

        rnn_hidden = 128
        self.rnn_layer = nn.GRU(
            input_size=input_dim + action_dim,
            hidden_size=rnn_hidden,
            num_layers=3,
            batch_first=True,
        )

        self.rnn_encoder = nn.Sequential(
            nn.Linear(rnn_hidden, goal_embed_dim), nn.Tanh()
        )
        self.merge_layer = nn.Sequential(
            nn.Linear(goal_embed_dim * 2, goal_embed_dim), nn.LeakyReLU()
        )

        feature_extractor = build_mlp_extractor(
            input_dim + goal_embed_dim + goal_embed_dim, hidden_size[:1], activation_fn
        )

        value_head = nn.Linear(
            hidden_size[-1] if len(hidden_size) > 0 else input_dim + goal_embed_dim,
            output_dim,
        )
        module_list = []
        atten_emb_dim = 128
        atten_emb_layer_num = 4
        num_heads = 16
        dropout = 0.1
        for i in range(atten_emb_layer_num):
            if i == 0:
                module_list.append(
                    nn.Sequential(
                        nn.Linear(
                            input_dim + goal_embed_dim + goal_embed_dim, atten_emb_dim
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
                        dim_feedforward=256,
                    )
                )
        self.value_net = nn.ModuleList(module_list)
        # model = feature_extractor + [value_head]
        #
        # self.value_net = nn.Sequential(*model)

        self.apply(weight_init)

    def forward(self, state, traj, action, map_info):
        traj = traj.detach().clone()
        traj = traj.transpose(1, 2)  # [1, seq_len, state_dim + action_dim]
        goal = traj[:, -1, :].clone()  # [1, state_dim + action_dim]

        # print('critic:',traj.shape,goal.shape,state.shape)
        # traj[:,:,-4:-2]=traj[:,:,-4:-2].detach().clone()*0
        # goal[:,-4:-2]=goal[:,-4:-2].detach().clone()*0
        # state[:,-4:-2]=state[:,-4:-2].detach().clone()*0

        rnn_output, h_state = self.rnn_layer(traj, None)
        lst_rnn_out = rnn_output[:, -1].repeat(state.shape[0], 1)
        lst_rnn_out_encoder = self.rnn_encoder(lst_rnn_out)
        goal = self.goal_encoder(goal)
        goal = goal.repeat(state.shape[0], 1)
        goal_cat = torch.cat([goal, lst_rnn_out_encoder], dim=1)
        goal_cat_emb = self.merge_layer(goal_cat)
        if self.map_type == MapType.ID:
            map_info = (
                F.one_hot(map_info, num_classes=self.map_num)
                .repeat(state.shape[0], 1)
                .float()
            )
        elif self.map_type == MapType.FIG:
            map_info = self.map_encoder(map_info).repeat(state.shape[0], 1)
        else:
            raise NotImplementedError
        map_info = self.map_feature_extractor(map_info)
        x = torch.cat([goal_cat_emb, state, action, map_info], dim=1)
        return self.value_net(x)


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
        # x = x.reshape([batch_size*task_size] + list(x.shape[2:]))
        return self.value_net(x)

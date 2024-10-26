import torch
import torch.nn as nn
import numpy as np

ROT_feature_dim = 50


def build_mlp_extractor(input_dim, hidden_size, activation_fn):
    """
    Create MLP feature extractor, code modified from:
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py
    """
    if len(hidden_size) > 0:
        mlp_extractor = [nn.Linear(input_dim, hidden_size[0]), activation_fn()]
    else:
        mlp_extractor = []

    for idx in range(len(hidden_size) - 1):
        mlp_extractor.append(nn.Linear(hidden_size[idx], hidden_size[idx + 1]))
        mlp_extractor.append(activation_fn())

    return mlp_extractor


# class FakeMLPRNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         """
#         batch first fake rnn
#         """
#         super().__init__()
#         self.feature_dim=ROT_feature_dim
#         self.input_size, self.hidden_size=input_size, self.feature_dim
#         self.rnn_layer=nn.ModuleList([nn.GRUCell(input_size=input_size, hidden_size=self.hidden_size) for _ in range(hidden_size)])


class meta_rot_contexter(nn.Module):
    def __init__(self, state_dim, embed_dim) -> None:
        super().__init__()
        self.state_dim = state_dim
        # self.feature_dim=ROT_feature_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.LayerNorm(embed_dim), nn.Tanh()
        )

    @torch.no_grad()
    def get_context(self, trajs):
        if trajs.ndim == 2:
            trajs = trajs.unsqueeze(0)
        assert trajs.shape[-1] >= self.state_dim
        trajs = trajs.clone()
        trajs = trajs[..., : self.state_dim]
        context = self.state_encoder(trajs)  # batch, seq_len, hidden_size
        return context.squeeze(0).detach()

    def forward(self, trajs):
        return self.get_context(trajs)


def soft_update(rho, net, target_net):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(rho * param.data + (1 - rho) * target_param.data)


def reset_logstd(policy):
    if policy.state_std_independent:
        policy.log_std = nn.Parameter(
            torch.zeros(1, policy.action_dim), requires_grad=True
        )
    else:
        policy.log_std.reset_parameters()


def weight_init(m):
    """Custom weight init for Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def net_l2_norm(network, mean=False):
    weights = 0
    param_num = 0
    for item in network.parameters():
        if item.requires_grad:
            weights = weights + item.pow(2).sum()
            param_num = param_num + np.prod(list(item.data.shape))
    if mean:
        param_num = max(param_num, 1)
        weights = weights / param_num
    return weights


def preprocess_traj(traj, device):
    """
    traj: [seq_len, state_dim+action_dim]

    convert from ndarray into tensor

    return: [1, state_dim+action_dim, seq_len]
    """
    traj = torch.FloatTensor(traj).to(device).unsqueeze(dim=0)
    traj = traj.transpose(1, 2)
    return traj

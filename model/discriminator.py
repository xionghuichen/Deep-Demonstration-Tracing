import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.model.critic import Critic


class GAILDisc(Critic):
    def __init__(
        self,
        state_dim,
        hidden_size,
        action_dim=None,
        activation_fn=nn.ReLU,
        output_dim=1,
    ):
        super.__init__(state_dim, hidden_size, action_dim, activation_fn, output_dim)

    @torch.no_grad()
    def gail_reward(self, state, action):
        d_sa = self.forward(state, action)
        return -F.logsigmoid(-d_sa)  # r(s, a) = -log(1-D(s,a))

# Created by xionghuichen at 2022/11/25
# Email: chenxh@lamda.nju.edu.cn
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from torch_utils import build_mlp_extractor


class MapEncoder(nn.Module):
    def __init__(self, depth=32, stride=2, shape=(3, 64, 64), activation=nn.ReLU):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(shape[0], 1 * depth, 4, stride),
            activation(),
            nn.Conv2d(1 * depth, 2 * depth, 4, stride),
            activation(),
            nn.Conv2d(2 * depth, 4 * depth, 4, stride),
            activation(),
            nn.Conv2d(4 * depth, 8 * depth, 4, stride),
            activation(),
        )

        self.shape = shape
        self.stride = stride
        self.depth = depth

    def forward(self, obs):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        embed = self.convolutions(obs.reshape(-1, *img_shape))
        embed = torch.reshape(embed, (*batch_shape, -1))
        return embed

    @property
    def embed_size(self):
        conv1_shape = conv_out_shape(self.shape[1:], 0, 4, self.stride)
        conv2_shape = conv_out_shape(conv1_shape, 0, 4, self.stride)
        conv3_shape = conv_out_shape(conv2_shape, 0, 4, self.stride)
        conv4_shape = conv_out_shape(conv3_shape, 0, 4, self.stride)
        embed_size = 8 * self.depth * np.prod(conv4_shape).item()
        return embed_size


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

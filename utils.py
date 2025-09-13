import torch
from torch import nn
import numpy as np
import random
import typing
from collections import defaultdict
from numpy.random import choice
import tabulate
import numpy as np
import random
import cv2
from collections import deque
import copy


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def padding_np_seq(x: np.array, max_len: int, dim: int = 0):
    """pad the np-array sequence in the last dim

    Arguments:
        x -- any-D array to be pad
        max_len -- the last dim expected
    """
    shape_expected = list(x.shape)
    shape_expected[dim] = max_len - x.shape[dim]
    shape_expected = tuple(shape_expected)
    return np.concatenate((x, np.zeros(shape_expected)), axis=dim)


def update_hist(
    hist_q: deque, state: np.array, pre_action: np.array, maxlen: int, dim: int = 0
):
    """update hist(a deque object), and return padded np.array sequence with
        (maxlen,state_dim+act_dim) and last valid id

    Arguments:
        hist_q -- deque object
        state -- current state
        pre_action -- previous action
        maxlen -- expected hist len

    Keyword Arguments:
        dim -- padding dim (default: {0})

    Returns:
        np.array history and last element index
    """
    hist_q.append(np.concatenate((state, pre_action), axis=-1))
    last_id = len(hist_q) - 1
    hist_np = padding_np_seq(np.array(hist_q), maxlen, dim=0)
    return hist_np, last_id


def parameter_count_filter(model: nn.Module, valid: lambda x: True):
    """Count parameters of a model and its submodules.

    Args:
        model: a torch module

    Returns:
        dict (str-> int): the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key  corresponds to the total
        number of parameters of the model.
    """

    r = defaultdict(int)
    for name, prm in model.named_parameters():
        if not valid(name):
            continue
        size = prm.numel()
        name = name.split(".")
        for k in range(0, len(name) + 1):
            prefix = ".".join(name[:k])
            r[prefix] += size
    total_num = r[""]
    if total_num > 1e8:
        return "{:.1f}G".format(total_num / 1e9)
    if total_num > 1e5:
        return "{:.1f}M".format(total_num / 1e6)
    if total_num > 1e2:
        return "{:.1f}K".format(total_num / 1e3)
    return str(total_num)


def soft_update(rho, net, target_net):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(rho * param.data + (1 - rho) * target_param.data)

def preprocess_traj(traj, device):
    """
    traj: [seq_len, state_dim+action_dim]

    convert from ndarray into tensor

    return: [1, state_dim+action_dim, seq_len]
    """
    traj = torch.FloatTensor(traj).to(device)
    if len(traj.shape) == 2:
        traj = traj.unsqueeze(dim=0)
    traj = traj.transpose(1, 2)
    return traj

def image_to_video(images, vedioPath):
        fileSize = images[0].shape[0:2]
        writer = cv2.VideoWriter(vedioPath + '.mp4',
                                 cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 12, (fileSize[1], fileSize[0]), True)
        total_frame = len(images)
        for i in range(total_frame):
            writer.write(images[i])
        writer.release()


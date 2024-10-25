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
from collections import deque


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    

def parameter_count_filter(model: nn.Module, valid: lambda x:True) -> typing.DefaultDict[str, int]:
    """
    Count parameters of a model and its submodules.

    Args:
        model: a torch module

    Returns:
        dict (str-> int): the key is either a parameter name or a module name.
        The value is the number of elements in the parameter, or in all
        parameters of the module. The key "" corresponds to the total
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
    total_num=r['']
    if total_num>1e8:
        return "{:.1f}G".format(total_num / 1e9)
    if total_num>1e5:
        return "{:.1f}M".format(total_num / 1e6)
    if total_num>1e2:
        return "{:.1f}K".format(total_num / 1e3) 
    return str(total_num)


import sys
from typing import Dict, List, Tuple, Union
from copy import deepcopy
import numpy as np
import torch
from scipy.spatial import KDTree
import gym
# import gym_continuous_maze
import random
from collections import deque
import pickle



def save_buffer(replay_buffer, buffer_path: str):
    with open(buffer_path, "wb") as f:
        pickle.dump(replay_buffer.buffer_info, f)


def load_buffer(replay_buffer, buffer_path: str):
    with open(buffer_path, "rb") as f:
        loaded_buffer_info = pickle.load(f)
        replay_buffer.load_buffer(loaded_buffer_info)
        return loaded_buffer_info

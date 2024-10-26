import copy 
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from einops import rearrange, repeat, parse_shape
from rl.utils.exp import preprocess_traj
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from rl.algo.sac_goalcritic import MT_GoalSACAgent

class ReplayAgent(MT_GoalSACAgent):
    def __init__(self, configs):
        super(ReplayAgent, self).__init__(configs)
        self.state_dim = configs["state_dim"]
        self.action_dim = configs["action_dim"]
        self.no_coor = configs["no_coordinate"]
        self.env_noise = configs["maze_env_noise"]
        self.action_ptr = 0
        self.action_seq = np.random.randn(220, self.action_dim)

    @torch.no_grad()
    def get_action(self,):
        self.eval()
        pass

    def reset_actions(self, traj):
        traj = traj.copy()
        s_seq, a_seq=traj[:, :-self.action_dim], traj[:, -self.action_dim:]
        self.action_seq = np.concatenate((a_seq, np.random.randn(220, self.action_dim)), axis=0)
        self.action_ptr = 0

    def select_action(self, obs, traj, *args, **kwargs):
        # ! traj is s_t, a_t-1 format, do transform first
        action = self.action_seq[self.action_ptr]+np.random.randn(self.action_dim)*self.env_noise
        self.action_ptr+=1
        return action

    def update(self, *args, **kwargs):
        return {}

    def BCupdate(self, state, action, next_state, reward, done, task_id, map_id, recent_buf_list):
        return {}
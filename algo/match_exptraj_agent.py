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

class MatchAgent(MT_GoalSACAgent):
    def __init__(self, configs):
        super(MatchAgent, self).__init__(configs)
        self.state_dim = configs["state_dim"]
        self.action_dim = configs["action_dim"]
        self.no_coor = configs["no_coordinate"]

    @torch.no_grad()
    def get_action(self,):
        self.eval()
        pass

    def select_action(self, obs, traj, *args, **kwargs):
        # ! traj is s_t, a_t-1 format, do transform first
        traj, obs = traj.copy(), obs.copy()
        s_seq, a_seq=traj[:, :-self.action_dim], traj[:, -self.action_dim:]
        if self.no_coor:
            s_seq[..., -2:]=0.0
            obs[..., -2:]=0.0
        dis = np.sum((s_seq-obs)**2, axis = -1)
        return a_seq[np.argmin(dis, axis=0)].copy()

    def update(self, *args, **kwargs):
        return {}

    def BCupdate(self, state, action, next_state, reward, done, task_id, map_id, recent_buf_list):
        return {}
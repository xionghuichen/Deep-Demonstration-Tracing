import sys
from typing import Dict, List, Tuple, Union
from copy import deepcopy
import numpy as np
import torch
from scipy.spatial import KDTree
import gym
import pickle

# import gym_continuous_maze
import random
from collections import deque

from utils import update_hist
from pickle import load, dump
from CONST import *


class TransitionBuffer:
    """
    Transition buffer for single task
    """

    def __init__(self, state_dim, action_dim, buffer_size, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        self.clear()

    def insert(self, state, action, next_state, reward, done):
        # insert (s, a, r, s', not_done)
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.next_state_buffer[self.ptr] = next_state
        self.reward_buffer[self.ptr] = reward
        self.mask_buffer[self.ptr] = 1.0 - done

        # update pointer and size
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def random_sample(self, batch_size=None):
        idx = list(range(self.size))
        if batch_size != None:
            idx = np.random.choice(idx, size=batch_size, replace=False)
        else:
            idx = np.array(idx)

        return (
            torch.FloatTensor(self.state_buffer[idx]).to(self.device),
            torch.FloatTensor(self.action_buffer[idx]).to(self.device),
            torch.FloatTensor(self.next_state_buffer[idx]).to(self.device),
            torch.FloatTensor(self.reward_buffer[idx]).to(self.device),
            torch.FloatTensor(self.mask_buffer[idx]).to(self.device),
        )

    def clear(self):
        self.state_buffer = np.zeros((self.buffer_size, self.state_dim))
        self.action_buffer = np.zeros((self.buffer_size, self.action_dim))
        self.next_state_buffer = np.zeros((self.buffer_size, self.state_dim))
        self.reward_buffer = np.zeros((self.buffer_size, 1))
        self.mask_buffer = np.zeros((self.buffer_size, 1))
        self.ptr = 0
        self.size = 0


class MT_TransitionBuffer:
    """
    Transition buffer for multi-task
    """

    def __init__(
        self, state_dim, action_dim, buffer_size, device, task_nums, segment_len=2000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        self.task_nums = task_nums
        self.stored_eps_num = (
            0  # NOTE: 这个属性现在不会在insert的过程中自动更新，是外部维护的数据变量
        )
        self.segment_len = segment_len
        self.initial_len = segment_len
        self.clear()

    def cat_new_mem(self, origin_data):
        sp = (
            origin_data.shape[0],
            self.segment_len,
            origin_data.shape[-1],
        )  # 3D data [task_nums, buffer_size, state_dim]
        new_fragment = np.zeros(sp)
        return np.concatenate((origin_data, new_fragment), axis=1)

    def insert(self, state, action, next_state, reward, done, task_id):
        # sanity check
        assert type(task_id) == int and task_id >= 0 and task_id < self.task_nums

        # insert (s, a, r, s', not_done)
        if self.ptr[task_id] >= len(self.state_buffer[task_id]):
            self.state_buffer = self.cat_new_mem(self.state_buffer)
            self.action_buffer = self.cat_new_mem(self.action_buffer)
            self.next_state_buffer = self.cat_new_mem(self.next_state_buffer)
            self.rew_buffer = self.cat_new_mem(self.rew_buffer)
            self.mask_buffer = self.cat_new_mem(self.mask_buffer)

        self.state_buffer[task_id][self.ptr] = state
        self.action_buffer[task_id][self.ptr] = action
        self.next_state_buffer[task_id][self.ptr] = next_state
        self.rew_buffer[task_id][self.ptr] = reward
        self.mask_buffer[task_id][self.ptr] = 1.0 - done

        # update pointer and size
        self.ptr[task_id] = (self.ptr[task_id] + 1) % self.buffer_size
        self.size[task_id] = min(self.size[task_id] + 1, self.buffer_size)

    def random_sample(self, task_id, batch_size=None, replace_sample=True):
        # sanity check
        assert type(task_id) == int and task_id >= 0 and task_id < self.task_nums
        idx = list(range(self.size[task_id]))
        if batch_size != None:
            idx = np.random.choice(idx, size=batch_size, replace=replace_sample)
        else:
            idx = np.array(idx)

        return (
            torch.FloatTensor(self.state_buffer[task_id][idx]).to(self.device),
            torch.FloatTensor(self.action_buffer[task_id][idx]).to(self.device),
            torch.FloatTensor(self.next_state_buffer[task_id][idx]).to(self.device),
            torch.FloatTensor(self.rew_buffer[task_id][idx]).to(self.device),
            torch.FloatTensor(self.mask_buffer[task_id][idx]).to(self.device),
        )

    def clear(self):
        self.state_buffer = np.zeros((self.task_nums, self.initial_len, self.state_dim))
        self.action_buffer = np.zeros(
            (self.task_nums, self.initial_len, self.action_dim)
        )
        self.next_state_buffer = np.zeros(
            (self.task_nums, self.initial_len, self.state_dim)
        )
        self.rew_buffer = np.zeros((self.task_nums, self.initial_len, 1))
        self.mask_buffer = np.zeros((self.task_nums, self.initial_len, 1))
        self.ptr = [0] * self.task_nums
        self.size = [0] * self.task_nums

    @property
    def buffer_info(self):
        return [
            self.state_buffer,
            self.action_buffer,
            self.next_state_buffer,
            self.rew_buffer,
            self.mask_buffer,
            self.ptr,
            self.size,
        ]

    def clear_task(self, task_id):
        self.ptr[task_id] = 0
        self.size[task_id] = 0

    def load_buffer(self, buffer_info: Tuple):
        (
            self.state_buffer,
            self.action_buffer,
            self.next_state_buffer,
            self.rew_buffer,
            self.mask_buffer,
            self.ptr,
            self.size,
        ) = buffer_info


class TrajectoryBuffer:
    def __init__(
        self,
        task_nums,
        env_name,
        device,
        action_weight,
        scale,
        add_bc_reward,
        do_scale,
        with_goal_distance,
        distance_weight,
        env_handler,
        demo_collect_env,
        hist_len=16,
        max_space_dist=2.0,
        reward_fun_type="origin",
        is_full_traj=False,
        add_local_view_error=False,
        local_view_error_weight=0.5,
        no_itor=False,
    ):
        demo_collect_env.reset()
        self.demo_collect_env = demo_collect_env
        self.task_nums = task_nums
        self.env_name = env_name
        self.coor_dim = env_handler.coor_dim
        self.state_dim = env_handler.state_dim
        self.action_dim = env_handler.action_dim
        self.env_handler = env_handler

        self.device = device
        self.action_weight = action_weight
        self.scale = scale
        self.add_bc_reward = add_bc_reward
        self.do_scale = do_scale
        self.with_goal_distance = with_goal_distance
        self.distance_weight = distance_weight
        self.hist_len = hist_len
        self.max_space_dist = max_space_dist
        self.reward_fun_type = reward_fun_type
        self.is_full_traj = is_full_traj
        self.add_local_view_error = add_local_view_error
        self.local_view_error_weight = local_view_error_weight
        self.no_itor = no_itor
        if self.no_itor:
            print("\n-------origin ilr reward-------\n")
        self.clear()

    def clear(self):
        self.trajectory_buffer = dict()
        self.kdtrees = dict()
        self._demo_buffer = dict()
        self.traj_lens = dict()
        self._init_buffer()

    def clear_insert(self, task_id, new_traj):
        self.trajectory_buffer[task_id] = []
        self._demo_buffer[task_id] = []
        self.insert(task_id, new_traj)

    def _init_buffer(self):
        """
        Buffer structure:
        {
            0: [...],
            1: [...],
            ...
            task_nums-1: [...]
        }
        """
        for i in range(self.task_nums):
            self.trajectory_buffer[i] = []
            self._demo_buffer[i] = []
            self.kdtrees[i] = []
            self.traj_lens[i] = []

    @property
    def buffer_info(self):
        return [self.trajectory_buffer, self._demo_buffer, self.kdtrees, self.traj_lens]

    def insert(self, task_id, new_traj, walls):
        """
        Example of new_traj:
        [
            [s_1, a_1],
            [s_2, a_2],
            ...
            [s_{T-1}, a_{T-1}]
        ]
        """
        self.trajectory_buffer[task_id].append(new_traj.copy())
        # self.kdtrees[task_id].append(
        #     KDTree(self._impose_action_weight(new_traj.copy()))
        # )
        self.traj_lens[task_id].append(new_traj.shape[0])
        self._demo_buffer[task_id].append(
            self._get_trans(new_traj.copy(), task_id, walls)
        )

    def random_sample(self, task_id):
        if len(self.trajectory_buffer[task_id]) == 0:
            return (None, None)
        else:
            idx = np.random.choice(np.arange(len(self.trajectory_buffer[task_id])))
            return idx, self.trajectory_buffer[task_id][idx]

    def random_sample_trajs(self, task_ids):
        trajs = []
        max_len = 0
        for task_id in task_ids:
            if torch.is_tensor(task_id) or (type(task_id) is np.ndarray):
                task_id = task_id.item()
            _, traj = self.random_sample(task_id)
            max_len = max(max_len, traj.shape[0])
            trajs.append(traj)
        padded_trajs, pad_lens = [], []
        for traj in trajs:
            pad_len = max_len - traj.shape[0]
            # we pad trajs in a batch into the same length
            padded_traj = np.pad(traj, ((pad_len, 0), (0, 0)), "constant")
            padded_trajs.append(padded_traj)
            pad_lens.append(pad_len)
        return np.array(padded_trajs).astype(np.float32), max_len, pad_lens

    def get_trans(self, task_id, idx):
        """
        :return: demos with (s, a, s', r, mask) style
        """
        return self._demo_buffer[task_id][idx]

    def _get_trans(self, traj, task_id, walls):
        env = self.demo_collect_env
        start, goal = self.env_handler.get_start_and_goal_from_demo(
            traj, random_start=False, noise_scaler=0.0
        )
        env.custom_walls(walls)
        env.reset(start=start, goal=goal, with_local_view=True)
        states, actions, next_states, rewards, masks = [], [], [], [], []
        pre_action = None

        for s_a in traj:
            state, action = s_a[:-2], s_a[-2:]
            env.set_pos(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward
            reward = self.env_handler.get_osil_reward(
                traj,
                state,
                action,
                done,
                reward,
            )

            if pre_action is None:
                pre_action = np.zeros(*action.shape)

            pre_action = action

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append([reward])
            masks.append([1.0 - done])

        res = (
            torch.FloatTensor(np.array(states)).to(self.device),
            torch.FloatTensor(np.array(actions)).to(self.device),
            torch.FloatTensor(np.array(next_states)).to(self.device),
            torch.FloatTensor(np.array(rewards)).to(self.device),
            torch.FloatTensor(np.array(masks)).to(self.device),
        )
        return res

    def load_buffer(self, buffer_info):
        (
            self.trajectory_buffer,
            self._demo_buffer,
            self.kdtrees,
            self.traj_lens,
        ) = buffer_info

    def get_full_traj(self, exp_traj, task_id):
        raise NotImplementedError


class MT_ShortTrajBuffer(MT_TransitionBuffer):
    def __init__(
        self, state_dim, action_dim, buffer_size, device, task_nums, hist_len=16
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.device = device
        self.task_nums = task_nums
        self.hist_len = hist_len
        self.clear()

    def insert(
        self,
        state,
        action,
        next_state,
        reward,
        done,
        cur_hist,
        cur_last_ind,
        next_hist,
        next_last_ind,
        task_id,
    ):
        # sanity check
        assert type(task_id) == int and task_id >= 0 and task_id < self.task_nums

        # insert (s, a, r, s', not_done)
        self.state_buffer[task_id][self.ptr] = state
        self.action_buffer[task_id][self.ptr] = action
        self.next_state_buffer[task_id][self.ptr] = next_state
        self.reward_buffer[task_id][self.ptr] = reward
        self.mask_buffer[task_id][self.ptr] = 1.0 - done
        self.cur_hist_buffer[task_id][self.ptr] = cur_hist
        self.cur_last_ind_buffer[task_id][self.ptr] = cur_last_ind
        self.next_hist_buffer[task_id][self.ptr] = next_hist
        self.next_last_ind_buffer[task_id][self.ptr] = next_last_ind

        # update pointer and size
        self.ptr[task_id] = (self.ptr[task_id] + 1) % self.buffer_size
        self.size[task_id] = min(self.size[task_id] + 1, self.buffer_size)

    def random_sample(self, task_id, batch_size=None):
        # sanity check
        assert type(task_id) == int and task_id >= 0 and task_id < self.task_nums

        idx = list(range(self.size[task_id]))
        if batch_size != None:
            idx = np.random.choice(idx, size=batch_size, replace=False)
        else:
            idx = np.array(idx)

        return (
            torch.FloatTensor(self.state_buffer[task_id][idx]).to(self.device),
            torch.FloatTensor(self.action_buffer[task_id][idx]).to(self.device),
            torch.FloatTensor(self.next_state_buffer[task_id][idx]).to(self.device),
            torch.FloatTensor(self.reward_buffer[task_id][idx]).to(self.device),
            torch.FloatTensor(self.mask_buffer[task_id][idx]).to(self.device),
            torch.FloatTensor(self.cur_hist_buffer[task_id][idx]).to(self.device),
            torch.LongTensor(self.cur_last_ind_buffer[task_id][idx]).to(self.device),
            torch.FloatTensor(self.next_hist_buffer[task_id][idx]).to(self.device),
            torch.LongTensor(self.next_last_ind_buffer[task_id][idx]).to(self.device),
        )

    def clear(self):
        super().clear()
        self.cur_hist_buffer = np.zeros(
            (
                self.task_nums,
                self.buffer_size,
                self.hist_len,
                self.state_dim + self.action_dim,
            )
        )
        self.cur_last_ind_buffer = np.zeros(
            (
                self.task_nums,
                self.buffer_size,
            )
        )
        self.next_hist_buffer = np.zeros(
            (
                self.task_nums,
                self.buffer_size,
                self.hist_len,
                self.state_dim + self.action_dim,
            )
        )
        self.next_last_ind_buffer = np.zeros(
            (
                self.task_nums,
                self.buffer_size,
            )
        )

    @property
    def buffer_info(self):
        return [
            self.state_buffer,
            self.action_buffer,
            self.next_state_buffer,
            self.reward_buffer,
            self.mask_buffer,
            self.cur_hist_buffer,
            self.cur_last_ind_buffer,
            self.next_hist_buffer,
            self.next_last_ind_buffer,
            self.ptr,
            self.size,
        ]

    def clear_task(self, task_id):
        self.ptr[task_id] = 0
        self.size[task_id] = 0

    def load_buffer(self, buffer_info: Tuple):
        (
            self.state_buffer,
            self.action_buffer,
            self.next_state_buffer,
            self.reward_buffer,
            self.mask_buffer,
            self.cur_hist_buffer,
            self.cur_last_ind_buffer,
            self.next_hist_buffer,
            self.next_last_ind_buffer,
            self.ptr,
            self.size,
        ) = buffer_info


class MT_FullTrajBuffer(MT_TransitionBuffer):
    def __init__(self, state_dim, action_dim, buffer_size, device, task_nums):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_buffer_size = buffer_size
        self.device = device
        self.task_nums = task_nums
        self.cur_buffer_sizes = [0] * task_nums
        self.clear()

    def insert(self, task_id, full_trajectory, reward_seq):
        ptr = self.ptrs[task_id]

        self.traj_buffer[task_id][ptr] = full_trajectory
        self.rew_buffer[task_id][ptr] = reward_seq

        self.ptrs[task_id] = (ptr + 1) % self.max_buffer_size
        self.cur_buffer_sizes[task_id] = min(
            self.cur_buffer_sizes[task_id] + 1, self.max_buffer_size
        )

    def padding_seq(self, seq: list, max_len: int) -> np.ndarray:
        new_seq = []

        for s in seq:
            new_seq.append(
                np.pad(
                    s,
                    ((0, max_len - len(s)), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
            )
        return np.array(new_seq)

    def random_sample(self, task_id, batch_size=None) -> tuple:
        if batch_size is None:
            batch_size = self.cur_buffer_sizes[task_id]

        idxs = np.random.choice(
            self.cur_buffer_sizes[task_id], size=batch_size, replace=False
        )

        raw_seq, max_len, seq_len, reward_seq = [], 0, [], []

        for idx in idxs:
            traj = self.traj_buffer[task_id][idx]
            reward_seq.append(self.rew_buffer[task_id][idx])
            raw_seq.append(traj)
            seq_len.append(len(traj))
            max_len = max(max_len, len(traj))
        pad_arr = self.padding_seq(raw_seq, max_len)  # np array detach from the buffer
        pad_rew_arr = self.padding_seq(reward_seq, max_len)
        mask = (
            (torch.arange(max_len)[None] < torch.tensor(seq_len).unsqueeze(-1) - 1)
            .float()
            .to(self.device)
        )

        return (
            torch.FloatTensor(pad_arr).to(self.device),
            torch.FloatTensor(pad_rew_arr).to(self.device),
            mask,
        )

    def random_sample_lst(self, task_id, batch_size=None) -> tuple:
        if batch_size is None:
            batch_size = self.cur_buffer_sizes[task_id]

        idxs = np.random.choice(
            self.cur_buffer_sizes[task_id], size=batch_size, replace=False
        )

        raw_seq, max_len, seq_len, reward_seq = [], 0, [], []

        for idx in idxs:
            traj = self.traj_buffer[task_id][idx]
            reward_seq.append(self.rew_buffer[task_id][idx])
            raw_seq.append(traj)
            seq_len.append(len(traj))
            max_len = max(max_len, len(traj))

        return (raw_seq, reward_seq, seq_len)

    def process_lst_data(self, traj_lst, rew_lst, seq_len) -> Tuple[torch.Tensor]:
        # [B,L,S+A], [B,L,1], [B,]
        assert len(traj_lst) == len(rew_lst) and len(traj_lst) == len(seq_len)
        max_len = 0
        for i in range(len(traj_lst)):
            max_len = max(max_len, len(traj_lst[i]))

        pad_traj_arr, pad_rew_arr = self.padding_seq(
            traj_lst, max_len
        ), self.padding_seq(rew_lst, max_len)
        mask = (
            (torch.arange(max_len)[None] < torch.tensor(seq_len).unsqueeze(-1) - 1)
            .float()
            .to(self.device)
        )  #! check

        return (
            torch.FloatTensor(pad_traj_arr).to(self.device),
            torch.FloatTensor(pad_rew_arr).to(self.device),
            mask,
        )

    def clear(self):
        self.traj_buffer = [[None] * self.max_buffer_size] * self.task_nums
        self.rew_buffer = [[0.0] * self.max_buffer_size] * self.task_nums
        self.cur_buffer_sizes = [0] * self.task_nums
        self.ptrs = [0] * self.task_nums

    @property
    def size(self):
        return self.cur_buffer_sizes

    @property
    def buffer_info(self):
        return [self.traj_buffer, self.rew_buffer, self.ptrs, self.cur_buffer_sizes]

    def clear_task(self, task_id):
        self.ptrs[task_id] = 0
        self.size[task_id] = 0

    def load_buffer(self, buffer_info: Tuple):
        (self.traj_buffer, self.rew_buffer, self.ptrs, self.cur_buffer_sizes) = (
            buffer_info
        )


def save_buffer(replay_buffer, buffer_path: str):
    with open(buffer_path, "wb") as f:
        pickle.dump(replay_buffer.buffer_info, f)


def load_buffer(replay_buffer, buffer_path: str):
    with open(buffer_path, "rb") as f:
        loaded_buffer_info = pickle.load(f)
        replay_buffer.load_buffer(loaded_buffer_info)
        return loaded_buffer_info

import os
import gym
import copy

import numpy as np
from gym import spaces
from pathlib import Path
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2

DEFAULT_SIZE = 500

class MetaEnv(gym.Env):
    def __init__(self,  reward_type="dense"):
        super(MetaEnv, self).__init__()

        self.model = SawyerPickPlaceEnvV2()
        self.task = None
        self.state_dim = 39
        self.action_space = self.model.action_space
        self.action_dim = self.action_space.shape[0]
        self.model._freeze_rand_vec = False
        self.model._set_task_called = True
        self.task_success = False
    def reset(self, init_state = None, goal = None, traj_for_env = None, apply_noise = False, reset_with_noise = False):
        # obs = self.model.reset(init_state = init_state, order = order)
        self.model.reset()
        obs = self.model.reset_model(given_goal = goal)
        if init_state is not None:
            self.model.set_env_state(init_state)
        self.task_success = False

        return obs

    def step(self, action):
        obs, reward, done, _, info = self.model.step(action)
        if int(info["success"]) == 1:
            self.task_success = True
            reward = 200
            done = True
        else:
            reward = 0
        return obs, reward, done, info

    def render(self, mode='human', width=500, height=500):
        return self.model.render(mode, width, height)

    def set_pos(self, state, reset = False):
        return self.model.set_env_state(state)

    def get_state(self):
        return self.model.get_env_state()
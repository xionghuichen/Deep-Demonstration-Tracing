import os
import gym
import copy

import numpy as np
from gym import spaces
from pathlib import Path
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_v2 import SawyerPickPlaceEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_place_wall_v2 import SawyerPickPlaceWallEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_pick_out_of_hole_v2 import SawyerPickOutOfHoleEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_basketball_v2 import SawyerBasketballEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_bin_picking_v2 import SawyerBinPickingEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_wall_v2 import SawyerPushWallEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_button_press_v2 import SawyerButtonPressEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_dial_turn_v2 import SawyerDialTurnEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_close_v2 import SawyerDrawerCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_peg_insertion_side_v2 import SawyerPegInsertionSideEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_reach_v2 import SawyerReachEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_push_v2 import SawyerPushEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_sweep_into_goal_v2 import SawyerSweepIntoGoalEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_window_open_v2 import SawyerWindowOpenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_door_close_v2 import SawyerDoorCloseEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_drawer_open_v2 import SawyerDrawerOpenEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_lever_pull_v2 import SawyerLeverPullEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_shelf_place_v2 import SawyerShelfPlaceEnvV2
from metaworld.envs.mujoco.sawyer_xyz.v2.sawyer_sweep_v2 import SawyerSweepEnvV2

DEFAULT_SIZE = 500

class MetaEnvUniform(gym.Env):
    def __init__(self,  reward_type="dense"):
        super(MetaEnvUniform, self).__init__()
        self.model_dict = {}

        self.model_dict['pick-place-v2'] = SawyerPickPlaceEnvV2()
        self.model_dict['pick-place-wall-v2'] = SawyerPickPlaceWallEnvV2()
        self.model_dict['pick-place-hole-v2'] = SawyerPickOutOfHoleEnvV2()
        self.model_dict['basketball-v2'] = SawyerBasketballEnvV2()
        self.model_dict['bin-picking-v2'] = SawyerBinPickingEnvV2()
        self.model_dict['pick-place-wall-v2'] = SawyerPushWallEnvV2()
        self.model_dict['button-press-v2'] = SawyerButtonPressEnvV2()
        self.model_dict['dial-turn-v2'] = SawyerDialTurnEnvV2()
        self.model_dict['drawer-close-v2'] = SawyerDrawerCloseEnvV2()
        self.model_dict['peg-insert-side-v2'] = SawyerPegInsertionSideEnvV2()
        self.model_dict['reach-v2'] = SawyerReachEnvV2()
        self.model_dict['push-v2'] = SawyerPushEnvV2()
        self.model_dict['sweep-into-v2'] = SawyerSweepIntoGoalEnvV2()
        self.model_dict['window-open-v2'] = SawyerWindowOpenEnvV2()
        self.model_dict['door-close-v2'] = SawyerDoorCloseEnvV2()
        self.model_dict['drawer-open-v2'] = SawyerDrawerOpenEnvV2()
        self.model_dict['lever-pull-v2'] = SawyerLeverPullEnvV2()
        self.model_dict['shelf-place-v2'] = SawyerShelfPlaceEnvV2()
        self.model_dict['sweep-v2'] = SawyerSweepEnvV2()

        self.task = None
        self.state_dim = 39
        self.action_space = self.model_dict['pick-place-v2'].action_space
        self.action_dim = self.action_space.shape[0]

        for k in self.model_dict.keys():
            self.model_dict[k]._freeze_rand_vec = False
            self.model_dict[k]._set_task_called = True

        self.task_success = False
        self.model = None

    def reset(self, init_state = None, goal = None, task = None, traj_for_env = None, apply_noise = False, reset_with_noise = False):
        # obs = self.model.reset(init_state = init_state, order = order)
        print('the current task is', task)
        assert task in ['pick-place-v2', 'pick-place-wall-v2', 'pick-place-hole-v2',
                        'basketball-v2', 'bin-picking-v2', 'pick-place-wall-v2',
                        'button-press-v2', 'dial-turn-v2', 'drawer-close-v2',
                        'peg-insert-side-v2', 'reach-v2', 'push-v2', 'sweep-into-v2',
                        'window-open-v2', 'door-close-v2', 'drawer-open-v2',
                        'lever-pull-v2', 'shelf-place-v2', 'sweep-v2'
                        ]
        self.model = self.model_dict[task]
        self.model.reset()
        self.model.reset_model(given_goal = goal)
        if init_state is not None:
            self.model.set_env_state(init_state)
        obs = self.model._get_obs()
        obs[18:36] = 0
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
        self.model.render_mode = mode
        if mode in self.model.mujoco_renderer._viewers.keys():
            self.model.mujoco_renderer._viewers[mode].cam.azimuth = -20
            self.model.mujoco_renderer._viewers[mode].cam.elevation = -20
        return self.model.render()

    def set_pos(self, state, reset = False):
        return self.model.set_env_state(state)

    def get_state(self):
        return self.model.get_env_state()
from typing import Dict, Optional, Tuple, List

import sys
import cv2
import gym
import numpy as np
import pygame
from gym import spaces
from pygame import gfxdraw

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

"""
Goal list, from easy to hard

------------- Easy ---------
[-3, 5]
------------ Medium --------
[-5, 5], [-5, 7], [-11, -7], [1, -7], [7, -7]
------------- Hard ---------
[-3, 11], [-11, -9], [-1, 3], [3, -3], [3, 9], [11, 11]
"""


def get_intersect(A, B, C, D):
    """
    *input_
    Get the intersection of [A, B] and [C, D]. Return False if segment don't cross.

    :param A: Point of the first segment
    :param B: Point of the first segment
    :param C: Point of the second segment
    :param D: Point of the second segment
    :return: The intersection if any, otherwise None.
    """
    # (A,B), (C,D) = input_[0]
    det = (B[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (B[1] - A[1])
    if det == 0:
        # Parallel
        return None
    else:
        t1 = ((C[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (C[1] - A[1])) / det
        t2 = ((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / det
        if t1 > 1 or t1 < 0 or t2 > 1 or t2 < 0:
            # not intersect
            return None
        else:
            xi = A[0] + t1 * (B[0] - A[0])
            yi = A[1] + t1 * (B[1] - A[1])
            # return np.array([xi, yi])
            return [xi, yi]


def rgb2grey(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


class ContinuousMaze(gym.Env):
    """Continuous maze environment."""

    action_space = spaces.Box(-1, 1, (2,))
    observation_space = spaces.Box(-12, 12, (2,))

    walls = np.array(
        [
            [[-12.0, -12.0], [-12.0, 12.0]],
            [[-10.0, 8.0], [-10.0, 10.0]],
            [[-10.0, 0.0], [-10.0, 6.0]],
            [[-10.0, -4.0], [-10.0, -2.0]],
            [[-10.0, -10.0], [-10.0, -6.0]],
            [[-8.0, 4.0], [-8.0, 8.0]],
            [[-8.0, -4.0], [-8.0, 0.0]],
            [[-8.0, -8.0], [-8.0, -6.0]],
            [[-6.0, 8.0], [-6.0, 10.0]],
            [[-6.0, 4.0], [-6.0, 6.0]],
            [[-6.0, 0.0], [-6.0, 2.0]],
            [[-6.0, -6.0], [-6.0, -4.0]],
            [[-4.0, 2.0], [-4.0, 8.0]],
            [[-4.0, -2.0], [-4.0, 0.0]],
            [[-4.0, -10.0], [-4.0, -6.0]],
            [[-2.0, 8.0], [-2.0, 12.0]],
            [[-2.0, 2.0], [-2.0, 6.0]],
            [[-2.0, -4.0], [-2.0, -2.0]], #
            [[0.0, 6.0], [0.0, 12.0]],
            [[0.0, 2.0], [0.0, 4.0]],
            [[0.0, -8.0], [0.0, -6.0]],
            [[2.0, 8.0], [2.0, 10.0]],
            [[2.0, -8.0], [2.0, 6.0]],
            [[4.0, 10.0], [4.0, 12.0]],
            [[4.0, 4.0], [4.0, 6.0]],
            [[4.0, 0.0], [4.0, 2.0]],
            [[4.0, -6.0], [4.0, -2.0]],
            [[4.0, -10.0], [4.0, -8.0]],
            [[6.0, 10.0], [6.0, 12.0]],
            [[6.0, 6.0], [6.0, 8.0]],
            [[6.0, 0.0], [6.0, 2.0]],
            [[6.0, -8.0], [6.0, -6.0]],
            [[8.0, 10.0], [8.0, 12.0]],
            [[8.0, 4.0], [8.0, 6.0]],
            [[8.0, -4.0], [8.0, 2.0]],
            [[8.0, -10.0], [8.0, -8.0]],
            [[10.0, 10.0], [10.0, 12.0]],
            [[10.0, 4.0], [10.0, 8.0]],
            [[10.0, -2.0], [10.0, 0.0]],
            [[12.0, -12.0], [12.0, 12.0]],
            [[-12.0, 12.0], [12.0, 12.0]],
            [[-12.0, 10.0], [-10.0, 10.0]],
            [[-8.0, 10.0], [-6.0, 10.0]],
            [[-4.0, 10.0], [-2.0, 10.0]],
            [[2.0, 10.0], [4.0, 10.0]],
            [[-8.0, 8.0], [-2.0, 8.0]],
            [[2.0, 8.0], [8.0, 8.0]],
            [[-10.0, 6.0], [-8.0, 6.0]],
            [[-6.0, 6.0], [-2.0, 6.0]],
            [[6.0, 6.0], [8.0, 6.0]],
            [[0.0, 4.0], [6.0, 4.0]],
            [[-10.0, 2.0], [-6.0, 2.0]],
            [[-2.0, 2.0], [0.0, 2.0]],
            [[8.0, 2.0], [10.0, 2.0]],
            [[-4.0, 0.0], [-2.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0]],
            [[6.0, 0.0], [8.0, 0.0]],
            [[-6.0, -2.0], [2.0, -2.0]],
            [[4.0, -2.0], [10.0, -2.0]],
            [[-12.0, -4.0], [-8.0, -4.0]],
            [[-4.0, -4.0], [-2.0, -4.0]],
            [[0.0, -4.0], [6.0, -4.0]],
            [[8.0, -4.0], [10.0, -4.0]],
            [[-8.0, -6.0], [-6.0, -6.0]],
            [[-2.0, -6.0], [0.0, -6.0]],
            [[6.0, -6.0], [10.0, -6.0]],
            [[-12.0, -8.0], [-6.0, -8.0]],
            [[-2.0, -8.0], [2.0, -8.0]],
            [[4.0, -8.0], [6.0, -8.0]],
            [[8.0, -8.0], [10.0, -8.0]],
            [[-10.0, -10.0], [-8.0, -10.0]],
            [[-4.0, -10.0], [4.0, -10.0]],
            [[-12.0, -12.0], [12.0, -12.0]],
        ]
    )

    def __init__(self) -> None:
        self.screen = None
        self.isopen = True
        self.all_pos = []
        self.global_view = None

        # param
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        # reward definition
        self.survival_reward = (
            0.0
        )  # -1.0 for every new step, to encourage shortest path
        self.hit_wall_reward = -20
        self.reach_goal_reward = 100
        self.move_forward_bonus = 1.0

        # for render
        self.screen_dim = 500
        self.bound = 13
        self.local_view_dim = 100
        self.local_view_size = (10, 10)
        self.with_local_view = False
        self.scale = self.screen_dim / (self.bound * 2)
        self.offset = self.screen_dim // 2

    def _execution(self, action):
        return self.pos + action

    def set_pos(self, state):
        self.pos = state[-2:].copy()

    def get_distance(self, corner, pos):
        min_dist = np.linalg.norm(pos - corner)
        for wall in self.walls:
            intersection = get_intersect(wall[0], wall[1], pos, corner)
            if intersection is not None:
                min_dist = min(min_dist, np.linalg.norm(intersection - pos))
        return min_dist

    def get_obstacle_local_view(self, pos):
        """
        Get local view with 8 corners
        """

        coord_offset = np.array(
            [
                [0.0, 1.0],
                [0.0, -1.0],
                [-1.0, 0.0],
                [1.0, 0.0],  # up, down, left, right
                [-1.0, 1.0],
                [1.0, 1.0],
                [-1.0, -1.0],
                [1.0, -1.0],  # # up-left, up-right, down-left, down-right
            ]
        )
        corners = coord_offset + np.array(
            [pos] * coord_offset.shape[0]
        )  # eight corner points

        local_view = []
        for corner in corners:
            dist = self.get_distance(corner, pos)
            local_view.append(dist)

        return np.append(np.array(local_view), pos)

    def get_obs(self, pos):
        if self.with_local_view:
            return self.get_obstacle_local_view(pos)
        else:
            return pos

    def check_hit_wall(self, pos, new_pos):
        hit_wall = False
        for wall in self.walls:
            intersection = get_intersect(wall[0], wall[1], pos, new_pos)
            if intersection is not None:
                hit_wall = True
                break
        return hit_wall

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Return (next_state, reward, done, info)
        """
        action = np.clip(action, -1.0, 1.0)  # for safety

        new_pos = self._execution(action)  # get next pos
        done = False
        reward = self.survival_reward

        hit_wall = self.check_hit_wall(self.pos, new_pos)

        if hit_wall:
            reward += self.hit_wall_reward
            new_pos = self.pos  # do not move if hit the wall
            done = True
        else:
            # # encourage to approach the goal
            # reward += (
            #     np.linalg.norm(self.pos - self.GOAL_POINT)
            #     - np.linalg.norm(new_pos - self.GOAL_POINT)
            # ) * self.move_forward_bonus
            reward += 0.0

        # whether reach the goal
        if np.linalg.norm(new_pos - self.GOAL_POINT) <= 0.5:
            reward += self.reach_goal_reward
            done = True

        self.pos = new_pos
        self.all_pos.append(self.pos.copy())

        next_state = self.get_obs(self.pos)
        # return self.pos.copy(), 0.0, False, {}
        return next_state, reward, done, {}

    def reset(
        self, seed=0, start=np.zeros(2), goal=np.ones(2), with_local_view=False
    ) -> np.ndarray:
        self.seed = seed

        # init start and end
        self.ORIGIN_POINT = start
        self.GOAL_POINT = goal
        self.with_local_view = with_local_view

        if with_local_view:
            self.state_dim = self.state_dim + 8

        self.pos = self.ORIGIN_POINT
        self.all_pos.append(self.pos.copy())
        return self.get_obs(self.pos)

    def render(self, mode: str = "human"):
        if self.screen is None:
            try:
                pygame.display.list_modes()
            except:
                import os

                os.environ["SDL_VIDEODRIVER"] = "dummy"

            # init screen
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill(BLACK)

        # history positions
        start_x, start_y = self.all_pos[0] * self.scale + self.offset  # start point
        gfxdraw.filled_circle(self.surf, int(start_x), int(start_y), 1, BLUE)
        for pos in self.all_pos[1:]:
            x, y = pos * self.scale + self.offset
            gfxdraw.filled_circle(self.surf, int(x), int(y), 1, RED)

        # walls
        for wall in self.walls:
            x1, y1 = wall[0] * self.scale + self.offset
            x2, y2 = wall[1] * self.scale + self.offset
            gfxdraw.line(self.surf, int(x1), int(y1), int(x2), int(y2), WHITE)

        # goals
        goal_x, goal_y = self.GOAL_POINT * self.scale + self.offset
        gfxdraw.filled_circle(self.surf, int(goal_x), int(goal_y), 1, GREEN)

        self.surf = pygame.transform.flip(self.surf, flip_x=False, flip_y=True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False

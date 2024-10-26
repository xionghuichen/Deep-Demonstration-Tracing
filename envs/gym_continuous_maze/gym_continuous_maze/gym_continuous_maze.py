from typing import Dict, Optional, Tuple, List

import sys
import cv2
import gym
import numpy as np
import pygame
import random
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


def quick_ex(A, B, C, D):
    """
    :return: True if [A,B] and [C,D] will never intersect, False if possible.
    """
    return (
        max(A[0], B[0]) < min(C[0], D[0])
        or max(C[0], D[0]) < min(A[0], B[0])
        or max(A[1], B[1]) < min(C[1], D[1])
        or max(C[1], D[1]) < min(A[1], B[1])
    )


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
    # if quick_ex(A,B,C,D):
    #     return None
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
    max_drift_scale = 0.1

    default_walls = np.array(
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
            [[-2.0, -4.0], [-2.0, -2.0]],  #
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

    def __init__(
        self,
        hit_wall_reward=-20,
        reach_goal_reward=100,
        obstacle_prob=0.0,
        local_view_num=-1,
        local_view_depth=1.5,
        action_disturbance=0.0,
    ) -> None:
        self.screen = None
        self.isopen = True
        self.all_pos = []
        self.global_view = None

        # param
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        # reward definition
        self.survival_reward = (
            0.0  # -1.0 for every new step, to encourage shortest path
        )
        self.hit_wall_reward = hit_wall_reward
        self.reach_goal_reward = reach_goal_reward
        self.move_forward_bonus = 1.0

        # for render
        self.screen_dim = 500
        self.bound = 13
        self.local_view_dim = 100
        self.local_view_size = (10, 10)
        self.with_local_view = False
        self.scale = self.screen_dim / (self.bound * 2)
        self.offset = self.screen_dim // 2
        self.walls = np.copy(self.default_walls)
        self.set_coordinate_drift()

        self.obstacle_prob = obstacle_prob

        self.local_view_num = local_view_num
        self.local_view_depth = local_view_depth

        self.action_disturbance = action_disturbance

    def set_max_drift_scale(self, custom_max_drift_scale):
        self.max_drift_scale = custom_max_drift_scale

    def set_coordinate_drift(
        self,
    ):
        self.x_drift, self.y_drift = (1 - 2 * np.random.rand(2)) * self.max_drift_scale

    def custom_walls(self, walls: np.ndarray) -> None:
        self.walls = np.copy(walls)

    def generate_random_maze(self):
        """generate maze by random bfs algo"""
        # mainly copy from https://blog.csdn.net/juzihongle1/article/details/73135920
        num_rows = num_cols = 12

        M = np.zeros((num_rows, num_cols, 5), dtype=np.uint8)

        image = np.zeros((num_rows * 10, num_cols * 10), dtype=np.uint8)

        r = 0
        c = 0
        history = [(r, c)]

        while history:
            r, c = random.choice(history)
            M[r, c, 4] = 1
            history.remove((r, c))
            check = []

            if c > 0:
                if M[r, c - 1, 4] == 1:
                    check.append("L")
                elif M[r, c - 1, 4] == 0:
                    history.append((r, c - 1))
                    M[r, c - 1, 4] = 2
            if r > 0:
                if M[r - 1, c, 4] == 1:
                    check.append("U")
                elif M[r - 1, c, 4] == 0:
                    history.append((r - 1, c))
                    M[r - 1, c, 4] = 2
            if c < num_cols - 1:
                if M[r, c + 1, 4] == 1:
                    check.append("R")
                elif M[r, c + 1, 4] == 0:
                    history.append((r, c + 1))
                    M[r, c + 1, 4] = 2
            if r < num_rows - 1:
                if M[r + 1, c, 4] == 1:
                    check.append("D")
                elif M[r + 1, c, 4] == 0:
                    history.append((r + 1, c))
                    M[r + 1, c, 4] = 2

            if len(check):
                move_direction = random.choice(check)
                if move_direction == "L":
                    M[r, c, 0] = 1
                    c = c - 1
                    M[r, c, 2] = 1
                if move_direction == "U":
                    M[r, c, 1] = 1
                    r = r - 1
                    M[r, c, 3] = 1
                if move_direction == "R":
                    M[r, c, 2] = 1
                    c = c + 1
                    M[r, c, 0] = 1
                if move_direction == "D":
                    M[r, c, 3] = 1
                    r = r + 1
                    M[r, c, 1] = 1

            for row in range(0, num_rows):
                for col in range(0, num_cols):
                    cell_data = M[row, col]
                    for i in range(10 * row + 2, 10 * row + 8):
                        image[i, range(10 * col + 2, 10 * col + 8)] = 255
                    if cell_data[0] == 1:
                        image[range(10 * row + 2, 10 * row + 8), 10 * col] = 255
                        image[range(10 * row + 2, 10 * row + 8), 10 * col + 1] = 255
                    if cell_data[1] == 1:
                        image[10 * row, range(10 * col + 2, 10 * col + 8)] = 255
                        image[10 * row + 1, range(10 * col + 2, 10 * col + 8)] = 255
                    if cell_data[2] == 1:
                        image[range(10 * row + 2, 10 * row + 8), 10 * col + 9] = 255
                        image[range(10 * row + 2, 10 * row + 8), 10 * col + 8] = 255
                    if cell_data[3] == 1:
                        image[10 * row + 9, range(10 * col + 2, 10 * col + 8)] = 255
                        image[10 * row + 8, range(10 * col + 2, 10 * col + 8)] = 255

            M[int(num_rows / 2), int(num_cols / 2), 0] = M[
                int(num_rows / 2), int(num_cols / 2), 1
            ] = 1
            M[int(num_rows / 2) - 1, int(num_cols / 2) - 1, 2] = M[
                int(num_rows / 2) - 1, int(num_cols / 2) - 1, 3
            ] = 1
            M[int(num_rows / 2), int(num_cols / 2) - 1, 1] = M[
                int(num_rows / 2), int(num_cols / 2) - 1, 2
            ] = 1
            M[int(num_rows / 2) - 1, int(num_cols / 2), 0] = M[
                int(num_rows / 2) - 1, int(num_cols / 2), 3
            ] = 1
            # Display the image

        return M

    def get_maze_walls(self, maze) -> np.array:
        """get walls from maze

        Arguments:
            maze -- np.array generated by function generate_random_maze

        Returns:
            np.array of walls
        """
        num_rows = num_cols = 12

        walls = []
        for y in range(num_cols):
            x = 0
            while x < num_rows:
                if maze[x, y, 0] == 1:  # no wall in the left
                    x += 1
                    continue
                start = x
                end = x + 1
                while end < num_rows and maze[end, y, 0] == 0:  # connect walls
                    end += 1
                walls.append([[start, y], [end, y]])
                x = end
        walls.append([[0, num_cols], [num_rows, num_cols]])

        for x in range(num_rows):
            y = 0
            while y < num_cols:
                if maze[x, y, 1] == 1:  # no wall in the left
                    y += 1
                    continue
                start = y
                end = y + 1
                while end < num_cols and maze[x, end, 1] == 0:  # connect walls
                    end += 1
                walls.append([[x, start], [x, end]])
                y = end
        walls.append([[num_rows, 0], [num_rows, num_cols]])

        walls = np.array(walls)
        walls[:, :, 0] = walls[:, :, 0] * 2 - num_rows
        walls[:, :, 1] = walls[:, :, 1] * 2 - num_cols
        return walls

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
                if min_dist == 0:
                    print("sfa")
        return min_dist

    def get_obstacle_local_view(self, pos):
        """
        Get local view with 8 corners
        """
        if self.local_view_num == -1:
            """
            Get local view with 8 corners
            """

            coord_offset = (
                np.array(
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
                * self.local_view_depth
            )
            corners = coord_offset + np.array(
                [pos] * coord_offset.shape[0]
            )  # eight corner points

            local_view = []
            for corner in corners:
                dist = self.get_distance(corner, pos)
                local_view.append(dist)

            return np.append(np.array(local_view), pos)
        else:
            theta = np.linspace(
                start=0, stop=2 * np.pi, num=self.local_view_num, endpoint=False
            )
            coord_offset = (
                np.vstack([np.cos(theta), np.sin(theta)]).transpose(1, 0)
                * self.local_view_depth
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
        if self.action_disturbance > 0:
            noise = np.clip(
                np.random.randn(*action.shape) * self.action_disturbance,
                -self.action_disturbance,
                self.action_disturbance,
            )
            action = action + noise
        action = np.clip(action, -1.0, 1.0)  # for safety

        new_pos, old_pos = self._execution(action), self.pos.copy()  # get next pos
        done = False
        reward = self.survival_reward

        hit_wall = self.check_hit_wall(self.pos, new_pos)

        if hit_wall:
            reward += self.hit_wall_reward
            done = True
        else:
            reward += 0.0

        # whether reach the goal
        reach_goal = False
        if (np.linalg.norm(new_pos - self.GOAL_POINT) <= 0.5) and (not hit_wall):
            reward += self.reach_goal_reward
            done = True
            reach_goal = True

        self.pos = new_pos
        self.all_pos.append(self.pos.copy())

        next_state = np.copy(self.get_obs(self.pos))
        return (
            next_state,
            reward,
            done,
            {
                "hit_wall": hit_wall,
                "reach_goal": reach_goal,
                "pos": old_pos,
                "next_pos": new_pos,
            },
        )

    def reset(
        self,
        seed=0,
        start=np.zeros(2),
        goal=np.ones(2),
        with_local_view=False,
        obs_drift=False,
    ) -> np.ndarray:
        self.seed = seed
        self.all_pos = []
        # if load_wall_config is not None:
        #     self.walls=np.copy(load_wall_config)

        # init start and end
        self.ORIGIN_POINT = start
        self.GOAL_POINT = goal
        self.with_local_view = with_local_view

        if with_local_view:
            self.state_dim = 2 + (
                8 if self.local_view_num == -1 else self.local_view_num
            )

        self.pos = self.ORIGIN_POINT
        self.all_pos.append(self.pos.copy())

        self.set_coordinate_drift()

        return self.get_obs(self.pos)

    def render(self, mode: str = "human", exp_traj=None):
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

        # exp traj
        if exp_traj is not None:
            for pos in exp_traj[:, -4:-2]:
                x, y = pos * self.scale + self.offset
                gfxdraw.filled_circle(self.surf, int(x), int(y), 1, WHITE)

        # history positions
        start_x, start_y = self.all_pos[0] * self.scale + self.offset  # start point
        gfxdraw.filled_circle(self.surf, int(start_x), int(start_y), 3, BLUE)
        for pos in self.all_pos[1:]:
            x, y = pos * self.scale + self.offset
            gfxdraw.filled_circle(self.surf, int(x), int(y), 1, RED)

        # connect the dots of the hist trajectory
        for pos1, pos2 in zip(self.all_pos[1:], self.all_pos[:-1]):
            x_pos1, y_pos1 = pos1 * self.scale + self.offset
            x_pos2, y_pos2 = pos2 * self.scale + self.offset
            gfxdraw.line(
                self.surf, int(x_pos1), int(y_pos1), int(x_pos2), int(y_pos2), RED
            )

        # walls
        for wall in self.walls:
            x1, y1 = wall[0] * self.scale + self.offset
            x2, y2 = wall[1] * self.scale + self.offset
            gfxdraw.line(self.surf, int(x1), int(y1), int(x2), int(y2), WHITE)

        # goals
        goal_x, goal_y = self.GOAL_POINT * self.scale + self.offset
        gfxdraw.filled_circle(self.surf, int(goal_x), int(goal_y), 3, GREEN)

        self.surf = pygame.transform.flip(self.surf, flip_x=False, flip_y=True)
        self.screen.blit(self.surf, (0, 0))
        res = None
        if mode == "human":
            pygame.display.flip()
        elif mode == "rgb_array":
            res = np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            res = self.isopen
        self.close()
        return res

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.isopen = False

    def get_drift_observations(self, state):
        obs = np.copy(state)
        obs[-2:] = obs[-2:] + np.array([self.x_drift, self.y_drift])
        return obs

    def add_one_extra_obstacle(self):
        """generate a random rectangle"""

        def generate_rect_vector():
            x = 1.15 + np.random.rand() * 0.1
            if np.random.rand() < 0.5:
                x = -x
            return x

        a, b = generate_rect_vector(), generate_rect_vector()
        extra_walls = np.array(
            [[[0, 0.0], [0, b]], [[0, 0], [a, 0]], [[0, b], [a, b]], [[a, 0], [a, b]]]
        )
        return extra_walls

    def add_extra_static_obstacles(self, exp_traj=None, start=None):
        if self.obstacle_prob == 0:
            return 0

        if (exp_traj is None) or (len(exp_traj) < 5):
            return 0

        extra_walls = []
        exp_traj = np.copy(exp_traj)

        begin_index = 0
        if (
            start is not None
        ):  # warning: only work when the start is in exp or generated by random_start logic
            dists = np.sum(np.abs(exp_traj[:, -4:-2] - start), axis=1, keepdims=False)
            begin_index = np.argmin(dists)

        cd_time = 0
        for exp_traj_index, (sa_pre, sa_1, sa_2, sa_next) in enumerate(
            zip(exp_traj[0:-4], exp_traj[1:-3], exp_traj[2:-2], exp_traj[3:-1])
        ):
            p1, action1 = sa_1[-4:-2], sa_1[-2:]
            p2, action2 = sa_2[-4:-2], sa_2[-2:]
            p_pre, action_pre = sa_pre[-4:-2], sa_pre[-2:]
            p_next, action_next = sa_next[-4:-2], sa_next[-2:]

            if len(extra_walls) >= 4:
                break

            if exp_traj_index < begin_index:
                continue

            if start is not None:
                if max(abs(p1 - start)) < 2.0:
                    continue

            if cd_time >= 1:
                cd_time = cd_time - 1
                continue

            # if not ((action1==action_pre)&(action_pre==action2)&(action2==action_next)).all():# corner!!!
            if not ((action1 == action_pre) & (action_pre == action2)).all():
                continue

            if np.random.rand() < self.obstacle_prob:
                # action1==action2 stands now!
                proj_direction = (
                    np.array([1.0, 0]) if action1[0] == 0 else np.array([0, 1.0])
                )
                proj_direction = (
                    -proj_direction if np.random.rand() > 0.5 else proj_direction
                )

                # (illustration.)
                #
                #   (1.1< <1.2)
                #    A------C
                # ---|P_int |---->>>(exp traj)
                #    |      |
                #    B------D
                #

                lam = 0.4 + 0.2 * np.random.rand()
                P_int = lam * p1 + (1 - lam) * p2
                # protruding_len=0.2+0.15*np.random.rand()# 0.3~0.5
                protruding_len = 0.35
                span_len = (
                    1.1 + 0.2 * np.random.rand()
                )  # 1.1~1.3, note that the rect will only cover one point for 0.6+1.3<2

                B_direction = (
                    -1.0 if np.random.rand() < 0.8 else 1.0
                )  # cover point with 0.8 prob

                A = P_int + protruding_len * proj_direction
                B = (
                    P_int + B_direction * proj_direction * 1.001
                )  # avoid to exception crossing a 0 width alley
                C = A + action1 * span_len
                D = B + action1 * span_len

                rect = np.array([[A, B], [A, C], [B, D], [C, D]])

                extra_walls.append(rect)

                cd_time = 4

        if len(extra_walls) > 0:
            extra_walls = np.vstack(extra_walls)
            self.walls = np.copy(self.walls)
            self.walls = np.vstack([self.walls, extra_walls])

        return len(extra_walls)

import gym
import numpy as np


class BasicMultiTaskEnvHandler:
    def __init__(self, configs) -> None:
        self.configs = configs
        self.dummy_env = self.create_env(configs)

    def create_env(self, *args, **kwargs):
        """
        Create the environment.
        """
        pass

    def get_start_and_goal_from_demo(self, *args, **kwargs):
        """
        get start and end of a traj
        """
        pass

    def config_env_through_demo(self, demo_traj, env):
        """
        Set the environment with the same configuration as the demonstration.
        """
        pass

    def get_osil_reward(
        self, demo_traj, state, action, done, task_reward, *args, **kwargs
    ):
        """
        Get the reward for the OSIL algorithm.
        """
        return self.reward_func(
            demo_traj, state, action, done, task_reward, *args, **kwargs
        )

    def reach_goal(self, goal, state):
        """
        Check if the agent has reached the goal.
        """
        pass


class VPAMMultiTaskEnvHandler(BasicMultiTaskEnvHandler):
    def __init__(self, configs) -> None:
        super().__init__(configs)
        self.coor_dim = (
            8 if self.dummy_env.local_view_num == -1 else self.dummy_env.local_view_num
        )
        self.state_dim = self.coor_dim + 2
        self.action_dim = self.dummy_env.action_dim

        self.reward_func = VPAMOSILRewardFunc(
            self.state_dim, self.action_dim, self.coor_dim, self.configs
        )

    def env_creator(self, raw_rew_func=False, collect_demo_data=False):
        """
        To compatible with the original code, we keep this function.
        """
        return self.create_env(raw_rew_func, collect_demo_data)

    def create_env(self, raw_rew_func=False, collect_demo_data=False):
        """
        We rescale the reward values for more stable RL training, but it is implmented in the environment class.
        This is a unnecessary and redundant implementation. We keep it here for compatibility with the original code.
        """
        env_name = "ValetParkingAssistMaze-v0"
        if collect_demo_data:
            env = gym.make(
                env_name,
                hit_wall_reward=self.configs["hit_wall_reward"],
                reach_goal_reward=self.configs["reach_goal_reward"],
                obstacle_prob=0.0,
                local_view_num=self.configs["local_view_num"],
                local_view_depth=self.configs["local_view_depth"],
            )
        elif raw_rew_func:
            env = gym.make(
                env_name,
                hit_wall_reward=-20,
                reach_goal_reward=100,
                obstacle_prob=self.configs["obstacle_prob"],
                local_view_num=self.configs["local_view_num"],
                local_view_depth=self.configs["local_view_depth"],
                action_disturbance=self.configs["action_disturbance"],
            )
        else:
            env = gym.make(
                env_name,
                hit_wall_reward=self.configs["hit_wall_reward"],
                reach_goal_reward=self.configs["reach_goal_reward"],
                obstacle_prob=self.configs["obstacle_prob"],
                local_view_num=self.configs["local_view_num"],
                local_view_depth=self.configs["local_view_depth"],
                action_disturbance=self.configs["action_disturbance"],
            )
        env.reset(with_local_view=True)
        env.action_space.seed(self.configs["seed"])
        return env

    def reach_goal(self, goal, state):
        if np.linalg.norm(state[-2:] - goal) <= 0.5:
            succeed = True
        else:
            succeed = False
        return succeed

    def get_start_and_goal_from_demo(
        self,
        traj,
        random_start=False,
        noise_scaler=0.1,
    ):
        """
        get start and end of a traj
        traj:
        [
            [s_0, a_0],
            [s_1, a_1],
            ...
            [s_{T-1}, a_{T-1}]
        ]
        """
        ind = np.random.choice(len(traj)) if random_start else int(0)
        start = np.array(traj[ind][self.coor_dim : -self.action_dim], dtype=np.int64)
        start = start.astype(np.float64)
        noise = (1 - 2 * np.random.rand(*start.shape)) * noise_scaler
        start += noise

        last_step, last_action = (
            np.array(traj[-1][self.coor_dim : -self.action_dim], dtype=np.int64),
            np.array(traj[-1][-self.action_dim :], dtype=np.int64),
        )
        goal = last_step + last_action
        return start, goal

    def config_env_through_demo(self, demo_traj, env, task_config):
        start, goal = self.get_start_and_goal_from_demo(
            demo_traj,
            random_start=(np.random.rand() < self.configs["random_start_rate"]),
            noise_scaler=self.configs["noise_scaler"],
        )
        env.custom_walls(task_config["map"])
        env.add_extra_static_obstacles(exp_traj=demo_traj, start=start)
        state, done = (
            env.reset(
                seed=self.configs["seed"],
                start=start,
                goal=goal,
                with_local_view=True,
            ),
            False,
        )
        return state, goal, done


class BasicOSILRewardFunc:
    def __init__(
        self,
        configs,
    ):
        self.configs = configs

    def __call__(self, demo_traj, state, action, done, hit_wall, task_reward):
        pass


class VPAMOSILRewardFunc(BasicOSILRewardFunc):
    def __init__(self, state_dim, action_dim, coor_dim, configs):
        super().__init__(configs)
        self.stand_dist_reward = False
        self.action_weight = self.configs["action_weight"]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.coor_dim = coor_dim

    def __call__(self, demo_traj, state, action, done, task_reward):
        min_dist_idx, min_dist = self.find_min_dist(demo_traj, state, action)
        manual_reward_fun_kwargs = {
            "min_dist": min_dist,
            "min_dist_idx": min_dist_idx,
            "traj_len": len(demo_traj),
            "distance_weight": self.configs["distance_weight"],
            "done": done,
            "max_space_dist": self.configs["max_space_dist"],
            "reward_fun_type": self.configs["reward_fun_type"],
        }
        rew = manual_reward_fun(**manual_reward_fun_kwargs)
        osil_reward = rew + task_reward
        osil_reward /= self.configs["scale"]
        return osil_reward

    def find_min_dist(self, target_traj, state, action):
        if self.stand_dist_reward:
            query_value = np.hstack((state, self.action_weight * action))
            dist_array = np.sqrt(np.mean(np.square(target_traj - query_value), axis=-1))

        else:
            query_value = state[self.coor_dim : self.state_dim]
            # compute euclidean distance
            dist_array = np.sqrt(
                np.sum(
                    np.square(
                        target_traj[:, self.coor_dim : self.state_dim] - query_value
                    ),
                    axis=-1,
                )
            )
        min_dist_idx = np.argmin(dist_array)
        target_s_a = target_traj[min_dist_idx]
        state_error = np.sqrt(
            np.square(
                target_s_a[self.coor_dim : self.state_dim]
                - state[self.coor_dim : self.state_dim]
            ).sum()
        )
        if self.stand_dist_reward:
            action_error = np.sqrt(
                np.square(target_s_a[self.state_dim :] - action).sum()
            )
        else:
            action_error = (
                1
                / np.exp(state_error)
                * np.sqrt(np.square(target_s_a[self.state_dim :] - action).sum())
            )
        if self.stand_dist_reward:
            min_dist = np.sqrt(np.square(target_s_a - query_value).sum())
        else:
            min_dist = state_error + action_error
        return min_dist_idx, min_dist


def get_env_handle(env_name, configs) -> BasicMultiTaskEnvHandler:
    if env_name == "ValetParkingAssistMaze-v0":
        return VPAMMultiTaskEnvHandler(configs)
    else:
        raise NotImplementedError


def get_bound_ilr_reward(
    min_dist, max_space_dist, raw_id, traj_len, distance_weight, done
):
    # min_dist_idx
    ilr_reward_base = 1.0
    base_ilr = ilr_reward_base
    base_dis = ilr_reward_base
    ilr_reward = base_ilr - min(min_dist, max_space_dist)  # [-1,1]
    distance_reward = base_dis - (traj_len - 1 - raw_id) / traj_len  # positive [0-1]
    if min_dist > max_space_dist:
        distance_reward = base_dis - 1
    if done:
        return ilr_reward + distance_weight * distance_reward
    return ilr_reward


def get_origin_ilr_reward(min_dist, raw_id, traj_len, distance_weight, done):
    ilr_reward = 1.0 - min_dist
    distance_reward = 1.0 - (traj_len - 1 - raw_id) / traj_len
    if done:
        return ilr_reward + distance_weight * distance_reward
    return ilr_reward


def manual_reward_fun(**kwargs):
    # "the final bonus only support one exp traj in KDTree"
    if "reward_fun_type" not in kwargs.keys():
        raise KeyError("Must specify reward_fun_type in params!")

    if kwargs["reward_fun_type"] == "origin":
        return get_origin_ilr_reward(
            kwargs["min_dist"],
            kwargs["min_dist_idx"],
            kwargs["traj_len"],
            kwargs["distance_weight"],
            kwargs["done"],
        )
    elif kwargs["reward_fun_type"] == "bound":
        return get_bound_ilr_reward(
            kwargs["min_dist"],
            kwargs["max_space_dist"],
            kwargs["min_dist_idx"],
            kwargs["traj_len"],
            kwargs["distance_weight"],
            kwargs["done"],
        )
    else:
        raise NotImplementedError(
            f"No reward function type {kwargs['reward_fun_type']}"
        )

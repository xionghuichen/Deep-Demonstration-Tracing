import os
import os.path as osp
import sys
project_dir = str(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

import numpy as np
import gym
import time
from collections import deque
import copy
import pickle

import gym
from RLA import exp_manager, time_tracker, logger

from utils import set_seed
from config_loader import get_alg_args, write_config
from buffer import save_buffer, load_buffer
from trainer import VPAMTrainer, ROBOTTrainer
from algo import MT_GoalMapPolicyMultiTaskSACAgent, MT_GoalPolicySACAgent
from multi_task_env_handler import get_env_handle, BasicMultiTaskEnvHandler
from envs.obstacle_policy import Initial_Obstacle_Policy
from CONST import *

gym.register(
    id="ValetParkingAssistMaze-v0",
    entry_point="envs.gym_continuous_maze.gym_continuous_maze.gym_continuous_maze:ContinuousMaze",
    max_episode_steps=100,
)

# get current file path
CURRENT_FILE_DIRNAME = os.path.dirname(os.path.abspath(__file__))

############################################## env specific functions ##############################################


def append_env_config(configs, env_name, env):
    if env_name == "ValetParkingAssistMaze-v0":
        configs["state_dim"] = env.state_dim
        configs["action_dim"] = env.action_dim
        eps = 0.2  # for maze env, we may want actor output boundary action value
        configs["action_high"] = float(env.action_space.high[0] + eps)
        configs["action_low"] = float(env.action_space.low[0] - eps)
    else:
        pass


def noisy_observation(env, state, configs):
    # next_obs = env.get_drift_observations(next_state) if configs['max_drift_scale'] > 0 else next_state
    return state

def get_trainer_policy(config):
    if config["benchmark_name"] == "ValetParkingAssistMaze-v0":
        return VPAMTrainer, MT_GoalMapPolicyMultiTaskSACAgent
    elif config["benchmark_name"] == "metaworld":
        return ROBOTTrainer, MT_GoalPolicySACAgent
    else:
        raise NotImplementedError


############################################## env specific functions ##############################################


def train(configs, env, env_handler: BasicMultiTaskEnvHandler):
    # init modules
    trainer_class, policy_class = get_trainer_policy(configs)
    trainer = trainer_class(CURRENT_FILE_DIRNAME, configs, env_handler)
    trainer.collect_demonstrations(configs)
    policy = policy_class(configs)
    data_collect_env = env_handler.env_creator(
        configs,
        collect_demo_data=True,
    )
    trainer.init_policy(policy, data_collect_env)
    trainer.init_training_setup()

    time.sleep(3.0)
    buffer_dir = trainer.buffer_dir
    demo_dir = trainer.demo_dir

    # save configs
    dump_configs = copy.copy(configs)
    del dump_configs["map_fig_dict"]
    write_config(dump_configs, os.path.join(exp_manager.results_dir, "configs.yml"))

    # TODO: 以下变量临时赋值，二次重构时要放到trainner内部
    iid_train_task_ids = trainer.iid_train_task_ids
    all_trajs = trainer.all_trajs
    policy = trainer.policy
    img_dir = trainer.img_dir
    if configs["benchmark_name"] == "ValetParkingAssistMaze-v0":
        task_id_to_task_config_list = trainer.task_id_to_task_config_list
    elif configs["benchmark_name"] == "metaworld":
        all_env_label_id = {'pick-place-v2': 0, 'pick-place-wall-v2': 1, 'pick-place-hole-v2':2, 'single': 3, configs['task']: 4}

    # start training and evaluation
    iid_train_task_ids = iid_train_task_ids

    logger.info("\n\nsample datas: ", all_trajs[0])

    best_eval_returns = dict()
    runned_episodes = dict()

    saved_maze_num = dict()
    for task_id in iid_train_task_ids:
        best_eval_returns[task_id] = float("-inf")
        runned_episodes[task_id] = 0
        saved_maze_num[task_id] = 0

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    t = 0

    recent_buffer_list = deque(maxlen=configs["recent_buf_len"])
    task_id = None
    task_config = None
    while t < int(configs["max_timesteps"] * 1.01):
        task_id, update = trainer.sample_next_task()
        if update:
            if t != 0:
                buffer_task_config = task_config if configs["benchmark_name"] == "ValetParkingAssistMaze-v0" else all_env_label_id[task_label]
                recent_buffer_list.append(
                    [
                        copy.deepcopy(policy.trans_buffer),
                        copy.deepcopy(policy.traj_buffer),
                        task_id,
                        buffer_task_config,
                    ]
                )
            traj_path = os.path.join(demo_dir, str(task_id) + "_traj")
            trans_path = os.path.join(buffer_dir, str(task_id) + "_trans")
            load_buffer(policy.traj_buffer, traj_path)
            if os.path.exists(trans_path):
                load_buffer(policy.trans_buffer, trans_path)
            else:
                policy.trans_buffer.clear()
            _, traj = policy.traj_buffer.random_sample(0)  # train traj

            if configs["benchmark_name"] == "metaworld":
                traj, traj_mujoco, goal, task_label = traj
        
        # Set Demo Config
        demo_traj = traj
        if configs["benchmark_name"] == "ValetParkingAssistMaze-v0":
            task_config = task_id_to_task_config_list[task_id]
            demo_config = {"task_config": task_config}
        elif configs["benchmark_name"] == "metaworld":
            demo_config = {
                "traj_mujoco": traj_mujoco,
                "goal": goal,
                "task_label": task_label,
            }

        logger.info(
            f"Training on training task {task_id}"
        )
        state, goal, done = env_handler.config_env_through_demo(
            demo_traj, env, demo_config
        )

        # Set obstacle
        if configs["benchmark_name"] == "metaworld":
            whether_obstacle = np.random.rand()<configs['obstacle_prob'] # whether add obstacle
            obstacle_policy = Initial_Obstacle_Policy(configs['task'])
            obstacle_try, obstacle_success = 0, False  
        else:
            whether_obstacle = False
            obstacle_success = None

        actions = []

        pre_action = None
        while not done:
            t += 1
            exp_manager.time_step_holder.set_time(t)
            episode_timesteps += 1

            if pre_action is None:
                pre_action = np.zeros(env.action_space.shape)
            obs = noisy_observation(env, state, configs)

            # Select action randomly or according to policy
            if policy.trans_buffer.size[0] < configs["start_timesteps"]:
                action = env.action_space.sample()
            else:
                action = policy.select_action(obs, traj, training=True)

            if whether_obstacle:
                action, obstacle_try, obstacle_success, obstacle_info = obstacle_policy(obs, action, obstacle_try, obstacle_success)

            actions.append(action)

            # Perform action
            next_state, reward, done, step_info = env.step(action)
            reward = env_handler.get_osil_reward(demo_traj, state, action, done, reward)

            logger.ma_record_tabular("mean_reward", reward, record_len=100, freq=1000)
            next_obs = noisy_observation(env, next_state, configs)

            # Store data in replay buffer
            with time_tracker("policy learning"):
                policy.trans_buffer.insert(obs, action, next_obs, reward, done, 0)
                if t % configs["update_freq"] == 0:
                    info = policy.update(
                        obs,
                        action,
                        next_obs,
                        reward,
                        done,
                        0,
                        task_config if configs["benchmark_name"] == "ValetParkingAssistMaze-v0" else all_env_label_id[task_label],
                        recent_buffer_list,
                    )
                    if info is not None:
                        for key, value in info.items():
                            logger.ma_record_tabular(
                                "mt_train_" + key + "/train_task_" + str(task_id),
                                value,
                                record_len=100,
                                freq=1000,
                                exclude=["csv"],
                            )
                            logger.ma_record_tabular(
                                "mt_train_" + key, value, record_len=100, freq=1000
                            )

            state = next_state
            pre_action = action
            episode_reward += reward

            if done:  # timelimit or hitwall
                obs = noisy_observation(env, state, configs)
                runned_episodes[task_id] += 1
                policy.trans_buffer.stored_eps_num = runned_episodes[task_id]
                succeed = env_handler.reach_goal(goal, state, env)
                saved_maze_num[task_id] = (saved_maze_num[task_id] + 1) % 5
                trainer.update_stats_after_episode(task_id, succeed, obstacle_success)

                if runned_episodes[task_id] % 50 < 5 and configs["benchmark_name"] == "ValetParkingAssistMaze-v0":
                    trainer.visualize_trajs(
                        env,
                        traj,
                        img_dir,
                        task_id,
                        runned_episodes[task_id],
                        task_config,
                        succeed,
                    )

                logger.info(
                    f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Task: {task_id} Reward: {episode_reward:.3f}"
                )
                if runned_episodes[task_id] % 10 == 0:
                    logger.record_tabular(
                        "mt_train_train_return/scene_",
                        episode_reward,
                        exclude=["csv"],
                    )  # too many data thus excluding
                    logger.record_tabular("mt_train_train_return", episode_reward)

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                trainer.evaluation(task_id, runned_episodes, task_config)
                # save data
                with time_tracker("save buffer io"):
                    save_buffer(policy.trans_buffer, trans_path)

            if (
                configs["time_tracker_log_freq"] != -1
                and (t + 1) % configs["time_tracker_log_freq"] == 0
            ):
                time_tracker.log()
            logger.dump_tabular()


if __name__ == "__main__":
    # get configs
    config_file = osp.join(CURRENT_FILE_DIRNAME, "configs/maze_mt.yml")
    configs = get_alg_args(config_file)
    set_seed(configs["seed"])  # set seed for reproduction
    configs["benchmark_name"] = configs["env_name"]
    env_handler = get_env_handle(configs["env_name"], configs=configs)
    assert isinstance(env_handler, BasicMultiTaskEnvHandler)
    env = env_handler.create_env(configs)
    append_env_config(configs, configs["env_name"], env)

    # init logger
    if configs["debug"]:  # whether we are debugging
        out_dir = osp.join(CURRENT_FILE_DIRNAME, EXP_LOG_NAME, "out_debug")
        # configs["eval_unseen_freq"] = 1
    else:
        out_dir = osp.join(CURRENT_FILE_DIRNAME, EXP_LOG_NAME, "exp")
    os.makedirs(out_dir, exist_ok=True)
    rla_data_root = out_dir
    exp_folder_name = configs["env_name"] + "-test-v0"
    exp_manager.configure(
        exp_folder_name,
        private_config_path=osp.join(CURRENT_FILE_DIRNAME, "rla_config.yml"),
        data_root=rla_data_root,
        code_root=CURRENT_FILE_DIRNAME,
    )
    exp_manager.set_hyper_param(**configs)
    exp_manager.add_record_param(
        [
            "description",
            "multi_map",
            "obstacle_prob",
            "no_coordinate",
            "batch_size",
        ]
    )
    exp_manager.log_files_gen()
    exp_manager.print_args()
    # train
    train(configs, env, env_handler)

import os
import os.path as osp
import numpy as np
import gym
import time
import random
from collections import deque
import copy

import gym
from fvcore.nn import parameter_count_table
from RLA import exp_manager, time_tracker, logger

from utils import set_seed
from config_loader import get_alg_args, write_config
from buffer import save_buffer, load_buffer
from trainer import VPAMTrainer
from algo.sac_goal_map_policy_critic_multi_task import MT_GoalMapPolicyMultiTaskSACAgent
from multi_task_env_handler import get_env_handle, BasicMultiTaskEnvHandler
from CONST import *

gym.register(
    id="ValetParkingAssistMaze-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze",
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
        raise NotImplementedError


def noisy_observation(env, state, configs):
    # next_obs = env.get_drift_observations(next_state) if configs['max_drift_scale'] > 0 else next_state
    return state


############################################## env specific functions ##############################################


def train(configs, env, env_handler: BasicMultiTaskEnvHandler):
    # init modules
    trainer = VPAMTrainer(CURRENT_FILE_DIRNAME, configs, env_handler)
    trainer.collect_demonstrations(configs)
    policy = MT_GoalMapPolicyMultiTaskSACAgent(configs)
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
    task_id_to_task_config_list = trainer.task_id_to_task_config_list
    policy = trainer.policy
    img_dir = trainer.img_dir

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
    while t < int(configs["max_timesteps"] * 1.01):
        task_id, update = trainer.sample_next_task()
        if update:
            if t != 0:
                recent_buffer_list.append(
                    [
                        copy.deepcopy(policy.trans_buffer),
                        copy.deepcopy(policy.traj_buffer),
                        task_id,
                        task_config,
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
        demo_traj = traj
        task_config = task_id_to_task_config_list[task_id]

        logger.info(
            f"Training on training task {task_id} (scene config {task_config['map_id']})"
        )
        state, goal, done = env_handler.config_env_through_demo(
            demo_traj, env, task_config
        )

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
                        task_config,
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
                succeed = env_handler.reach_goal(goal, state)
                saved_maze_num[task_id] = (saved_maze_num[task_id] + 1) % 5
                trainer.update_stats_after_episode(task_id, succeed)

                if runned_episodes[task_id] % 50 < 5:
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
                        "mt_train_train_return/"
                        + "scene_"
                        + str(task_id_to_task_config_list[task_id]["map_id"]),
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

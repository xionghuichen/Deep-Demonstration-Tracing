from cgi import test
import os
from pickle import FALSE
import gym

# import gym_continuous_maze
import numpy as np
import matplotlib.pyplot as plt
import random

from collections import deque
from utils import update_hist

# from multi_task_env_handler import VPAMMultiTaskEnvHandler

# from algo.sac_mlp import SACAgent


# def save_result(configs, model_path, img_path, Model, traj=None, walls=None, env_creator=None, txt_path=None):
#     env = env_creator(raw_rew_func=True) # gym.make(configs["env_name"])
#     env.reset(with_local_view=configs["with_local_view"])
#     if walls is None:
#         walls=env.default_walls

#     get_env_config(env, configs)

#     if traj is None:
#         policy = SACAgent(configs)
#         start, goal = (
#             np.array(configs["start"], dtype=np.int64),
#             np.array(configs["goal"], dtype=np.int64),
#         )
#     else:
#         policy = Model(configs)
#         start, goal = get_start_goal(traj, in_eval=True)

#     policy.init_component()
#     policy.load_model(model_path)
#     policy.actor.eval()  # for dropout

#     env.custom_walls(walls)
#     env.add_extra_static_obstacles(exp_traj=traj,start=start)
#     state, done = (
#         env.reset(
#             seed=100, start=start, goal=goal, with_local_view=configs["with_local_view"]
#         ),
#         False,
#     )

#     if configs['only_replay']:
#         policy.reset_actions(traj)


#     hist_len=configs["hist_len"]
#     cur_hist=deque(maxlen=hist_len)
#     pre_action=None
#     full_trajectory=[]

#     actions_txt,obs_txt=[],[]
#     while not done:
#         if pre_action is None:
#             pre_action=np.zeros(*env.action_space.shape)

#         obs=env.get_drift_observations(state) if configs['max_drift_scale']>0 else state
#         full_trajectory.append(np.hstack([np.copy(obs),np.copy(pre_action)]))

#         cur_hist_np,cur_last_id=update_hist(cur_hist,obs,pre_action,hist_len,dim=0)

#         if traj is None:
#             action = policy.select_action(obs, training=False)
#         else:
#             action = policy.select_action(obs, traj, training=False)
#         actions_txt.append(action),obs_txt.append(obs)
#         pre_action=action
#         state, _, done, _ = env.step(action)
#         if done:
#             obs_txt.append(state)

#     rollout_maze = env.render(mode="rgb_array",exp_traj=traj)
#     env.close()
#     policy.actor.train()


#     with open(txt_path,'w') as f:
#         f.write('exp traj'+str(traj))
#     with open(txt_path,'a') as f:
#         f.write('obs'+str(obs_txt))
#     with open(txt_path,'a') as f:
#         f.write('\n\naction'+str(actions_txt))
#     with open(txt_path,'a') as f:
#         f.write('\n\nenv.all_pos'+str(env.all_pos))


#     plt.imsave(img_path, rollout_maze)
#     print(f"Result saved at {img_path}")

                    
                    
def eval_policy(policy, configs, env_handler, eval_episodes=3, traj=None, walls=None):
    # todo: add hist
    hist_len = 3
    eval_env = env_handler.env_creator(
        raw_rew_func=True
    )  # gym.make(configs["env_name"])

    if walls is not None:
        eval_env.custom_walls(walls)
    avg_return = 0.0
    if traj is None:
        start, goal = (
            np.array(configs["start"], dtype=np.int64),
            np.array(configs["goal"], dtype=np.int64),
        )
    else:
        start, goal = env_handler.get_start_and_goal_from_demo(traj)

    policy.actor.eval()  # for dropout

    short_exp_log = {"return": [], "obstacle_num": []}
    for _ in range(eval_episodes):
        if walls is not None:
            eval_env.custom_walls(walls)
        extra_obstacle_num = eval_env.add_extra_static_obstacles(
            exp_traj=traj, start=start
        )
        state, done = (
            eval_env.reset(
                seed=configs["seed"],
                start=start,
                goal=goal,
                with_local_view=True,
            ),
            False,
        )
        if configs["only_replay"]:
            policy.reset_actions(traj)

        cur_hist = deque(maxlen=hist_len)
        pre_action = None
        full_trajectory = []
        single_epi_return = 0.0
        while not done:
            obs = (
                eval_env.get_drift_observations(state)
                if configs["max_drift_scale"] > 0
                else state
            )
            if pre_action is None:
                pre_action = np.zeros(eval_env.action_space.shape)
            full_trajectory.append(np.hstack([np.copy(obs), np.copy(pre_action)]))

            cur_hist_np, cur_last_id = update_hist(
                cur_hist, obs, pre_action, hist_len, dim=0
            )
            if traj is None:
                action = policy.select_action(obs, training=False)
            else:
                action = policy.select_action(obs, traj, training=False)

            state, reward, done, _ = eval_env.step(action)
            single_epi_return += reward
            pre_action = action
            avg_return += reward

        short_exp_log["return"].append(single_epi_return), short_exp_log[
            "obstacle_num"
        ].append(extra_obstacle_num)

    policy.actor.train()

    avg_return /= eval_episodes

    # logger.info("---------------------------------------")
    # logger.info(f"Evaluation over {eval_episodes} episodes: {avg_return:.3f}")
    # logger.info("---------------------------------------")
    return avg_return, short_exp_log


# def eval_random_all(all_trajs, policy, configs, logger, unseen_num=10):
#     """Evaluate policy on random samples of all goals"""
#     avg_return = 0.0
#     for _ in range(unseen_num):
#         new_traj = random.choice(all_trajs)
#         avg_return += eval_policy(policy, configs, logger, 5, traj=new_traj.copy())
#     avg_return /= unseen_num
#     return avg_return


def test_unseen(
    all_trajs,
    test_task_ids,
    policy,
    configs,
    img_dir,
    map_id_lst,
    all_maps,
    env_handler,
):
    """Test policy on the unseen tasks
    total_short_exp_logs: task_ids:list,return:float,obstacle_num:int
    """
    ret_dict = {}
    avg_return = 0.0
    policy.actor.eval()

    # eval_unseen_task_ids=random.sample(test_task_ids,k=min(50,len(test_task_ids)))
    total_short_exp_logs = {"task_ids": [], "return": [], "obstacle_num": []}
    eval_unseen_task_ids = test_task_ids
    for task_id in eval_unseen_task_ids:
        env = env_handler.env_creator(
            raw_rew_func=True
        )  # gym.make(configs["env_name"])
        env.custom_walls(map_id_lst[task_id]["map"])
        # env.walls=all_maps[map_id_lst[task_id]]

        new_traj = all_trajs[task_id]
        cur_return, short_exp_log = eval_policy(
            policy,
            configs,
            env_handler,
            3,
            traj=new_traj.copy(),
            walls=map_id_lst[task_id]["map"],
        )

        # add details
        total_short_exp_logs["task_ids"] += [task_id] * len(short_exp_log["return"])
        total_short_exp_logs["return"] += short_exp_log["return"]
        total_short_exp_logs["obstacle_num"] += short_exp_log["obstacle_num"]

        avg_return += cur_return
        ret_dict[task_id] = cur_return

        # save result
        start, goal = env_handler.get_start_and_goal_from_demo(new_traj)
        env.custom_walls(map_id_lst[task_id]["map"])
        obstacle_num = env.add_extra_static_obstacles(exp_traj=new_traj, start=start)
        state, done = (
            env.reset(
                seed=100,
                start=start,
                goal=goal,
                with_local_view=True,
            ),
            False,
        )

        pre_action = None
        hist_len = 3
        cur_hist = deque(maxlen=hist_len)
        full_trajectory = []

        actions_txt, obs_txt, epi_return = [], [], 0.0

        if configs["only_replay"]:
            policy.reset_actions(new_traj)

        while not done:
            obs = (
                env.get_drift_observations(state)
                if configs["max_drift_scale"] > 0
                else state
            )

            if pre_action is None:
                pre_action = np.zeros(env.action_space.shape)
            full_trajectory.append(np.hstack([np.copy(obs), np.copy(pre_action)]))

            cur_hist_np, cur_last_id = update_hist(
                cur_hist, obs, pre_action, hist_len, dim=0
            )
            action = policy.select_action(obs, new_traj, training=False)

            obs_txt.append(obs), actions_txt.append(action)
            pre_action = action
            state, reward, done, _ = env.step(action)
            epi_return += reward
            if done:
                obs_txt.append(state)

        total_short_exp_logs["task_ids"].append(task_id)
        total_short_exp_logs["return"].append(epi_return)
        total_short_exp_logs["obstacle_num"].append(obstacle_num)

        rollout_maze = env.render(mode="rgb_array", exp_traj=new_traj)
        img_path = os.path.join(img_dir, str(task_id) + "_test_unseen.png")
        txt_path = os.path.join(img_dir, str(task_id) + "_test_unseen_traj.txt")

        with open(txt_path, "w") as f:
            f.write("exp traj" + str(new_traj))
        with open(txt_path, "a") as f:
            f.write("obs" + str(obs_txt))
        with open(txt_path, "a") as f:
            f.write("\n\naction" + str(actions_txt))
        with open(txt_path, "a") as f:
            f.write("\n\nenv.all_pos" + str(env.all_pos))

        plt.imsave(img_path, rollout_maze)
        env.close()

    policy.actor.train()
    avg_return /= len(eval_unseen_task_ids)

    total_short_exp_logs["return"] = np.array(
        total_short_exp_logs["return"], dtype=np.float32
    )
    total_short_exp_logs["obstacle_num"] = np.array(
        total_short_exp_logs["obstacle_num"], dtype=np.int32
    )

    return avg_return, total_short_exp_logs

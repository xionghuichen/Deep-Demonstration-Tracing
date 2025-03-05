from cgi import test
import os
from pickle import FALSE
import gym

import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import pickle
from collections import deque

import torch

from utils import image_to_video
from obstacle_policy import Initial_Obstacle_Policy


def eval_policy(policy, configs, logger, add_obstacle, eval_episodes=1, traj=None, walls=None,
                env_handler=None, reward_fun=None, eval_env = None, draw_env = None, visual = False, videopath = None, task_id = 0, prefix = '', save_traj = True):
    # !warning minor bug, add_obstacle then we refer to success_obstacle
    hist_len = 3 if not configs["is_sequence_data"] else policy.hist_len

    videoimages = []
    agent_traj = []
    agent_obs_traj = []
    agent_action_traj = []
    agent_state_traj = []

    atten_wei_lst = []
    if videopath is not None and not os.path.exists(videopath):
        os.makedirs(videopath)

    if env_handler is not None:
        assert eval_env is None
        eval_env = env_handler.create_prototype()  # gym.make(configs["env_name"])

    # if visual:
    #     eval_env.eval()

    traj_mujoco = None
    goal = None
    task_label = None

    if isinstance(traj, list):
        if len(traj) == 2:
            traj, traj_mujoco = traj
        elif len(traj) == 3:
            traj, traj_mujoco, goal = traj
        elif len(traj) == 4:
            traj, traj_mujoco, goal, task_label = traj

    if walls is not None:
        eval_env.custom_walls(walls)
    avg_return = 0.0
    avg_return_il = 0.0
    assert traj_mujoco is not None
    env_state = traj_mujoco[0]

    policy.actor.eval()  # for dropout
    for epIdx in range(eval_episodes):

        state, done = (
        eval_env.reset(
            init_state = env_state,
            goal = goal,
            task = task_label,
        ),
        False,
        )
        # ---------------------obstacle logic---------------------
        whether_obstacle = add_obstacle
        obstacle_policy = Initial_Obstacle_Policy(configs['task'])
        obstacle_try, obstacle_success = 0, False  
        # obstacle_try: try to add obstacle for how many times, in pick place refer to how many times we set action[3]=-1
        # success_obstacle: whether we successfully add obstacle, in pick place refer to whether we successfully [drop] the block
        # ---------------------obstacle logic end----------------------------

        ######################
        if draw_env is not None:
            draw_env.reset(
                            init_state=env_state,
                            goal = goal,
                            task = task_label)

        episode_irl = []
        cur_hist=deque(maxlen=hist_len)
        pre_action=None
        stepIdx = 0

        while not done:
            obs = state
            if pre_action is None:
                pre_action = np.zeros(eval_env.action_space.shape)
                
            if traj is None:
                policy_results = policy.select_action(obs, training=False, return_atten_wei_lst = True)
            else:
                policy_results = policy.select_action(obs, traj, training=False, return_atten_wei_lst = False) ############yjy mod return actio and log_prob, atten_wei_lst qishi cun l log_prob

            atten_weights = None
            action = policy_results
            atten_wei_lst.append(atten_weights)
            # ---------------------obstacle logic---------------------
            if whether_obstacle:
                action, obstacle_try, obstacle_success, obstacle_info = obstacle_policy(obs, action, obstacle_try, obstacle_success)
            # ---------------------obstacle logic end---------------------

            if visual and videopath is not None:
                image = eval_env.render(mode = "rgb_array")
                image = np.ascontiguousarray(image)
            agent_obs_traj.append(obs)
            agent_action_traj.append(action)

            agent_traj.append(action)
            next_state, reward, done, _ = eval_env.step(action)

            if done:
                agent_traj.append(action)

            if visual and videopath is not None:
                ilr_reward_scaled, reward_info = reward_fun(traj, state, action, done, task_reward=0.0, return_info=True)
                ilr_reward = ilr_reward_scaled*configs['scale']
                episode_irl.append(ilr_reward)
                idx = reward_info['raw_idx']

                # cv2.putText(image, 'step: {}'.format(stepIdx, ilr_reward), (50, 40),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                # cv2.putText(image, 'il rew: {:.4f}'.format(ilr_reward, ilr_reward), (50, 80),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                # cv2.putText(image, 'env rew: {:.4f}'.format(reward, ilr_reward), (50, 120),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                # cv2.putText(image, 'min dist: {:.4f}/{:.2f}'.format(ilr_info[0], configs['max_space_dist'][0]), (50, 160),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                # # cv2.putText(image, 'gripper dist: {:.4f}'.format(ilr_info[1]), (50, 200),
                # #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                # # cv2.putText(image, 'ctrl dist: {:.4f}/{:.2f}'.format(ilr_info[2], configs['max_space_dist'][1]), (50, 240),
                # #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                # # cv2.putText(image, 'item dist: {:.4f}'.format(ilr_info[3]), (50, 280),
                # #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                # # cv2.putText(image, 'action dist: {:.4f}'.format(ilr_info[4]), (50, 320),
                # #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                # cv2.putText(image, 'state error: {:.4f}'.format(ilr_info[5]), (50, 200),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                # cv2.putText(image, 'action error: {:.4f}'.format(ilr_info[6]), (50, 240),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                # cv2.putText(image, 'policy: {:.2f},{:.2f},{:.2f},{:.2f}'.format(*action), (50, 280),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)
                # cv2.putText(image, 'demo: {:.2f},{:.2f},{:.2f},{:.2f}'.format(*ilr_info[-1]), (50, 320),
                #             cv2.FONT_HERSHEY_COMPLEX, 0.75, (255,255,255), 2)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if draw_env is not None:
                    if stepIdx + 1 <= len(traj_mujoco):
                        demo_state = traj_mujoco[stepIdx]
                        draw_env.set_pos(demo_state)
                        demoimage = draw_env.render(mode="rgb_array")
                        demoimage = np.ascontiguousarray(demoimage)
                        cv2.putText(demoimage, 'demonstration {}'.format(stepIdx), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0,
                                    (255,255,255), 3)
                        demoimage = cv2.cvtColor(demoimage, cv2.COLOR_BGR2RGB)

                    draw_state = traj_mujoco[idx]
                    draw_env.set_pos(draw_state)
                    cmpimage = draw_env.render(mode="rgb_array")
                    cmpimage = np.ascontiguousarray(cmpimage)
                    cv2.putText(cmpimage, 'matched state {}'.format(idx), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0,
                                (255,255,255), 3)
                    cmpimage = cv2.cvtColor(cmpimage, cv2.COLOR_BGR2RGB)

                    image = np.concatenate([image, cmpimage, demoimage], axis=1).astype(np.uint8)
                videoimages.append(image)

                if done:
                    image_to_video(videoimages, os.path.join(videopath, prefix + '{}_{}'.format(task_id, epIdx)))
                    videoimages = []
                    stepIdx = 0
                    print('episode_irl', episode_irl)

            state = next_state
            avg_return += reward
            if visual and videopath is not None:
                avg_return_il = avg_return_il + reward + ilr_reward
            stepIdx += 1

    policy.actor.train()

    avg_return /= eval_episodes
    avg_return_il /= eval_episodes

    print('avg_return: ', avg_return)
    print('avg_return_il: ', avg_return_il)

    logger.info("---------------------------------------")
    logger.info(f"Evaluation over {eval_episodes} episodes: {avg_return:.3f}")
    logger.info("---------------------------------------")
    return avg_return


def eval_random_all(all_trajs, policy, configs, logger, unseen_num=10):
    """Evaluate policy on random samples of all goals
    """
    avg_return = 0.0
    for _ in range(unseen_num):
        new_traj = random.choice(all_trajs)
        avg_return += eval_policy(policy, configs, logger, 5, traj=new_traj.copy())
    avg_return /= unseen_num
    return avg_return

def test_unseen(all_trajs, test_task_ids, policy, configs, logger, img_dir, env_handler, reward_fun, add_obstacle):
    """Test policy on the unseen tasks
    """
    ret_dict = {}
    avg_return = 0.0
    policy.actor.eval()

    eval_unseen_task_ids=test_task_ids
    for task_id in eval_unseen_task_ids:
        print('debug', task_id, len(all_trajs[0]), len(all_trajs[1]), len(all_trajs[2]))
        new_traj, traj_mujoco, goal, task_label = all_trajs[0][task_id], all_trajs[1][task_id], all_trajs[2][task_id], all_trajs[3][task_id]
        traj_length = len(new_traj)
        new_traj, traj_mujoco = new_traj[0:traj_length], traj_mujoco[0:traj_length]
        cur_return = eval_policy(policy, configs, logger, add_obstacle=add_obstacle, eval_episodes=1, traj=[new_traj, traj_mujoco, goal, task_label],
                        env_handler=env_handler, reward_fun=reward_fun)
        avg_return += cur_return
        ret_dict[task_id]=cur_return

    policy.actor.train()
    avg_return /= len(eval_unseen_task_ids)
    return avg_return, ret_dict

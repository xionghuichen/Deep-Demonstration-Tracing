
import os
import os.path as osp
import numpy as np
import gym
import time 
import random
from collections import deque
import matplotlib.pyplot as plt

from fvcore.nn import parameter_count_table
import copy 
from RLA import exp_manager, time_tracker, logger
from utils import set_seed, parameter_count_filter
from config_loader import get_alg_args, write_config
from demo_collector.vpam import get_dataset, get_start_goal
from buffer import save_buffer, load_buffer
# get current file path
CURRENT_FILE_DIRNAME = os.path.dirname(os.path.abspath(__file__))

############################################## env specific functions ##############################################

def env_creator(env_name, configs, raw_rew_func=False):
    if env_name == 'ValetParkingAssistMaze-v0':
        if raw_rew_func:
            env = gym.make(env_name, hit_wall_reward=-20, reach_goal_reward=100, obstacle_prob=configs['obstacle_prob'], 
                    local_view_num=configs['local_view_num'], local_view_depth=configs['local_view_depth'],
                    action_disturbance=configs['action_disturbance'])
        else:
            env = gym.make(env_name, hit_wall_reward=configs['hit_wall_reward'],
                    reach_goal_reward=configs['reach_goal_reward'],
                    obstacle_prob=configs['obstacle_prob'],
                    local_view_num=configs['local_view_num'], 
                    local_view_depth=configs['local_view_depth'],
                    action_disturbance=configs['action_disturbance']
                    )
        env.reset(with_local_view=configs["with_local_view"])
        env.set_max_drift_scale(configs['max_drift_scale'])
        env.action_space.seed(configs["seed"])
    else:
        raise NotImplementedError
    return env

def append_env_config(configs, env_name, env):
    if env_name == 'ValetParkingAssistMaze-v0':
        configs["state_dim"] = env.state_dim
        configs["action_dim"] = env.action_dim
        eps = 0.2  # for maze env, we may want actor output boundary action value
        configs["action_high"] = float(env.action_space.high[0] + eps)
        configs["action_low"] = float(env.action_space.low[0] - eps)
    else:
        raise NotImplementedError


def collect_demonstrations(configs):
    set_seed(0) # set seed 0 for same task selection
    if configs["env_name"] == 'ValetParkingAssistMaze-v0':
        env_for_refresh = env_creator(configs["env_name"], configs, raw_rew_func=True)
        all_trajs, train_map_ids, map_shape, map_fig_dict = get_dataset(
            configs["data_root"], configs["multi_map"], configs["raw_data_file_path"], env_for_refresh, 
            configs["task_num_per_map"], configs["train_ratio"]
        )
        assert all_trajs[0].shape[-1] == (8 if configs['local_view_num']==-1 else configs['local_view_num']) + 4
        configs["map_num"] = len(train_map_ids)  # len(iid_train_m2ts.keys())
        configs["map_shape"] = map_shape
        configs["map_fig_dict"] = map_fig_dict

    else:
        raise NotImplementedError
    
    set_seed(configs["seed"])
    return all_trajs, train_map_ids, map_shape, map_fig_dict


def config_env_through_demo(env, env_name, demo, task_config):
    if env_name == 'ValetParkingAssistMaze-v0':
        start, goal = get_start_goal(
            demo, random_start=(np.random.rand() < configs["random_start_rate"]), noise_scaler=configs["noise_scaler"])        
        env.custom_walls(task_config['map'])
        env.add_extra_static_obstacles(exp_traj=demo, start=start)
        state, done = (
            env.reset(
                seed=configs["seed"],
                start=start,
                goal=goal,
                with_local_view=configs["with_local_view"],
            ),
            False,
        )
    else:
        raise NotImplementedError
    return state, goal, done

def get_osil_reward(env_name, state, action, policy, done, step_info, demo_traj):
    if env_name == 'ValetParkingAssistMaze-v0':
        _,ilr_info = policy.traj_buffer.get_ilr(state, action, 0, done)
        manual_reward_fun_kwargs = {
                                'min_dist':ilr_info['min_dist'],
                                'min_raw_idx':ilr_info['min_raw_idx'],
                                'traj_len':len(demo_traj),
                                'distance_weight':configs['distance_weight'],
                                'done':done,
                                'max_space_dist':configs['max_space_dist'],
                                'is_hit':step_info['hit_wall'],
                                'reward_fun_type':configs['reward_fun_type']
                                }
        def get_bound_ilr_reward(min_dist, max_space_dist, raw_id, traj_len, distance_weight, done):
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
        manual_reward=get_bound_ilr_reward(**manual_reward_fun_kwargs)
        reward += manual_reward
        reward /= configs["scale"]
    else:
        raise NotImplementedError
    return reward

def noisy_observation(env, state, configs):
    # next_obs = env.get_drift_observations(next_state) if configs['max_drift_scale'] > 0 else next_state
    return state

def reach_goal(env_name, state):
    if env_name == 'ValetParkingAssistMaze-v0':
        if np.linalg.norm(state[-2:] - env.GOAL_POINT) <= 0.5:
            succeed=True
        else:
            succeed=False
    else:
        raise NotImplementedError
    return succeed

def visualize_trajs(env_name, traj, img_dir, task_id, run_num, task_config, succeed):
    if env_name == 'ValetParkingAssistMaze-v0':
        rollout_maze = env.render(mode="rgb_array",exp_traj=traj)
        img_path = os.path.join(
                img_dir, '_'.join([str(task_id), "lastest-run-", str(run_num % 5), "map-", str(task_config['map_id'])]) \
                + ("_succeed" if succeed else "_fail") + ".png",
            )# too many data thus excluding
        plt.imsave(img_path, rollout_maze)
    else:
        raise NotImplementedError

def evaluation():
    # Evaluate episode
    if (runned_episodes[task_id]) % configs["eval_freq"] == 0:
        with time_tracker('eval policy'):
            _, eval_traj = policy.traj_buffer.random_sample(0)
            avg_return, eval_exp_logs = eval_policy(policy, configs, logger, traj=eval_traj, walls=cur_task_wall, env_creator=env_creator)
            logger.record_tabular(
                "mt_train_eval_return/" + "scene_" + str(task_id_to_task_config_list[task_id]['map_id']), avg_return, exclude=['csv']
            )
            logger.record_tabular("mt_train_eval_return", avg_return)
            with_obstacle_returns,without_obstacle_returns=np.array(eval_exp_logs['return'])[np.array(eval_exp_logs['obstacle_num'])>0],np.array(eval_exp_logs['return'])[np.array(eval_exp_logs['obstacle_num'])==0]
            if len(with_obstacle_returns)>0:
                logger.record_tabular("mt_train_eval_return/with_obstacle", np.mean(with_obstacle_returns))
            if len(without_obstacle_returns)>0:
                logger.record_tabular("mt_train_eval_return/without_obstacle", np.mean(without_obstacle_returns))


    if sum(runned_episodes.values()) % configs["eval_unseen_freq"] == 0:
        with time_tracker("test unseen"):
            iid_test_ids_activate = iid_test_task_ids
            iid_test_ids_choose = random.sample(list(iid_test_ids_activate), k=min(50, len(iid_test_ids_activate)))
            unseen_result, unseen_exp_log = test_unseen(
                all_trajs, iid_test_ids_choose, policy, configs, logger,
                img_dir, task_id_to_map_list, all_maps, env_creator
            )
        logger.record_tabular("eval_unseen_return", unseen_result)

        temp_map_ids=task_id_to_map_list[unseen_exp_log['task_ids']]
        for mid in set(temp_map_ids):
            logger.record_tabular(f"eval_unseen_return/map_{mid}", 
                                np.mean(unseen_exp_log['return'][temp_map_ids==mid]), exclude=['csv'])

        with_obstacle_returns,without_obstacle_returns=unseen_exp_log['return'][unseen_exp_log['obstacle_num']>0],unseen_exp_log['return'][unseen_exp_log['obstacle_num']==0]
        if len(with_obstacle_returns)>0:
            logger.record_tabular(f"eval_unseen_return/with_obstacle", np.mean(with_obstacle_returns))
        if len(without_obstacle_returns)>0:
            logger.record_tabular(f"eval_unseen_return/without_obstacle", np.mean(without_obstacle_returns))

        if len(ood_tasks) > 0:
            ood_test_ids_choose = random.sample(list(ood_tasks), k=min(50,len(ood_tasks)))
            new_map_unseen_result, unseen_exp_log = test_unseen(
                all_trajs, ood_test_ids_choose, policy, configs, logger, img_dir,
                task_id_to_map_list, all_maps, env_creator)
            logger.record_tabular("eval_unseen_return_new_map", new_map_unseen_result)
            
            temp_map_ids=task_id_to_map_list[unseen_exp_log['task_ids']]
            for mid in set(temp_map_ids):
                logger.record_tabular(f"eval_unseen_return_new_map/new_map_{mid}", np.mean(unseen_exp_log['return'][temp_map_ids==mid]), exclude=['csv'])

            with_obstacle_returns,without_obstacle_returns=unseen_exp_log['return'][unseen_exp_log['obstacle_num']>0],unseen_exp_log['return'][unseen_exp_log['obstacle_num']==0]
            if len(with_obstacle_returns)>0:
                logger.record_tabular(f"eval_unseen_return_new_map/with_obstacle", np.mean(with_obstacle_returns))
            if len(without_obstacle_returns)>0:
                logger.record_tabular(f"eval_unseen_return_new_map/without_obstacle", np.mean(without_obstacle_returns))
            
            if new_map_unseen_result > best_ood_test_return:
                policy.save_model(os.path.join(result_dir, model_name + "_best_ood_test.pth"))
                best_ood_test_return = new_map_unseen_result

        if unseen_result > best_iid_test_return:
            policy.save_model(os.path.join(result_dir, model_name + "_best_iid_test.pth"))
            best_iid_test_return = unseen_result
        
        hard_to_complete_task = []
        for m_a in activate_map_lst:
            activate_failure_rate_cur_map = []

            for t_id in map_to_task_dict[m_a]:
                if t_id in iid_train_task_ids_activate:

                    logger.record_tabular(
                        "eval_failure_rates-detailed/" + "task_"+str(t_id), failure_rates[t_id], exclude=['csv']
                    )
                    if failure_rates[t_id] > 0.5:
                        hard_to_complete_task.append(t_id)
                    activate_failure_rate_cur_map.append(failure_rates[t_id])
            logger.record_tabular(
                "eval_failure_rates/" + "map_"+str(m_a), np.mean(activate_failure_rate_cur_map), exclude=['csv']
            )
            logger.record_tabular("eval_failure_rates/hard_to_complete_task_rate", len(hard_to_complete_task)/len(iid_train_task_ids_activate))
        logger.info("hard_to_complete_task", hard_to_complete_task)
        logger.record_tabular("eval_failure_rates", np.mean(get_failure_rate_list(iid_train_task_ids_activate)))

############################################## env specific functions ##############################################


def load_demo_for_policy(policy, iid_train_task_ids, all_trajs, demo_dir, all_maps, task_id_to_map_list, reuse_existing_buf=True):
    gen_traj_buffer = []
    skip_traj_buffer = []
    for task_id in iid_train_task_ids:
        buf_path = os.path.join(demo_dir, str(task_id) + "_traj")
        if os.path.exists(buf_path) and reuse_existing_buf:
            skip_traj_buffer.append(task_id)
            continue
        else:
            policy.traj_buffer.insert(0, all_trajs[task_id], all_maps[task_id_to_map_list[task_id]])
            save_buffer(policy.traj_buffer, buf_path)
            policy.traj_buffer.clear()
            gen_traj_buffer.append(task_id)
    logger.info(f"gen traj buffer {len(gen_traj_buffer)}", gen_traj_buffer)
    logger.info(f"reuse traj buffer {len(skip_traj_buffer)}", skip_traj_buffer)



def train(configs, data_file_path, env):
    train_map_ids, iid_train_task_ids, all_trajs, all_maps, task_id_to_task_config_list = collect_demonstrations(configs, env)
    policy = Model(configs)
    policy.init_component()
    time.sleep(3.0)
    tmp_data_dir = exp_manager.tmp_data_dir
    gen_data_dir = data_file_path + '_data'
    img_dir = os.path.join(tmp_data_dir, "imgs")
    buffer_dir = os.path.join(tmp_data_dir, "train_buffers")
    demo_dir = os.path.join(tmp_data_dir, "demo_buffer")
    load_demo_for_policy(policy, iid_train_task_ids, all_trajs, demo_dir, all_maps, task_id_to_task_config_list, reuse_existing_buf=True)
    with open(os.path.join(exp_manager.log_dir,"model_parameters.txt"),"w+") as f:
            f.write(''.join(['Actor:\n',parameter_count_table(policy.actor,max_depth=1),'\n\n']))
            f.write(''.join(['Critic drop map net:\n',parameter_count_filter(policy.critic_1, valid= lambda x: 'map_' not in x),'\n\n\n\n\n\n']))
            f.write(''.join(['Critic1:\n',parameter_count_table(policy.critic_1,max_depth=1),'\n\n\n\n\n\n']))
            f.write(''.join(['General Info:\n',parameter_count_table(policy,max_depth=2)]))
    
    # save configs
    dump_configs = copy.copy(configs)
    del dump_configs['map_fig_dict']
    del dump_configs['env_creator']
    write_config(dump_configs, os.path.join(exp_manager.results_dir, "configs.yml"))

    # start training and evaluation
    iid_maps_num = len(train_map_ids)
    activate_map_lst, remain_map_lst = [], list(range(iid_maps_num))

    remain_map_lst, activate_map_lst = [], train_map_ids
    iid_train_task_ids_activate = iid_train_task_ids
    
    logger.info('\n\nsample datas: ',all_trajs[0])

    best_iid_test_return = float("-inf")
    best_ood_test_return = float("-inf")
    best_eval_returns = dict()
    runned_episodes = dict()
    failure_episodes = dict()
    saved_maze_num = dict()
    for task_id in iid_train_task_ids:
        best_eval_returns[task_id] = float("-inf")
        runned_episodes[task_id] = 0
        failure_episodes[task_id] = deque(maxlen=100)
        saved_maze_num[task_id] = 0

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    t = 0
    model_name = "best_model"
    failure_rates = {}
    for tid in iid_train_task_ids:
        failure_rates[tid] = 1

    def get_failure_rate_list(task_ids):
        return [failure_rates[tid] if tid in failure_rates else 1 for tid in task_ids]
    eps = 0 # 5e-4/(len(iid_train)/200)
    task_repeat_counter = 0
    recent_buffer_list = deque(maxlen=configs["recent_buf_len"])
    task_id = None
    map_id = None
    while t < int(configs["max_timesteps"]*1.01):
        if task_repeat_counter % configs["task_repeated"] == 0:
            if task_repeat_counter != 0:
                recent_buffer_list.append([copy.deepcopy(policy.trans_buffer), copy.deepcopy(policy.traj_buffer), task_id, map_id])
            if np.random.random_sample() < configs['task_random_sample_prob']:
                task_id = random.sample(list(iid_train_task_ids_activate), k=1)[0]
            else:
                failure_rate_list = np.array([failure_rates[tid] if tid in failure_rates else 1 for tid in iid_train_task_ids_activate])
                task_sample_prob = failure_rate_list + eps
                if sum(task_sample_prob) != 0:
                    task_sample_prob = task_sample_prob / sum(task_sample_prob)
                    task_id = np.random.choice(iid_train_task_ids_activate, p=task_sample_prob, replace=False)
                else:
                    task_id = random.sample(list(iid_train_task_ids_activate), k=1)[0]  # degrade
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
        logger.info(f"Training on training task {task_id} (config {task_config})")
        state, goal, done = config_env_through_demo(env, configs['env_name'], demo_traj, task_config)
        
        actions = []

        pre_action = None
        while not done:
            t += 1
            exp_manager.time_step_holder.set_time(t)
            episode_timesteps += 1

            if pre_action is None:
                pre_action=np.zeros(env.action_space.shape)
            obs = noisy_observation(env, state, configs)

            # Select action randomly or according to policy
            if policy.trans_buffer.size[0] < configs["start_timesteps"]:
                action = env.action_space.sample()
            else:
                action = policy.select_action(obs, traj, training=True)

            actions.append(action)

            # Perform action
            p = time_tracker.add("step env and get reward")
            with p:
                next_state, reward, done, step_info = env.step(action) 
                reward = get_osil_reward(configs['env_name'], state, action, policy, done, step_info, demo_traj)                
            logger.ma_record_tabular("mean_reward", reward, record_len=100, freq=1000)
            next_obs = noisy_observation(env, next_state, configs)

            # Store data in replay buffer
            with time_tracker("policy learning"):
                policy.trans_buffer.insert(obs, action, next_obs, reward, done, 0)
                if t % configs["update_freq"] == 0:
                    info = policy.update(obs, action, next_obs, reward, done, 0, map_id, recent_buffer_list)
                    if info is not None:
                        for key, value in info.items():
                            logger.ma_record_tabular( "mt_train_" + key + "/train_task_" + str(task_id),
                                value, record_len=100, freq=1000, exclude=['csv'])
                            logger.ma_record_tabular("mt_train_" + key, value, record_len=100, freq=1000)

            state = next_state
            pre_action = action
            episode_reward += reward

            if done:# timelimit or hitwall
                obs= noisy_observation(env, state, configs)
                runned_episodes[task_id] += 1
                policy.trans_buffer.stored_eps_num = runned_episodes[task_id]
                task_repeat_counter += 1
                succeed = reach_goal(configs['env_name'], state)
                failure_episodes[task_id].append(0 if succeed else 1)
                failure_rates[task_id] = np.mean(failure_episodes[task_id])
                
                saved_maze_num[task_id] = (saved_maze_num[task_id] + 1) % 5
                if runned_episodes[task_id] % 50 < 5:
                    visualize_trajs(configs['env_name'], traj, img_dir, task_id, task_config, succeed)
                logger.info(
                    f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Task: {task_id} Reward: {episode_reward:.3f}"
                )
                if runned_episodes[task_id] % 10 == 0:
                    logger.record_tabular(
                        "mt_train_train_return/" + "scene_" + str(task_id_to_task_config_list[task_id]['map_id']), episode_reward, exclude=['csv']) # too many data thus excluding
                    logger.record_tabular("mt_train_train_return", episode_reward)

                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                evaluation()
                # save data
                with time_tracker("save buffer io"):
                    save_buffer(policy.trans_buffer, trans_path)

            if (configs["time_tracker_log_freq"] != -1 and (t + 1) % configs["time_tracker_log_freq"] == 0):
                time_tracker.log()
            logger.dump_tabular()

        if (episode_num + 1) % 500 == 0:
            logger.record_tabular('iid_train_task_num',len(iid_train_task_ids_activate))
            logger.info(f'current trainning on {len(iid_train_task_ids_activate)} tasks')
            logger.record_tabular('iid_train_map_num',len(activate_map_lst))
    f.close()
    
if __name__ == "__main__":
    # get configs
    config_file = osp.join(CURRENT_FILE_DIRNAME,  "config/maze_mt.yml")
    configs = get_alg_args(config_file)
    set_seed(configs["seed"])  # set seed for reproduction
    env = env_creator(configs["env_name"])
    append_env_config(configs, configs["env_name"], env)
    
    # init logger
    if configs["debug"]:  # whether we are debugging
        out_dir = osp.join(CURRENT_FILE_DIRNAME, 'RLA_LOG', 'out_debug')
    else:
        out_dir = osp.join(CURRENT_FILE_DIRNAME, 'RLA_LOG', 'exp')
    os.makedirs(out_dir, exist_ok=True)
    rla_data_root = out_dir
    exp_folder_name = configs["env_name"]
    exp_manager.configure(
        exp_folder_name,
        private_config_path= osp.join(CURRENT_FILE_DIRNAME, 'rla_config.yml'),
        data_root=rla_data_root,
        code_root=CURRENT_FILE_DIRNAME,
    )
    exp_manager.set_hyper_param(**configs)
    exp_manager.add_record_param(
        [
            "description",
            'multi_map',
            'obstacle_prob',
            'no_coordinate', 
            'batch_size',
        ]
    )
    exp_manager.log_files_gen()
    exp_manager.print_args()
    # train
    train(configs)
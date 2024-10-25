import os
import numpy as np
import gym

import cv2

def get_start_goal(traj, random_start=False, noise_scaler=0.1, in_eval=False):
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
    action_dim, coor_dim= 2,2
    ind = np.random.choice(len(traj)) if random_start else int(0)
    start = np.array(traj[ind][-(action_dim+coor_dim):-action_dim], dtype=np.int64)
    start = start.astype(np.float64)
    noise = (1-2*np.random.rand(*start.shape)) * noise_scaler
    start += noise

    last_step, last_action = (
        np.array(traj[-1][-(action_dim+coor_dim):-action_dim], dtype=np.int64),
        np.array(traj[-1][-action_dim:], dtype=np.int64),
    )
    goal = last_step + last_action
    return start, goal

def _refresh_maze_local_view(data_pkg, raw_rew_env, with_local_view):
    """
    Refresh dataset with new localview configs
    """
    print('-------------------')
    print('Refresh dataset with current localview configs!')
    print('-------------------')
    env = raw_rew_env
    new_data_pkg={
        "local_view_num":env.local_view_num,
        "local_view_depth":env.local_view_depth
    }
    all_trajs,map_id_lst,all_maps=data_pkg['all_trajs'],data_pkg['map_id_lst'],data_pkg['all_maps']

    new_all_trajs=[]

    for i,exp_traj in enumerate(all_trajs):
        cur_wall=all_maps[map_id_lst[i]]

        start, goal = get_start_goal(exp_traj)
        env.custom_walls(cur_wall)
        env.reset(start=start, goal=goal, with_local_view=with_local_view)

        new_exp_traj=[]
        for s_a in exp_traj:
            pos,a = s_a[-4:-2], s_a[-2:]
            obs = env.get_obs(pos)
            new_s_a=np.append(obs,a)
            new_exp_traj.append(new_s_a)
        new_all_trajs.append(np.array(new_exp_traj))

    new_data_pkg["all_trajs"],new_data_pkg["all_maps"], new_data_pkg["map_id_lst"]=new_all_trajs, all_maps, map_id_lst
    print('-------------------')
    print('Finish refreshing!')
    print('-------------------')
    return new_data_pkg

import pickle



def _get_raw_data_pkg(data_root, multi_map, with_local_view, raw_data_file_path, env_for_refresh):
    """
    Get raw data package from file or refresh it if necessary
    """
    if multi_map:
        raw_data_file_path=os.path.join(data_root, "multi_map250_data_pkg")
    else:
        raw_data_file_path = os.path.join(data_root, "single_map_data_pkg")        
    data_file_suffix=f"(ln_{env_for_refresh.local_view_num}_ld_{env_for_refresh.local_view_depth})"

    data_file_path=raw_data_file_path+data_file_suffix
    if os.path.exists(data_file_path):
        f = open(data_file_path, "rb")
        data_pkg = pickle.load(f)
    else:
        f = open(raw_data_file_path, "rb")
        data_pkg = pickle.load(f)
        data_pkg = _refresh_maze_local_view(data_pkg, env_for_refresh)

        with open(data_file_path, "wb") as f:
            pickle.dump(data_pkg, f)
    return data_pkg

def get_dataset(data_root, multi_map, raw_data_file_path, env_for_refresh, 
                task_num_per_map, train_ratio):
    data_pkg = _get_raw_data_pkg(data_root, multi_map, raw_data_file_path, env_for_refresh)
    img_recale = 0.2
    map_shape = (int(width * img_recale), int(height * img_recale))
    if multi_map:
        unseen_map_num = 10
        all_trajs = data_pkg['all_trajs']
        task_id_to_map_list = np.array(data_pkg['map_id_lst'])
        all_maps = data_pkg['all_maps']
        all_map_id_list = list(set(data_pkg['map_id_lst']))
        map_to_task_dict = {}
        for mid in all_map_id_list:
            map_to_task_dict[mid] = np.where(task_id_to_map_list == mid)[0]
        test_map_ids = np.random.choice(all_map_id_list, size=unseen_map_num, replace=False)
        train_map_ids = list(set(all_map_id_list) - set(test_map_ids))
        iid_tasks = np.concatenate([np.random.choice(map_to_task_dict[mid], size=task_num_per_map, replace=False) for mid in train_map_ids])
        iid_tasks = np.random.permutation(iid_tasks)
        iid_train_task_ids = iid_tasks[:int(len(iid_tasks) * train_ratio)]
        iid_test_task_ids = iid_tasks[int(len(iid_tasks) * train_ratio):]
        ood_tasks = np.concatenate([map_to_task_dict[mid] for mid in test_map_ids])
        train_task_len_dict = {}
        for tid in iid_train_task_ids:
            train_task_len_dict[tid] = all_trajs[tid].shape[0]
        map_fig_dict = {}
        for mid in train_map_ids:
            env_for_refresh.reset()
            env_for_refresh.custom_walls(all_maps[mid])
            img = env_for_refresh.render(mode="rgb_array")
            env_for_refresh.close()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            width, height = img.shape[:2][::-1]
            img_resize = cv2.resize(img, map_shape, interpolation=cv2.INTER_AREA)
            img_resize[img_resize > 0] = 255
            map_fig_dict[mid] = np.expand_dims(img_resize, axis=0)
        print('-------------------')
        print('Successfully load multi-map dataset!')
        print('-------------------')
    else:
        all_trajs = data_pkg['all_trajs']
        task_id_to_map_list = np.array(data_pkg['map_id_lst'])
        all_maps = data_pkg['all_maps']
        all_map_id_list = list(set(data_pkg['map_id_lst']))
        map_to_task_dict = {}
        for mid in all_map_id_list:
            map_to_task_dict[mid] = np.where(task_id_to_map_list == mid)[0]
        train_map_ids = all_map_id_list
        iid_tasks = np.array(range(len(all_trajs)))
        iid_tasks = np.random.permutation(iid_tasks)
        iid_train_task_ids = iid_tasks[:int(len(iid_tasks) * train_ratio)]
        iid_test_task_ids = iid_tasks[int(len(iid_tasks) * train_ratio):]
        ood_tasks = []
        train_task_len_dict = {}
        for tid in iid_train_task_ids:
            train_task_len_dict[tid] = all_trajs[tid].shape[0]
        map_fig_dict = {}
        for mid in train_map_ids:
            env_for_refresh.reset()
            env_for_refresh.custom_walls(all_maps[mid])
            img = env_for_refresh.render(mode="rgb_array")
            env_for_refresh.close()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            width, height = img.shape[:2][::-1]
            img_resize = cv2.resize(img, map_shape, interpolation=cv2.INTER_AREA)
            img_resize[img_resize > 0] = 255
            map_fig_dict[mid] = np.expand_dims(img_resize, axis=0)

        print('-------------------')
        print('Successfully load single map dataset!')
        print('-------------------')
    return all_trajs, task_id_to_map_list, all_maps, map_to_task_dict, iid_train_task_ids, iid_test_task_ids, ood_tasks, train_task_len_dict, map_fig_dict

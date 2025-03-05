import os
from demo_collector.vpam import get_dataset
import gym
from RLA import logger, time_tracker, exp_manager
import numpy as np
import random
import matplotlib.pyplot as plt


from fvcore.nn import parameter_count_table
from utils import set_seed, parameter_count_filter
# from eval.vpam import eval_policy, test_unseen
from eval.metaworld import eval_policy, test_unseen
from buffer import save_buffer
from task_sampler import TaskSampler
from multi_task_env_handler import BasicMultiTaskEnvHandler


class BasicTrainer:
    """
    BasicTrainer is responsible for managing the training process of a model within a multi-task environment.
    Attributes:
        project_root (str): The root directory of the project.
        configs (dict): Configuration parameters for the trainer.
        env_handler (BasicMultiTaskEnvHandler): Handler for the multi-task environment.
        env_name (str): Name of the environment.
        model_name (str): Name of the model.
        result_dir (str): Directory to store results.
        tmp_data_dir (str): Directory to store temporary data.
        img_dir (str): Directory to store images.
        demo_dir (str): Directory to store demonstration buffers.
        buffer_dir (str): Directory to store training buffers.
        task_sampler (TaskSampler): Sampler for tasks based on configuration parameters.
    Methods:
        __init__(project_root: str, configs: dict, env_handler: BasicMultiTaskEnvHandler):
            Initializes the BasicTrainer with the given project root, configurations, and environment handler.
        collect_demonstrations(configs):
            Collects demonstrations based on the provided configurations.
        init_policy(policy, data_collect_env):
            Initializes the policy with the environment handler and data collection environment, loads demonstrations for the policy, and logs parameters.
        init_training_setup():
            Sets up the training environment by constructing the task set.
        sample_next_task():
            Samples the next task to be performed and returns the task ID and update status.
        log_parameters():
            Logs the parameters of the policy components to a file.
        load_demo_for_policy():
            Loads demonstrations for the policy.
        update_stats_after_episode(task_id, succeed):
            Updates the failure rate of a task based on the success of the episode.
    """

    def __init__(
        self, project_root: str, configs: dict, env_handler: BasicMultiTaskEnvHandler
    ) -> None:
        self.configs = configs
        self.project_root = project_root
        self.env_name = configs["env_name"]
        self.model_name = "best_model"
        self.env_handler = env_handler

        self.result_dir = exp_manager.results_dir
        self.tmp_data_dir = exp_manager.tmp_data_dir
        self.img_dir = os.path.join(self.result_dir, "imgs")
        self.demo_dir = os.path.join(self.tmp_data_dir, "demo_buffer")
        self.buffer_dir = os.path.join(self.tmp_data_dir, "train_buffers")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.demo_dir, exist_ok=True)
        os.makedirs(self.buffer_dir, exist_ok=True)
        self.task_sampler = TaskSampler(
            configs["task_repeated"], configs["task_random_sample_prob"]
        )

    def collect_demonstrations(self, configs):
        self.iid_train_task_ids = None
        self.all_trajs = None
        pass

    def init_policy(self, policy, data_collect_env):
        self.policy = policy
        self.policy.init_component(self.env_handler, data_collect_env)
        self.load_demo_for_policy()
        self.log_parameters()

    def init_training_setup(self):
        self.task_sampler.construct_taskset(self.iid_train_task_ids)

    def sample_next_task(self):
        task_id, update = self.task_sampler.sample_next()
        return task_id, update

    def log_parameters(self):
        with open(
            os.path.join(exp_manager.hyparameter_dir, "model_parameters.txt"), "w+"
        ) as f:
            f.write(
                "".join(
                    [
                        "Actor:\n",
                        parameter_count_table(self.policy.actor, max_depth=1),
                        "\n\n",
                    ]
                )
            )
            f.write(
                "".join(
                    [
                        "Critic drop map net:\n",
                        parameter_count_filter(
                            self.policy.critic_1, valid=lambda x: "map_" not in x
                        ),
                        "\n\n\n\n\n\n",
                    ]
                )
            )
            f.write(
                "".join(
                    [
                        "Critic1:\n",
                        parameter_count_table(self.policy.critic_1, max_depth=1),
                        "\n\n\n\n\n\n",
                    ]
                )
            )
            f.write(
                "".join(
                    ["General Info:\n", parameter_count_table(self.policy, max_depth=2)]
                )
            )

    def load_demo_for_policy(self):
        pass

    def update_stats_after_episode(self, task_id, succeed):
        self.task_sampler.update_failure_rate(task_id, succeed)


class VPAMTrainer(BasicTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.best_ood_test_return = -np.inf
        self.best_iid_test_return = -np.inf

    def collect_demonstrations(self, configs):
        super().collect_demonstrations(configs)
        env_for_refresh = self.env_handler.env_creator()

        if self.configs["multi_map"]:
            raw_data_file_path = os.path.join(
                self.project_root, "data/vpam", "multi_map250_data_pkg"
            )
        else:
            raw_data_file_path = os.path.join(
                self.project_root, "data/vpam", "single_map_data_pkg"
            )
        (
            iid_train_task_ids,
            iid_test_task_ids,
            ood_tasks,
            all_trajs,
            task_id_to_config_list,
            train_map_ids,
            map_shape,
            map_fig_dict,
            all_maps,
            map_to_task_dict,
        ) = get_dataset(
            configs["multi_map"],
            raw_data_file_path,
            env_for_refresh,
            configs["task_num_per_map"],
            configs["train_ratio"],
            configs["seed"],
        )
        assert (
            all_trajs[0].shape[-1]
            == (8 if configs["local_view_num"] == -1 else configs["local_view_num"]) + 4
        )
        # TODO: 这里不应该耦合
        self.configs["map_num"] = len(all_maps)  # len(iid_train_m2ts.keys())
        self.configs["map_shape"] = map_shape
        self.configs["map_fig_dict"] = map_fig_dict
        self.iid_train_task_ids = iid_train_task_ids
        self.iid_test_task_ids = iid_test_task_ids
        self.task_id_to_task_config_list = task_id_to_config_list
        self.ood_tasks = ood_tasks
        self.all_trajs = all_trajs
        self.all_maps = all_maps
        self.train_map_ids = train_map_ids
        self.map_to_task_dict = map_to_task_dict

    def load_demo_for_policy(self):
        """
        TODO: demo 不需要存储也不需要复用，这里的buffer逻辑需要被简化
        """
        gen_traj_buffer = []
        skip_traj_buffer = []
        for task_id in self.iid_train_task_ids:
            buf_path = os.path.join(self.demo_dir, str(task_id) + "_traj")
            if os.path.exists(buf_path):
                skip_traj_buffer.append(task_id)
                continue
            else:
                self.policy.traj_buffer.insert_mujoco(
                    0,
                    self.all_trajs[task_id],
                    self.task_id_to_task_config_list[task_id]["map"],
                )
                save_buffer(self.policy.traj_buffer, buf_path)
                self.policy.traj_buffer.clear()
                gen_traj_buffer.append(task_id)
        logger.info(f"gen traj buffer {len(gen_traj_buffer)}", gen_traj_buffer)
        logger.info(f"reuse traj buffer {len(skip_traj_buffer)}", skip_traj_buffer)

    def visualize_trajs(
        self, env, traj, img_dir, task_id, run_num, task_config, succeed
    ):
        rollout_maze = env.render(mode="rgb_array", exp_traj=traj)
        img_path = os.path.join(
            img_dir,
            "_".join(
                [
                    str(task_id),
                    "lastest-run-",
                    str(run_num % 5),
                    "map-",
                    str(task_config["map_id"]),
                ]
            )
            + ("_succeed" if succeed else "_fail")
            + ".png",
        )  # too many data thus excluding
        plt.imsave(img_path, rollout_maze)

    def evaluation(self, task_id, runned_episodes, task_config):
        """
        TODO: 这里eval 和 test的逻辑过于复杂，后面需要进行重构
        """
        if (runned_episodes[task_id]) % self.configs["eval_freq"] == 0:
            with time_tracker("eval policy"):
                _, eval_traj = self.policy.traj_buffer.random_sample(0)
                avg_return, eval_exp_logs = eval_policy(
                    self.policy,
                    self.configs,
                    traj=eval_traj,
                    walls=task_config["map"],
                    env_handler=self.env_handler,
                )
                logger.record_tabular(
                    "mt_train_eval_return/" + "scene_" + str(task_config["map_id"]),
                    avg_return,
                    exclude=["csv"],
                )
                logger.record_tabular("mt_train_eval_return", avg_return)
                with_obstacle_returns, without_obstacle_returns = (
                    np.array(eval_exp_logs["return"])[
                        np.array(eval_exp_logs["obstacle_num"]) > 0
                    ],
                    np.array(eval_exp_logs["return"])[
                        np.array(eval_exp_logs["obstacle_num"]) == 0
                    ],
                )
                if len(with_obstacle_returns) > 0:
                    logger.record_tabular(
                        "mt_train_eval_return/with_obstacle",
                        np.mean(with_obstacle_returns),
                    )
                if len(without_obstacle_returns) > 0:
                    logger.record_tabular(
                        "mt_train_eval_return/without_obstacle",
                        np.mean(without_obstacle_returns),
                    )

        if sum(runned_episodes.values()) % self.configs["eval_unseen_freq"] == 0:
            with time_tracker("test unseen"):
                iid_test_ids_activate = self.iid_test_task_ids
                iid_test_ids_choose = random.sample(
                    list(iid_test_ids_activate), k=min(50, len(iid_test_ids_activate))
                )
                unseen_result, unseen_exp_log = test_unseen(
                    self.all_trajs,
                    iid_test_ids_choose,
                    self.policy,
                    self.configs,
                    self.img_dir,
                    self.task_id_to_task_config_list,
                    self.all_maps,
                    env_handler=self.env_handler,
                )
            logger.record_tabular("eval_unseen_return", unseen_result)

            temp_map_info = np.array(self.task_id_to_task_config_list)[
                unseen_exp_log["task_ids"]
            ]
            temp_map_ids = []
            for map_info in temp_map_info:
                temp_map_ids.append(map_info["map_id"])
            for mid in set(temp_map_ids):
                logger.record_tabular(
                    f"eval_unseen_return/map_{mid}",
                    np.mean(unseen_exp_log["return"][temp_map_ids == mid]),
                    exclude=["csv"],
                )

            with_obstacle_returns, without_obstacle_returns = (
                unseen_exp_log["return"][unseen_exp_log["obstacle_num"] > 0],
                unseen_exp_log["return"][unseen_exp_log["obstacle_num"] == 0],
            )
            if len(with_obstacle_returns) > 0:
                logger.record_tabular(
                    f"eval_unseen_return/with_obstacle", np.mean(with_obstacle_returns)
                )
            if len(without_obstacle_returns) > 0:
                logger.record_tabular(
                    f"eval_unseen_return/without_obstacle",
                    np.mean(without_obstacle_returns),
                )

            if len(self.ood_tasks) > 0:
                ood_test_ids_choose = random.sample(
                    list(self.ood_tasks), k=min(50, len(self.ood_tasks))
                )
                new_map_unseen_result, unseen_exp_log = test_unseen(
                    self.all_trajs,
                    ood_test_ids_choose,
                    self.policy,
                    self.configs,
                    self.img_dir,
                    self.task_id_to_task_config_list,
                    self.all_maps,
                    self.env_handler,
                )
                logger.record_tabular(
                    "eval_unseen_return_new_map", new_map_unseen_result
                )

                temp_map_info = np.array(self.task_id_to_task_config_list)[
                    unseen_exp_log["task_ids"]
                ]
                temp_map_ids = []
                for map_info in temp_map_info:
                    temp_map_ids.append(map_info["map_id"])

                for mid in set(temp_map_ids):
                    logger.record_tabular(
                        f"eval_unseen_return_new_map/new_map_{mid}",
                        np.mean(unseen_exp_log["return"][temp_map_ids == mid]),
                        exclude=["csv"],
                    )

                with_obstacle_returns, without_obstacle_returns = (
                    unseen_exp_log["return"][unseen_exp_log["obstacle_num"] > 0],
                    unseen_exp_log["return"][unseen_exp_log["obstacle_num"] == 0],
                )
                if len(with_obstacle_returns) > 0:
                    logger.record_tabular(
                        f"eval_unseen_return_new_map/with_obstacle",
                        np.mean(with_obstacle_returns),
                    )
                if len(without_obstacle_returns) > 0:
                    logger.record_tabular(
                        f"eval_unseen_return_new_map/without_obstacle",
                        np.mean(without_obstacle_returns),
                    )

                if new_map_unseen_result > self.best_ood_test_return:
                    self.policy.save_model(
                        os.path.join(
                            self.result_dir, self.model_name + "_best_ood_test.pth"
                        )
                    )
                    self.best_ood_test_return = new_map_unseen_result

            if unseen_result > self.best_iid_test_return:
                self.policy.save_model(
                    os.path.join(
                        self.result_dir, self.model_name + "_best_iid_test.pth"
                    )
                )
            self.best_iid_test_return = unseen_result
            hard_to_complete_task = []
            for m_a in self.train_map_ids:
                activate_failure_rate_cur_map = []
                for t_id in self.map_to_task_dict[m_a]:
                    if t_id in self.iid_train_task_ids:

                        logger.record_tabular(
                            "eval_failure_rates-detailed/" + "task_" + str(t_id),
                            self.task_sampler.failure_rates[t_id],
                            exclude=["csv"],
                        )
                        if self.task_sampler.failure_rates[t_id] > 0.5:
                            hard_to_complete_task.append(t_id)
                        activate_failure_rate_cur_map.append(
                            self.task_sampler.failure_rates[t_id]
                        )
                logger.record_tabular(
                    "eval_failure_rates/" + "map_" + str(m_a),
                    np.mean(activate_failure_rate_cur_map),
                    exclude=["csv"],
                )
                logger.record_tabular(
                    "eval_failure_rates/hard_to_complete_task_rate",
                    len(hard_to_complete_task) / len(self.iid_train_task_ids),
                )
            logger.info("hard_to_complete_task", hard_to_complete_task)
            logger.record_tabular(
                "eval_failure_rates",
                np.mean(
                    self.task_sampler.get_failure_rate_list(self.iid_train_task_ids)
                ),
            )

class ROBOTTrainer(BasicTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.best_ood_test_return = -np.inf
        self.best_iid_test_return = -np.inf
        self.failure_episodes_total_with_obstacle = []
        self.failure_episodes_total_without_obstacle = []
        
    def collect_demonstrations(self, configs):
        super().collect_demonstrations(configs)
        # load demonstrations
        all_trajs = []
        all_mujoco_infos = []
        all_collect_goals = []
        all_task_labels = []
        all_uniform_task_list = ['pick-place-v2', 'pick-place-wall-v2', 'pick-place-hole-v2']

        if configs['task'] != 'metauniform':
            dataPath = f'./robotics_demonstrations/{configs["task"]}-clip-filt'
            dataPath = dataPath.replace('Clean', '')
        else:
            assert configs['task'] == 'metauniform'
            dataPath = None

        # 这里的逻辑是把所有 demo 都加载完
        if configs['shot_test']:
            max_data_num = 30
        else:
            max_data_num = 300
        all_lengths = []

        while True:
            fileIdx = len(all_trajs)
            # fileName = 'traj_{}'.format(fileIdx)
            if  fileIdx >= max_data_num:
                break
            else:
                # filt trajs
                if configs['task'] == 'metauniform':
                    temp_env = all_uniform_task_list[fileIdx % len(all_uniform_task_list)]
                    if temp_env == "pick-place-v2":
                        dataPath = f'./robotics_demonstrations/pick-place-v2-clip-filt'
                    elif temp_env == "pick-place-wall-v2":
                        dataPath = f'./robotics_demonstrations/pick-place-wall-v2-clip-filt'
                    else:
                        assert temp_env == "pick-place-hole-v2"
                        dataPath = f'./robotics_demonstrations/pick-place-hole-v2-clip-filt'
                    print(dataPath)
                    all_task_labels.append(temp_env)
                else:
                    all_task_labels.append(configs['task'])

                f = open(os.path.join(dataPath, 'traj_{}'.format(fileIdx // len(all_uniform_task_list))), 'rb')
                import pickle
                load_state, load_action, mujoco_arr, goal = pickle.load(f)
                print('traj length: {}'.format(len(mujoco_arr)))
                all_lengths.append(len(mujoco_arr))
                if isinstance(load_state, list): load_state = np.array(load_state)
                if isinstance(load_action, list): load_action = np.array(load_action)
                all_trajs.append(np.concatenate((load_state.reshape((-1, load_state.shape[-1])),
                                                 load_action.reshape((-1, load_action.shape[-1]))
                                                 ), axis=1))
                all_mujoco_infos.append(mujoco_arr)
                all_collect_goals.append(goal)
                f.close()

        print('traj length mean: {} max:{} min:{}'.format(np.mean(all_lengths), np.max(all_lengths), np.min(all_lengths)))
        task_ids = list(range(len(all_trajs)))

        if not configs['large_dataset']:
            if configs['multi_traj']:
                iid_train_ids = [0,1,2,3,4,5, -5,-4,-3,-2,-1]  # 220 90
            else:
                iid_train_ids = [configs['demo_idx']]
        else:
            if configs['test'] and not configs['test_without_draw']:
                iid_train_ids = [0,1,2,3,4,5, -5,4,-3,-2,-1]  # 220 90
            else:
                iid_train_ids = list(range(configs['demo_num']))

        print(iid_train_ids)
        # iid_train_ids = random.sample(task_ids, k=220)  # 220 90
                
        def pack_new_list(old_lst:list,ids:list):
            new_lst=[]
            for id in ids:
                new_lst.append(old_lst[id])
            return new_lst
        iid_test_ids = list(set(task_ids) - set(iid_train_ids))
        iid_train, iid_test = pack_new_list(all_trajs, iid_train_ids), pack_new_list(all_trajs, iid_test_ids)
        iid_train_mujoco = pack_new_list(all_mujoco_infos, iid_train_ids)
        iid_test_mujoco = pack_new_list(all_mujoco_infos, iid_test_ids)
        iid_train_goal = pack_new_list(all_collect_goals, iid_train_ids)
        iid_test_goal = pack_new_list(all_collect_goals, iid_test_ids)
        iid_train_task_label = pack_new_list(all_task_labels, iid_train_ids)
        iid_test_task_label = pack_new_list(all_task_labels, iid_test_ids)

        self.all_trajs = all_trajs
        self.all_mujoco_infos = all_mujoco_infos
        self.all_collect_goals = all_collect_goals
        self.all_task_labels = all_task_labels
        
        self.iid_test_ids = iid_test_ids
        self.iid_train, self.iid_test = iid_train, iid_test
        self.iid_train_mujoco, self.iid_test_mujoco = iid_train_mujoco, iid_test_mujoco
        self.iid_train_goal, self.iid_test_goal = iid_train_goal, iid_test_goal
        self.iid_train_task_label, self.iid_test_task_label = iid_train_task_label, iid_test_task_label

        iid_train_m2ts = {0: list(range(len(iid_train)))}
        iid_train_t2m, iid_test_t2m = [0] * len(iid_train), [0] * len(iid_test)
        iid_train_tasks_num = len(iid_train)
        
        iid_maps_num = len(iid_train_m2ts.keys())
        self.activate_map_lst, self.remain_map_lst =  list(range(iid_maps_num)),  []
        self.iid_train_task_ids_activate = list(range(iid_train_tasks_num))
        self.iid_train_m2ts = iid_train_m2ts
        self.iid_train_t2m, self.iid_test_t2m = iid_train_t2m, iid_test_t2m

        print('-------------------')
        print('Successfully load single task dataset!')
        print('-------------------')

        print('iid_train', len(iid_train))
        configs["task_nums"] = 1  # we in fact has only one task in the buffer every time
        # configs["map_num"] = len(configs["env_name_list"])  # maze
        configs["map_num"] = len(all_uniform_task_list)  # maze
        # TODO: 这里不应该耦合
        
        self.iid_train_task_ids = iid_train_ids
        self.iid_test_task_ids = iid_test_ids
        # task config
        self.task_id_to_mujoco_arr_list = all_mujoco_infos
        self.task_id_to_goal_list = all_collect_goals
        self.task_id_to_task_label = all_task_labels
        #
        self.all_trajs = all_trajs

    def load_demo_for_policy(self):
        new_trajs = []
        traj_lengths = []
        for task_id in self.iid_train_task_ids:
            new_traj = self.policy.traj_buffer.insert_mujoco(0, self.all_trajs[task_id], self.task_id_to_mujoco_arr_list[task_id],
                                                      self.task_id_to_goal_list[task_id], self.task_id_to_task_label[task_id])
            save_buffer(
                self.policy.traj_buffer, os.path.join(self.demo_dir, str(task_id) + "_traj")
            )
            self.policy.traj_buffer.clear()
            new_trajs.append(new_traj)
            traj_lengths.append(len(new_traj))
            print('max traj length: ', np.max(traj_lengths))

    def visualize_trajs(
        self, env, traj, img_dir, task_id, run_num, task_config, succeed
    ):
        rollout_maze = env.render(mode="rgb_array", exp_traj=traj)
        img_path = os.path.join(
            img_dir,
            "_".join(
                [
                    str(task_id),
                    "lastest-run-",
                    str(run_num % 5),
                    "map-",
                    str(task_config["map_id"]),
                ]
            )
            + ("_succeed" if succeed else "_fail")
            + ".png",
        )  # too many data thus excluding
        plt.imsave(img_path, rollout_maze)

    def evaluation(self, task_id, runned_episodes):
        reward_fun = self.env_handler.reward_func
        if (runned_episodes[task_id]) % self.configs["eval_freq"] == 0: # self.configs["eval_freq"] !!!!!!!!!!!!!!!!!!!!!!!
            p = time_tracker("eval self.policy [line 292]")
            with p:
                _, eval_traj = self.policy.traj_buffer.random_sample(0)
                avg_return = eval_policy(
                    self.policy, self.configs, logger, add_obstacle=False, eval_episodes = 1,
                    traj=eval_traj, walls=None, env_handler=self.env_handler, reward_fun=reward_fun
                )
                logger.record_tabular("mt_train_eval_return/without_obstacle", avg_return)

                # ---------------------obstacle logic---------------------
                if self.configs['obstacle_prob']>0:
                    _, eval_traj = self.policy.traj_buffer.random_sample(0)
                    avg_return_with_obstacle = eval_policy(
                        self.policy, self.configs, logger, add_obstacle=True, eval_episodes = 1,
                        traj=eval_traj, walls=None, env_handler=self.env_handler)
                    logger.record_tabular("mt_train_eval_return/with_obstacle", avg_return_with_obstacle)

                    # override
                    avg_return = self.configs['obstacle_prob']*avg_return_with_obstacle + (1-self.configs['obstacle_prob'])*avg_return
                # ---------------------obstacle logic end----------------------------


                logger.record_tabular(
                    "mt_train_eval_return/" + "map_" + str(self.iid_train_t2m[task_id]), avg_return
                )
                logger.record_tabular("mt_train_eval_return", avg_return)

                if avg_return >= self.best_iid_test_return:
                    p = time_tracker("save trained models")
                    with p:
                        self.policy.save_model(
                            os.path.join(
                                self.result_dir,  "best_iid_test.pth"
                            )
                        )
                    self.best_iid_test_return = avg_return

        if sum(runned_episodes.values()) % self.configs["eval_unseen_freq"] == 0: # self.configs["eval_unseen_freq"]
            unseen_test_num=50
            p = time_tracker("test unseen [line 341]")
            with p:
                iid_test_ids_choose = random.sample(self.iid_test_ids,
                                                    k=min(unseen_test_num, len(self.iid_test_ids)))
                unseen_result, unseen_ret_dict = test_unseen(
                    [self.all_trajs, self.all_mujoco_infos, self.all_collect_goals, self.all_task_labels], iid_test_ids_choose, self.policy, self.configs, logger,
                    self.img_dir, self.env_handler, reward_fun, add_obstacle=False)
                logger.record_tabular("eval_unseen_return/without_obstacle", unseen_result)

                # ---------------------obstacle logic---------------------
                if self.configs['obstacle_prob']>0:
                    iid_test_ids_choose = random.sample(self.iid_test_ids,
                                                        k=min(unseen_test_num, len(self.iid_test_ids)))
                    unseen_result_with_obstacle, unseen_ret_dict = test_unseen(
                        [self.all_trajs, self.all_mujoco_infos, self.all_collect_goals, self.all_task_labels], iid_test_ids_choose, self.policy, self.configs, logger,
                        self.img_dir, self.env_handler, reward_fun, add_obstacle=True)
                    logger.record_tabular("eval_unseen_return/with_obstacle", unseen_result_with_obstacle)
                    # override
                    unseen_result = self.configs['obstacle_prob']*unseen_result_with_obstacle + (1-self.configs['obstacle_prob'])*unseen_result
                # ---------------------obstacle logic end----------------------------
                logger.record_tabular("eval_unseen_return", unseen_result)
            temp_map_record = {}
            for kk, vv in unseen_ret_dict.items():
                c_m = 0
                if c_m not in temp_map_record.keys():
                    temp_map_record[c_m] = []
                temp_map_record[c_m].append(vv)
            for kk, vv in temp_map_record.items():
                logger.record_tabular(f"eval_unseen_return/map_{kk}", np.mean(vv))

            failure_rates = self.task_sampler.failure_rates
            for m_a in self.activate_map_lst:
                activate_task_ids_cur_map = []
                for t_id in self.iid_train_m2ts[m_a]:
                    if t_id in self.iid_train_task_ids_activate:
                        logger.record_tabular(
                            "eval_failure_rates/" + "task_" + str(t_id), self.task_sampler.failure_rates[t_id], exclude=['csv']
                        )
                        activate_task_ids_cur_map.append(t_id)
                temp_lst = [failure_rates[tid] for tid in activate_task_ids_cur_map]
                logger.record_tabular(
                    "eval_failure_rates/" + "map_" + str(m_a), np.mean(temp_lst)
                )
            temp_lst = [failure_rates[tid] for tid in self.iid_train_task_ids_activate]
            logger.record_tabular("eval_failure_rates", np.mean(temp_lst))

    def update_stats_after_episode(self, task_id, succeed, obstacle_success):
        self.task_sampler.update_failure_rate(task_id, succeed)
        if obstacle_success:
            self.failure_episodes_total_with_obstacle.append(0 if succeed else 1)
        else:
            self.failure_episodes_total_without_obstacle.append(0 if succeed else 1)
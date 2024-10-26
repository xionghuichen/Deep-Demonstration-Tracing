import os
from demo_collector.vpam import get_dataset
import gym
from RLA import logger, time_tracker, exp_manager
import numpy as np
import random

from fvcore.nn import parameter_count_table
from utils import set_seed, parameter_count_filter
from eval.vpam import eval_policy, test_unseen
from buffer import save_buffer


class BasicTrainer:
    def __init__(self, project_root, configs, env_handler) -> None:
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

    def init_policy(self, policy, data_collect_env):
        self.policy = policy
        self.policy.init_component(self.env_handler, data_collect_env)
        self.load_demo_for_policy()
        self.log_parameters()

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

    def collect_demonstrations(self, configs):
        self.iid_train_task_ids = None
        self.all_trajs = None
        pass

    def load_demo_for_policy(self):
        pass


class VPAMTrainer(BasicTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # def env_creator(self, raw_rew_func=False):
    #     """
    #     We rescale the reward values for more stable RL training, but it is implmented in the environment class.
    #     This is a unnecessary and redundant implementation. We keep it here for compatibility with the original code.
    #     """
    #     if raw_rew_func:
    #         env = gym.make(
    #             self.env_name,
    #             hit_wall_reward=-20,
    #             reach_goal_reward=100,
    #             obstacle_prob=self.configs["obstacle_prob"],
    #             local_view_num=self.configs["local_view_num"],
    #             local_view_depth=self.configs["local_view_depth"],
    #             action_disturbance=self.configs["action_disturbance"],
    #         )
    #     else:
    #         env = gym.make(
    #             self.env_name,
    #             hit_wall_reward=self.configs["hit_wall_reward"],
    #             reach_goal_reward=self.configs["reach_goal_reward"],
    #             obstacle_prob=self.configs["obstacle_prob"],
    #             local_view_num=self.configs["local_view_num"],
    #             local_view_depth=self.configs["local_view_depth"],
    #             action_disturbance=self.configs["action_disturbance"],
    #         )
    #     env.reset(with_local_view=True)
    #     env.action_space.seed(self.configs["seed"])
    #     return env

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
        ) = get_dataset(
            configs["multi_map"],
            raw_data_file_path,
            env_for_refresh,
            configs["task_num_per_map"],
            configs["train_ratio"],
        )
        assert (
            all_trajs[0].shape[-1]
            == (8 if configs["local_view_num"] == -1 else configs["local_view_num"]) + 4
        )
        # TODO: 这里不应该耦合
        self.configs["map_num"] = len(train_map_ids)  # len(iid_train_m2ts.keys())
        self.configs["map_shape"] = map_shape
        self.configs["map_fig_dict"] = map_fig_dict
        self.iid_train_task_ids = iid_train_task_ids
        self.iid_test_task_ids = iid_test_task_ids
        self.task_id_to_task_config_list = task_id_to_config_list
        self.ood_tasks = ood_tasks
        self.all_trajs = all_trajs
        self.all_maps = all_maps

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
                self.policy.traj_buffer.insert(
                    0,
                    self.all_trajs[task_id],
                    self.task_id_to_task_config_list[task_id]["map"],
                )
                save_buffer(self.policy.traj_buffer, buf_path)
                self.policy.traj_buffer.clear()
                gen_traj_buffer.append(task_id)
        logger.info(f"gen traj buffer {len(gen_traj_buffer)}", gen_traj_buffer)
        logger.info(f"reuse traj buffer {len(skip_traj_buffer)}", skip_traj_buffer)

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
                    logger,
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
                    logger,
                    self.img_dir,
                    self.task_id_to_task_config_list,
                    self.all_maps,
                    env_handler=self.env_handler,
                )
            logger.record_tabular("eval_unseen_return", unseen_result)

            temp_map_ids = self.task_id_to_task_config_list[unseen_exp_log["task_ids"]][
                "map_id"
            ]
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
                    logger,
                    self.img_dir,
                    self.task_id_to_task_config_list,
                    self.all_maps,
                    self.env_handler,
                )
                logger.record_tabular(
                    "eval_unseen_return_new_map", new_map_unseen_result
                )

                temp_map_ids = self.task_id_to_task_config_list[
                    unseen_exp_log["task_ids"]
                ]["map_id"]
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

                if new_map_unseen_result > best_ood_test_return:
                    self.policy.save_model(
                        os.path.join(
                            self.result_dir, self.model_name + "_best_ood_test.pth"
                        )
                    )
                    best_ood_test_return = new_map_unseen_result

            if unseen_result > best_iid_test_return:
                self.policy.save_model(
                    os.path.join(
                        self.result_dir, self.model_name + "_best_iid_test.pth"
                    )
                )
                best_iid_test_return = unseen_result

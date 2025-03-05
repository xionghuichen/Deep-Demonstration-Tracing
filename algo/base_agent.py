# Created by xionghuichen at 2022/9/21
# Email: chenxh@lamda.nju.edu.cn
import os
import numpy as np
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from torch_utils import preprocess_traj
from CONST import MapType
from buffer import MT_TransitionBuffer, TrajectoryBuffer

# used
class BaseAgent(nn.Module):
    """
    Base Soft Actor Critic
    """

    def __init__(self, configs: dict):
        super(BaseAgent, self).__init__()
        # init hyper param
        ## sac mlp
        self.state_dim = configs["state_dim"]
        self.action_dim = configs["action_dim"]
        self.action_high = configs["action_high"]
        self.action_low = configs["action_low"]
        self.device = configs["device"]
        ## algo
        self.gamma = configs["gamma"]
        self.learn_alpha = configs["learn_alpha"]
        self.start_timesteps = configs["start_timesteps"]
        self.start_epi = configs["start_epi"]
        self.updates_per_step = configs["updates_per_step"]
        self.batch_size = configs["batch_size"]
        self.entropy_target = -self.action_dim
        self.rho = configs["rho"]
        self.buffer_size = configs["buffer_size"]
        ## alpha
        self.alpha_lr = configs["alpha_lr"]
        self.alpha = configs["alpha"]
        ## actor
        self.actor_hidden_size = configs["actor_hidden_size"]
        self.actor_lr = configs["actor_lr"]
        ## critic
        self.critic_hidden_size = configs["critic_hidden_size"]
        self.critic_lr = configs["critic_lr"]
        ## for tensorboard
        self.info = dict()

        ## for reward
        self.action_weight = configs["action_weight"]

        ## for pretrain
        self.batch_with_demo = configs["batch_with_demo"]
        self.env_name = configs["env_name"]
        self.with_local_view = True
        self.scale = configs["scale"]
        self.add_bc_reward = True
        self.do_scale = True

        # for net
        self.share_state_encoder = configs["share_state_encoder"]

        # imitation
        self.with_distance_reward = configs["with_distance_reward"]
        self.distance_weight = configs["distance_weight"]

        self.no_coordinate = configs["no_coordinate"]

        
        
        
        ### sac atten
        self.task_nums = configs["task_nums"]
        ## actor
        self.actor_embed_dim = configs["actor_embed_dim"]
        self.actor_num_heads = configs["actor_num_heads"]
        self.actor_dropout = configs["actor_dropout"]
        self.actor_pos_encode = configs["actor_pos_encode"]

        ## critic
        self.critic_embed_dim = configs["critic_embed_dim"]
        self.critic_num_heads = configs["critic_num_heads"]
        self.critic_dropout = configs["critic_dropout"]
        self.critic_pos_encode = configs["critic_pos_encode"]
        self.alpha_init = configs["alpha"]
        
        ### sac goalcritic
        self.actor_num_encoder_layers = configs["actor_num_encoder_layers"]
        self.actor_num_decoder_layers = configs["actor_num_decoder_layers"]
        self.actor_dim_feedforward = configs["actor_dim_feedforward"]

        ## critic
        self.critic_goal_embed_dim = configs["critic_goal_embed_dim"]
        self.critic_embed_goal = configs["critic_embed_goal"]
        self.alpha_init = configs["alpha"]

        self.critic_num_encoder_layers = configs["critic_num_encoder_layers"]
        self.critic_num_decoder_layers = configs["critic_num_decoder_layers"]
        self.critic_dim_feedforward = configs["critic_dim_feedforward"]

        self.distance_weight = configs["distance_weight"]
        self.reward_fun_type = configs["reward_fun_type"]
        self.max_space_dist = configs["max_space_dist"]
        # self.env_creator = configs["env_creator"]
        self.map_num = configs["map_num"]
        self.no_itor = configs["no_itor"]
        
        
        self.use_transformer = configs["use_transformer"]
        self.use_rnn_critic = configs["use_rnn_critic"]
        self.use_only_decoder = configs["use_only_decoder"]
        self.use_rnn_actor = configs["use_rnn_actor"]
        self.configs = configs
        self.map_type = configs["map_type"]
        self.replace_sample = True

    def init_component(self, env_handler, demo_collect_env):
        # replay buffer
        self.trans_buffer = MT_TransitionBuffer(
            self.state_dim,
            self.action_dim,
            self.buffer_size,
            self.device,
            self.task_nums,
        )

        self.traj_buffer = TrajectoryBuffer(
            self.task_nums,
            self.env_name,
            self.device,
            self.action_weight,
            self.scale,
            self.add_bc_reward,
            self.do_scale,
            self.with_distance_reward,
            self.distance_weight,
            env_handler=env_handler,
            demo_collect_env=demo_collect_env,
            max_space_dist=self.max_space_dist,
            reward_fun_type=self.reward_fun_type,
            no_itor=self.no_itor,
        )

        # alpha, the entropy coefficient
        print(self.alpha_init)
        self.log_alpha = torch.log(
            torch.ones(1, device=self.device) * self.alpha_init
        ).requires_grad_(True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        with torch.no_grad():
            if self.learn_alpha:
                self.alpha = self.log_alpha.exp().item()
            else:
                self.alpha = self.alpha
        self.init_critic()
        self.init_policy()

        self.models = {
            "actor": self.actor,
            "actor_optim": self.actor_optim,
            "critic_1": self.critic_1,
            "critic_1_target": self.critic_1_target,
            "critic_2": self.critic_2,
            "critic_2_target": self.critic_2_target,
            "critic_optim": self.critic_optim,
            "log_alpha": self.log_alpha,
            "alpha_optim": self.alpha_optim,
        }

    ### sample data
    def get_demo_trans(self, task_id, traj_buffer, demo_num=-1):
        if traj_buffer is None:
            traj_buffer = self.traj_buffer
        idx, traj = traj_buffer.random_sample(task_id)
        (
            demo_states,
            demo_actions,
            demo_next_states,
            demo_rewards,
            demo_masks,
        ) = traj_buffer.get_trans(task_id, idx)
        if demo_num > 0:
            sample_idx = torch.randint(demo_states.shape[0], (demo_num,))
            demo_states = demo_states[sample_idx]
            demo_actions = demo_actions[sample_idx]
            demo_next_states = demo_next_states[sample_idx]
            demo_rewards = demo_rewards[sample_idx]
            demo_masks = demo_masks[sample_idx]
        else:
            raise NotImplementedError

        traj = preprocess_traj(traj, self.device)
        return demo_states, demo_actions, traj

    def get_samples(self, task_id, trans_buffer=None, traj_buffer=None, demo_num=-1):
        if trans_buffer is None:
            trans_buffer = self.trans_buffer
        if traj_buffer is None:
            traj_buffer = self.traj_buffer

        states, actions, next_states, rewards, masks = trans_buffer.random_sample(
            task_id, self.batch_size, replace_sample=self.replace_sample
        )
        idx, traj = traj_buffer.random_sample(task_id)

        # add demo to batch
        if self.batch_with_demo:
            (
                demo_states,
                demo_actions,
                demo_next_states,
                demo_rewards,
                demo_masks,
            ) = traj_buffer.get_trans(task_id, idx)
            if demo_num >= 0:
                sample_idx = torch.randint(demo_states.shape[0], (demo_num,))
                demo_states = demo_states[sample_idx]
                demo_actions = demo_actions[sample_idx]
                demo_next_states = demo_next_states[sample_idx]
                demo_rewards = demo_rewards[sample_idx]
                demo_masks = demo_masks[sample_idx]
            states = torch.cat((states, demo_states), dim=0)
            actions = torch.cat((actions, demo_actions), dim=0)
            next_states = torch.cat((next_states, demo_next_states), dim=0)
            rewards = torch.cat((rewards, demo_rewards), dim=0)
            masks = torch.cat((masks, demo_masks), dim=0)

        traj = preprocess_traj(traj, self.device)
        return states, actions, next_states, rewards, masks, traj

    def get_multi_map_samples(self, recent_buf_list):
        res_dict = {}
        for k in [
            "map_info",
            "states",
            "actions",
            "next_states",
            "rewards",
            "masks",
            "traj",
            "goal",
        ]:
            res_dict[k] = []
        do_train = False
        replace_sample = self.replace_sample
        max_len = 0
        for item in recent_buf_list:
            trans_buffer, traj_buffer, task_id, task_config = item
            map_id = task_config["map_id"]
            if replace_sample:
                if (
                    trans_buffer.stored_eps_num < self.start_epi
                    or trans_buffer.size[0] < 2.0
                ):
                    continue
            else:
                if (
                    trans_buffer.stored_eps_num < self.start_epi
                    or trans_buffer.size[0] < self.batch_size
                ):
                    continue
            do_train = True
            if self.map_type == MapType.ID:
                map_info = (
                    torch.tensor(map_id, dtype=torch.int32).to(self.device).long()
                )
            elif self.map_type == MapType.FIG:
                map_info = torch.from_numpy(
                    (self.map_fig_dict[map_id] / 255).astype(np.float32)
                ).to(self.device)
            else:
                raise RuntimeError
            states, actions, next_states, rewards, masks, traj = self.get_samples(
                0,
                trans_buffer,
                traj_buffer,
                int(self.batch_size / len(recent_buf_list)),
            )
            res_dict["map_info"].append(map_info)
            res_dict["states"].append(states)
            res_dict["actions"].append(actions)
            res_dict["next_states"].append(next_states)
            res_dict["rewards"].append(rewards)
            res_dict["masks"].append(masks)
            res_dict["traj"].append(traj.transpose(1, 2)[0])
            res_dict["goal"].append(traj.transpose(1, 2)[0, -1])
            max_len = np.maximum(max_len, traj[0].shape[0])

        if do_train:
            for k, v in res_dict.items():
                if k in ["states", "actions", "next_states", "rewards", "masks"]:
                    res_dict[k] = torch.stack(v, dim=1)
                elif k in ["traj"]:
                    res_dict[k] = pad_sequence(
                        res_dict["traj"], batch_first=True
                    ).transpose(1, 2)
                    pass
                else:
                    res_dict[k] = torch.stack(v, dim=0)
        return res_dict, do_train

    def init_critic(self, *args, **kwargs):
        raise NotImplementedError

    def init_policy(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def update(
        self, *args, **kwargs
    ):
        raise NotImplementedError

    ### decision making
    def load_actor(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found: {}".format(model_path))
        else:
            state_dicts = torch.load(model_path)
            for model in self.models:
                if model == "actor":
                    if isinstance(
                        self.models[model], torch.Tensor
                    ):  # especially for sac, which has log_alpha to be loaded
                        self.models[model] = state_dicts[model][model]
                    else:
                        self.models[model].load_state_dict(state_dicts[model])
        # self.log_alpha.data = self.models["log_alpha"].data
        # self.alpha = self.log_alpha.clone().detach().exp().item()

        print(f"Successfully load model from {model_path}!")

    def squash_action(self, action):
        action = self.action_high * torch.tanh(
            action
        )  # [-inf, +inf] -> [-1.0, 1.0] -> [-1.0-eps, +1.0+eps]
        # action = (self.action_high - self.action_low) / 2.0 * action + (
        #     self.action_high + self.action_low
        # ) / 2.0
        return action

    ### model utils
    def save_model(self, model_path):
        if not self.models:
            raise ValueError("Models to be saved is None!")
        state_dicts = {}
        for model in self.models:
            if isinstance(self.models[model], torch.Tensor):
                state_dicts[model] = {model: self.models[model]}
            else:
                state_dicts[model] = self.models[model].state_dict()
        torch.save(state_dicts, model_path)
        return state_dicts

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model file not found: {}".format(model_path))
        else:
            state_dicts = torch.load(model_path)
            for model in self.models:
                if isinstance(
                    self.models[model], torch.Tensor
                ):  # especially for sac, which has log_alpha to be loaded
                    self.models[model] = state_dicts[model][model]
                else:
                    self.models[model].load_state_dict(state_dicts[model])
        self.log_alpha.data = self.models["log_alpha"].data
        self.alpha = self.log_alpha.clone().detach().exp().item()

        print(f"Successfully load model from {model_path}!")

    @torch.no_grad()
    def select_action(self, state, traj, training):
        action, _ = self(state, traj, training)
        return action.cpu().data.numpy().flatten()

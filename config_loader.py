import argparse
from typing import Dict
from yaml import load, dump, FullLoader


def write_config(configs, yml_file_path):
    with open(yml_file_path, "w", encoding="utf8") as f:
        dump(configs, f, allow_unicode=True)


def _read_config(config_yml_path) -> Dict:
    with open(config_yml_path, "r", encoding="utf-8") as f:
        configs = load(f, Loader=FullLoader)
    return configs


def _parse_args():
    # read config from CLI
    parser = argparse.ArgumentParser()

    # Experiment logging config
    parser.add_argument(
        "--sync_logs", default=False, action="store_true"
    )  # the log will be send to imitator server /out_debug, configs are in rla_config.yml
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument(
        "--description",
        default="default-exp-info",
        type=str,
        help="Make a detailed description about your experiment",
    )

    # experiment setting config
    parser.add_argument("--multi_map", default=False, action="store_true")
    parser.add_argument("--obstacle_prob", default=0.1, type=float)
    parser.add_argument("--no_coordinate", default=False, action="store_true")
    
    parser.add_argument("--demo_num", default=60, type=int)
    
    parser.add_argument("--eval_freq", default=20, type=int)
    parser.add_argument("--eval_unseen_freq", default=1000, type=int)
    
    parser.add_argument("--task", default="maze", type=str)

    args = parser.parse_args()
    return args


def get_env_config(env, configs):
    configs["state_dim"] = env.state_dim
    configs["action_dim"] = env.action_dim
    eps = 0.2  # for maze env, we may want actor output boundary action value
    configs["action_high"] = float(env.action_space.high[0] + eps)
    configs["action_low"] = float(env.action_space.low[0] - eps)


def get_alg_args(config_file):
    args = _parse_args()
    configs = _read_config(config_file)
    configs.update(vars(args))
    if configs["multi_map"]:
        if configs["update_freq"] != 3:
            configs["update_freq"] = 3
            print("The multi_map update_freq will be set to 3")
        if configs["recent_buf_len"] != 5:
            configs["recent_buf_len"] = 5
            print("The multi_map recent_buf_len will be set to 5")
    # if configs['algo']=='DCRL':
    #     configs['baseline']=True
    #     configs['add_bc_reward']=False
    # elif configs['algo']=='DDT':
    #     pass
    # else:
    #     raise NotImplementedError("Unknown algorithm")

    return configs

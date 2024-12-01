# Deep-Demonstration-Tracing

The  Official Code of "Deep Demonstration Tracing: Learning Generalizable Imitator Policy for Runtime Imitation from a Single Demonstration" (ICML'24). It will be released soon.

## quick start


Step1: download demo dataset from [Google Drive](https://drive.google.com/drive/folders/1GIzGRSqSdF7-KkN5dVF_SZHrN3hePmjb?usp=sharing) to `.data`

Step2: install related packages

```
conda create -n ddt python=3.10
pip install -r requirement.txt
```

install environments

```
cd envs/gym_continuous_maze
pip install -e .
```

install experiments management tool RLAssitant

```
git clone https://github.com/polixir/RLAssistant.git
cd RLAssistant
pip install -e .
```

step3: 在Maze环境中运行SAC

```bash
# 默认实验结果会保存到out_mt文件夹下, --description用于标记当前实验结果
python main_ddt.py --device cuda:0 --description "sac maze"

# 默认实验结果会保存到out_test文件夹下，可用于debug
python main_ddt.py --device cuda:0 --description "sac maze" --test
```

**备注**：模型在开始运行前会加载configs/maze_mt.yml中的相关配置，关于相关参数的含义请看对应注释或通过该参数的名字即可得知

## cmd cheetsheet

```bash
# enable multi-map
--multi_map

# add_obstacle prob
--obstacle_prob 0.1

# dont containing coordinate in obs
--no_coordinate

# reward fun
--reward_fun_type bound

# essential exp setup
--debug/test/mt --device cuda:0/1/2/3 --seed 0
```

## 在Maze环境运行实验
```bash
# simplest setting
python -m sac_maze.sac_maze_tp --description "sm_DDT"  --debug --device cuda:0  --seed 0   --reward_fun_type bound --training_method origin  --max_timesteps 1000000

# multi map setting
python -m sac_maze.sac_maze_tp --description "2500_DDT"  --debug --device cuda:0  --seed 0   --reward_fun_type bound --training_method origin  --max_timesteps 1000000

# for more setting checkout out cmd cheetsheet
```

python -m sac_maze.sac_maze_tp --description "sm_DDT"  --test --device cuda:0  --seed 0   --reward_fun_type bound --training_method origin  --max_timesteps 1000000 



删除无用的临时数据：rm -rf RLA_LOG/exp/tmp_data/

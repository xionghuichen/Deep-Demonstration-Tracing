from gym import register

register(
    id="ContinuousMaze-v0",
    entry_point="envs.gym_continuous_maze.gym_continuous_maze.gym_continuous_maze:ContinuousMaze",
    max_episode_steps=100,
)

# Past env for upper level agent decision
# DIY
register(
    id="Planning-v0",
    entry_point="envs.robotics:PlanningEnv",
    max_episode_steps=8,
)

# DIY
register(
    id="Stack-v0",
    entry_point="envs.robotics:StackEnv",
    max_episode_steps=8,
)

# DIY
register(
    id="Collect-v0",
    entry_point="envs.robotics:CollectEnv",
    max_episode_steps=8,
)

# Past env for lower level agent decision
# DIY
register(
    id="Hrl-v0",
    entry_point="envs.robotics:HrlEnv",
    max_episode_steps=50,
)

# DIY
register(
    id="StackHrl-v0",
    entry_point="envs.robotics:StackHrlEnv",
    max_episode_steps=50,
)

# DIY
register(
    id="CollectHrl-v0",
    entry_point="envs.robotics:CollectHrlEnv",
    max_episode_steps=50,
)

# DIY
register(
    id="RenderHrl-v0",
    entry_point="envs.robotics:RenderHrlEnv",
    max_episode_steps=50,
)

# DIY
register(
    id="TestHrl-v0",
    entry_point="envs.robotics:TestHrlEnv",
    max_episode_steps=50,
)

# My clean env for imitator learning
register(
    id="PlanningClean-v0",
    entry_point="envs.robotics:TestCleanEnv",
    max_episode_steps=100,
)

register(
    id="StackClean-v0",
    entry_point="envs.robotics:StackCleanEnv",
    max_episode_steps=100,
)

register(
    id="CollectClean-v0",
    entry_point="envs.robotics:CollectCleanEnv",
    max_episode_steps=100,
)

register(
    id="UniformClean-v0",
    entry_point="envs.robotics:UniformEnv",
    max_episode_steps=100,
)

register(
    id="Pusher-v6",
    entry_point="envs.mujocoEnv:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="Reacher-v6",
    entry_point="envs.mujocoEnv:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)


register(
    id="Pusher-v8",
    entry_point="envs.mujocoEnvforDemo:PusherEnv",
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id="Reacher-v8",
    entry_point="envs.mujocoEnvforDemo:ReacherEnv",
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id="Meta-v2",
    entry_point="envs.metaEnv:MetaEnv",
    max_episode_steps=100,
)

register(
    id="MetaBasketball-v2",
    entry_point="envs.metaEnv:MetaEnvBasketball",
    max_episode_steps=130,
)

register(
    id="MetaBinpicking-v2",
    entry_point="envs.metaEnv:MetaEnvBinPicking",
    max_episode_steps=220,
)

register(
    id="MetaBinpicking-v2",
    entry_point="envs.metaEnv:MetaEnvBinPicking",
    max_episode_steps=220,
)

register(
    id="UniformMeta-v0",
    entry_point="envs.metaEnv:MetaEnvUniform",
    max_episode_steps=150,
)
import gym
import gymnasium as gym1

def convert_space(space):
    """Convert gymnasium space to gym space."""
    if isinstance(space, gym.spaces.Space):
        return space
    elif isinstance(space, gym1.spaces.Box):
        return gym.spaces.Box(
            low=space.low,
            high=space.high,
            shape=space.shape,
            dtype=space.dtype
        )
    elif isinstance(space, gym1.spaces.Discrete):
        return gym.spaces.Discrete(space.n)
    elif isinstance(space, gym1.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete(space.nvec)
    elif isinstance(space, gym1.spaces.MultiBinary):
        return gym.spaces.MultiBinary(space.n)
    elif isinstance(space, gym1.spaces.Tuple):
        return gym.spaces.Tuple([convert_space(s) for s in space.spaces])
    elif isinstance(space, gym1.spaces.Dict):
        return gym.spaces.Dict({key: convert_space(value) for key, value in space.spaces.items()})
    else:
        raise NotImplementedError(f"Conversion for space type {type(space)} not implemented.")
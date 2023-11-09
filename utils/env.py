import gym
import gym_minigrid


def make_env(env_key, seed=None, env_args={}):
    env = gym.make(env_key, **env_args)
    env.seed(seed)
    return env


def env_args_str_to_dict(env_args_str):
    if not env_args_str:
        return {}
    keys = env_args_str[::2]  # Every even element is a key
    vals = env_args_str[1::2]  # Every odd element is a value
    return dict(zip(keys, [eval(v) for v in vals]))

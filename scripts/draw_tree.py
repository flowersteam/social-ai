#!/usr/bin/env python3

import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='SocialAI-DrawingEnv-v1',
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)

# Put all env related arguments after --env_args, e.g. --env_args nb_foo 1 is_bar True
parser.add_argument("--env-args", nargs='*', default=None)


args = parser.parse_args()


env = gym.make(args.env, **env_args_str_to_dict(args.env_args))

# draw tree
env.parameter_tree.draw_tree(
    filename="viz/SocialAIParam/{}_raw_tree".format(args.env),
    ignore_labels=["Num_of_colors"],
    folded_nodes=["Collaboration", "AppleStealing"],
    label_parser={
        "AppleStealing": "Adversarial",
        "Pragmatic_frame_complexity": "Introductory_sequence",
},
    selected_parameters={
        "Env_type": "Information_seeking",
        "Pragmatic_frame_complexity": "Eye_contact",
        "Peer_help": "N",
        "Cue_type": "Pointing",
        "Problem": "Doors",
        "N": "1",
        "Peer": "N",
    }
)

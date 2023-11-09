#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
from utils import *
from models import MultiModalBaby11ACModel
from collections import Counter
import torch_ac
import json
from termcolor import colored, COLORS

from functools import partial
from tkinter import *

from torch.distributions import Categorical

inter_acl = False
draw_tree = True

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size, mask_unobserved=args.mask_unobserved)

    window.show_img(img)

def reset():
    # if args.seed != -1:
    #     env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


tot_bonus = [0]

prev = {
    "prev_obs": None,
    "prev_info": {},
}
shortened_obj_names = {
    'lockablebox'      : 'loc_box',
    'applegenerator'   : 'app_gen',
    'generatorplatform': 'gen_pl',
    'marbletee'        : 'tee',
    'remotedoor'       : 'rem_door',
}

IDX_TO_OBJECT = {v: shortened_obj_names.get(k, k) for k, v in OBJECT_TO_IDX.items()}
# no duplicates
assert len(IDX_TO_OBJECT) == len(OBJECT_TO_IDX)

IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
assert len(IDX_TO_COLOR) == len(COLOR_TO_IDX)


# def to_string(enc):
#     s = "{:<8} {} {} {} {} {:3} {:3} {}\t".format(
#         IDX_TO_OBJECT.get(enc[0], enc[0]),  # obj
#         *enc[1:3],  # x, y
#         IDX_TO_COLOR.get(enc[3], enc[3])[:1].upper(),  # color
#         *enc[4:]  #
#     )
#
#     if IDX_TO_OBJECT.get(enc[0], enc[0]) == "unseen":
#         pass
#         # s = colored(s, "on_grey")
#
#     elif IDX_TO_OBJECT.get(enc[0], enc[0]) != "empty":
#         col = IDX_TO_COLOR.get(enc[3], enc[3])
#         if col in COLORS:
#             s = colored(s, col)
#
#     return s


def step(action):
    if type(action) == np.ndarray:
        obs, reward, done, info = env.step(action)
    else:
        action = [int(action), np.nan, np.nan]
        obs, reward, done, info = env.step(action)


    redraw(obs)

    if done:
        print('done!')
        print('Reward=%.2f' % (reward))
        print('Exploration_bonus=%.2f' % (tot_bonus[0]))
        tot_bonus[0] = 0

        with open(output_file, "a") as f:
            if reward > 0:
                f.write("Success!\n")
            f.write("New episode.\n")

        reset()

    else:
        print('\nStep=%s' % (env.step_count))

        # print to screen
        print("Obs : ", end="")
        print("".join(info["descriptions"]), end="")
        if obs["utterance_history"] != "Conversation: \n":
            print(obs['utterance_history'])
        print("Act : ", end="")

        # write to file
        with open(output_file, "a") as f:
            f.write("Obs : ")
            f.write("".join(info["descriptions"]))
            if obs["utterance_history"] != "Conversation: \n":
                f.write(obs['utterance_history'])
            # f.write("Your possible actions are:\n")
            # f.write("(a) move forward\n")
            # f.write("(b) turn left\n")
            # f.write("(c) turn right\n")
            # f.write("(d) toggle\n")
            # f.write("(e) no_op\n")
            f.write("Act : ")

    print('Full reward (undiminshed)=%.2f' % (reward))


def key_handler(event):

    # if hasattr(event.canvas, "_event_loop") and event.canvas._event_loop.isRunning():
    #     return

    print('pressed', event.key)

    action_dict = {
        "up": "a) move forward",
        "left": "b) turn left",
        "right": "c) turn right",
        " ": "d) toggle",
        "shift": "e) no_op",
    }
    action_dict = {
        "up": "move forward",
        "left": "turn left",
        "right": "turn right",
        " ": "toggle",
        "shift": "no_op",
    }

    if event.key in action_dict:
        your_action = action_dict[event.key]

        with open(output_file, "a") as f:
            f.write("{}\n".format(your_action))

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'r':
        reset()
        return

    if event.key == 'tab':
        step(np.array([np.nan, np.nan, np.nan]))
        return

    if event.key == 'shift':
        step(np.array([np.nan, np.nan, np.nan]))
        return

    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return
    if event.key == 't':
        step(env.actions.speak)
        return

    if event.key == '1':
        step(np.array([np.nan, 0, 0]))
        return
    if event.key == '2':
        step(np.array([np.nan, 0, 1]))
        return
    if event.key == '3':
        step(np.array([np.nan, 1, 0]))
        return
    if event.key == '4':
        step(np.array([np.nan, 1, 1]))
        return
    if event.key == '5':
        step(np.array([np.nan, 2, 2]))
        return
    if event.key == '6':
        step(np.array([np.nan, 1, 2]))
        return
    if event.key == '7':
        step(np.array([np.nan, 2, 1]))
        return
    if event.key == '8':
        step(np.array([np.nan, 1, 3]))
        return
    if event.key == 'p':
        step(np.array([np.nan, 3, 3]))
        return

    # Spacebar
    if event.key == ' ':
        step(env.actions.toggle)
        return
    if event.key == '9':
        step(env.actions.pickup)
        return
    if event.key == '0':
        step(env.actions.drop)
        return

    if event.key == 'enter':
        step(env.actions.done)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    # default="SocialAI-AsocialBoxInformationSeekingParamEnv-v1",
    default="SocialAI-ColorBoxesLLMCSParamEnv-v1",
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
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)
parser.add_argument(
    '--print_grid',
    default=False,
    help="print the grid with symbols",
    action='store_true'
)
parser.add_argument(
    '--calc-bonus',
    default=False,
    help="calculate explo bonus",
    action='store_true'
)
parser.add_argument(
    '--mask-unobserved',
    default=False,
    help="mask cells that are not observed by the agent",
    action='store_true'
)
parser.add_argument(
    '--output-file',
    default="./llm_data/in_context_color_test.txt",
    help="file where to save episodes",
)


# Put all env related arguments after --env_args, e.g. --env_args nb_foo 1 is_bar True
parser.add_argument("--env-args", nargs='*', default=None)

args = parser.parse_args()

output_file=args.output_file

env = gym.make(args.env, **env_args_str_to_dict(args.env_args))

if draw_tree:
    # draw tree
    env.parameter_tree.draw_tree(
        filename="viz/SocialAIParam/{}_raw_tree".format(args.env),
        ignore_labels=["Num_of_colors"],
    )

if args.seed >= 0:
    env.seed(args.seed)

with open(output_file, "a") as f:
    f.write("New episode.\n")

window = Window('gym_minigrid - ' + args.env, figsize=(4, 4))
window.reg_key_handler(key_handler)
env.window = window

# Blocking event loop
window.show(block=True)

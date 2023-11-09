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

class InteractiveACL:

    def choose(self, node):

        def pop_up(options):
            pop_data = {}

            def setVar(value):
                pop_data["var"] = value
                root.destroy()

            root = Tk()
            root.title(node.label)
            root.geometry('600x{}'.format(50*len(options)))

            for i, o in enumerate(options):
                fn = partial(setVar, value=i)
                Button(root, text='{}'.format(o), command=fn).pack()

            root.mainloop()

            return pop_data["var"]

        chosen_ind = pop_up([n.label for n in node.children])

        ch = node.children[chosen_ind]

        return ch


if inter_acl:
    interactive_acl = InteractiveACL()
else:
    interactive_acl = None


def redraw(img):
    if not args.agent_view:
        img = env.render('human', tile_size=args.tile_size, mask_unobserved=args.mask_unobserved)

    window.show_img(img)

def reset():
    # if args.seed != -1:
    #     env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

    if draw_tree:
        # draw tree
        params = env.current_env.parameters
        env.parameter_tree.draw_tree(
            filename="viz/SocialAIParam/parameters_{}_{}".format(params["Env_type"], hash(str(params))),
            ignore_labels=["Num_of_colors"],
            selected_parameters=params
        )

        with open('viz/SocialAIParam/parameters_{}_{}.json'.format(params["Env_type"], hash(str(params))), 'w') as fp:
            json.dump(params, fp)


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


def to_string(enc):
    s =  "{:<8} {} {} {} {} {:3} {:3} {}\t".format(
        IDX_TO_OBJECT.get(enc[0], enc[0]),  # obj
        *enc[1:3],  # x, y
        IDX_TO_COLOR.get(enc[3], enc[3])[:1].upper(),  # color
        *enc[4:]  #
    )

    if IDX_TO_OBJECT.get(enc[0], enc[0]) == "unseen":
        pass
        # s = colored(s, "on_grey")

    elif IDX_TO_OBJECT.get(enc[0], enc[0]) != "empty":
        col = IDX_TO_COLOR.get(enc[3], enc[3])
        if col in COLORS:
            s = colored(s, col)

    return s


def step(action):
    if type(action) == np.ndarray:
        obs, reward, done, info = env.step(action)
    else:
        action = [int(action), np.nan, np.nan]
        obs, reward, done, info = env.step(action)

    print('\nStep=%s' % (env.step_count))

    # print("".join(info["descriptions"]))
    print(obs['utterance_history'])
    print("")
    # print("Your possible actions are:")
    # print("a) move forward")
    # print("b) turn left")
    # print("c) turn right")
    # print("d) toggle")
    # print("e) no_op")
    # print("Your next action is: ")

    if args.print_grid:
        grid = obs['image'].transpose((1, 0, 2))
        for row_i, row in enumerate(grid):

            # if row_i == 0:
            #     for _ in row:
            #         print(to_string(["OBJECT", "X", "Y", "C", "-", "---", "---", "-"]), end="")
            #         # print("{:<8} {} {} {} {:2} {:2} {} {}\t".format("Object", "X", "Y", "C", "", "", "", ""), end="")
            # print(end="\n")

            for col_i, enc in enumerate(row):
                print(str(enc), end=" | ")
                # if row_i == len(grid) - 1 and col_i == len(row) // 2:
                #     # gent
                #     print(to_string(["^^^^^^", "^", "^", "^", "^", "^^^", "^^^", "^"]), end="")
                # else:
                #     print(to_string(enc), end="")
            print(end="\n")

    if not args.agent_view:

        nvec = algo.acmodel.model_raw_action_space.nvec

        raw_action = (
            5 if np.isnan(action[0]) else 1,  # speak switch
            0 if np.isnan(action[1]) else 1,  # speak switch
            0 if np.isnan(action[1]) else action[1],  # template
            0 if np.isnan(action[2]) else action[2],  # word
        )


        dist = []
        for a, n in zip(raw_action, nvec):
            logits = torch.ones(n)[None, :]
            logits[0][int(a)] *= 10

            d = Categorical(logits=logits)
            dist.append(d)
        if args.calc_bonus:
            bonus = algo.calculate_exploration_bonus(
                obs=[obs],
                embeddings=torch.zeros([1,128]),
                done=[done],
                prev_obs=[prev["prev_obs"]],
                prev_info=[prev["prev_info"]],
                agent_actions=torch.tensor([raw_action]),
                dist=dist,
                i_step=0,
            )

        else:
            bonus = [0]

        prev["prev_obs"] = obs
        prev["prev_info"] = info

        tot_bonus[0] = tot_bonus[0]+bonus[0]
        print('expl_bonus_step=%.2f' % (bonus[0]))
        print('tot_bonus=%.2f' % (tot_bonus[0]))

        if done:
            for v in algo.visitation_counter.values():
                v[0] = Counter()

    print('Full reward (undiminshed)=%.2f' % (reward))

    redraw(obs)

    if done:
        print('done!')
        print('Reward=%.2f' % (reward))
        print('Exploration_bonus=%.2f' % (tot_bonus[0]))
        tot_bonus[0] = 0

        if draw_tree:
            # draw tree
            params = env.current_env.parameters
            env.parameter_tree.draw_tree(
                filename="viz/SocialAIParam/parameters_{}_{}".format(params["Env_type"], hash(str(params))),
                ignore_labels=[],
                selected_parameters=params,
            )

            with open('viz/SocialAIParam/parameters_{}_{}.json'.format(params["Env_type"], hash(str(params))),
                      'w') as fp:
                json.dump(params, fp)

        reset()

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

    if event.key in action_dict:
        your_action = action_dict[event.key]
        print("Your next action is: {}".format(your_action))

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
    default='SocialAI-ELangColorBoxesTestInformationSeekingParamEnv-v1',
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

# Put all env related arguments after --env_args, e.g. --env_args nb_foo 1 is_bar True
parser.add_argument("--env-args", nargs='*', default=None)

parser.add_argument("--exploration-bonus", action="store_true", default=False,
                    help="Use a count based exploration bonus")
parser.add_argument("--exploration-bonus-type", nargs="+", default=["lang"],
                    help="modality on which to use the bonus (lang/grid/cell)")
parser.add_argument("--exploration-bonus-params", nargs="+", type=float, default=(30., 50.),  # lang
                    help="parameters for a count based exploration bonus (C, M)")
# parser.add_argument("--exploration-bonus-params", nargs="+", type=float, default=(3, 50.),  # cell
#                     help="parameters for a count based exploration bonus (C, M)")
# parser.add_argument("--exploration-bonus-params", nargs="+", type=float, default=(1.5, 50.), # grid
#                     help="parameters for a count based exploration bonus (C, M)")
parser.add_argument("--exploration-bonus-tanh", nargs="+", type=float, default=None,
                    help="tanh expl bonus scale, None means no tanh")
parser.add_argument("--intrinsic-reward-coef", type=float, default=0.1,
                    help="tanh expl bonus scale, None means no tanh")

args = parser.parse_args()

if interactive_acl:
    env = gym.make(args.env, curriculum=interactive_acl, **env_args_str_to_dict(args.env_args))
else:
    env = gym.make(args.env, **env_args_str_to_dict(args.env_args))

if draw_tree:
    # draw tree
    env.parameter_tree.draw_tree(
        filename="viz/SocialAIParam/{}_raw_tree".format(args.env),
        ignore_labels=["Num_of_colors"],
    )


# if hasattr(env, "draw_tree"):
#     env.draw_tree(ignore_labels=["Num_of_colors"])

# if hasattr(env, "print_tree"):
#     env.print_tree()

if args.seed >= 0:
    env.seed(args.seed)

# dummy just algo instance just to enable exploration bonus calculation
algo = torch_ac.PPOAlgo(
    envs=[env],
    acmodel=MultiModalBaby11ACModel(
        obs_space=utils.get_obss_preprocessor(
            obs_space=env.observation_space,
            text=False,
            dialogue_current=False,
            dialogue_history=True,
        )[0],
        action_space=env.action_space,
    ),
    exploration_bonus=True,
    exploration_bonus_tanh=args.exploration_bonus_tanh,
    exploration_bonus_type=args.exploration_bonus_type,
    exploration_bonus_params=args.exploration_bonus_params,
    expert_exploration_bonus=False,
    episodic_exploration_bonus=True,
    intrinsic_reward_coef=args.intrinsic_reward_coef,
    num_frames_per_proc=40,
)

# if args.agent_view:
#     env = RGBImgPartialObsWrapper(env)
#     env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env, figsize=(4, 4))
window.reg_key_handler(key_handler)
env.window = window

# Blocking event loop
window.show(block=True)

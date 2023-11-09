#!/usr/bin/env python3
import argparse
from gym_minigrid.window import Window
from utils import *
import gym
import pickle
from datetime import datetime

episodes = []
record = [False]


def update_caption_with_recording_indicator():
    new_caption = f"Recoding {'ON' if record[0] else 'OFF'}\n------------------\n\n" + window.caption.get_text()
    window.set_caption(new_caption)

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size, mask_unobserved=args.mask_unobserved)

    # adds the rocding
    update_caption_with_recording_indicator()

    window.show_img(img)

def start_recording():
    record[0] = True
    print("Recording started")

    episodes[-1][-1]["record"]=True

def reset():
    episodes.append([])
    obs, info = env.reset_with_info()
    record[0] = False
    redraw(obs)

    episodes[-1].append(
        {
            "action": None,
            "info": info,
            "obs": obs,
            "reward": None,
            "done": None,
            "record": record[0],
        }
    )


def step(action):
    if type(action) == np.ndarray:
        obs, reward, done, info = env.step(action)
    else:
        action = [int(action), np.nan, np.nan]
        obs, reward, done, info = env.step(action)

    episodes[-1].append(
        {
            "action": action,
            "info": info,
            "obs": obs,
            "reward": reward,
            "done": done,
            "record": record[0],
        }
    )
    redraw(obs)

    if done:
        print('done!')
        print('Reward=%.2f' % (reward))

        # reset and add initial state to episodes
        reset()

    else:
        print('\nStep=%s' % (env.step_count))


    # filter steps without recording
    episodes_to_save = [[s for s in ep if s["record"]] for ep in episodes]
    episodes_to_save = [ep for ep in episodes_to_save if len(ep) > 0]

    # set first recording step to be as if it was just reset (the real first step)
    for ep_to_save in episodes_to_save:
        ep_to_save[0]["action"]=None
        ep_to_save[0]["reward"]=None
        ep_to_save[0]["done"]=None


    # picle the episodes
    dump_pickle = Path(output_dir) / "episodes.pkl"
    print(f"Saving {len(episodes_to_save)} episodes ({[len(e) for e in episodes_to_save]}) to : {dump_pickle}")

    with open(dump_pickle, 'wb') as f:
        pickle.dump(episodes_to_save, f)


def key_handler(event):

    print('pressed', event.key)

    if event.key == 'r':
        start_recording()
        return

    if event.key == 'escape':
        window.close()
        return

    if event.key == 's':
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        help="gym environment to load",
        # default="SocialAI-AsocialBoxInformationSeekingParamEnv-v1",
        # default="SocialAI-ColorBoxesLLMCSParamEnv-v1",
        default="SocialAI-ColorLLMCSParamEnv-v1",
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
        '--mask-unobserved',
        default=False,
        help="mask cells that are not observed by the agent",
        action='store_true'
    )
    parser.add_argument(
        '--save-dir',
        default="./llm_data/in_context_examples/",
        help="file where to save episodes",
    )
    parser.add_argument(
        '--load',
        default=None,
        help="Load in context examples to append to",
    )
    parser.add_argument(
        '--name',
        default="in_context",
        help="additional name tag for the episodes",
    )
    parser.add_argument(
        '--draw-tree',
        action="store_true",
        help="Draw the sampling treee",
    )

    # Put all env related arguments after --env_args, e.g. --env_args nb_foo 1 is_bar True
    parser.add_argument("--env-args", nargs='*', default=None)

    args = parser.parse_args()

    env = gym.make(args.env, **env_args_str_to_dict(args.env_args))

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = Path(args.save_dir) / f"{args.name}_{args.env}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    if args.load:
        with open(args.load, 'rb') as f:
            episodes = pickle.load(f)

    if args.draw_tree:
        # draw tree
        env.parameter_tree.draw_tree(
            filename=output_dir / f"/{args.env}_raw_tree",
            ignore_labels=["Num_of_colors"],
        )

    if args.seed >= 0:
        env.seed(args.seed)

    window = Window('gym_minigrid - ' + args.env, figsize=(6, 4))
    window.reg_key_handler(key_handler)
    env.window = window

    reset()
    # # a trick to make the first image appear right away
    # # this action is not saved
    # obs, _, _, _ = env.step(np.array([np.nan, np.nan, np.nan]))
    # redraw(obs)

    # Blocking event loop
    window.show(block=True)

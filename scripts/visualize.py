import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

from utils.babyai_utils.baby_agent import load_agent
from utils.env import make_env
from utils.other import seed
from utils.storage import get_model_dir
from utils.storage import get_status
from models import *
import subprocess

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--max-steps", type=int, default=None,
                    help="max num of steps")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.5,
                    help="pause duration between two consequent actions of the agent (default: 0.5)")
parser.add_argument("--env-name", type=str, default=None, required=True,
                    help="env name")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename", required=True)
parser.add_argument("--episodes", type=int, default=10,
                    help="number of episodes to visualize")

args = parser.parse_args()

# Set seed for all randomness sources

seed(args.seed)

save = args.gif
if save:
    savename = args.gif
    if savename == "model_id":
        savename = args.model.replace('storage/', '')
        savename = savename.replace('/','_')
        savename += '_{}'.format(args.seed)




# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment

if str(args.model).startswith("./storage/"):
    args.model = args.model.replace("./storage/", "")

if str(args.model).startswith("storage/"):
    args.model = args.model.replace("storage/", "")

with open(Path("./storage") / args.model / "config.json") as f:
    conf = json.load(f)

if args.env_name is None:
    # load env_args from status
    env_args = {}
    if not "env_args" in conf.keys():
        env_args = get_status(get_model_dir(args.model), None)['env_args']
    else:
        env_args = conf["env_args"]

    env = make_env(args.env_name, args.seed, env_args=env_args)
else:
    env_name = args.env_name
    env = make_env(args.env_name, args.seed)

for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Define agent
model_dir = get_model_dir(args.model)
num_frames = None
agent = load_agent(env, model_dir, args.argmax, num_frames)

print("Agent loaded\n")

# Run the agent

if save:
   from imageio import mimsave
   old_frames = []
   frames = []

# Create a window to view the environment
env.render(mode='human')

def plt_2_rgb(env):
    data = np.frombuffer(env.window.fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(env.window.fig.canvas.get_width_height()[::-1] + (3,))
    return data


for episode in range(args.episodes):
    print("episode:", episode)
    obs = env.reset()

    env.render(mode='human')
    if save:
        frames.append(plt_2_rgb(env))

    i = 0
    while True:
        i += 1

        action = agent.get_action(obs)
        obs, reward, done, _ = env.step(action)
        agent.analyze_feedback(reward, done)
        env.render(mode='human')

        if save:
            img = plt_2_rgb(env)
            frames.append(img)
            if done:
                # quadruple last frame to pause between episodes
                for i in range(3):
                    same_img = np.copy(img)
                    # toggle a pixel between frames to avoid cropping when going from gif to mp4
                    same_img[0,0,2] = 0 if (i % 2) == 0 else 255
                    frames.append(same_img)

        if done or env.window.closed:
            break

        if args.max_steps is not None:
            if i > args.max_steps:
                break


    if env.window.closed:
        break

if save:
    # from IPython import embed; embed()
    gifpath = save+".gif"
    print(f"Saving to {gifpath} ", end="")
    mimsave(gifpath, frames, duration=args.pause)
    # Reduce gif size
    # bashCommand = "gifsicle -O3 --colors 32 -o {}.gif {}.gif".format(savename, savename)
    # process = subprocess.run(bashCommand.split(), stdout=subprocess.PIPE)

    print("Done.")

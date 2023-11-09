# python -m scripts.LLM_test  --gif test_GPT_boxes --episodes 1 --max-steps 8 --model text-davinci-003 --env-args size 6 --env-name SocialAI-ColorBoxesLLMCSParamEnv-v1 --in-context-path llm_data/in_context_color_boxes.txt
# python -m scripts.LLM_test  --gif test_GPT_asoc --episodes 1 --max-steps 8 --model text-ada-001 --env-args size 6  --env-name SocialAI-AsocialBoxInformationSeekingParamEnv-v1 --in-context-path llm_data/in_context_asocial_box.txt --feed-full-ep

# python -m scripts.LLM_test  --gif test_GPT_boxes --episodes 1 --max-steps 8 --model bloom_560m --env-args size 6 --env-name SocialAI-ColorBoxesLLMCSParamEnv-v1 --in-context-path llm_data/in_context_color_boxes.txt
# python -m scripts.LLM_test  --gif test_GPT_asoc --episodes 1 --max-steps 8 --model bloom_560m --env-args size 6  --env-name SocialAI-AsocialBoxInformationSeekingParamEnv-v1 --in-context-path llm_data/in_context_asocial_box.txt --feed-full-ep

## bloom 560m
# boxes
# python -m scripts.LLM_test --log llm_log/bloom_560m_boxes_no_hist  --gif evaluation --episodes 20 --max-steps 10 --model bloom_560m --env-args size 6 --env-name SocialAI-ColorBoxesLLMCSParamEnv-v1 --in-context-path llm_data/in_context_color_boxes.txt

# asocial
# python -m scripts.LLM_test --log llm_log/bloom_560m_asocial_no_hist   --gif evaluation --episodes 20 --max-steps 10 --model bloom_560m --env-args size 6  --env-name SocialAI-AsocialBoxInformationSeekingParamEnv-v1 --in-context-path llm_data/in_context_asocial_box.txt

# random
# python -m scripts.LLM_test --log llm_log/random_boxes  --gif evaluation --episodes 20 --max-steps 10 --model random --env-args size 6 --env-name SocialAI-ColorBoxesLLMCSParamEnv-v1 --in-context-path llm_data/in_context_color_boxes.txt

import argparse
import json
import requests
import time
import warnings
from n_tokens import estimate_price

import numpy as np
import torch
from pathlib import Path

from utils.babyai_utils.baby_agent import load_agent
from utils import *
from models import *
import subprocess
import os

from matplotlib import pyplot as plt

from gym_minigrid.wrappers import *
from gym_minigrid.window import Window
from datetime import datetime

from imageio import mimsave

def prompt_preprocessor(llm_prompt):
    # remove peer observations
    lines = llm_prompt.split("\n")
    new_lines = []
    for line in lines:
        if line.startswith("#"):
            continue

        elif line.startswith("Conversation"):
            continue

        elif "peer" in line:
            caretaker = True
            if caretaker:
                # show only the location of the caretaker

                # this is very ugly, todo: refactor this
                assert "there is a" in line
                start_index = line.index('there is a') + 11
                new_line = line[:start_index] + 'caretaker'

                new_lines.append(new_line)

            else:
                # no caretaker at all
                if line.startswith("Obs :") and "peer" in line:
                    # remove only the peer descriptions
                    line = "Obs :"
                    new_lines.append(line)
                else:
                    assert "peer" in line

        elif "Caretaker:" in line:
            # line = line.replace("Caretaker:", "Caretaker says: '") + "'"
            new_lines.append(line)

        else:
            new_lines.append(line)

    return "\n".join(new_lines)


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=False,
                    help="text-ada-001")
parser.add_argument("--seed", type=int, default=0,
                    help="Seed of the first episode. The seed for the following episodes will be used in order: seed, seed + 1, ... seed + (n_episodes-1) (default: 0)")
parser.add_argument("--max-steps", type=int, default=5,
                    help="max num of steps")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.5,
                    help="pause duration between two consequent actions of the agent (default: 0.5)")
parser.add_argument("--env-name", type=str,
                    # default="SocialAI-ELangColorBoxesTestInformationSeekingParamEnv-v1",
                    # default="SocialAI-AsocialBoxInformationSeekingParamEnv-v1",
                    default="SocialAI-ColorBoxesLLMCSParamEnv-v1",
                    required=False,
                    help="env name")
parser.add_argument("--in-context-path", type=str,
                    # default='llm_data/short_in_context_boxes.txt'
                    # default='llm_data/in_context_asocial_box.txt'
                    default='llm_data/in_context_color_boxes.txt',
                    required=False,
                    help="path to in context examples")
parser.add_argument("--gif", type=str, default="visualization",
                    help="store output as gif with the given filename", required=False)
parser.add_argument("--episodes", type=int, default=1,
                    help="number of episodes to visualize")
parser.add_argument("--env-args", nargs='*', default=None)
parser.add_argument("--agent_view", default=False, help="draw the agent sees (partially observable view)", action='store_true' )
parser.add_argument("--tile_size", type=int, help="size at which to render tiles", default=32 )
parser.add_argument("--mask-unobserved", default=False, help="mask cells that are not observed by the agent", action='store_true' )
parser.add_argument("--log", type=str, default="llm_log/episodes_log", help="log from the run", required=False)
parser.add_argument("--feed-full-ep", default=False, help="weather to append the whole episode to the prompt", action='store_true')
parser.add_argument("--skip-check", default=False, help="Don't estimate the price.", action="store_true")

args = parser.parse_args()

# Set seed for all randomness sources

seed(args.seed)

model = args.model


in_context_examples_path = args.in_context_path

print("env name:", args.env_name)
print("examples:", in_context_examples_path)
print("model:", args.model)

# datetime
now = datetime.now()
datetime_string = now.strftime("%d_%m_%Y_%H:%M:%S")
print(datetime_string)

# log filenames

log_folder = args.log+"_"+datetime_string+"/"
os.mkdir(log_folder)
evaluation_log_filename = log_folder+"evaluation_log.json"
prompt_log_filename = log_folder + "prompt_log.txt"
ep_h_log_filename = log_folder+"episode_history_query.txt"
gif_savename = log_folder + args.gif + ".gif"

assert "viz" not in gif_savename # don't use viz anymore


env_args = env_args_str_to_dict(args.env_args)
env = make_env(args.env_name, args.seed, env_args)

# env = gym.make(args.env_name, **env_args)
print(f"Environment {args.env_name} and args: {env_args_str_to_dict(args.env_args)}\n")

# Define agent
print("Agent loaded\n")

# prepare models

if args.model in ["text-davinci-003", "text-ada-001", "gpt-3.5-turbo-0301"]:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

elif args.model in ["gpt2_large", "api_bloom"]:
    HF_TOKEN = os.getenv("HF_TOKEN")

elif args.model in ["bloom_560m"]:
    from transformers import BloomForCausalLM
    from transformers import BloomTokenizerFast

    hf_tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m", cache_dir=".cache/huggingface/")
    hf_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", cache_dir=".cache/huggingface/")

elif args.model in ["bloom"]:
    from transformers import BloomForCausalLM
    from transformers import BloomTokenizerFast

    hf_tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom", cache_dir=".cache/huggingface/")
    hf_model = BloomForCausalLM.from_pretrained("bigscience/bloom", cache_dir=".cache/huggingface/")


def plt_2_rgb(env):
    # data = np.frombuffer(env.window.fig.canvas.tostring_rgb(), dtype=np.uint8)
    # data = data.reshape(env.window.fig.canvas.get_width_height()[::-1] + (3,))

    width, height = env.window.fig.get_size_inches() * env.window.fig.get_dpi()
    data = np.fromstring(env.window.fig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return data

def generate(text_input, model):
    # return "(a) move forward"
    if model == "dummy":
        print("dummy action forward")
        return "move forward"

    elif model == "random":
        print("random agent")
        return random.choice([
            "move forward",
            "turn left",
            "turn right",
            "toggle",
        ])

    elif model in ["gpt-3.5-turbo-0301"]:
        while True:
            try:
                c = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        # {"role": "system", "content": ""},
                        # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                        # {"role": "user", "content": "Continue the following text in the most logical way.\n"+text_input}
                        {"role": "user", "content": text_input}
                    ],
                    max_tokens=3,
                    n=1,
                    temperature=0,
                    request_timeout=30,
                )
                break
            except Exception as e:
                print(e)
                print("Pausing")
                time.sleep(10)
                continue
        print("generation: ", c['choices'][0]['message']['content'])
        return c['choices'][0]['message']['content']

    elif model in ["text-davinci-003", "text-ada-001"]:
        while True:
            try:
                response = openai.Completion.create(
                    model=model,
                    prompt=text_input,
                    # temperature=0.7,
                    temperature=0.0,
                    max_tokens=3,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    timeout=30
                )
                break

            except Exception as e:
                print(e)
                print("Pausing")
                time.sleep(10)
                continue

        choices = response["choices"]
        assert len(choices) == 1
        return choices[0]["text"].strip().lower()  # remove newline from the end

    elif model in ["gpt2_large", "api_bloom"]:
        # HF_TOKEN = os.getenv("HF_TOKEN")
        if model == "gpt2_large":
            API_URL = "https://api-inference.huggingface.co/models/gpt2-large"

        elif model == "api_bloom":
            API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"

        else:
            raise ValueError(f"Undefined model {model}.")

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        def query(text_prompt, n_tokens=3):

            input = text_prompt

            # make n_tokens request and append the output each time - one request generates one token

            for _ in range(n_tokens):
                # prepare request
                payload = {
                    "inputs": input,
                    "parameters": {
                        "do_sample": False,
                        'temperature': 0,
                        'wait_for_model': True,
                        # "max_length": 500,  # for gpt2
                        # "max_new_tokens": 250  # fot gpt2-xl
                    },
                }
                data = json.dumps(payload)

                # request
                response = requests.request("POST", API_URL, headers=headers, data=data)
                response_json = json.loads(response.content.decode("utf-8"))

                if type(response_json) is list and len(response_json) == 1:
                    # generated_text contains the input + the response
                    response_full_text = response_json[0]['generated_text']

                    # we use this as the next input
                    input = response_full_text

                else:
                    print("Invalid request to huggingface api")
                    from IPython import embed; embed()

            # remove the prompt from the beginning
            assert response_full_text.startswith(text_prompt)
            response_text = response_full_text[len(text_prompt):]

            return response_text

        response = query(text_input).strip().lower()
        return response

    elif model in ["bloom_560m"]:
        # from transformers import BloomForCausalLM
        # from transformers import BloomTokenizerFast
        #
        # tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m", cache_dir=".cache/huggingface/")
        # model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m", cache_dir=".cache/huggingface/")

        inputs = hf_tokenizer(text_input, return_tensors="pt")
        # 3 words
        result_length = inputs['input_ids'].shape[-1]+3
        full_output = hf_tokenizer.decode(hf_model.generate(inputs["input_ids"], max_length=result_length)[0])

        assert full_output.startswith(text_input)
        response = full_output[len(text_input):]

        response = response.strip().lower()

        return response

    else:
        raise ValueError("Unknown model.")

def get_parsed_action(text_action):
    if "move forward" in text_action:
        return "move forward"

    elif "turn left" in text_action:
        return "turn left"

    elif "turn right" in text_action:
        return "turn right"

    elif "toggle" in text_action:
        return "toggle"

    elif "no_op" in text_action:
        return "no_op"
    else:
        warnings.warn(f"Undefined action {text_action}")
        return "no_op"


def step(text_action):
    text_action = get_parsed_action(text_action)

    if "move forward" == text_action:
        action = [int(env.actions.forward), np.nan, np.nan]

    elif "turn left" == text_action:
        action = [int(env.actions.left), np.nan, np.nan]

    elif "turn right" == text_action:
        action = [int(env.actions.right), np.nan, np.nan]

    elif "toggle" == text_action:
        action = [int(env.actions.toggle), np.nan, np.nan]

    elif "no_op" == text_action:
        action = [np.nan, np.nan, np.nan]

    # if text_action.startswith("a"):
    #     action = [int(env.actions.forward), np.nan, np.nan]
    #
    # elif text_action.startswith("b"):
    #     action = [int(env.actions.left), np.nan, np.nan]
    #
    # elif text_action.startswith("c"):
    #     action = [int(env.actions.right), np.nan, np.nan]
    #
    # elif text_action.startswith("d"):
    #     action = [int(env.actions.toggle), np.nan, np.nan]
    #
    # elif text_action.startswith("e"):
    #     action = [np.nan, np.nan, np.nan]
    #
    # else:
    #     print("Unknown action.")

    obs, reward, done, info = env.step(action)

    return obs, reward, done, info



def reset(env):
    env.reset()
    # a dirty trick just to get obs and info
    return step("no_op")


def generate_text_obs(obs, info):
    llm_prompt = "Obs : "
    llm_prompt += "".join(info["descriptions"])
    if obs["utterance_history"] != "Conversation: \n":
        utt_hist = obs['utterance_history']
        utt_hist = utt_hist.replace("Conversation: \n","")
        llm_prompt += utt_hist

    return llm_prompt


def action_query():
    # llm_prompt = ""
    # llm_prompt += "Your possible actions are:\n"
    # llm_prompt += "(a) move forward\n"
    # llm_prompt += "(b) turn left\n"
    # llm_prompt += "(c) turn right\n"
    # llm_prompt += "(d) toggle\n"
    # llm_prompt += "(e) no_op\n"
    # llm_prompt += "Your next action is: ("
    llm_prompt = "Act :"
    return llm_prompt

# lod context examples
with open(in_context_examples_path, "r") as f:
    in_context_examples = f.read()

with open(prompt_log_filename, "a+") as f:
    f.write(datetime_string)

with open(ep_h_log_filename, "a+") as f:
    f.write(datetime_string)

feed_episode_history = args.feed_full_ep

# asoc
in_context_n_tokens = 800
ep_obs_len = 50 * 3

# color
in_context_n_tokens = 1434
# ep_obs_len = 70

# feed only current obs
if feed_episode_history:
    ep_obs_len = 50

else:
    # last_n = 1
    # last_n = 2
    last_n = 3
    ep_obs_len = 50 * last_n

_, price = estimate_price(
    num_of_episodes=args.episodes,
    in_context_len=in_context_n_tokens,
    ep_obs_len=ep_obs_len,
    n_steps=args.max_steps,
    model=args.model,
    feed_episode_history=feed_episode_history
)
if not args.skip_check:
    input(f"You will spend: {price} dollars. (in context: {in_context_n_tokens} obs: {ep_obs_len}), ok?")

# prepare frames list to save to gif
frames = []

assert args.max_steps <= 20

success_rates = []
# episodes start
for episode in range(args.episodes):
    print("Episode:", episode)
    new_episode_text = "New episode.\n"
    episode_history_text = new_episode_text

    success = False
    episode_seed = args.seed + episode
    env = make_env(args.env_name, episode_seed, env_args)

    with open(prompt_log_filename, "a+") as f:
        f.write("\n\n")

    observations = []
    actions = []
    for i in range(int(args.max_steps)):
        if i == 0:
            obs, reward, done, info = reset(env)
            action_text = ""

        else:
            with open(prompt_log_filename, "a+") as f:
                f.write("\nnew prompt: -----------------------------------\n")
                f.write(llm_prompt)

            text_action = generate(llm_prompt, args.model)
            obs, reward, done, info = step(text_action)
            action_text = f"Act : {get_parsed_action(text_action)}\n"
            actions.append(action_text)

            print(action_text)

        text_obs = generate_text_obs(obs, info)
        observations.append(text_obs)
        print(prompt_preprocessor(text_obs))

        # feed the full episode history
        episode_history_text += prompt_preprocessor(action_text + text_obs)  # append to history of this episode

        if feed_episode_history:
            # feed full episode history
            llm_prompt = in_context_examples + episode_history_text + action_query()

        else:
            n = min(last_n, len(observations))
            obs = observations[-n:]
            act = (actions + [action_query()])[-n:]

            episode_text = "".join([o+a for o,a in zip(obs, act)])

            llm_prompt = in_context_examples + new_episode_text + episode_text

        llm_prompt = prompt_preprocessor(llm_prompt)


        # save the image
        env.render(mode="human")
        rgb_img = plt_2_rgb(env)
        frames.append(rgb_img)

        if env.current_env.box.blocked and not env.current_env.box.is_open:
            # target box is blocked -> apple can't be obtained
            # break to save compute
            break

        if done:
            # quadruple last frame to pause between episodes
            for i in range(3):
                same_img = np.copy(rgb_img)
                # toggle a pixel between frames to avoid cropping when going from gif to mp4
                same_img[0, 0, 2] = 0 if (i % 2) == 0 else 255
                frames.append(same_img)

            if reward > 0:
                print("Success!")
                episode_history_text += "Success!\n"
                success = True
            else:
                episode_history_text += "Failure!\n"

            with open(ep_h_log_filename, "a+") as f:
                f.write("\nnew prompt: -----------------------------------\n")
                f.write(episode_history_text)

            break

        else:
            with open(ep_h_log_filename, "a+") as f:
                f.write("\nnew prompt: -----------------------------------\n")
                f.write(episode_history_text)

    print(f"{'Success' if success else 'Failure'}")
    success_rates.append(success)

mean_success_rate =  np.mean(success_rates)
print("Success rate:", mean_success_rate)
print(f"Saving gif to {gif_savename}.")
mimsave(gif_savename, frames, duration=args.pause)

print("Done.")

log_data_dict = vars(args)
log_data_dict["success_rates"] = success_rates
log_data_dict["mean_success_rate"] = mean_success_rate

print("Evaluation log: ", evaluation_log_filename)
with open(evaluation_log_filename, "w") as f:
    f.write(json.dumps(log_data_dict))

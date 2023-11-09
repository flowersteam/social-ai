import argparse
import matplotlib.pyplot as plt
import json
import time
import numpy as np
import torch
from pathlib import Path

from utils.babyai_utils.baby_agent import load_agent
from utils.storage import get_status
from utils.env import make_env
from utils.other import seed
from utils.storage import get_model_dir
from models import *

from scipy import stats
print("Wrong script. This is from VIGIL")
exit()

start = time.time()

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--random-agent", action="store_true", default=False,
                    help="random actions")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes to test")
parser.add_argument("--test-p", type=float, default=0.05,
                    help="p value")
parser.add_argument("--n-seeds", type=int, default=16,
                    help="number of episodes to test")
parser.add_argument("--subsample-step", type=int, default=1,
                    help="subsample step")
parser.add_argument("--start-step", type=int, default=1,
                    help="at which step to start the curves")

args = parser.parse_args()

# Set seed for all randomness sources

seed(args.seed)

assert args.seed == 1
assert not args.argmax
# assert args.num_frames == 28000000
# assert args.episodes == 1000

test_p = args.test_p
n_seeds = args.n_seeds
subsample_step = args.subsample_step
start_step = args.start_step

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# what to load
models_to_evaluate = [
    "25-03_RERUN_WizardGuide_lang64_mm_baby_short_rec_env_MiniGrid-TalkItOutNoLiar-8x8-v0_multi-modal-babyai11-agent_arch_original_endpool_res_custom-ppo-2_exploration-bonus-params_5_50",
    "25-03_RERUN_WizardTwoGuides_lang64_mm_baby_short_rec_env_MiniGrid-TalkItOut-8x8-v0_multi-modal-babyai11-agent_arch_original_endpool_res_custom-ppo-2_exploration-bonus-params_5_50"
]
print("evaluating models: ", models_to_evaluate)

# what to put in the legend
label_parser_dict = {
    "RERUN_WizardGuide_lang64_no_explo": "Abl-MH-BabyAI",
    "RERUN_WizardTwoGuides_lang64_no_explo": "MH-BabyAI",

    "RERUN_WizardGuide_lang64_mm_baby_short_rec_env": "Abl-MH-BabyAI-ExpBonus",
    "RERUN_WizardTwoGuides_lang64_mm_baby_short_rec_env": "MH-BabyAI-ExpBonus",

    "RERUN_WizardGuide_lang64_deaf_no_explo": "Abl-Deaf-MH-BabyAI",
    "RERUN_WizardTwoGuides_lang64_deaf_no_explo": "Deaf-MH-BabyAI",

    "RERUN_WizardGuide_lang64_bow": "Abl-MH-BabyAI-ExpBonus-BOW",
    "RERUN_WizardTwoGuides_lang64_bow": "MH-BabyAI-ExpBonus-BOW",

    "RERUN_WizardGuide_lang64_no_mem": "Abl-MH-BabyAI-ExpBonus-no-mem",
    "RERUN_WizardTwoGuides_lang64_no_mem": "MH-BabyAI-ExpBonus-no-mem",

    "RERUN_WizardGuide_lang64_bigru": "Abl-MH-BabyAI-ExpBonus-bigru",
    "RERUN_WizardTwoGuides_lang64_bigru": "MH-BabyAI-ExpBonus-bigru",

    "RERUN_WizardGuide_lang64_attgru": "Abl-MH-BabyAI-ExpBonus-attgru",
    "RERUN_WizardTwoGuides_lang64_attgru": "MH-BabyAI-ExpBonus-attgru",

    "RERUN_WizardGuide_lang64_curr_dial": "Abl-MH-BabyAI-ExpBonus-current-dialogue",
    "RERUN_WizardTwoGuides_lang64_curr_dial": "MH-BabyAI-ExpBonus-current-dialogue",

    "RERUN_WizardTwoGuides_lang64_mm_baby_short_rec_100M": "MH-BabyAI-ExpBonus-100M"
}

# how do to stat tests
compare = {
    "MH-BabyAI-ExpBonus": "Abl-MH-BabyAI-ExpBonus",
}

COLORS = ["red", "blue", "green", "black", "purpule", "brown", "orange", "gray"]
label_color_dict = {l: c for l, c in zip(label_parser_dict.values(), COLORS)}


test_set_check_path = Path("test_set_check_{}_nep_{}.json".format(args.seed, args.episodes))

def calc_perf_for_seed(i, model_name, num_frames, seed, argmax, episodes, random_agent=False):
    print("seed {}".format(i))
    model = Path(model_name) / str(i)
    model_dir = get_model_dir(model)

    if test_set_check_path.exists():
        with open(test_set_check_path, "r") as f:
            check_loaded = json.load(f)
        print("check loaded")
    else:
        print("check not loaded")
        check_loaded = None

    # Load environment
    with open(model_dir+"/config.json") as f:
        conf = json.load(f)

    env_name = conf["env"]

    env = make_env(env_name, seed)
    print("Environment loaded\n")

    # load agent
    agent = load_agent(env, model_dir, argmax, num_frames)
    status = get_status(model_dir, num_frames)
    assert status["num_frames"] == num_frames
    print("Agent loaded\n")

    check = {}

    seed_rewards = []
    for episode in range(episodes):
        print("[{}/{}]: ".format(episode, episodes), end="", flush=True)

        obs = env.reset()

        # check envs are the same during seeds
        if episode in check:
            assert check[episode] == int(obs['image'].sum())
        else:
            check[episode] = int(obs['image'].sum())

        if check_loaded is not None:
            assert check[episode] == int(obs['image'].sum())

        while True:
            if random_agent:
                action = agent.get_random_action(obs)
            else:
                action = agent.get_action(obs)

            obs, reward, done, _ = env.step(action)
            print(".", end="", flush=True)
            agent.analyze_feedback(reward, done)

            if done:
                seed_rewards.append(reward)
                break

        print()

    seed_rewards = np.array(seed_rewards)
    seed_success_rates = seed_rewards > 0

    if not test_set_check_path.exists():
        with open(test_set_check_path, "w") as f:
            json.dump(check, f)
            print("check saved")

    print("seed success rate:", seed_success_rates.mean())
    print("seed reward:", seed_rewards.mean())

    return seed_rewards.mean(), seed_success_rates.mean()



def get_available_steps(model):
    model_dir = Path(get_model_dir(model))
    per_seed_available_steps = {}
    for seed_dir in model_dir.glob("*"):
        per_seed_available_steps[seed_dir] = sorted([
           int(str(p.with_suffix("")).split("status_")[-1])
           for p in seed_dir.glob("status_*")
        ])

    num_steps = min([len(steps) for steps in per_seed_available_steps.values()])

    steps = list(per_seed_available_steps.values())[0][:num_steps]

    for available_steps in per_seed_available_steps.values():
        s_steps = available_steps[:num_steps]
        assert steps == s_steps

    return steps

def plot_with_shade(subplot_nb, ax, x, y, err, color, shade_color, label,
                    legend=False, leg_size=30, leg_loc='best', title=None,
                    ylim=[0, 100], xlim=[0, 40], leg_args={}, leg_linewidth=8.0, linewidth=7.0, ticksize=30,
                    zorder=None, xlabel='perf', ylabel='env steps', smooth_factor=1000):
    # plt.rcParams.update({'font.size': 15})
    ax.locator_params(axis='x', nbins=6)
    ax.locator_params(axis='y', nbins=5)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)

    # smoothing
    def smooth(x_, n=50):
        return np.array([x_[max(i - n, 0):i + 1].mean() for i in range(len(x_))])

    if smooth_factor > 0:
        y = smooth(y, n=smooth_factor)
        err = smooth(err, n=smooth_factor)

    ax.plot(x, y, color=color, label=label, linewidth=linewidth, zorder=zorder)
    ax.fill_between(x, y - err, y + err, color=shade_color, alpha=0.2)
    if legend:
        leg = ax.legend(loc=leg_loc, fontsize=leg_size, **leg_args)  # 34
        for legobj in leg.legendHandles:
            legobj.set_linewidth(leg_linewidth)
    ax.set_xlabel(xlabel, fontsize=30)
    if subplot_nb == 0:
        ax.set_ylabel(ylabel, fontsize=30)
    ax.set_xlim(xmin=xlim[0], xmax=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if title:
        ax.set_title(title, fontsize=22)


def label_parser(label, label_parser_dict):
    if sum([1 for k, v in label_parser_dict.items() if k in label]) != 1:
        print("ERROR")
        print(label)
        exit()

    for k, v in label_parser_dict.items():
        if k in label: return v

    return label


f, ax = plt.subplots(1, 1, figsize=(10.0, 6.0))
ax = [ax]

performances = {}
per_seed_performances = {}
stds = {}


label_parser_dict_reverse = {v: k for k, v in label_parser_dict.items()}
assert len(label_parser_dict_reverse) == len(label_parser_dict)

label_to_model = {}
# evaluate and draw curves
for model in models_to_evaluate:
    label = label_parser(model, label_parser_dict)
    label_to_model[label] = model

    color = label_color_dict[label]
    performances[label] = []
    per_seed_performances[label] = []
    stds[label] = []

    steps = get_available_steps(model)
    steps = steps[::subsample_step]
    steps = [s for s in steps if s > start_step]

    print("steps:", steps)

    for step in steps:
        results = []
        for s in range(n_seeds):
            results.append(calc_perf_for_seed(
                s,
                model_name=model,
                num_frames=step,
                seed=args.seed,
                argmax=args.argmax,
                episodes=args.episodes,
            ))

        rewards, success_rates = zip(*results)
        rewards = np.array(rewards)
        success_rates = np.array(success_rates)
        per_seed_performances[label].append(success_rates)
        performances[label].append(success_rates.mean())
        stds[label].append(success_rates.std())

    means = np.array(performances[label])
    err = np.array(stds[label])
    label = label_parser(str(model), label_parser_dict)
    max_steps = np.max(steps)
    min_steps = np.min(steps)
    min_y = 0.0
    max_y = 1.0
    ylabel = "performance"
    smooth_factor = 0

    plot_with_shade(0, ax[0], steps, means, err, color, color, label,
                    legend=True, xlim=[min_steps, max_steps], ylim=[min_y, max_y],
                    leg_size=20, xlabel="Env steps (millions)", ylabel=ylabel, linewidth=5.0, smooth_factor=smooth_factor)

assert len(label_to_model) == len(models_to_evaluate)


def get_compatible_steps(model1, model2, subsample_step):
    steps_1 = get_available_steps(model1)[::subsample_step]
    steps_2 = get_available_steps(model2)[::subsample_step]

    min_steps = min(len(steps_1), len(steps_2))
    steps_1 = steps_1[:min_steps]
    steps_2 = steps_2[:min_steps]
    assert steps_1 == steps_2

    return steps_1


# stat tests
for k, v in compare.items():
    dist_1_steps = per_seed_performances[k]
    dist_2_steps = per_seed_performances[v]

    model_k = label_to_model[k]
    model_v = label_to_model[v]
    steps = get_compatible_steps(model_k, model_v, subsample_step)
    steps = [s for s in steps if s > start_step]

    for step, dist_1, dist_2 in zip(steps, dist_1_steps, dist_2_steps):
        assert len(dist_1) == n_seeds
        assert len(dist_2) == n_seeds

        p = stats.ttest_ind(
            dist_1,
            dist_2,
            equal_var=False
        ).pvalue

        if np.isnan(p):
            from IPython import embed; embed()

        if p < test_p:
            plt.scatter(step, 0.8, color=label_color_dict[k], s=50, marker="x")

        print("{} (m:{}) <---> {} (m:{}) = p: {}  result: {}".format(
            k, np.mean(dist_1), v, np.mean(dist_2), p,
            "Distributions different(p={})".format(test_p) if p < test_p else "Distributions same(p={})".format(test_p)
        ))
        print()

f.savefig('graphics/test.png')
f.savefig('graphics/test.svg')

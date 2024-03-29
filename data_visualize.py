#!/usr/bin/env python
import re
import itertools
import math
from itertools import chain
import time

# import seaborn
import numpy as np
import os
from collections import OrderedDict, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import sys
from termcolor import cprint, colored
from pathlib import Path
import pickle
from scipy import stats


metrics = ['success_rate_mean']


eval_metric = "test_success_rates"

super_title = ""

# statistical evaluation p-value
test_p = 0.05

agg_title = ""

color_dict = None
eval_filename = None

max_frames = 20_000_000

legend_show_n_seeds = False
draw_legend = True

study_train = False
study_eval = True

plot_test = True

plot_aggregated_test = True
plot_only_aggregated_test = True


xnbins = 4
ynbins = 3

steps_denom = 1e6

# Global vas for tracking and labeling data at load time.
exp_idx = 0
label_parser_dict = None
label_parser = lambda l, _, label_parser_dict: l

smooth_factor = 10 # used
# smooth_factor = 0
print("smooth factor:", smooth_factor)
eval_smooth_factor = None
leg_size = 30

def smooth(x_, n=50):
    if n is None:
        return x_

    if type(x_) == list:
        x_ = np.array(x_)
    return np.array([x_[max(i - n, 0):i + 1].mean() for i in range(len(x_))])

sort_test = False

def sort_test_set(env_name):
    helps = [
        "LanguageFeedback",
        "LanguageColor",
        "Pointing",
        "Emulation",
    ]
    problems = [
        "Boxes",
        "Switches",
        "Generators",
        "Marble",
        "Doors",
        "Levers",
    ]

    env_names = []
    for p in problems:
        for h in helps:
            env_names.append(h+p)

    env_names.extend([
        "LeverDoorColl",
        "MarblePushColl",
        "MarblePassColl",
        "AppleStealing"
    ])

    for i, en in enumerate(env_names):
        if en in env_name:
            return i

    raise ValueError(f"Test env {env_name} not known")


subsample_step = 1
load_subsample_step = 1

x_lim = 0
max_x_lim = np.inf

summary_dict = {}
summary_dict_colors = {}
to_plot_dict = {}


default_colors_ = ["blue","orange","green","magenta", "brown", "red",'black',"grey",u'#ff7f0e',
                  "cyan", "pink",'purple', u'#1f77b4',
                  "darkorchid","sienna","lightpink", "indigo","mediumseagreen",'aqua',
                  'deeppink','silver','khaki','goldenrod'] * 100


def get_eval_data(logdir, eval_metric):
    eval_data = defaultdict(lambda :defaultdict(list))

    for root, _, files in os.walk(logdir):
        for file in files:
            if 'testing_' in file:
                assert ".pkl" in file
                test_env_name = file.lstrip("testing_").rstrip(".pkl")
                try:
                    with open(root+"/"+file, "rb") as f:
                        seed_eval_data = pickle.load(f)
                except:
                    print("Pickle not loaded: ", root+"/"+file)
                    time.sleep(1)
                    continue

                eval_data[test_env_name]["values"].append(seed_eval_data[eval_metric])
                eval_data[test_env_name]["steps"].append(seed_eval_data["test_step_nb"])

    for test_env, seed_data in eval_data.items():
        min_len_seed = min([len(s) for s in seed_data['steps']])
        eval_data[test_env]["values"] = np.array([s[:min_len_seed] for s in eval_data[test_env]["values"]])
        eval_data[test_env]["steps"] = np.array([s[:min_len_seed] for s in eval_data[test_env]["steps"]])

    return eval_data

def get_all_runs(logdir, load_subsample_step=1):
    """
    Recursively look through logdir for output files produced by
    Assumes that any file "log.csv" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'log.csv' in files:
            if (Path(root) / 'log.csv').stat().st_size == 0:
                print("CSV {} empty".format(os.path.join(root, 'log.csv')))
                continue

            run_name = root[8:]

            exp_name = None

            config = None
            exp_idx += 1

            # load progress data
            try:
                exp_data = pd.read_csv(os.path.join(root, 'log.csv'))
                print("Loaded:", os.path.join(root, 'log.csv'))
            except:
                raise ValueError("CSV {} faulty".format(os.path.join(root, 'log.csv')))

            exp_data = exp_data[::load_subsample_step]
            data_dict = exp_data.to_dict("list")

            data_dict['config'] = config
            nb_epochs = len(data_dict['frames'])
            if nb_epochs == 1:
                print(f'{run_name} -> {colored(f"nb_epochs {nb_epochs}", "red")}')
            else:
                print('{} -> nb_epochs {}'.format(run_name, nb_epochs))

            datasets.append(data_dict)

    return datasets


def get_datasets(rootdir, load_only="", load_subsample_step=1, ignore_patterns=("ignore"), require_patterns=()):
    _, models_list, _ = next(os.walk(rootdir))
    for dir_name in models_list.copy():
        # add "ignore" in a directory name to avoid loading its content
        for ignore_pattern in ignore_patterns:
            if ignore_pattern in dir_name or load_only not in dir_name:
                if dir_name in models_list:
                    models_list.remove(dir_name)

        if len(require_patterns) > 0:
            if not any([require_pattern in dir_name for require_pattern in require_patterns]):
                if dir_name in models_list:
                    models_list.remove(dir_name)

    for expe_name in list(labels.keys()):
        if expe_name not in models_list:
            del labels[expe_name]

    # setting per-model type colors
    for i, m_name in enumerate(models_list):
        for m_type, m_color in per_model_colors.items():
            if m_type in m_name:
                colors[m_name] = m_color
        print("extracting data for {}...".format(m_name))
        m_id = m_name
        models_saves[m_id] = OrderedDict()
        models_saves[m_id]['data'] = get_all_runs(rootdir+m_name, load_subsample_step=load_subsample_step)
        print("done")

        if m_name not in labels:
            labels[m_name] = m_name

        model_eval_data[m_id] = get_eval_data(logdir=rootdir+m_name, eval_metric=eval_metric)

    """
    retrieve all experiences located in "data to vizu" folder
    """
labels = OrderedDict()
per_model_colors = OrderedDict()

# LOAD DATA
models_saves = OrderedDict()
colors = OrderedDict()
model_eval_data = OrderedDict()

static_lines = {}

ignore_patterns = ["_ignore_"]

to_compare = None
load_pattern = sys.argv[1]

test_envs_to_plot = None  # plot all

min_y, max_y = 0.0, 1.1


def label_parser(label):
    label = label.replace("04-01_Pointing_CB_heldout_doors", "PPO_CB")
    label = label.replace("19-01_Color_CB_heldout_doors", "PPO_CBL")
    label = label.replace("19-01_Feedback_CB_heldout_doors_20M", "PPO_CBL")

    label = label.replace("20-01_JA_Color_CB_heldout_doors", "JA_PPO_CBL")

    label = label.replace("05-01_scaffolding_50M_no_acl", "PPO_no_scaf")
    label = label.replace("05-01_scaffolding_50M_acl_4_acl-type_intro_seq", "PPO_scaf_4")
    label = label.replace("05-01_scaffolding_50M_acl_8_acl-type_intro_seq_scaf", "PPO_scaf_8")


    label = label.replace("03-01_RR_ft_single_CB_marble_pass_A_soc_exp", "PPO_CB_role_B")
    label = label.replace("03-01_RR_ft_single_CB_marble_pass_A_asoc_contr", "PPO_CB_asocial")

    label = label.replace("05-01_RR_ft_group_50M_CB_marble_pass_A_soc_exp", "PPO_CB_role_B")
    label = label.replace("05-01_RR_ft_group_50M_CB_marble_pass_A_asoc_contr", "PPO_CB_asocial")

    label = label.replace("20-01_Imitation_PPO_CB_exploration-bonus-type_cell_exploration-bonus-params__0.25_50",
                          "PPO_CB_0.25")
    label = label.replace("20-01_Imitation_PPO_CB_exploration-bonus-type_cell_exploration-bonus-params__0.5_50",
                          "PPO_CB_0.5")
    label = label.replace("20-01_Imitation_PPO_CB_exploration-bonus-type_cell_exploration-bonus-params__1_50",
                          "PPO_CB_1")

    return label

color_dict = { }

title_tag = None


require_patterns = ["_"]

study_train = True
study_eval = False

plot_aggregated_test = False
plot_only_aggregated_test = False

smooth_factor = 0

test_envs_to_plot = None

color_dict = None


to_compare = [
    # ("26-01_Adversarial_5M_Stumps_PPO_CB", "26-01_Adversarial_5M_Stumps_PPO_CB_hidden_npc", "orange"),
    # ("26-01_Adversarial_5M_Stumps_PPO_CB", "26-01_Adversarial_5M_Stumps_PPO_CB_asoc", "green")
]

if to_compare is None and len(require_patterns) == 2 and "_" not in require_patterns:
    # if only two curves compare those two automatically
    to_compare = [(require_patterns[0], require_patterns[1], "black")]


plot_path = "plots/testplot"
show_plot = True


# all of those
include_patterns = []

fontsize = 20
legend_fontsize = 20
linewidth = 5

leg_args = {
    'fontsize': legend_fontsize
}

title_fontsize = int(fontsize*1.2)


storage_dir = "storage/"
if load_pattern.startswith(storage_dir):
    load_pattern = load_pattern[len(storage_dir):]

if load_pattern.startswith("./storage/"):
    load_pattern = load_pattern[len("./storage/"):]

get_datasets(storage_dir, str(load_pattern), load_subsample_step=load_subsample_step, ignore_patterns=ignore_patterns, require_patterns=require_patterns)

label_parser_dict = {
    # "PPO_CB": "PPO_CB",
    # "02-06_AppleStealing_experiments_cb_bonus_angle_occ_env_SocialAI-OthersPerceptionInferenceParamEnv-v1_exploration-bonus-type_cell": "NPC_visible",
}

env_type = str(load_pattern)

fig_type = "test"
try:
    top_n = int(sys.argv[2])
except:
    top_n = 8


print("Loaded:")
print("\n".join(list(models_saves.keys())))

if per_model_colors:  # order runs for legend order as in per_models_colors, with corresponding colors
    ordered_labels = OrderedDict()
    for teacher_type in per_model_colors.keys():
        for k,v in labels.items():
            if teacher_type in k:
                ordered_labels[k] = v
    labels = ordered_labels
else:
    print('not using per_model_color')
    for k in models_saves.keys():
        labels[k] = k

# Plot utils
def plot_with_shade(subplot_nb, ax, x, y, err, color, shade_color, label,
                    legend=False, leg_loc='best', title=None,
                    ylim=[0, 100], xlim=[0, 40], leg_args={}, leg_linewidth=13.0, linewidth=10.0, labelsize=20, fontsize=20, title_fontsize=30,
                    zorder=None, xlabel='Perf', ylabel='Env steps', linestyle="-", xnbins=3, ynbins=3):

    ax.locator_params(axis='x', nbins=xnbins)
    ax.locator_params(axis='y', nbins=ynbins)

    ax.tick_params(axis='y', which='both', labelsize=labelsize)
    ax.tick_params(axis='x', which='both', labelsize=labelsize*0.8)

    ax.plot(x, y, color=color, label=label, linewidth=linewidth, zorder=zorder, linestyle=linestyle)

    if not np.array_equal(err, np.zeros_like(err)):
        ax.fill_between(x, y-err, y+err, color=shade_color, alpha=0.2)

    if legend:
        leg = ax.legend(loc=leg_loc, **leg_args)  # 34
        for legobj in leg.legendHandles:
            legobj.set_linewidth(leg_linewidth)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    if subplot_nb == 0:
        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=2)

    ax.set_xlim(xmin=xlim[0], xmax=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if title:
        ax.set_title(title, fontsize=title_fontsize, y=1.03)


# only one figure is drawn -> maybe we can add loops later
assert len(metrics) == 1
f, ax = plt.subplots(1, 1, figsize=(9.0, 9.0))
ax = [ax]

min_y = np.inf

max_steps = 0
exclude_patterns = []

metric = metrics[0]

ylabel = {
    "success_rate_mean": "Success rate (%)",
    "exploration_bonus_mean": "Exploration bonus",
    "NPC_intro": "Successful introduction (%)",
}.get(metric, metric)

# for metric_i, metric in enumerate(metrics):
default_colors = default_colors_.copy()

if study_train:
    for model_i, model_id in enumerate(models_saves.keys()):

        #excluding some experiments
        if any([ex_pat in model_id for ex_pat in exclude_patterns]):
            continue

        if len(include_patterns) > 0:
            if not any([in_pat in model_id for in_pat in include_patterns]):
                continue

        runs_data = models_saves[model_id]['data']
        ys = []

        if runs_data[0]['frames'][1] == 'frames':
            runs_data[0]['frames'] = list(filter(('frames').__ne__, runs_data[0]['frames']))

        # determine minimal run length across seeds
        lens = [len(run['frames']) for run in runs_data if len(run['frames'])]
        minimum = sorted(lens)[-min(top_n, len(lens))]
        min_len = np.min([len(run['frames']) for run in runs_data if len(run['frames']) >= minimum])

        # keep only top k
        runs_data = [run for run in runs_data if len(run['frames']) >= minimum]

        # min_len = np.min([len(run['frames']) for run in runs_data if len(run['frames']) > 10])

        # compute env steps (x axis)
        longest_id = np.argmax([len(rd['frames']) for rd in runs_data])
        steps = np.array(runs_data[longest_id]['frames'], dtype=np.int) / steps_denom
        steps = steps[:min_len]

        for run in runs_data:
            if metric not in run:
                raise ValueError(f"Metric {metric} not found. Possible metrics: {list(run.keys())}")

            data = run[metric]

            # checking for header
            if data[1] == metric:
                data = np.array(list(filter((metric).__ne__, data)), dtype=np.float16)

            if len(data) >= min_len:
                # discard extra
                if len(data) > min_len:
                    print("run has too many {} datapoints ({}). Discarding {}".format(model_id, len(data),
                                                                                      len(data) - min_len))
                    data = data[0:min_len]
                ys.append(data)
            else:
                raise ValueError("How can data be < min_len if it was capped above")

        ys_same_len = ys

        # computes stats
        n_seeds = len(ys_same_len)

        sems = np.std(ys_same_len, axis=0)/np.sqrt(len(ys_same_len))  # sem
        stds = np.std(ys_same_len, axis=0)  # std
        means = np.mean(ys_same_len, axis=0)
        color = default_colors[model_i]

        # plot x y bounds
        curr_max_steps = np.max(steps)

        if curr_max_steps > max_steps:
            max_steps = curr_max_steps

        if subsample_step:
            steps = steps[0::subsample_step]
            means = means[0::subsample_step]
            stds = stds[0::subsample_step]
            sems = sems[0::subsample_step]
            ys_same_len = [y[0::subsample_step] for y in ys_same_len]

        # display seeds separately
        label = label_parser(model_id)

        if study_eval:
            label = label+"(train)"

        if color_dict:
            color = color_dict[label]
        else:
            color = default_colors[model_i]

        if smooth_factor:
            means = smooth(means, smooth_factor)
            stds = smooth(stds, smooth_factor)

        to_plot_dict[model_id] = {
            "label": label,
            "steps": steps,
            "means": means,
            "stds": stds,
            "sems": sems,
            "ys": ys_same_len,
            "color": color,
        }


if study_eval:
    print("Evaluation")
    # evaluation sets
    number_of_eval_envs = max(list([len(v.keys()) for v in model_eval_data.values()]))

    if plot_aggregated_test:
        number_of_eval_envs += 1

    if number_of_eval_envs == 0:
        print("No eval envs")
        exit()

    default_colors = default_colors_.copy()

    test_summary_dict = defaultdict(dict)
    test_summary_dict_colors = defaultdict(dict)

    for model_i, model_id in enumerate(model_eval_data.keys()):
        # excluding some experiments
        if any([ex_pat in model_id for ex_pat in exclude_patterns]):
            continue
        if len(include_patterns) > 0:
            if not any([in_pat in model_id for in_pat in include_patterns]):
                continue

        # test envs
        test_envs = model_eval_data[model_id].items()

        # filter unwanted eval envs
        if test_envs_to_plot is not None:
            test_envs = [(name, data) for name, data in test_envs if name in test_envs_to_plot]

        # computes stats
        if sort_test:
            test_envs_sorted = list(sorted(test_envs, key=lambda kv: sort_test_set(kv[0])))
        else:
            test_envs_sorted = list(test_envs)

        if plot_aggregated_test:
            agg_means = []

        for env_i, (test_env, env_data) in enumerate(test_envs_sorted):
            ys_same_len = env_data["values"]
            steps = env_data["steps"].mean(0) / steps_denom
            n_seeds = len(ys_same_len)

            sems = np.std(ys_same_len, axis=0) / np.sqrt(len(ys_same_len))  # sem
            stds = np.std(ys_same_len, axis=0)  # std
            means = np.mean(ys_same_len, axis=0)
            color = default_colors[model_i]

            curr_max_steps = np.max(steps)

            if plot_aggregated_test:
                agg_means.append(means)


            x_lim = max(steps[-1], x_lim)
            x_lim = min(max_x_lim, x_lim)

            eval_metric_name = {
                "test_success_rates": "Success rate",
                'exploration_bonus_mean': "Exploration bonus",
            }.get(eval_metric, eval_metric)

            test_env_name = test_env.replace("Env", "").replace("Test", "")

            env_types = ["InformationSeeking", "Collaboration", "PerspectiveTaking"]
            for env_type in env_types:
                if env_type in test_env_name:
                    test_env_name = test_env_name.replace(env_type, "")
                    test_env_name += f"\n({env_type})"

            if len(test_envs_sorted) > 1:
                label = label_parser(model_id) + f"_{test_env}"
            else:
                label = label_parser(model_id)

            if study_train:
                label=label+"(test)"

            if not plot_only_aggregated_test:

                if label in color_dict:
                    color = color_dict[label]
                else:
                    color = default_colors[model_i*len(test_envs_sorted)+env_i]

                if legend_show_n_seeds:
                    label = label + "({})".format(n_seeds)

                if eval_smooth_factor:
                    means = smooth(means, eval_smooth_factor)
                    stds = smooth(stds, eval_smooth_factor)
                    sems = smooth(sems, eval_smooth_factor)

                to_plot_dict[model_id+f"_{test_env}"] = {
                    "label": label,
                    "steps": steps,
                    "means": means,
                    "stds": stds,
                    "sems": sems,
                    "ys": ys_same_len,
                    "color": color,
                }

        if plot_aggregated_test:

            ys_same_len = agg_means
            agg_means = np.array(agg_means)
            agg_mean = agg_means.mean(axis=0)
            agg_std = agg_means.std(axis=0)  # std
            agg_sems = ...

            label = label_parser(model_id)

            if study_train:
                label = label + "(train)"

            if eval_smooth_factor:
                agg_mean = smooth(agg_mean, eval_smooth_factor)
                agg_std = smooth(agg_std, eval_smooth_factor)
                agg_sems = smooth(agg_sems, eval_smooth_factor)

            if label in color_dict:
                color = color_dict[label]

            else:
                color = default_colors[model_i]

            to_plot_dict[model_id+"_agg_test"] = {
                "label": label,
                "steps": steps,
                "means": agg_mean,
                "stds": agg_std,
                "sems": agg_sems,
                "ys": ys_same_len,
                "color": color,
            }


# should be labels
to_scatter_dict = {}

if to_compare is not None:
    for comp_i, (a_model_id, b_model_id, color) in enumerate(to_compare):

        a_data = to_plot_dict[a_model_id]["ys"]
        b_data = to_plot_dict[b_model_id]["ys"]

        steps = to_plot_dict[a_model_id]["steps"]

        if color == "auto_color":
            color = to_plot_dict[a_model_id]["color"]

        if len(a_data[0]) != len(b_data[0]):
            # extract steps present in both
            a_steps = to_plot_dict[a_model_id]["steps"]
            b_steps = to_plot_dict[b_model_id]["steps"]

            steps = list(set(a_steps) & set(b_steps))

            # keep only the values for those steps
            mask_a = [(a_s in steps) for a_s in a_steps]
            a_data = np.array(a_data)[:, mask_a]

            mask_b = [(b_s in steps) for b_s in b_steps]
            b_data = np.array(b_data)[:, mask_b]

        p = stats.ttest_ind(
            a_data,
            b_data,
            equal_var=False
        ).pvalue

        steps = [s for s, p in zip(steps, p) if p < test_p]

        ys = [1.02+0.02*comp_i]*len(steps)

        to_scatter_dict[f"compare_{a_model_id}_{b_model_id}"] = {
            "label": "",
            "xs": steps,
            "ys": ys,
            "color": color,
        }

for scatter_i, (scatter_ID, scatter_id_data) in enumerate(to_scatter_dict.items()):

    # unpack data
    label, xs, ys, color = (
        scatter_id_data["label"],
        scatter_id_data["xs"],
        scatter_id_data["ys"],
        scatter_id_data["color"],
    )

    xlabel = f"Env steps (1e6)"

    plt.scatter(
        xs,
        ys,
        color=color,
        marker="x"
    )

    summary_dict[label] = xs[-1]
    summary_dict_colors[label] = color

for curve_i, (curve_ID, model_id_data) in enumerate(to_plot_dict.items()):

    # unpack data
    label, steps, means, stds, sems, ys, color = (
        model_id_data["label"],
        model_id_data["steps"],
        model_id_data["means"],
        model_id_data["stds"],
        model_id_data["sems"],
        model_id_data["ys"],
        model_id_data["color"]
    )

    # if smooth_factor:
    #     means = smooth(means, smooth_factor)
    #     stds = smooth(stds, smooth_factor)

    if legend_show_n_seeds:
        n_seeds = len(ys)
        label = label+"({})".format(n_seeds)


    x_lim = max(steps[-1], x_lim)
    x_lim = min(max_x_lim, x_lim)

    xlabel = f"Env steps (1e6)"


    plot_with_shade(
        0, ax[0], steps, means, stds, color, color, label,
        # 0, ax[0], steps, means, sems, color, color, label,
        legend=draw_legend,
        xlim=[0, x_lim],
        ylim=[0, max_y],
        xlabel=xlabel,
        ylabel=ylabel,
        title=title_tag,
        labelsize=fontsize,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        linewidth=linewidth,
        leg_linewidth=5,
        leg_args=leg_args,
        xnbins=xnbins,
        ynbins=ynbins,
    )

    summary_dict[label] = means[-1]
    summary_dict_colors[label] = color

# plot static lines
if static_lines:
    for label, (mean, std, color) in static_lines.items():

        if label == "":
            label = None

        plot_with_shade(
            0, ax[0], steps, np.array([mean]*len(steps)), np.array([std]*len(steps)), color, color, label,
            legend=True,
            xlim=[0, x_lim],
            ylim=[0, 1.0],
            xlabel=f"Env steps (1e6)",
            ylabel=ylabel,
            linestyle=":",
            leg_args=leg_args,
            fontsize=fontsize,
            title_fontsize=title_fontsize,
            xnbins=xnbins,
            ynbins=ynbins,
        )


if plot_path:
    f.savefig(plot_path+".png")
    f.savefig(plot_path+".svg")
    f.savefig(plot_path+".jpeg", dpi=300)
    print(f"Plot saved to {plot_path}.[png/svg/jpeg].")


if show_plot:
    plt.tight_layout()
    plt.subplots_adjust(hspace=1.5, wspace=0.5, left=0.1, right=0.9, bottom=0.1, top=0.85)
    plt.suptitle(super_title)
    plt.show()
plt.close()


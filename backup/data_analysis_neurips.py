#!/usr/bin/env python
import seaborn
import numpy as np
import os
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import sys
from termcolor import cprint

# Load data

# Global vars for tracking and labeling data at load time.
exp_idx = 0
label_parser_dict = None

smooth_factor = 10
leg_size = 30

subsample_step = 1
load_subsample_step = 50

default_colors = ["blue","orange","green","magenta", "brown", "red",'black',"grey",u'#ff7f0e',
                  "cyan", "pink",'purple', u'#1f77b4',
                  "darkorchid","sienna","lightpink", "indigo","mediumseagreen",'aqua',
                  'deeppink','silver','khaki','goldenrod','y','y','y','y','y','y','y','y','y','y','y','y' ]  + ['y']*50

def get_all_runs(logdir, load_subsample_step=1):
    """
    Recursively look through logdir for output files produced by
    Assumes that any file "progress.txt" is a valid hit. 
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'log.csv' in files:
            run_name = root[8:]
            exp_name = None
            
            # try to load a config file containing hyperparameters
            config = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']       
            except:
                print('No file named config.json')
                
            exp_idx += 1

            # load progress data
            try:
                print(os.path.join(root,'log.csv'))
                exp_data = pd.read_csv(os.path.join(root,'log.csv'))
            except:
                raise ValueError("CSV {} faulty".format(os.path.join(root, 'log.csv')))
            
            exp_data = exp_data[::load_subsample_step]
            data_dict = exp_data.to_dict("list")

            data_dict['config'] = config
            nb_epochs = len(data_dict['frames'])
            print('{} -> {}'.format(run_name, nb_epochs))


            datasets.append(data_dict)

    return datasets

def get_datasets(rootdir, load_only="", load_subsample_step=1, ignore_pattern="ignore"):
    _, models_list, _ = next(os.walk(rootdir))
    print(models_list)
    for dir_name in models_list.copy():
        # add "ignore" in a directory name to avoid loading its content
        if ignore_pattern in dir_name or load_only not in dir_name:
            models_list.remove(dir_name)
    for expe_name in list(labels.keys()):
        if expe_name not in models_list:
            del labels[expe_name]
            
    # setting per-model type colors    
    for i,m_name in enumerate(models_list):
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

    """
    retrieve all experiences located in "data to vizu" folder
    """
labels = OrderedDict()
per_model_colors = OrderedDict()
# per_model_colors = OrderedDict([('ALP-GMM',u'#1f77b4'),
#                                 ('hmn','pink'),
#                                 ('ADR','black')])

# LOAD DATA
models_saves = OrderedDict()
colors = OrderedDict()

static_lines = {}
# get_datasets("storage/",load_only="RERUN_WizardGuide")
# get_datasets("storage/",load_only="RERUN_WizardTwoGuides")
try:
    figure_id = eval(sys.argv[1])
except:
    figure_id = sys.argv[1]

print("fig:", figure_id)
if figure_id == 0:
    # train change
    env_type = "No_NPC_environment"
    fig_type = "train"

    get_datasets("storage/", "RERUN_WizardGuide_lang64_mm", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardGuide_lang64_deaf_no_explo", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardGuide_lang64_no_explo", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardGuide_lang64_curr_dial", load_subsample_step=load_subsample_step)
    top_n = 16
elif figure_id == 1:
    # arch change
    env_type = "No_NPC_environment"
    fig_type = "arch"

    get_datasets("storage/", "RERUN_WizardGuide_lang64_mm", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardGuide_lang64_bow", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardGuide_lang64_no_mem", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardGuide_lang64_bigru", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardGuide_lang64_attgru", load_subsample_step=load_subsample_step)
    top_n = 16
elif figure_id == 2:
    # train change FULL
    env_type = "FULL_environment"
    fig_type = "train"

    get_datasets("storage/", "RERUN_WizardTwoGuides_lang64_mm", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardTwoGuides_lang64_deaf_no_explo", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardTwoGuides_lang64_no_explo", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardTwoGuides_lang64_curr_dial", load_subsample_step=load_subsample_step)
    top_n = 16
elif figure_id == 3:
    # arch change FULL
    env_type = "FULL_environment"
    fig_type = "arch"

    get_datasets("storage/", "RERUN_WizardTwoGuides_lang64_mm", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardTwoGuides_lang64_bow", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardTwoGuides_lang64_no_mem", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardTwoGuides_lang64_bigru", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "RERUN_WizardTwoGuides_lang64_attgru", load_subsample_step=load_subsample_step)
    top_n = 16
elif str(figure_id) == "ShowMe":

    get_datasets("storage/", "20-05_NeurIPS_ShowMe_ABL_CEB", load_subsample_step=load_subsample_step, ignore_pattern="tanh_0.3")
    get_datasets("storage/", "20-05_NeurIPS_ShowMe_NO_BONUS_ABL", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "20-05_NeurIPS_ShowMe_CEB", load_subsample_step=load_subsample_step, ignore_pattern="tanh_0.3")
    get_datasets("storage/", "20-05_NeurIPS_ShowMe_NO_BONUS_env", load_subsample_step=load_subsample_step)

    label_parser_dict = {
        "20-05_NeurIPS_ShowMe_ABL_CEB" : "ShowMe_exp_bonus_no_social_skills_required",
        "20-05_NeurIPS_ShowMe_NO_BONUS_ABL" : "ShowMe_no_bonus_no_social_skills_required",
        "20-05_NeurIPS_ShowMe_CEB" : "ShowMe_exp_bonus",
        "20-05_NeurIPS_ShowMe_NO_BONUS_env" : "ShowMe_no_bonus",
    }

    env_type = str(figure_id)

    fig_type = "test"
    top_n = 16

elif str(figure_id) == "Help":

    # env_type = "Bobo"
    # get_datasets("storage/", "Bobo")
    get_datasets("storage/", "24-05_NeurIPS_Help", load_subsample_step=load_subsample_step, ignore_pattern="ABL")
    # get_datasets("storage/", "26-05_NeurIPS_gpu_Help_NoSocial_NO_BONUS_ABL", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "26-05_NeurIPS_gpu_Help_NoSocial_NO_BONUS_env", load_subsample_step=load_subsample_step)

    label_parser_dict = {
        "Help_NO_BONUS_env": "PPO",
        "Help_BONUS_env": "PPO+Explo",
        # "Help_NO_BONUS_ABL_env": "ExiterRole_no_bonus_no_NPC",
        # "Help_BONUS_ABL_env": "ExiterRole_bonus_no_NPC",
        "26-05_NeurIPS_gpu_Help_NoSocial_NO_BONUS_env": "Unsocial PPO",
        # "26-05_NeurIPS_gpu_Help_NoSocial_NO_BONUS_ABL": "ExiterRole_Insocial_ABL"
    }

    static_lines = {
        "PPO (helper)": (0.12, 0.05, "#1f77b4"),
        "PPO+Explo (helper)": (0.11, 0.04, "indianred"),
        # "Help_exp_bonus": (0.11525, 0.04916 , default_colors[2]),
        # "HelperRole_ABL_no_exp_bonus": (0.022375, 0.01848, default_colors[3]),
        "Unsocial PPO (helper)": (0.15, 0.06, "grey"),
        # "HelperRole_ABL_Insocial": (0.01775, 0.010544, default_colors[4]),
    }

    env_type = str(figure_id)

    fig_type = "test"
    top_n = 16

elif str(figure_id) == "TalkItOut":
    print("You mean Polite")
    exit()

elif str(figure_id) == "TalkItOutPolite":
    # env_type = "TalkItOut"
    # get_datasets("storage/", "ORIENT_env_MiniGrid-TalkItOut")

    # env_type = "GuideThief"
    # get_datasets("storage/", "GuideThief")

    # env_type = "Bobo"
    # get_datasets("storage/", "Bobo")
    get_datasets("storage/", "20-05_NeurIPS_TalkItOutPolite", load_subsample_step=load_subsample_step)
    # get_datasets("storage/", "21-05_NeurIPS_small_bonus_TalkItOutPolite")
    get_datasets("storage/", "26-05_NeurIPS_gpu_TalkItOutPolite_NoSocial_NO_BONUS_env", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "26-05_NeurIPS_gpu_TalkItOutPolite_NoSocial_NO_BONUS_NoLiar", load_subsample_step=load_subsample_step)

    label_parser_dict = {
        "TalkItOutPolite_NO_BONUS_env": "PPO",
        "TalkItOutPolite_e": "PPO+Explo",
        "TalkItOutPolite_NO_BONUS_NoLiar": "PPO (no liar)",
        "TalkItOutPolite_NoLiar_e": "PPO+Explo (no liar)",
        "26-05_NeurIPS_gpu_TalkItOutPolite_NoSocial_NO_BONUS_env": "Unsocial PPO",
        "26-05_NeurIPS_gpu_TalkItOutPolite_NoSocial_NO_BONUS_NoLiar": "Unsocial PPO (no liar)",
    }


    env_type = str(figure_id)

    fig_type = "test"
    top_n = 16

elif str(figure_id) == "DiverseExit":
    get_datasets("storage/", "24-05_NeurIPS_DiverseExit", load_subsample_step=load_subsample_step)
    get_datasets("storage/", "26-05_NeurIPS_gpu_DiverseExit", load_subsample_step=load_subsample_step)

    label_parser_dict = {
        "DiverseExit_NO_BONUS": "No_bonus",
        "DiverseExit_BONUS": "BOnus",
        "gpu_DiverseExit_NoSocial": "No_social",
    }

    env_type = str(figure_id)

    fig_type = "test"
    top_n = 16

else:
    get_datasets("storage/", str(figure_id), load_subsample_step=load_subsample_step)

    env_type = str(figure_id)

    fig_type = "test"
    top_n = 8

#### get_datasets("storage/", "RERUN_WizardGuide_lang64_nameless")
#### get_datasets("storage/", "RERUN_WizardTwoGuides_lang64_nameless")


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

def plot_with_shade(subplot_nb, ax,x,y,err,color,shade_color,label,
                  y_min=None,y_max=None, legend=False, leg_size=30, leg_loc='best', title=None,
                  ylim=[0,100], xlim=[0,40], leg_args={}, leg_linewidth=13.0, linewidth=10.0, ticksize=20,
                   zorder=None, xlabel='perf',ylabel='env steps'):
    #plt.rcParams.update({'font.size': 15})
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=3)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.plot(x,y, color=color, label=label,linewidth=linewidth,zorder=zorder)
    ax.fill_between(x,y-err,y+err,color=shade_color,alpha=0.2)
    if legend:
        leg = ax.legend(loc=leg_loc, **leg_args) #34
        for legobj in leg.legendHandles:
            legobj.set_linewidth(leg_linewidth)
    ax.set_xlabel(xlabel, fontsize=30)
    if subplot_nb == 0:
        ax.set_ylabel(ylabel, fontsize=30,labelpad=-4)
    ax.set_xlim(xmin=xlim[0],xmax=xlim[1])
    ax.set_ylim(bottom=ylim[0],top=ylim[1])
    if title:
        ax.set_title(title, fontsize=22)
# Plot utils
def plot_with_shade_grg(subplot_nb, ax,x,y,err,color,shade_color,label,
                  y_min=None,y_max=None, legend=False, leg_size=30, leg_loc='best', title=None,
                  ylim=[0,100], xlim=[0,40], leg_args={}, leg_linewidth=13.0, linewidth=10.0, ticksize=20,
                   zorder=None, xlabel='perf',ylabel='env steps', linestyle="-"):
    #plt.rcParams.update({'font.size': 15})
    ax.locator_params(axis='x', nbins=4)
    ax.locator_params(axis='y', nbins=3)
    ax.tick_params(axis='both', which='major', labelsize=ticksize)


    ax.plot(x, y, color=color, label=label,linewidth=linewidth,zorder=zorder, linestyle=linestyle)
    ax.fill_between(x, y-err, y+err,color=shade_color,alpha=0.2)
    if legend:
        leg = ax.legend(loc=leg_loc, **leg_args) #34
        for legobj in leg.legendHandles:
            legobj.set_linewidth(leg_linewidth)
    ax.set_xlabel(xlabel, fontsize=30)
    if subplot_nb == 0:
        ax.set_ylabel(ylabel, fontsize=30, labelpad=-4)
    ax.set_xlim(xmin=xlim[0],xmax=xlim[1])
    ax.set_ylim(bottom=ylim[0],top=ylim[1])
    if title:
        ax.set_title(title, fontsize=22)
        

# Metric plot
metric = 'bin_extrinsic_return_mean'
# metric = 'mission_string_observed_mean'
# metric = 'extrinsic_return_mean'
# metric = 'extrinsic_return_max'
# metric = "rreturn_mean"
# metric = 'rreturn_max'
# metric = 'FPS'

f, ax = plt.subplots(1, 1, figsize=(10.0, 6.0))
ax = [ax]
max_y = -np.inf
min_y = np.inf
# hardcoded
min_y, max_y = 0.0, 1.0
max_steps = 0
exclude_patterns = []
include_patterns = []


def label_parser(label, figure_id, label_parser_dict=None):
    if label_parser_dict:
        if sum([1 for k, v in label_parser_dict.items() if k in label]) != 1:
            if label in label_parser_dict:
                # see if there is an exact match
                return label_parser_dict[label]
            else:
                print("ERROR multiple curves match a lable and there is no exact match")
                print(label)
                exit()

        for k, v in label_parser_dict.items():
            if k in label: return v

    else:
        # return label.split("_env_")[1]
        if figure_id not in [1,2,3,4]:
            return label
        else:
            label_parser_dict = {
                "RERUN_WizardGuide_lang64_no_explo": "MH-BabyAI",
                "RERUN_WizardTwoGuides_lang64_no_explo": "MH-BabyAI",

                "RERUN_WizardGuide_lang64_mm_baby_short_rec_env": "MH-BabyAI-ExpBonus",
                "RERUN_WizardTwoGuides_lang64_mm_baby_short_rec_env": "MH-BabyAI-ExpBonus",

                "RERUN_WizardGuide_lang64_deaf_no_explo": "Deaf-MH-BabyAI",
                "RERUN_WizardTwoGuides_lang64_deaf_no_explo": "Deaf-MH-BabyAI",

                "RERUN_WizardGuide_lang64_bow": "MH-BabyAI-ExpBonus-BOW",
                "RERUN_WizardTwoGuides_lang64_bow": "MH-BabyAI-ExpBonus-BOW",

                "RERUN_WizardGuide_lang64_no_mem": "MH-BabyAI-ExpBonus-no-mem",
                "RERUN_WizardTwoGuides_lang64_no_mem": "MH-BabyAI-ExpBonus-no-mem",

                "RERUN_WizardGuide_lang64_bigru": "MH-BabyAI-ExpBonus-bigru",
                "RERUN_WizardTwoGuides_lang64_bigru": "MH-BabyAI-ExpBonus-bigru",

                "RERUN_WizardGuide_lang64_attgru": "MH-BabyAI-ExpBonus-attgru",
                "RERUN_WizardTwoGuides_lang64_attgru": "MH-BabyAI-ExpBonus-attgru",

                "RERUN_WizardGuide_lang64_curr_dial": "MH-BabyAI-ExpBonus-current-dialogue",
                "RERUN_WizardTwoGuides_lang64_curr_dial": "MH-BabyAI-ExpBonus-current-dialogue",

                "RERUN_WizardTwoGuides_lang64_mm_baby_short_rec_100M": "MH-BabyAI-ExpBonus-100M"
            }
            if sum([1 for k, v in label_parser_dict.items() if k in label]) != 1:
                print("ERROR multiple curves match a lable")
                print(label)
                exit()

            for k, v in label_parser_dict.items():
                if k in label: return v

    return label

per_seed=False

for i, m_id in enumerate(models_saves.keys()):
    #excluding some experiments
    if any([ex_pat in m_id for ex_pat in exclude_patterns]):
        continue
    if len(include_patterns) > 0:
        if not any([in_pat in m_id for in_pat in include_patterns]):
            continue
    runs_data = models_saves[m_id]['data']
    ys = []

    # DIRTY FIX FOR FAULTY LOGGING
    print("m_id:", m_id)
    if runs_data[0]['frames'][1] == 'frames':
        runs_data[0]['frames'] = list(filter(('frames').__ne__, runs_data[0]['frames']))
    ###########################################    


    # determine minimal run length across seeds
    minimum = sorted([len(run['frames']) for run in runs_data if len(run['frames'])])[-top_n]
    min_len = np.min([len(run['frames']) for run in runs_data if len(run['frames']) >= minimum])

#     min_len = np.min([len(run['frames']) for run in runs_data if len(run['frames']) > 10])


    print("min_len:", min_len)

    #compute env steps (x axis)
    longest_id = np.argmax([len(rd['frames']) for rd in runs_data])
    steps = np.array(runs_data[longest_id]['frames'], dtype=np.int) / 1000000
    steps = steps[:min_len]
    for run in runs_data:  
        data = run[metric]
        # DIRTY FIX FOR FAULTY LOGGING (headers in data)
        if data[1] == metric:
            data = np.array(list(filter((metric).__ne__, data)), dtype=np.float16)
        ###########################################
        if len(data) >= min_len:
            if len(data) > min_len:
                print("run has too many {} datapoints ({}). Discarding {}".format(m_id, len(data),
                                                                                  len(data)-min_len))
                data = data[0:min_len]
            ys.append(data)
    ys_same_len = ys  # RUNS MUST HAVE SAME LEN

    # computes stats
    n_seeds = len(ys_same_len)
    sems = np.std(ys_same_len,axis=0)/np.sqrt(len(ys_same_len)) # sem
    stds = np.std(ys_same_len,axis=0) # std
    means = np.mean(ys_same_len,axis=0)
    color = default_colors[i]

    # per-metric adjusments
    ylabel=metric
    if metric == 'bin_extrinsic_return_mean':
        ylabel = "success rate"
    if metric == 'duration':
        ylabel = "time (hours)"
        means = means / 3600
        sems = sems / 3600
        stds = stds / 3600

    #plot x y bounds
    curr_max_y = np.max(means)
    curr_min_y = np.min(means)
    curr_max_steps = np.max(steps)
    if curr_max_y > max_y:
        max_y = curr_max_y
    if curr_min_y < min_y:
        min_y = curr_min_y
    if curr_max_steps > max_steps:
        max_steps = curr_max_steps

    if subsample_step:
        steps = steps[0::subsample_step]
        means = means[0::subsample_step]
        stds = stds[0::subsample_step]
        sems = sems[0::subsample_step]
        ys_same_len = [y[0::subsample_step] for y in ys_same_len]

    # display seeds separtely
    if per_seed:
        for s_i, seed_ys in enumerate(ys_same_len):
            seed_c = default_colors[i+s_i]
            label = m_id#+"(s:{})".format(s_i)
            plot_with_shade(0, ax[0], steps, seed_ys, stds*0, seed_c, seed_c, label,
                legend=False, xlim=[0, max_steps], ylim=[min_y, max_y],
                        leg_size=leg_size, xlabel="env steps (millions)", ylabel=ylabel, smooth_factor=smooth_factor,
                            )
    else:
        label = label_parser(m_id, figure_id, label_parser_dict=label_parser_dict)
        label = label #+"({})".format(n_seeds)


        def smooth(x_, n=50):
            if type(x_) == list:
                x_ = np.array(x_)
            return np.array([x_[max(i - n, 0):i + 1].mean() for i in range(len(x_))])
        if smooth_factor:
            means = smooth(means,smooth_factor)
            stds = smooth(stds,smooth_factor)
        x_lim = 30
        if figure_id == "TalkItOutPolite":
            leg_args = {
                'ncol': 1,
                'columnspacing': 1.0,
                'handlelength': 1.0,
                'frameon': False,
                # 'bbox_to_anchor': (0.00, 0.23, 0.10, .102),
                'bbox_to_anchor': (0.55, 0.35, 0.10, .102),
                'labelspacing': 0.2,
                'fontsize': 27
            }
        elif figure_id == "Help":
            leg_args = {
                'ncol': 1,
                'columnspacing': 1.0,
                'handlelength': 1.0,
                'frameon': False,
                # 'bbox_to_anchor': (0.00, 0.23, 0.10, .102),
                'bbox_to_anchor': (0.39, 0.20, 0.10, .102),
                'labelspacing': 0.2,
                'fontsize': 27
            }
        else:
            leg_args = {}

        color_code = dict([
            ('PPO+Explo', 'indianred'),
            ('PPO', "#1f77b4"),
            ('Unsocial PPO', "grey"),
            ('PPO (no liar)', "#043252"),
            ('PPO+Explo (no liar)', "darkred"),
            ('Unsocial PPO (no liar)', "black"),
            ('PPO+Explo (helper)', 'indianred'),
            ('PPO (helper)', "#1f77b4"),
            ('Unsocial PPO (helper)', "grey")]
        )
        color = color_code.get(label, np.random.choice(default_colors))
        print("C:",color)
        plot_with_shade_grg(
            0, ax[0], steps, means, stds, color, color, label,
                    legend=True,
                    xlim=[0, steps[-1] if not x_lim else x_lim],
                    ylim=[0, 1.0], xlabel="env steps (millions)", ylabel=ylabel, title=None,
                        leg_args =leg_args)
        #
        # plot_with_shade(0, ax[0], steps, means, stds, color, color,label,
        #         legend=True, xlim=[0, max_steps], ylim=[min_y, max_y],
        #                 leg_size=leg_size, xlabel="Env steps (millions)", ylabel=ylabel, linewidth=5.0, smooth_factor=smooth_factor)


for label, (mean, std, color) in static_lines.items():
    plot_with_shade_grg(
        0, ax[0], steps, np.array([mean]*len(steps)), np.array([std]*len(steps)), color, color, label,
                    legend=True,
                    xlim=[0, max_steps],
                    ylim=[0, 1.0],
                    xlabel="env steps (millions)", ylabel=ylabel, linestyle=":",
                    leg_args=leg_args)

plt.tight_layout()
f.savefig('graphics/{}_results.svg'.format(str(figure_id)))
f.savefig('graphics/{}_results.png'.format(str(figure_id)))
plt.show()
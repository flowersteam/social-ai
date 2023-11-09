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

eval_metric = "test_success_rates"
# eval_metric = "exploration_bonus_mean"

super_title = ""
# super_title = "PPO - No exploration bonus"
# super_title = "Count Based exploration bonus (Grid Search)"
# super_title = "PPO + RND"
# super_title = "PPO + RIDE"

agg_title = ""

color_dict = None
eval_filename = None

max_frames = 20_000_000

draw_legend = True
per_seed = False
study_eval = True

plot_train = True
plot_test = True

plot_aggregated_test = False
plot_only_aggregated_test = False


train_inc_font = 3

xnbins = 4
ynbins = 3

steps_denom = 1e6

# Global vas for tracking and labeling data at load time.
exp_idx = 0
label_parser_dict = None
label_parser = lambda l, _, label_parser_dict: l

# smooth_factor = 100
smooth_factor = 10
smooth_factor = 0
print("smooth factor:", smooth_factor)
eval_smooth_factor = 1
leg_size = 30

def smooth(x_, n=50):
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
max_x_lim = 17
max_x_lim = np.inf
# x_lim = 100

summary_dict = {}
summary_dict_colors = {}


# default_colors = ["blue","orange","green","magenta", "brown", "red",'black',"grey",u'#ff7f0e',
#                   "cyan", "pink",'purple', u'#1f77b4',
#                   "darkorchid","sienna","lightpink", "indigo","mediumseagreen",'aqua',
#                   'deeppink','silver','khaki','goldenrod','y','y','y','y','y','y','y','y','y','y','y','y' ] + ['y']*50
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

        # if 'log.csv' in files:
        #     run_name = root[8:]
        #     exp_name = None
        #
        #     config = None
        #     exp_idx += 1
        #
        #     # load progress data
        #     try:
        #         print(os.path.join(root, 'log.csv'))
        #         exp_data = pd.read_csv(os.path.join(root, 'log.csv'))
        #     except:
        #         size = (Path(root) / 'log.csv').stat().st_size
        #         if size == 0:
        #             raise ValueError("CSV {} empty".format(os.path.join(root, 'log.csv')))
        #         else:
        #             raise ValueError("CSV {} faulty".format(os.path.join(root, 'log.csv')))
        #
        #     exp_data = exp_data[::load_subsample_step]
        #     data_dict = exp_data.to_dict("list")
        #
        #     data_dict['config'] = config
        #     nb_epochs = len(data_dict['frames'])
        #     print('{} -> {}'.format(run_name, nb_epochs))

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
# per_model_colors = OrderedDict([('ALP-GMM',u'#1f77b4'),
#                                 ('hmn','pink'),
#                                 ('ADR','black')])

# LOAD DATA
models_saves = OrderedDict()
colors = OrderedDict()
model_eval_data = OrderedDict()

static_lines = {}
# get_datasets("storage/",load_only="RERUN_WizardGuide")
# get_datasets("storage/",load_only="RERUN_WizardTwoGuides")
try:
    load_pattern = eval(sys.argv[1])

except:
    load_pattern = sys.argv[1]

ignore_patterns = ["_ignore_"]
require_patterns = [
    "_"
]

# require_patterns = [
    # "dummy_cs_jz_scaf_A_E_N_A_E",
    # "03-12_dummy_cs_jz_formats_AE",
# ]
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     if "single" in label:
#         ty = "single"
#     elif "group" in label:
#         ty = "group"
#
#     if "asoc" in label:
#         return f"Asocial_pretrain({ty})"
#
#     if "exp_soc" in label:
#         return f"Role_B_pretrain({ty})"
#
#     return label


#
# # DUMMY FORMATS
# require_patterns = [
#     "03-12_dummy_cs_formats_CBL",
#     "dummy_cs_formats_CBL_N_rec_5"
    # "03-12_dummy_cs_jz_formats_",
    # "dummy_cs_jz_formats_N_rec_5"
# ]
# def label_parser(label, figure_id, label_parser_dict=None):
#     if "CBL" in label:
#         eb = "CBL"
#     else:
#         eb = "no_bonus"
#
#     if "AE" in label:
#         label = f"AE_PPO_{eb}"
#     elif "E" in label:
#         label = f"E_PPO_{eb}"
#     elif "A" in label:
#         label = f"A_PPO_{eb}"
#     elif "N" in label:
#         label = f"N_PPO_{eb}"
#
#     return label
#

# DUMMY CLASSIC
# require_patterns = [
    # "07-12_dummy_cs_NEW2_Pointing_sm_CB_very_small",
    # "dummy_cs_JA_Pointing_CB_sm",

    # "06-12_dummy_cs_NEW_Color_CBL",
    # "dummy_cs_JA_Color_CBL_new"

    # "07-12_dummy_cs_NEW2_Feedback_CBL",
    # "dummy_cs_JA_Feedback_CBL_new"

    # "08-12_dummy_cs_emulation_no_distr_rec_5_CB_exploration-bonus-type_cell_exploration-bonus-params__1_50",
    # "08-12_dummy_cs_emulation_no_distr_rec_5_CB",
    
    # "dummy_cs_RR_ft_NEW_single_CB_marble_pass_B_exp_soc",
    # "dummy_cs_RR_ft_NEW_single_CB_marble_pass_B_contr_asoc",

    # "dummy_cs_RR_ft_NEW_group_CB_marble_pass_A_exp_soc",
    # "dummy_cs_RR_ft_NEW_group_CB_marble_pass_A_contr_asoc"

    # "03-12_dummy_cs_jz_formats_A",
    # "03-12_dummy_cs_jz_formats_E",
    # "03-12_dummy_cs_jz_formats_AE",
    # "dummy_cs_jz_formats_N_rec_5"

    # "03-12_dummy_cs_formats_CBL_A",
    # "03-12_dummy_cs_formats_CBL_E",
    # "03-12_dummy_cs_formats_CBL_AE",
    # "dummy_cs_formats_CBL_N_rec_5"

    # "03-12_dummy_cs_jz_formats_AE",
    # "dummy_cs_jz_scaf_A_E_N_A_E_full-AEfull",
    # "dummy_cs_jz_scaf_A_E_N_A_E_scaf_full-AEfull",
# ]

# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.replace("07-12_dummy_cs_NEW2_Pointing_sm_CB_very_small", "PPO_CB")
#     label = label.replace("dummy_cs_JA_Pointing_CB_sm", "JA_PPO_CB")
#
#     label = label.replace("06-12_dummy_cs_NEW_Color_CBL", "PPO_CBL")
#     label = label.replace("dummy_cs_JA_Color_CBL_new", "JA_PPO_CBL")
#
#     label = label.replace("07-12_dummy_cs_NEW2_Feedback_CBL", "PPO_CBL")
#     label = label.replace("dummy_cs_JA_Feedback_CBL_new", "JA_PPO_CBL")
#
#     label = label.replace(
#         "08-12_dummy_cs_emulation_no_distr_rec_5_CB_exploration-bonus-type_cell_exploration-bonus-params__1_50",
#         "PPO_CB_1")
#     label = label.replace(
#         "08-12_dummy_cs_emulation_no_distr_rec_5_CB_exploration-bonus-type_cell_exploration-bonus-params__1_50",
#         "PPO_CB_1")
#
#     label = label.replace("dummy_cs_RR_ft_NEW_single_CB_marble_pass_B_exp_soc", "PPO_CB_role_B_single")
#     label = label.replace("dummy_cs_RR_ft_NEW_single_CB_marble_pass_B_contr_asoc", "PPO_CB_asoc_single")
#
#     label = label.replace("dummy_cs_RR_ft_NEW_group_CB_marble_pass_A_exp_soc", "PPO_CB_role_B_group")
#     label = label.replace("dummy_cs_RR_ft_NEW_group_CB_marble_pass_A_contr_asoc", "PPO_CB_asoc_group")
#
#     label = label.replace(
#         "03-12_dummy_cs_formats_CBL_A_rec_5_env_SocialAI-ALangFeedbackTrainFormatsCSParamEnv-v1_recurrence_5_test-set-name_AFormatsTestSet_exploration-bonus-type_lang",
#         "PPO_CBL_Ask")
#     label = label.replace(
#         "03-12_dummy_cs_formats_CBL_E_rec_5_env_SocialAI-ELangFeedbackTrainFormatsCSParamEnv-v1_recurrence_5_test-set-name_EFormatsTestSet_exploration-bonus-type_lang",
#         "PPO_CBL_Eye_contact")
#     label = label.replace(
#         "03-12_dummy_cs_formats_CBL_AE_rec_5_env_SocialAI-AELangFeedbackTrainFormatsCSParamEnv-v1_recurrence_5_test-set-name_AEFormatsTestSet_exploration-bonus-type_lang",
#         "PPO_CBL_Ask_Eye_contact")
#     label = label.replace("dummy_cs_formats_CBL_N_rec_5", "PPO_CBL_No")
#
#     label = label.replace(
#         "03-12_dummy_cs_jz_formats_E_rec_5_env_SocialAI-ELangFeedbackTrainFormatsCSParamEnv-v1_recurrence_5_test-set-name_EFormatsTestSet",
#         "PPO_no_bonus_Eye_contact")
#     label = label.replace(
#         "03-12_dummy_cs_jz_formats_A_rec_5_env_SocialAI-ALangFeedbackTrainFormatsCSParamEnv-v1_recurrence_5_test-set-name_AFormatsTestSet",
#         "PPO_no_bonus_Ask")
#     label = label.replace(
#         "03-12_dummy_cs_jz_formats_AE_rec_5_env_SocialAI-AELangFeedbackTrainFormatsCSParamEnv-v1_recurrence_5_test-set-name_AEFormatsTestSet",
#         "PPO_no_bonus_Ask_Eye_contact")
#     label = label.replace("dummy_cs_jz_formats_N_rec_5", "PPO_no_bonus_No")
#
#     label = label.replace("03-12_dummy_cs_jz_formats_AE", "PPO_no_bonus_no_scaf")
#     label = label.replace("dummy_cs_jz_scaf_A_E_N_A_E_full-AEfull", "PPO_no_bonus_scaf_4")
#     label = label.replace("dummy_cs_jz_scaf_A_E_N_A_E_scaf_full-AEfull", "PPO_no_bonus_scaf_8")
#
#     return label


# Final case studies
require_patterns = [
    "_",
    # pointing
    # "04-01_Pointing_CB_heldout_doors",

    # # role reversal
    # "03-01_RR_ft_single_CB_marble_pass_A_asoc_contr",
    # "03-01_RR_ft_single_CB_marble_pass_A_soc_exp",

    # "05-01_RR_ft_group_50M_CB_marble_pass_A_asoc_contr",
    # "05-01_RR_ft_group_50M_CB_marble_pass_A_soc_exp",

    # scaffolding
    # "05-01_scaffolding_50M_no",
    # "05-01_scaffolding_50M_acl_4_acl-type_intro_seq",
    # "05-01_scaffolding_50M_acl_8_acl-type_intro_seq_scaf",
]

def label_parser(label, figure_id, label_parser_dict=None):
    label = label.replace("04-01_Pointing_CB_heldout_doors", "PPO_CB")

    label = label.replace("05-01_scaffolding_50M_no_acl", "PPO_no_scaf")
    label = label.replace("05-01_scaffolding_50M_acl_4_acl-type_intro_seq", "PPO_scaf_4")
    label = label.replace("05-01_scaffolding_50M_acl_8_acl-type_intro_seq_scaf", "PPO_scaf_8")

    label = label.replace("03-01_RR_ft_single_CB_marble_pass_A_soc_exp", "PPO_CB_role_B")
    label = label.replace("03-01_RR_ft_single_CB_marble_pass_A_asoc_contr", "PPO_CB_asocial")

    label = label.replace("05-01_RR_ft_group_50M_CB_marble_pass_A_soc_exp", "PPO_CB_role_B")
    label = label.replace("05-01_RR_ft_group_50M_CB_marble_pass_A_asoc_contr", "PPO_CB_asocial")

    return label


color_dict = {

    # JA
    # "JA_PPO_CBL": "blue",
    # "PPO_CBL": "orange",

    # RR group
    # "PPO_CB_role_B_group": "orange",
    # "PPO_CB_asoc_group": "blue"

    # formats No
    # "PPO_no_bonus_No": "blue",
    # "PPO_no_bonus_Eye_contact": "magenta",
    # "PPO_no_bonus_Ask": "orange",
    # "PPO_no_bonus_Ask_Eye_contact": "green"

    # formats CBL
    # "PPO_CBL_No": "blue",
    # "PPO_CBL_Eye_contact": "magenta",
    # "PPO_CBL_Ask": "orange",
    # "PPO_CBL_Ask_Eye_contact": "green"
}

# # POINTING_GENERALIZATION (DUMMY)
# require_patterns = [
#     "29-10_SAI_Pointing_CS_PPO_CB_",
#     "29-10_SAI_LangColor_CS_PPO_CB_"
# ]
#
# color_dict = {
#     "dummy_cs_JA_Feedback_CBL_new": "blue",
#     "dummy_cs_Feedback_CBL": "orange",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("Pointing_CS_PPO_CB", "PPO_CB_train(DUMMY)")
#     label=label.replace("LangColor_CS_PPO_CB", "PPO_CB_test(DUMMY)")
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/Pointing_gen_eval.png"

# # FEEDBACK GENERALIZATION (DUMMY)
# require_patterns = [
#     "29-10_SAI_LangFeedback_CS_PPO_CBL_",
#     "29-10_SAI_LangColor_CS_PPO_CB_"
# ]
#
# color_dict = {
#     "PPO_CBL_train(DUMMY)": "blue",
#     "PPO_CBL_test(DUMMY)": "maroon",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("LangFeedback_CS_PPO_CBL", "PPO_CBL_train(DUMMY)")
#     label=label.replace("LangColor_CS_PPO_CB", "PPO_CBL_test(DUMMY)")
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/Feedback_gen_eval.png"

# # COLOR GENERALIZATION (DUMMY)
# require_patterns = [
#     "29-10_SAI_LangColor_CS_PPO_CBL_",
#     "29-10_SAI_LangColor_CS_PPO_CB_"
# ]
#
# color_dict = {
#     "PPO_CBL_train(DUMMY)": "blue",
#     "PPO_CBL_test(DUMMY)": "maroon",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("LangColor_CS_PPO_CBL", "PPO_CBL_train(DUMMY)")
#     label=label.replace("LangColor_CS_PPO_CB", "PPO_CBL_test(DUMMY)")
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/Color_gen_eval.png"

# # POINTING - PILOT
# require_patterns = [
#     "29-10_SAI_Pointing_CS_PPO_",
# ]
#
# color_dict = {
#     "PPO_RIDE": "orange",
#     "PPO_RND": "magenta",
#     "PPO_no": "maroon",
#     "PPO_CBL": "green",
#     "PPO_CB": "blue",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("Pointing_CS_", "")
#     return label
# #
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/Pointing_eval.png"


# LANGCOLOR - 7 Colors - PILOT
# require_patterns = [
#     "29-10_SAI_LangColor_CS_PPO_",
# ]
#
# color_dict = {
#     "PPO_RIDE": "orange",
#     "PPO_RND": "magenta",
#     "PPO_no": "maroon",
#     "PPO_CBL": "green",
#     "PPO_CB": "blue",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("LangColor_CS_", "")
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/Color_eval.png"

# # LangColor - CBL - 3 5 7
# require_patterns = [
#     "02-11_SAI_LangColor_CS_5C_PPO_CBL",
#     "02-11_SAI_LangColor_CS_3C_PPO_CBL",
#     "29-10_SAI_LangColor_CS_PPO_CBL"
# ]

# RND RIDE reference : RIDE > RND > no
# require_patterns = [
#     "24-08_new_ref",
# ]


# # # LANG FEEDBACK
# require_patterns = [
#     "24-10_SAI_LangFeedback_CS_PPO_",
#     "29-10_SAI_LangFeedback_CS_PPO_",
# ]
# color_dict = {
#     "PPO_RIDE": "orange",
#     "PPO_RND": "magenta",
#     "PPO_no": "maroon",
#     "PPO_CBL": "green",
#     "PPO_CB": "blue",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("LangFeedback_CS_", "")
#     return label
#
# # eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/Feedback_eval.png"
#

# # ROLE REVERSAL - group (DUMMY)
# require_patterns = [
#     "24-10_SAI_LangFeedback_CS_PPO_CB_",
#     "29-10_SAI_LangFeedback_CS_PPO_CBL_",
# ]
# color_dict = {
#     "PPO_CB_experimental": "green",
#     "PPO_CB_control": "blue",
# }
# color_dict=None
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("LangFeedback_CS_", "")
#
#     label=label.replace("PPO_CB", "PPO_CB_control")
#     label=label.replace("controlL", "experimental")
#
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/RR_dummy_group.png"

# # ROLE REVERSAL - single (DUMMY)
# require_patterns = [
#     "24-10_SAI_LangFeedback_CS_PPO_CB_",
#     "24-10_SAI_LangFeedback_CS_PPO_no_",
# ]
# color_dict = {
#     "PPO_CB_experimental": "green",
#     "PPO_CB_control": "blue",
# }
# color_dict=None
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("LangFeedback_CS_", "")
#
#     label=label.replace("PPO_CB", "PPO_CB_control")
#     label=label.replace("PPO_no", "PPO_CB_experimental")
#
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/RR_dummy_single.png"

# # IMITATION train (DUMMY)
# require_patterns = [
#     "29-10_SAI_LangFeedback_CS_PPO_CBL_",
#     "29-10_SAI_Pointing_CS_PPO_RIDE",
# ]
#
# color_dict = {
#     "PPO_CB_no_distr(DUMMY)": "magenta",
#     "PPO_CB_distr(DUMMY)": "orange",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("LangFeedback_CS_PPO_CBL", "PPO_CB_no_distr(DUMMY)")
#     label=label.replace("Pointing_CS_PPO_RIDE", "PPO_CB_distr(DUMMY)")
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/Imitation_train.png"

# # IMITATION test (DUMMY)
# require_patterns = [
#     "29-10_SAI_LangFeedback_CS_PPO_CBL_",
#     "29-10_SAI_Pointing_CS_PPO_RIDE",
# ]
#
# color_dict = {
#     "PPO_CB_no_distr(DUMMY)": "magenta",
#     "PPO_CB_distr(DUMMY)": "orange",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("LangFeedback_CS_PPO_CBL", "PPO_CB_no_distr(DUMMY)")
#     label=label.replace("Pointing_CS_PPO_RIDE", "PPO_CB_distr(DUMMY)")
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/Imitation_test.png"


# JA_POINTING
# require_patterns = [
#     "29-10_SAI_Pointing_CS_PPO_CB_",
#     "04-11_SAI_JA_Pointing_CS_PPO_CB_less",  # less reward
# ]
# color_dict = {
#     "JA_Pointing_PPO_CB": "orange",
#     "Pointing_PPO_CB": "blue",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("_CS_", "_")
#     label=label.replace("_less_", "")
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/JA_Pointing_eval.png"


# # JA_COLORS (JA, no)  x (3,5,7)
# max_x_lim = 17
# require_patterns = [
#     # "02-11_SAI_JA_LangColor", # max_x_lim = 17
#     "02-11_SAI_JA_LangColor_CS_3C", # max_x_lim = 17
#     # "02-11_SAI_LangColor_CS_5C_PPO_CBL", # max_x_lim = 17
#     "02-11_SAI_LangColor_CS_3C_PPO_CBL",
#     # "29-10_SAI_LangColor_CS_PPO_CBL"
# ]
# color_dict = {
#     "JA_LangColor_PPO_CBL": "orange",
#     "LangColor_PPO_CBL": "blue",
# }

# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("_CS_", "_")
#     label=label.replace("_3C_", "_")
#     return label

# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/JA_Color_eval.png"


# JA_FEEDBACK -> max_xlim=17
# max_x_lim = 17
# require_patterns = [
#     "02-11_SAI_JA_LangFeedback_CS_PPO_CBL_",
#     "29-10_SAI_LangFeedback_CS_PPO_CBL_",
#     "dummy_cs_F",
#     "dummy_cs_JA_F"
# ]
# color_dict = {
#     "JA_LangFeedback_PPO_CBL": "orange",
#     "LangFeedback_PPO_CBL": "blue",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("_CS_", "_")
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/JA_Feedback_eval.png"

# # Formats CBL
# require_patterns = [
#     "03-11_SAI_LangFeedback_CS_F_NO_PPO_CBL_env_SocialAI",
#     "29-10_SAI_LangFeedback_CS_PPO_CBL_env_SocialAI",
#     "03-11_SAI_LangFeedback_CS_F_ASK_PPO_CBL_env_SocialAI",
#     "03-11_SAI_LangFeedback_CS_F_ASK_EYE_PPO_CBL_env_SocialAI",
# ]
# color_dict = {
#     "LangFeedback_Eye_PPO_CBL": "blue",
#     "LangFeedback_Ask_PPO_CBL": "orange",
#     "LangFeedback_NO_PPO_CBL": "green",
#     "LangFeedback_AskEye_PPO_CBL": "magenta",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("_CS_", "_")
#     label=label.replace("_F_", "_")
#
#     label=label.replace("LangFeedback_PPO", "LangFeedback_EYE_PPO")
#
#     label=label.replace("EYE", "Eye")
#     label=label.replace("No", "No")
#     label=label.replace("ASK", "Ask")
#     label=label.replace("Ask_Eye", "AskEye")
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/Formats_CBL_eval.png"

# # Formats NO
# require_patterns = [
#    "24-10_SAI_LangFeedback_CS_PPO_no", # EYE
#    "04-11_SAI_LangFeedback_CS_F_NO_PPO_NO_env_SocialAI",
#    "04-11_SAI_LangFeedback_CS_F_ASK_PPO_NO_env_SocialAI",
#    "04-11_SAI_LangFeedback_CS_F_ASK_EYE_PPO_NO_env_SocialAI",
# ]
#
# color_dict = {
#     "LangFeedback_Eye_PPO_no": "blue",
#     "LangFeedback_Ask_PPO_no": "orange",
#     "LangFeedback_NO_PPO_no": "green",
#     "LangFeedback_AskEye_PPO_no": "magenta",
# }
#
# def label_parser(label, figure_id, label_parser_dict=None):
#     label = label.split("_env_")[0].split("SAI_")[1]
#     label=label.replace("_CS_", "_")
#     label=label.replace("_F_", "_")
#     #
#     label=label.replace("LangFeedback_PPO", "LangFeedback_EYE_PPO")
#     label=label.replace("PPO_NO", "PPO_no")
#
#     label=label.replace("EYE", "Eye")
#     label=label.replace("No", "No")
#     label=label.replace("ASK", "Ask")
#     label=label.replace("Ask_Eye", "AskEye")
#     return label
#
# eval_filename = f"/home/flowers/Documents/projects/embodied_acting_and_speaking/case_studies_figures/Formats_no_eval.png"


#
# require_patterns = [
#     "11-07_bAI_cb_GS_param_tanh_env_SocialAI-SocialAIParamEnv-v1_exploration-bonus-type_cell_exploration-bonus-params__2_50_exploration-bonus-tanh_0.6",
#     # "04-11_SAI_ImitationDistr_CS_PPO_CB_small_env_SocialAI-EEmulationDistrInformationSeekingParamEnv-v1_recurrence_10",
#     # "04-11_SAI_ImitationDistr_CS_PPO_CB_small_env_SocialAI-EEmulationDistrInformationSeekingParamEnv-v1_recurrence_10",
#     "03-11_SAI_ImitationDistr_CS_PPO_CB_env_SocialAI-EEmulationDistrInformationSeekingParamEnv-v1_recurrence_10",
#     # "04-11_SAI_ImitationNoDistr_CS_PPO_CB_small_env_SocialAI-EEmulationNoDistrInformationSeekingParamEnv-v1_recurrence_10",
# ]

# require_patterns = [
#    "02-11_SAI_LangColor_CS_3C_PPO_CBL",
#     "02-11_SAI_JA_LangColor_CS_3C_PPO_CBL",
# ]  # at least one of those


# all of those
include_patterns = [
    "_"
]
#include_patterns = ["rec_5"]

if eval_filename:
    # saving
    fontsize = 40
    legend_fontsize = 30
    linewidth = 10
else:
    fontsize = 5
    legend_fontsize = 5
    linewidth = 1

fontsize = 5
legend_fontsize = 5
linewidth = 1

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

to_remove = []

for tr_ in to_remove:
    if tr_ in models_saves:
        del models_saves[tr_]

print("Loaded:")
print("\n".join(list(models_saves.keys())))

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

def plot_with_shade_seed(subplot_nb, ax, x, y, err, color, shade_color, label,
                         y_min=None, y_max=None, legend=False, leg_size=30, leg_loc='best', title=None,
                         ylim=[0,100], xlim=[0,40], leg_args={}, leg_linewidth=13.0, linewidth=10.0, labelsize=20,
                         filename=None,
                         zorder=None, xlabel='perf', ylabel='Env steps'):

    plt.rcParams.update({'font.size': 15})

    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0

    ax.locator_params(axis='x', nbins=3)
    ax.locator_params(axis='y', nbins=3)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    x = x[:len(y)]

    # ax.scatter(x, y, color=color, linewidth=linewidth, zorder=zorder)
    ax.plot(x, y, color=color, label=label, linewidth=linewidth, zorder=zorder)

    if err is not None:
        ax.fill_between(x, y-err, y+err, color=shade_color, alpha=0.2)

    if legend:
        leg = ax.legend(loc=leg_loc, **leg_args) #34
        for legobj in leg.legendHandles:
            legobj.set_linewidth(leg_linewidth)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    if subplot_nb == 0:
        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=4)

    ax.set_xlim(xmin=xlim[0],xmax=xlim[1])
    ax.set_ylim(bottom=ylim[0],top=ylim[1])
    if title:
        ax.set_title(title, fontsize=fontsize)

    # if filename is not None:
    #     f.savefig(filename)


# Plot utils
def plot_with_shade_grg(subplot_nb, ax, x, y, err, color, shade_color, label,
                        legend=False, leg_loc='best', title=None,
                        ylim=[0, 100], xlim=[0, 40], leg_args={}, leg_linewidth=13.0, linewidth=10.0, labelsize=20, fontsize=20, title_fontsize=30,
                        zorder=None, xlabel='Perf', ylabel='Env steps', linestyle="-", xnbins=3, ynbins=3, filename=None):

    #plt.rcParams.update({'font.size': 15})
    ax.locator_params(axis='x', nbins=xnbins)
    ax.locator_params(axis='y', nbins=ynbins)

    ax.tick_params(axis='y', which='both', labelsize=labelsize)
    ax.tick_params(axis='x', which='both', labelsize=labelsize*0.8)
    # ax.tick_params(axis='both', which='both', labelsize="small")

    # ax.scatter(x, y, color=color,linewidth=linewidth,zorder=zorder, linestyle=linestyle)
    ax.plot(x, y, color=color, label=label, linewidth=linewidth, zorder=zorder, linestyle=linestyle)

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
        ax.set_title(title, fontsize=title_fontsize)

    # if filename is not None:
    #     f.savefig(filename)


# Metric plot
# metric = 'success_rate_mean'
# metric = 'mission_string_observed_mean'
# metric = 'extrinsic_return_mean'
# metric = 'extrinsic_return_max'
# metric = "rreturn_mean"
# metric = 'rreturn_max'
# metric = 'FPS'
# metric = 'duration'
# metric = 'intrinsic_reward_perf2_'
# metric = 'NPC_intro'


metrics = [
    'success_rate_mean',
    # 'FPS',
    # 'extrinsic_return_mean',
    # 'exploration_bonus_mean',
    'NPC_intro',
    # 'curriculum_param_mean',
    # 'curriculum_max_success_rate_mean',
    # 'rreturn_mean'
]

# f, ax = plt.subplots(1, len(metrics), figsize=(15.0, 9.0))
f, ax = plt.subplots(1, len(metrics), figsize=(9.0, 9.0))
# f, ax = plt.subplots(1, len(metrics), figsize=(20.0, 20.0))
# f, ax = plt.subplots(1, 1, figsize=(5.0, 3.0))

if len(metrics) == 1:
    ax = [ax]

max_y = -np.inf
min_y = np.inf
# hardcoded
min_y, max_y = 0.0, 1.0
max_steps = 0
exclude_patterns = []


# def label_parser(label, figure_id, label_parser_dict=None):
#
#     label = label.split("_env_")[0].split("SAI_")[1]
#
#     # # Pointing
#     # label=label.replace("Pointing_CS_", "")
#
#     # Feedback
#     label=label.replace("LangFeedback_CS_", "")
#
#
#     # label=label.replace("CS_PPO", "7COL_PPO")
#     # label=label.replace("CS_3C_PPO", "3COL_PPO")
#     # label=label.replace("CS_5C_PPO", "5COL_PPO")
#
#     # label=label.replace("CS_PPO", "Eye_contact_PPO")
#     # label=label.replace("CS_F_ASK_PPO", "Ask_PPO")
#     # label=label.replace("CS_F_NO_PPO", "NO_PPO")
#     # label=label.replace("CS_F_ASK_EYE_PPO", "Ask_Eye_contact_PPO")
#     #
#     # label=label.replace("PPO_no", "PPO_no_bonus")
#     # label=label.replace("PPO_NO", "PPO_no_bonus")
#
#     if label_parser_dict:
#         if sum([1 for k, v in label_parser_dict.items() if k in label]) != 1:
#             if label in label_parser_dict:
#                 # see if there is an exact match
#                 return label_parser_dict[label]
#             else:
#                 print("ERROR multiple curves match a lable and there is no exact match for {}".format(label))
#                 exit()
#
#         for k, v in label_parser_dict.items():
#             if k in label: return v
#
#     else:
#         # return label.split("_env_")[1]
#         if figure_id not in [1, 2, 3, 4]:
#             return label
#         else:
#             # default
#             pass
#
#     return label


for metric_i, metric in enumerate(metrics):
    min_y, max_y = 0.0, 1.0
    default_colors = default_colors_.copy()
    for model_i, m_id in enumerate(models_saves.keys()):

        #excluding some experiments
        if any([ex_pat in m_id for ex_pat in exclude_patterns]):
            continue
        if len(include_patterns) > 0:
            if not any([in_pat in m_id for in_pat in include_patterns]):
                continue
        runs_data = models_saves[m_id]['data']
        ys = []

        if runs_data[0]['frames'][1] == 'frames':
            runs_data[0]['frames'] = list(filter(('frames').__ne__, runs_data[0]['frames']))
        ###########################################

        if per_seed:
            min_len = None

        else:
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
                # succes_rate_mean <==> bin_extrinsic_return_mean
                if metric == 'success_rate_mean':
                    metric_ = "bin_extrinsic_return_mean"
                    if metric_ not in run:
                        raise ValueError("Neither {} or {} is present: {} Possible metrics: {}. ".format(metric, metric_, list(run.keys())))

                    data = run[metric_]

                else:
                    raise ValueError("Unknown metric: {} Possible metrics: {}. ".format(metric, list(run.keys())))
            else:
                data = run[metric]

            if data[1] == metric:
                data = np.array(list(filter((metric).__ne__, data)), dtype=np.float16)
            ###########################################
            if per_seed:
                ys.append(data)
            else:
                if len(data) >= min_len:
                    if len(data) > min_len:
                        print("run has too many {} datapoints ({}). Discarding {}".format(m_id, len(data),
                                                                                          len(data)-min_len))
                        data = data[0:min_len]
                    ys.append(data)
                else:
                    raise ValueError("How can data be < min_len if it was capped above")

        ys_same_len = ys

        # computes stats
        n_seeds = len(ys_same_len)

        if per_seed:
            sems = np.array(ys_same_len)
            stds = np.array(ys_same_len)
            means = np.array(ys_same_len)
            color = default_colors[model_i]

        else:
            sems = np.std(ys_same_len, axis=0)/np.sqrt(len(ys_same_len))  # sem
            stds = np.std(ys_same_len, axis=0)  # std
            means = np.mean(ys_same_len, axis=0)
            color = default_colors[model_i]

        # per-metric adjustments
        ylabel = metric

        ylabel = {
           "success_rate_mean" : "Success rate",
            "exploration_bonus_mean": "Exploration bonus",
            "NPC_intro": "Successful introduction (%)",
        }.get(ylabel, ylabel)


        if metric == 'duration':
            ylabel = "time (hours)"
            means = means / 3600
            sems = sems / 3600
            stds = stds / 3600

        if per_seed:
            #plot x y bounds
            curr_max_y = np.max(np.max(means))
            curr_min_y = np.min(np.min(means))
            curr_max_steps = np.max(np.max(steps))

        else:
            # plot x y bounds
            curr_max_y = np.max(means+stds)
            curr_min_y = np.min(means-stds)
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
                seed_c = default_colors[model_i+s_i]
                # label = m_id#+"(s:{})".format(s_i)
                label = str(s_i)
                seed_ys = smooth(seed_ys, smooth_factor)
                plot_with_shade_seed(0, ax[metric_i], steps, seed_ys, None, seed_c, seed_c, label,
                                     legend=draw_legend, xlim=[0, max_steps], ylim=[min_y, max_y],
                                     leg_size=leg_size, xlabel=f"Env steps (1e6)", ylabel=ylabel, linewidth=linewidth,
                                     labelsize=fontsize,
                                     # fontsize=fontsize,
                                     )

                summary_dict[s_i] = seed_ys[-1]
                summary_dict_colors[s_i] = seed_c
        else:
            label = label_parser(m_id, load_pattern, label_parser_dict=label_parser_dict)

            if color_dict:
                color = color_dict[label]
            else:
                color = default_colors[model_i]

            label = label+"({})".format(n_seeds)


            if smooth_factor:
                means = smooth(means, smooth_factor)
                stds = smooth(stds, smooth_factor)

            x_lim = max(steps[-1], x_lim)
            x_lim = min(max_x_lim, x_lim)

            leg_args = {
                'fontsize': legend_fontsize
            }

            plot_with_shade_grg(
                0, ax[metric_i], steps, means, stds, color, color, label,
                legend=draw_legend and metric_i == 0,
                xlim=[0, x_lim],
                ylim=[0, max_y],
                xlabel=f"Env steps (1e6)",
                ylabel=ylabel,
                title=None,
                labelsize=fontsize*train_inc_font,
                fontsize=fontsize*train_inc_font,
                title_fontsize=title_fontsize,
                linewidth=linewidth,
                leg_linewidth=5,
                leg_args=leg_args,
                xnbins=xnbins,
                ynbins=ynbins,
            )
            summary_dict[label] = means[-1]
            summary_dict_colors[label] = color

    if len(summary_dict) == 0:
        raise ValueError(f"No experiments found for {load_pattern}.")

    # print summary
    best = max(summary_dict.values())

    pc = 0.3
    n = int(len(summary_dict)*pc)
    print("top n: ", n)

    top_pc = sorted(summary_dict.values())[-n:]
    bottom_pc = sorted(summary_dict.values())[:n]

    print("legend:")
    cprint("\tbest", "green")
    cprint("\ttop {} %".format(pc), "blue")
    cprint("\tbottom {} %".format(pc), "red")
    print("\tothers")
    print()


    for l, p in sorted(summary_dict.items(), key=lambda kv: kv[1]):

        c = summary_dict_colors[l]
        if p == best:
            cprint("label: {} ({})".format(l, c), "green")
            cprint("\t {}:{}".format(metric, p), "green")

        elif p in top_pc:
            cprint("label: {} ({})".format(l, c), "blue")
            cprint("\t {}:{}".format(metric, p), "blue")

        elif p in bottom_pc:
            cprint("label: {} ({})".format(l, c), "red")
            cprint("\t {}:{}".format(metric, p), "red")

        else:
            print("label: {} ({})".format(l, c))
            print("\t {}:{}".format(metric, p))

    for label, (mean, std, color) in static_lines.items():
        plot_with_shade_grg(
            0, ax[metric_i], steps, np.array([mean]*len(steps)), np.array([std]*len(steps)), color, color, label,
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

# plt.tight_layout()
# f.savefig('graphics/{}_{}_results.svg'.format(str(figure_id, metric)))
# f.savefig('graphics/{}_{}_results.png'.format(str(figure_id, metric)))
cprint("Ignore pattern: {}".format(ignore_patterns), "blue")
if plot_train:
    plt.tight_layout()
    # plt.subplots_adjust(hspace=1.5, wspace=0.5, left=0.1, right=0.9, bottom=0.1, top=0.85)
    plt.subplots_adjust(hspace=1.5, wspace=0.5, left=0.1, right=0.9, bottom=0.1, top=0.85)
    plt.suptitle(super_title)
    plt.show()
plt.close()

curr_max_y = 0
x_lim = 0

max_y = -np.inf
min_y = np.inf
# hardcoded
min_y, max_y = 0.0, 1.0

grid = True
draw_eval_legend = True

if study_eval:
    print("Evaluation")
    # evaluation sets
    number_of_eval_envs = max(list([len(v.keys()) for v in model_eval_data.values()]))

    if plot_aggregated_test:
        number_of_eval_envs += 1

    if number_of_eval_envs == 0:
        print("No eval envs")
        exit()

    if plot_only_aggregated_test:
        f, ax = plt.subplots(1, 1, figsize=(9.0, 9.0))

    else:
        if grid:
            # grid
            subplot_y = math.ceil(math.sqrt(number_of_eval_envs))
            subplot_x = math.ceil(number_of_eval_envs / subplot_y)
            # from IPython import embed; embed()

            while subplot_x % 1 != 0:
                subplot_y -= 1
                subplot_x = number_of_eval_envs / subplot_y

            if subplot_x == 1:
                subplot_y = math.ceil(math.sqrt(number_of_eval_envs))
                subplot_x = math.floor(math.sqrt(number_of_eval_envs))

            subplot_y = int(subplot_y)
            subplot_x = int(subplot_x)

            assert subplot_y * subplot_x >= number_of_eval_envs

            f, ax_ = plt.subplots(subplot_y, subplot_x, figsize=(6.0, 6.0), sharey=False)  #, sharex=True, sharey=True)

            if subplot_y != 1:
                ax = list(chain.from_iterable(ax_))
            else:
                ax=ax_

        else:
            # flat
            f, ax = plt.subplots(1, number_of_eval_envs, figsize=(15.0, 9.0)) #), sharey=True, sharex=True)

    if number_of_eval_envs == 1:
        ax = [ax]

    default_colors = default_colors_.copy()

    test_summary_dict = defaultdict(dict)
    test_summary_dict_colors = defaultdict(dict)

    for model_i, m_id in enumerate(model_eval_data.keys()):
        # excluding some experiments
        if any([ex_pat in m_id for ex_pat in exclude_patterns]):
            continue
        if len(include_patterns) > 0:
            if not any([in_pat in m_id for in_pat in include_patterns]):
                continue

        # computes stats
        if sort_test:
            test_envs_sorted = enumerate(sorted(model_eval_data[m_id].items(), key=lambda kv: sort_test_set(kv[0])))
        else:
            test_envs_sorted = enumerate(model_eval_data[m_id].items())

        if plot_aggregated_test:
            agg_means = []

        for env_i, (test_env, env_data) in  test_envs_sorted:
            ys_same_len = env_data["values"]
            steps = env_data["steps"].mean(0) / steps_denom
            n_seeds = len(ys_same_len)

            if per_seed:
                sems = np.array(ys_same_len)
                stds = np.array(ys_same_len)
                means = np.array(ys_same_len)
                color = default_colors[model_i]

            else:
                sems = np.std(ys_same_len, axis=0) / np.sqrt(len(ys_same_len))  # sem
                stds = np.std(ys_same_len, axis=0)  # std
                means = np.mean(ys_same_len, axis=0)
                color = default_colors[model_i]

            # per-metric adjusments

            if per_seed:
                # plot x y bounds
                curr_max_y = np.max(np.max(means))
                curr_min_y = np.min(np.min(means))
                curr_max_steps = np.max(np.max(steps))

            else:
                # plot x y bounds
                curr_max_y = np.max(means + stds)
                curr_min_y = np.min(means - stds)
                curr_max_steps = np.max(steps)

            if plot_aggregated_test:
                agg_means.append(means)

            if curr_max_y > max_y:
                max_y = curr_max_y
            if curr_min_y < min_y:
                min_y = curr_min_y

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

            if grid:
                ylabel = eval_metric_name
                title = test_env_name

            else:
                # flat
                ylabel = test_env_name
                title = eval_metric_name

            leg_args = {
                'fontsize': legend_fontsize // 1
            }

            if per_seed:
                for s_i, seed_ys in enumerate(ys_same_len):
                    seed_c = default_colors[model_i + s_i]
                    # label = m_id#+"(s:{})".format(s_i)
                    label = str(s_i)

                    if not plot_only_aggregated_test:
                        seed_ys = smooth(seed_ys, eval_smooth_factor)
                        plot_with_shade_seed(0, ax[env_i], steps, seed_ys, None, seed_c, seed_c, label,
                                             legend=draw_eval_legend, xlim=[0, x_lim], ylim=[min_y, max_y],
                                             leg_size=leg_size, xlabel=f"Steps (1e6)", ylabel=ylabel, linewidth=linewidth, title=title)

                    test_summary_dict[s_i][test_env] = seed_ys[-1]
                    test_summary_dict_colors[s_i] = seed_c
            else:
                label = label_parser(m_id, load_pattern, label_parser_dict=label_parser_dict)

                if not plot_only_aggregated_test:

                    if color_dict:
                        color = color_dict[label]
                    else:
                        color = default_colors[model_i]

                    label = label + "({})".format(n_seeds)

                    if smooth_factor:
                        means = smooth(means, eval_smooth_factor)
                        stds = smooth(stds, eval_smooth_factor)

                    plot_with_shade_grg(
                        0, ax[env_i], steps, means, stds, color, color, label,
                        legend=draw_eval_legend,
                        xlim=[0, x_lim+1],
                        ylim=[0, max_y],
                        xlabel=f"Env steps (1e6)" if env_i // (subplot_x) == subplot_y -1 else None,  # only last line
                        ylabel=ylabel if env_i % subplot_x == 0 else None,  # only first row
                        title=title,
                        title_fontsize=title_fontsize,
                        labelsize=fontsize,
                        fontsize=fontsize,
                        linewidth=linewidth,
                        leg_linewidth=5,
                        leg_args=leg_args,
                        xnbins=xnbins,
                        ynbins=ynbins,
                    )

                test_summary_dict[label][test_env] = means[-1]
                test_summary_dict_colors[label] = color

        if plot_aggregated_test:
            if plot_only_aggregated_test:
                agg_env_i = 0
            else:
                agg_env_i = number_of_eval_envs - 1 # last one

            agg_means = np.array(agg_means)
            agg_mean = agg_means.mean(axis=0)
            agg_std = agg_means.std(axis=0)  # std

            if smooth_factor and not per_seed:
                agg_mean = smooth(agg_mean, eval_smooth_factor)
                agg_std = smooth(agg_std, eval_smooth_factor)

            if color_dict:
                color = color_dict[re.sub("\([0-9]\)", '', label)]
            else:
                color = default_colors[model_i]

            if per_seed:
                print("Not smooth aggregated because of per seed")
                for s_i, (seed_ys, seed_st) in enumerate(zip(agg_mean, agg_std)):
                    seed_c = default_colors[model_i + s_i]
                    # label = m_id#+"(s:{})".format(s_i)
                    label = str(s_i)
                    # seed_ys = smooth(seed_ys, eval_smooth_factor)
                    plot_with_shade_seed(0,
                                         ax if plot_only_aggregated_test else ax[agg_env_i],
                                         steps, seed_ys, seed_st, seed_c, seed_c, label,
                                         legend=draw_eval_legend, xlim=[0, x_lim], ylim=[min_y, max_y],
                                         labelsize=fontsize,
                                         filename=eval_filename,
                                         leg_size=leg_size, xlabel=f"Steps (1e6)", ylabel=ylabel, linewidth=1, title=agg_title)
            else:

                #   just used for creating a dummy Imitation test figure -> delete
                # agg_mean = agg_mean * 0.1
                # agg_std = agg_std * 0.1
                # max_y = 1

                plot_with_shade_grg(
                    0,
                    ax if plot_only_aggregated_test else ax[agg_env_i],
                    steps, agg_mean, agg_std, color, color, label,
                    legend=draw_eval_legend,
                    xlim=[0, x_lim + 1],
                    ylim=[0, max_y],
                    xlabel=f"Steps (1e6)" if plot_only_aggregated_test or (agg_env_i // (subplot_x) == subplot_y - 1) else None,  # only last line
                    ylabel=ylabel if plot_only_aggregated_test or (agg_env_i % subplot_x == 0) else None,  # only first row
                    title_fontsize=title_fontsize,
                    title=agg_title,
                    labelsize=fontsize,
                    fontsize=fontsize,
                    linewidth=linewidth,
                    leg_linewidth=5,
                    leg_args=leg_args,
                    xnbins=xnbins,
                    ynbins=ynbins,
                    filename=eval_filename,
                )

    # print summary

    means_dict = {
        lab: np.array(list(lab_sd.values())).mean() for lab, lab_sd in test_summary_dict.items()
    }
    best = max(means_dict.values())

    pc = 0.3
    n = int(len(means_dict) * pc)
    print("top n: ", n)

    top_pc = sorted(means_dict.values())[-n:]
    bottom_pc = sorted(means_dict.values())[:n]

    print("Legend:")
    cprint("\tbest", "green")
    cprint("\ttop {} %".format(pc), "blue")
    cprint("\tbottom {} %".format(pc), "red")
    print("\tothers")
    print()

    for l, l_mean in sorted(means_dict.items(), key=lambda kv: kv[1]):

        l_summary_dict = test_summary_dict[l]

        c = test_summary_dict_colors[l]
        print("label: {} ({})".format(l, c))

        #print("\t{}({}) - Mean".format(l_mean, metric))
        
        if l_mean == best:
            cprint("\t{}({}) - Mean".format(l_mean, eval_metric), "green")

        elif l_mean in top_pc:
            cprint("\t{}({}) - Mean".format(l_mean, eval_metric), "blue")

        elif l_mean in bottom_pc:
            cprint("\t{}({}) - Mean".format(l_mean, eval_metric), "red")

        else:
            print("\t{}({})".format(l_mean, eval_metric))

        n_over_50 = 0

        if sort_test:
            sorted_envs = sorted(l_summary_dict.items(), key=lambda kv: sort_test_set(env_name=kv[0]))
        else:
            sorted_envs = l_summary_dict.items()

        for tenv, p in sorted_envs:
            if p < 0.5:
                print("\t{:4f}({}) - \t{}".format(p, eval_metric, tenv))
            else:
                print("\t{:4f}({}) -*\t{}".format(p, eval_metric, tenv))
                n_over_50 += 1
        print("\tenv over 50 - {}/{}".format(n_over_50, len(l_summary_dict)))

    if plot_test:
        plt.tight_layout()
        # plt.subplots_adjust(hspace=0.8, wspace=0.15, left=0.035, right=0.99, bottom=0.065, top=0.93)
        plt.show()

    if eval_filename is not None:
        plt.subplots_adjust(hspace=0.8, wspace=0.15, left=0.15, right=0.99, bottom=0.15, top=0.93)

        res= input(f"Save to {eval_filename} (y/n)?")
        if res == "y":
            f.savefig(eval_filename)
            print(f'saved to {eval_filename}')
        else:
            print('not saved')

import os
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from scipy import stats

experiments = Path("./results_1000/")

results_dict = {}

def label_parser(label):
    label_parser_dict = {
        "VIGIL4_WizardGuide_lang64_no_explo": "ABL_MH-BabyAI",
        "VIGIL4_WizardTwoGuides_lang64_no_explo": "FULL_MH-BabyAI",

        "VIGIL4_WizardGuide_lang64_mm": "ABL_MH-BabyAI-ExpBonus",
        "VIGIL4_WizardTwoGuides_lang64_mm": "FULL_MH-BabyAI-ExpBonus",

        "VIGIL4_WizardGuide_lang64_deaf_no_explo": "ABL_Deaf-MH-BabyAI",
        "VIGIL4_WizardTwoGuides_lang64_deaf_no_explo": "FULL_Deaf-MH-BabyAI",

        "VIGIL4_WizardGuide_lang64_bow": "ABL_MH-BabyAI-BOW",
        "VIGIL4_WizardTwoGuides_lang64_bow": "FULL_MH-BabyAI-BOW",

        "VIGIL4_WizardGuide_lang64_no_mem": "ABL_MH-BabyAI-no-mem",
        "VIGIL4_WizardTwoGuides_lang64_no_mem": "FULL_MH-BabyAI-no-mem",

        "VIGIL5_WizardGuide_lang64_bigru": "ABL_MH-BabyAI-bigru",
        "VIGIL5_WizardTwoGuides_lang64_bigru": "FULL_MH-BabyAI-bigru",

        "VIGIL5_WizardGuide_lang64_attgru": "ABL_MH-BabyAI-attgru",
        "VIGIL5_WizardTwoGuides_lang64_attgru": "FULL_MH-BabyAI-attgru",

        "VIGIL4_WizardGuide_lang64_curr_dial": "ABL_MH-BabyAI-current-dialogue",
        "VIGIL4_WizardTwoGuides_lang64_curr_dial": "FULL_MH-BabyAI-current-dialogue",

        "random_WizardGuide": "ABL_Random-agent",
        "random_WizardTwoGuides": "FULL_Random-agent",
    }
    if sum([1 for k, v in label_parser_dict.items() if k in label]) != 1:
        print("ERROR")
        print(label)
        exit()

    for k, v in label_parser_dict.items():
        if k in label: return v

    return label

for experiment_out_file in experiments.iterdir():
    results_dict[label_parser(str(experiment_out_file))] = []
    with open(experiment_out_file) as f:
        for line in f:
            if "seed success rate" in line:
                seed_success_rate = float(re.search('[0-9]\.[0-9]*', line).group())
                results_dict[label_parser(str(experiment_out_file))].append(seed_success_rate)

assert set([len(v) for v in results_dict.values()]) == set([16])

test_p = 0.05
compare = {
    "ABL_MH-BabyAI-ExpBonus": "ABL_MH-BabyAI",
    "ABL_MH-BabyAI": "ABL_Deaf-MH-BabyAI",
    "ABL_Deaf-MH-BabyAI": "ABL_Random-agent",
    "FULL_MH-BabyAI-ExpBonus": "FULL_MH-BabyAI",
    "FULL_MH-BabyAI": "FULL_Deaf-MH-BabyAI",
    "FULL_Deaf-MH-BabyAI": "FULL_Random-agent",
}
for k, v in compare.items():
    p = stats.ttest_ind(
        results_dict[k],
        results_dict[v],
        equal_var=False
    ).pvalue
    if np.isnan(p):
        from IPython import embed; embed()
    print("{} (m:{}) <---> {} (m:{}) = p: {}  result: {}".format(
        k, np.mean(results_dict[k]), v, np.mean(results_dict[v]), p,
        "Distributions different(p={})".format(test_p) if p < test_p else "Distributions same(p={})".format(test_p)
    ))
    print()
# from IPython import embed; embed()
import sys
from pathlib import Path
from datetime import date
import subprocess
import shutil
import os
import stat
import getpass
import re
import glob


def process_arg_string(expe_args):  # function to extract flagged (with a *) arguments as details for experience name
    details_string = ''
    processed_arg_string = expe_args.replace('*', '')  # keep a version of args cleaned from exp name related flags
    # args = [arg_chunk.split(' -') for arg_chunk in expe_args.split(' --')]
    arg_chunks = [arg_chunk for arg_chunk in expe_args.split(' --')]
    args_list = []
    for arg in arg_chunks:
        if ' -' in arg and arg.split(' -')[1].isalpha():
            args_list.extend(arg.split(' -'))
        else:
            args_list.append(arg)
    # args_list = [item for sublist in args for item in sublist]  # flatten
    for arg in args_list:
        if arg == '':
            continue
        if arg[0] == '*':
            if arg[-1] == ' ':
                arg = arg[:-1]
            details_string += '_' + arg[1:].replace(' ', '_').replace('/', '-')
    return details_string, processed_arg_string


slurm_confs = {'curta_extra_long': "#SBATCH -p inria\n"
                                   "#SBATCH -t 119:00:00\n",
               'curta_long': "#SBATCH -p inria\n"
                             "#SBATCH -t 72:00:00\n",
               'curta_medium': "#SBATCH -p inria\n"
                               "#SBATCH -t 48:00:00\n",
               'curta_short': "#SBATCH -p inria\n"
                              "#SBATCH -t 24:00:00\n",
               'jz_super_short_gpu':
                                '#SBATCH -A imi@v100\n'
                                '#SBATCH --gres=gpu:1\n'
                                "#SBATCH -t 9:59:00\n"
                                "#SBATCH --qos=qos_gpu-t3\n",
               'jz_short_gpu': '#SBATCH -A imi@v100\n'
                               '#SBATCH --gres=gpu:1\n'
                               "#SBATCH -t 19:59:00\n"
                               "#SBATCH --qos=qos_gpu-t3\n",
               'jz_short_gpu_chained': '#SBATCH -A imi@v100\n'
                               '#SBATCH --gres=gpu:1\n'
                               "#SBATCH -t 19:59:00\n"
                               "#SBATCH --qos=qos_gpu-t3\n",
               'jz_short_2gpus_chained': '#SBATCH -A imi@v100\n'
                                       '#SBATCH --gres=gpu:2\n'
                                       "#SBATCH -t 19:59:00\n"
                                       "#SBATCH --qos=qos_gpu-t3\n",
               'jz_medium_gpu': '#SBATCH -A imi@v100\n' 
                                '#SBATCH --gres=gpu:1\n'
                                "#SBATCH -t 48:00:00\n"
                                "#SBATCH --qos=qos_gpu-t4\n",
               'jz_super_short_2gpus': '#SBATCH -A imi@v100\n'
                                 '#SBATCH --gres=gpu:2\n'
                                 "#SBATCH -t 14:59:00\n"
                                 "#SBATCH --qos=qos_gpu-t3\n",
               'jz_short_2gpus': '#SBATCH -A imi@v100\n'
                               '#SBATCH --gres=gpu:2\n'
                               "#SBATCH -t 19:59:00\n"
                               "#SBATCH --qos=qos_gpu-t3\n",
               'jz_short_2gpus_32g': '#SBATCH -A imi@v100\n'
                                 '#SBATCH -C v100-32g\n'
                                 '#SBATCH --gres=gpu:2\n'
                                 "#SBATCH -t 19:59:00\n"
                                 "#SBATCH --qos=qos_gpu-t3\n",
               'jz_medium_2gpus': '#SBATCH -A imi@v100\n'
                                '#SBATCH --gres=gpu:2\n'
                                "#SBATCH -t 48:00:00\n"
                                "#SBATCH --qos=qos_gpu-t4\n",
               'jz_medium_2gpus_32g': '#SBATCH -A imi@v100\n'
                                '#SBATCH -C v100-32g\n'
                                '#SBATCH --gres=gpu:2\n'
                                "#SBATCH -t 48:00:00\n"
                                "#SBATCH --qos=qos_gpu-t4\n",
               'jz_long_gpu': '#SBATCH -A imi@v100\n'
                              '#SBATCH --gres=gpu:1\n'
                              "#SBATCH -t 72:00:00\n"
                              "#SBATCH --qos=qos_gpu-t4\n",
               'jz_long_2gpus': '#SBATCH -A imi@v100\n'
                                '#SBATCH --gres=gpu:2\n'
                                '#SBATCH -t 72:00:00\n'
                                '#SBATCH --qos=qos_gpu-t4\n',
               'jz_long_2gpus_32g': '#SBATCH -A imi@v100\n'
                              '#SBATCH -C v100-32g\n'
                              '#SBATCH --gres=gpu:2\n'
                              "#SBATCH -t 72:00:00\n"
                              "#SBATCH --qos=qos_gpu-t4\n",
               'jz_super_long_2gpus_32g': '#SBATCH -A imi@v100\n'
                                    '#SBATCH -C v100-32g\n'
                                    '#SBATCH --gres=gpu:2\n'
                                    "#SBATCH -t 99:00:00\n"
                                    "#SBATCH --qos=qos_gpu-t4\n",
               'jz_short_cpu': '#SBATCH -A imi@cpu\n'
                                "#SBATCH -t 19:59:00\n"
                                "#SBATCH --qos=qos_cpu-t3\n",
               'jz_medium_cpu': '#SBATCH -A imi@cpu\n' 
                                "#SBATCH -t 48:00:00\n"
                                "#SBATCH --qos=qos_cpu-t4\n",
               'jz_long_cpu': '#SBATCH -A imi@cpu\n'
                               "#SBATCH -t 72:00:00\n"
                               "#SBATCH --qos=qos_cpu-t4\n",
               'plafrim_cpu_medium': "#SBATCH -t 48:00:00\n",
               'plafrim_cpu_long': "#SBATCH -t 72:00:00\n",
               'plafrim_gpu_medium': '#SBATCH -p long_sirocco\n'
                                     "#SBATCH -t 48:00:00\n"
                                     '#SBATCH --gres=gpu:1\n'
               }

cur_path = str(Path.cwd())
date = date.today().strftime("%d-%m")
# create campain log dir if not already done
Path(cur_path + "/campain_logs/jobouts/").mkdir(parents=True, exist_ok=True)
Path(cur_path + "/campain_logs/scripts/").mkdir(parents=True, exist_ok=True)
# Load txt file containing experiments to run (give it as argument to this script)
filename = 'to_run.txt'
if len(sys.argv) >= 2:
    filename = sys.argv[1]
launch = True
# Save a copy of txt file
shutil.copyfile(cur_path + "/" + filename, cur_path + '/campain_logs/scripts/' + date + '_' + filename)

# one_launch_per_n_seeds = 8
one_launch_per_n_seeds = 4

global_seed_offset = 0
incremental = False
if len(sys.argv) >= 3:
    if sys.argv[2] == 'nolaunch':
        launch = False
    if sys.argv[2] == 'seed_offset':
        global_seed_offset = int(sys.argv[3])
    if sys.argv[2] == 'incremental_seed_offset':
        global_seed_offset = int(sys.argv[3])
        incremental = True
if launch:
    print('Creating and Launching slurm scripts given arguments from {}'.format(filename))
    # time.sleep(1.0)
expe_list = []
with open(filename, 'r') as f:
    expe_list = [line.rstrip() for line in f]

exp_names = set()
for expe_args in expe_list:
    seed_offset_to_use = global_seed_offset

    if len(expe_args) == 0:
        # empty line
        continue

    if expe_args[0] == '#':
        # comment line
        continue

    exp_config = expe_args.split('--')[1:5]

    if not [arg.split(' ')[0] for arg in exp_config] == ['slurm_conf', 'nb_seeds', 'frames', 'model']:
        raise ValueError("Arguments must be in the following order {}".format(
            ['slurm_conf', 'nb_seeds', 'frames', 'model']))

    slurm_conf_name, nb_seeds, frames, exp_name = [arg.split(' ')[1] for arg in exp_config]

    user = getpass.getuser()
    if 'curta' in slurm_conf_name:
        gpu = ''
        PYTHON_INTERP = "$HOME/anaconda3/envs/act_and_speak/bin/python"
        n_cpus = 1
    elif 'plafrim' in slurm_conf_name:
        gpu = ''
        PYTHON_INTERP = '/home/{}/USER/conda/envs/act_and_speak/bin/python'.format(user)
        n_cpus = 1
    elif 'jz' in slurm_conf_name:

        if user == "utu57ed":
            PYTHON_INTERP='/gpfsscratch/rech/imi/{}/miniconda3/envs/social_ai/bin/python'.format(user)
        elif user == "uxo14qj":
            PYTHON_INTERP='/gpfswork/rech/imi/{}/miniconda3/envs/act_and_speak/bin/python'.format(user)
        else:
            if user != "flowers":
                raise ValueError("Who are you? User {} unknown.".format(user))

        gpu = ''  # '--gpu_id 0'
        n_cpus = 2

        n_cpus = 4
        assert n_cpus*one_launch_per_n_seeds == 16  # cpus_per_task is 8 will result in 16 cpus
    else:
        raise Exception("Unrecognized conf name: {} ".format(slurm_conf_name))

    # assert ((int(nb_seeds) % 8) == 0), 'number of seeds should be divisible by 8'
    assert ((int(nb_seeds) % 4) == 0), 'number of seeds should be divisible by 8'
    run_args = expe_args.split(exp_name, 1)[
        1]  # WARNING: assumes that exp_name comes after slurm_conf and nb_seeds and frames in txt

    # prepare experiment name formatting (use --* or -* instead of -- or - to use argument in experiment name
    # print(expe_args.split(exp_name))
    exp_details, run_args = process_arg_string(run_args)
    exp_name = date + '_' + exp_name + exp_details

    # no two trains are to be put in the same dir
    assert exp_names not in exp_names
    exp_names.add(exp_name)

    slurm_script_fullname = cur_path + "/campain_logs/scripts/{}".format(exp_name) + ".sh"
    # create corresponding slurm script

    # calculate how many chained jobs we need
    chained_training = "chained" in slurm_conf_name
    frames = int(frames)

    if chained_training:
        # assume 10M frames per 20h (fps 140 - very conservative)
        timelimit = slurm_confs[slurm_conf_name].split("-t ")[-1].split("\n")[0]
        assert timelimit == '19:59:00'
        one_script_frames = 10000000
        print(f"One script frames: {one_script_frames}")

        num_chained_jobs = frames // one_script_frames + bool(frames % one_script_frames)

    else:
        one_script_frames = frames
        num_chained_jobs = 1  # no chaining

    assert "--frames " not in run_args

    current_script_frames = min(one_script_frames, frames)

    # launch scripts (1 launch per 4 seeds)
    if launch:
        for i in range(int(nb_seeds) // one_launch_per_n_seeds):

            # continue jobs
            cont_job_i = num_chained_jobs  # last job

            exp_name_no_date = exp_name[5:]
            continue_slurm_script_fullname = cur_path + "/campain_logs/scripts/*{}_continue_{}".format(exp_name_no_date, "*")
            matched_scripts = glob.glob(continue_slurm_script_fullname)
            matched_scripts.sort(key=os.path.getctime)

            for last_script in reversed(matched_scripts):
                # start from the latest written script and start the first encountered that has a err file (that was ran)

                p = re.compile("continue_(.*).sh")
                last_job_id = int(p.search(last_script).group(1))

                last_script_name = os.path.basename(last_script)[:-3].replace("_continue_", "_cont_")
                if len(glob.glob(cur_path + "/campain_logs/jobouts/"+last_script_name+"*.sh.err")) == 1:
                    # error file found -> script was ran -> this is the script that crashed
                    break

            print(f"Continuing job id: {last_job_id}")
            # last_err_log = glob.glob(cur_path + "/campain_logs/jobouts/"+last_script_name+"*.sh.err")[0]
            #
            # print("Then ended with:\n")
            # print('"""\n')
            # for l in open(last_err_log).readlines():
            #     print("\t"+l, end='')
            # print('"""\n')

            # write continue script
            cont_script_name = "{}_continue_{}.sh".format(exp_name, last_job_id)
            continue_slurm_script_fullname = cur_path + "/campain_logs/scripts/"+cont_script_name

            current_script_frames = min(one_script_frames*(2+cont_job_i), frames)
            # run continue job
            sbatch_pipe = subprocess.Popen(
                ['sbatch', 'campain_logs/scripts/{}'.format(os.path.basename(last_script)), str((i * one_launch_per_n_seeds) + seed_offset_to_use)],  # 0 4 8 12
                stdout=subprocess.PIPE
            )

    if incremental:
        global_seed_offset += int(nb_seeds)

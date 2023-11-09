import sys
import time
from pathlib import Path
from datetime import date
import subprocess
import shutil
import os
import stat
import getpass

def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def write_script(script_fullname, exp_name, PYTHON_INTERP, n_cpu_cores, slurm_conf_name, run_args, script_frames,
                 is_continue=False, dependecy_jobid=None):

    print('creating slurm script with: --model {} {} --frames {} {}'.format(exp_name, run_args, script_frames, "--continue-train auto" if is_continue else ""))
    logfile_name = "{}{}_jid_%A".format(exp_name, "_cont_"+dependecy_jobid if is_continue else "")
    with open(script_fullname, 'w') as f:
        f.write('#!/bin/sh\n')

        if is_continue:
            f.write('#SBATCH --dependency=afterok:{}\n'.format(dependecy_jobid))
            f.write('#SBATCH --kill-on-invalid-dep=yes\n')

        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --cpus-per-task={}\n'.format((n_cpu_cores * n_seeds_per_one_launch)//2))  # cpus asked = num_cores // 2
        if "jz" in slurm_conf_name:
            f.write('#SBATCH --hint=nomultithread\n')
        f.write(slurm_confs[slurm_conf_name])
        f.write('#SBATCH --open-mode=append\n')  # append logs in logs files instead of truncating
        f.write('#SBATCH -o campain_logs/jobouts/{}.sh.out\n'
                '#SBATCH -e campain_logs/jobouts/{}.sh.err\n'.format(logfile_name, logfile_name))
        f.write("export EXP_INTERP='{}' ;\n".format(PYTHON_INTERP))
        f.write('# Launch !\n')
        f.write(
            'cpu_list=$(taskset -pc $$ | sed -E "s/(.*): (.*)/\\2/g" | tr "," "\\n" | sed -E "s/^[0-9]*$/&-&/g" | sed -E "s/-/ /g" | xargs -l seq | tr "\\n" " ")\n')
        f.write('echo "cpu list: $cpu_list"\n')
        f.write('COUNT=${1:-0}\n')
        f.write('i=0\n')
        f.write('cpus=""\n')
        f.write('for cpu in $cpu_list; do\n')
        f.write('cpus="$cpus$cpu"\n')
        f.write('i=$(($i+1))\n')
        f.write('if [ "$i" = "{}" ]; then\n'.format(n_cpu_cores))

        if "2gpus" in slurm_conf_name:
            f.write(
                "{}".format('CUDA_VISIBLE_DEVICES=$(( $COUNT % 2 )); ') +
                'taskset -c $cpus $EXP_INTERP -m scripts.train --model {}/$COUNT --seed $COUNT'.format(exp_name) +
                run_args + " --frames {}".format(script_frames) + "{}".format(" --continue-train auto" if is_continue else "") + ' &\n')

        elif "4gpus" in slurm_conf_name:
            f.write(
                "{}".format('CUDA_VISIBLE_DEVICES=$(( $COUNT % 4 )); ') +
                'taskset -c $cpus $EXP_INTERP -m scripts.train --model {}/$COUNT --seed $COUNT'.format(exp_name) +
                run_args + " --frames {}".format(script_frames) + "{}".format(" --continue-train auto" if is_continue else "") + ' &\n')

        else:
            f.write(
                # "{}".format('CUDA_VISIBLE_DEVICES=$(( $COUNT % 2 )); ' if "2gpus" in slurm_conf_name else "") +
                'taskset -c $cpus $EXP_INTERP -m scripts.train --model {}/$COUNT --seed $COUNT'.format(exp_name) +
                run_args + " --frames {}".format(script_frames) + "{}".format(" --continue-train auto" if is_continue else "") + ' &\n')

        f.write('echo "Using cpus $cpus for seed $COUNT"\n')
        f.write('COUNT=$(( $COUNT + 1 ))\n')
        f.write('cpus=""\n')
        f.write('i=0\n')
        f.write('else\n')
        f.write('cpus="$cpus,"\n')
        f.write('fi\n')
        f.write('done\n')
        f.write('wait\n')
        f.close()

    st = os.stat(script_fullname)
    os.chmod(script_fullname, st.st_mode | stat.S_IEXEC)

def write_script_one_seed(script_fullname, exp_name, PYTHON_INTERP, n_cpu_cores, slurm_conf_name, run_args, script_frames,
                 is_continue=False, dependecy_jobid=None):

    n_cpus = n_cpu_cores//2

    assert n_seeds_per_one_launch == 1, "Use write_script_old"
    print('creating slurm script with: --model {} {} --frames {} {}'.format(exp_name, run_args, script_frames, "--continue-train auto" if is_continue else ""))
    logfile_name = "{}{}_jid_%A".format(exp_name, "_cont_"+dependecy_jobid if is_continue else "")
    with open(script_fullname, 'w') as f:
        f.write('#!/bin/sh\n')

        if is_continue:
            f.write('#SBATCH --dependency=afterok:{}\n'.format(dependecy_jobid))
            f.write('#SBATCH --kill-on-invalid-dep=yes\n')

        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --cpus-per-task={}\n'.format((n_cpus)))
        if "jz" in slurm_conf_name:
            f.write('#SBATCH --hint=nomultithread\n')
        f.write(slurm_confs[slurm_conf_name])
        f.write('#SBATCH --open-mode=append\n')  # append logs in logs files instead of truncating
        f.write('#SBATCH -o campain_logs/jobouts/{}.sh.out\n'
                '#SBATCH -e campain_logs/jobouts/{}.sh.err\n'.format(logfile_name, logfile_name))
        f.write("export EXP_INTERP='{}' ;\n".format(PYTHON_INTERP))
        f.write('SEED=${1:-0}\n')
        f.write('# Launch !\n')
        f.write(
            '$EXP_INTERP -m scripts.train --model {}/$SEED --seed $SEED'.format(exp_name) +
            run_args + " --frames {}".format(script_frames) + "{}".format(" --continue-train auto" if is_continue else ""))
        f.close()

    st = os.stat(script_fullname)
    os.chmod(script_fullname, st.st_mode | stat.S_IEXEC)


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
                                "#SBATCH -t 3:59:00\n"
                                "#SBATCH --qos=qos_gpu-t3\n",
               'jz_short_gpu': '#SBATCH -A imi@v100\n'
                               '#SBATCH --gres=gpu:1\n'
                               "#SBATCH -t 19:59:00\n"
                               "#SBATCH --qos=qos_gpu-t3\n",
               'jz_super_short_gpu_chained':
                               '#SBATCH -A imi@v100\n'
                               '#SBATCH --gres=gpu:1\n'
                               "#SBATCH -t 3:59:00\n"
                               "#SBATCH -C v100\n" 
                               "#SBATCH --qos=qos_gpu-t3\n",
               'jz_short_gpu_chained': '#SBATCH -A imi@v100\n'
                                         '#SBATCH --gres=gpu:1\n'
                                         "#SBATCH -t 19:59:00\n"
                                         "#SBATCH -C v100\n"
                                         "#SBATCH --qos=qos_gpu-t3\n",
               'jz_short_gpu_chained_a100_4h': '#SBATCH -A imi@a100\n'
                                            '#SBATCH --gres=gpu:1\n'
                                            "#SBATCH -t 3:59:00\n"
                                            "#SBATCH -C a100\n"
                                            "#SBATCH --qos=qos_gpu-t3\n",
               'jz_short_gpu_chained_a100': '#SBATCH -A imi@a100\n'
                                       '#SBATCH --gres=gpu:1\n'
                                       "#SBATCH -t 19:59:00\n"
                                       "#SBATCH -C a100\n"
                                       "#SBATCH --qos=qos_gpu-t3\n",
               'jz_short_2gpus_chained': '#SBATCH -A imi@v100\n'
                                        '#SBATCH --gres=gpu:2\n'
                                        "#SBATCH -t 19:59:00\n"
                                        "#SBATCH -C v100\n"
                                       "#SBATCH --qos=qos_gpu-t3\n",
               'jz_short_4gpus_chained': '#SBATCH -A imi@v100\n'
                                         '#SBATCH --gres=gpu:4\n'
                                         "#SBATCH -t 19:59:00\n"
                                         "#SBATCH -C v100\n"
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
               'jz_short_cpu_chained': '#SBATCH -A imi@cpu\n'
                                       "#SBATCH -t 19:59:00\n"
                                       "#SBATCH --qos=qos_cpu-t3\n",
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

# how many seeds does one launch launch
# one_launch_per_n_seeds = 8

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

    arguments = ['slurm_conf', 'nb_seeds', 'cpu_cores_per_seed', 'gpus_per_seed', 'seeds_per_launch', 'frames', 'model']
    exp_config = expe_args.split('--')[1:len(arguments)+1]
    given_args = [arg.split(' ')[0] for arg in exp_config]

    if not given_args == arguments:
        raise ValueError("Arguments must be in the following order {}, and are {}".format(arguments, given_args))

    slurm_conf_name, nb_seeds, n_cpu_cores_per_seed, n_gpus_per_seed, n_seeds_per_one_launch, frames, exp_name = [arg.split(' ')[1] for arg in exp_config]

    n_seeds_per_one_launch = int(n_seeds_per_one_launch)
    n_cpu_cores_per_seed = int(n_cpu_cores_per_seed)

    user = getpass.getuser()
    if 'curta' in slurm_conf_name:
        gpu = ''
        PYTHON_INTERP = "$HOME/anaconda3/envs/act_and_speak/bin/python"
        n_cpu_cores_per_seed = 1

    elif 'plafrim' in slurm_conf_name:
        gpu = ''
        PYTHON_INTERP = '/home/{}/USER/conda/envs/act_and_speak/bin/python'.format(user)
        n_cpu_cores_per_seed = 1

    elif 'jz' in slurm_conf_name:
        if user == "utu57ed" or user == 'flowers':
            PYTHON_INTERP='/gpfsscratch/rech/imi/{}/miniconda3/envs/social_ai/bin/python'.format(user)
        elif user == "uxo14qj":
            PYTHON_INTERP='/gpfswork/rech/imi/{}/miniconda3/envs/act_and_speak/bin/python'.format(user)
        else:
            if user != "flowers":
                raise ValueError("Who are you? User {} unknown.".format(user))

        gpu = ''  # '--gpu_id 0'
        # n_cpus = 2

        # n_seeds_per_one_launch = 4
        # n_cpu_cores = 16 # n cpu cores for one seed
        # assert n_cpu_cores * n_seeds_per_one_launch == 64

        # n_seeds_per_one_launch = 2
        # n_cpu_cores = 16 # n cpu cores for one seed
        # assert n_cpu_cores * n_seeds_per_one_launch == 32

        # n_seeds_per_one_launch = 2
        # n_cpu_cores = 32 # n cpu cores for one seed
        # assert n_cpu_cores * n_seeds_per_one_launch == 64

        # n_seeds_per_one_launch = 1
        # n_cpu_cores = 16 # n cpu cores for one seed
        # assert n_cpu_cores * n_seeds_per_one_launch == 16
        #
        # n_seeds_per_one_launch = 1
        # n_cpu_cores = 32  # n cpu cores for one seed
        # assert n_cpu_cores * n_seeds_per_one_launch == 32
        #
        # assert n_seeds_per_one_launch == 1
        # assert n_cpu_cores_per_seed == 64  # n cpu cores for one seed
        # assert n_cpu_cores_per_seed * n_seeds_per_one_launch == 64

        # n_cpus = 64 # n cpu cores for one seed
        # assert n_cpus*one_launch_per_n_seeds == 256  # cpus_per_task is 8 will result in 16 cpu cores

        if "2gpus" in slurm_conf_name:
            job_gpus = 2
        elif "4gpus" in slurm_conf_name:
            job_gpus = 4
        elif "gpu" in slurm_conf_name:
            job_gpus = 1
        else:
            print("No gpus used")
            job_gpus = 1

        assert float(n_gpus_per_seed) == float(job_gpus / n_seeds_per_one_launch)


        print(f"\nJob configuration (1 launch):")
        print(f"\tSeeds: {n_seeds_per_one_launch}")
        print(f"\tGPUs: {job_gpus}")

        print(f"\n1 seed configuration:")
        print(f"\tCPU cores {n_cpu_cores_per_seed}")
        print(f"\tGPUs {job_gpus / n_seeds_per_one_launch}")
        time.sleep(0.5)

    else:
        raise Exception("Unrecognized conf name: {} ".format(slurm_conf_name))

    # assert ((int(nb_seeds) % 8) == 0), 'number of seeds should be divisible by 8'
    assert ((int(nb_seeds) % 4) == 0) or (int(nb_seeds) == 1), f'number of seeds should be divisible by 4 or 1 and is {nb_seeds}'
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
    print(chained_training)
    if chained_training:
        # assume 10M frames per 20h (fps 140 - very conservative)
        timelimit = slurm_confs[slurm_conf_name].split("-t ")[-1].split("\n")[0]
        if timelimit == '19:59:00':
            one_script_frames = 10000000

        elif timelimit == "3:59:00":
            one_script_frames = 2500000
        else:
            raise ValueError(f"Bad timelimit {timelimit}.")

        print(f"One script frames: {one_script_frames}")

        num_chained_jobs = frames // one_script_frames + bool(frames % one_script_frames)

        # # assume conservative fps - 300 (for one seed per gpu)
        # fps = 300
        # timelimit = slurm_confs[slurm_conf_name].split("-t ")[-1].split("\n")[0]
        # assert timelimit == '3:59:00'
        # timelimit_secs = get_sec(timelimit)
        #
        # one_script_frames = fps*timelimit_secs
        #
        # num_chained_jobs = frames // one_script_frames + bool(frames % one_script_frames)
        #
        # print(f"One script frames: {one_script_frames} -> num chained jobs {num_chained_jobs}")

    else:
        one_script_frames = frames
        num_chained_jobs = 1  # no chaining

    assert "--frames " not in run_args

    current_script_frames = min(one_script_frames, frames)
    if n_seeds_per_one_launch == 1:
        write_script_one_seed(slurm_script_fullname, exp_name, PYTHON_INTERP, n_cpu_cores_per_seed,
                              slurm_conf_name, run_args, current_script_frames, is_continue=False,
                              dependecy_jobid=None)
    else:
        write_script(slurm_script_fullname, exp_name, PYTHON_INTERP, n_cpu_cores_per_seed, slurm_conf_name,
                     run_args, current_script_frames, is_continue=False, dependecy_jobid=None)

    # launch scripts
    if launch:
        for i in range(int(nb_seeds) // n_seeds_per_one_launch):


            print('starting from seed {}'.format((i * n_seeds_per_one_launch) + global_seed_offset))
            # run start job
            sbatch_pipe = subprocess.Popen(
                ['sbatch', 'campain_logs/scripts/{}.sh'.format(exp_name), str((i * n_seeds_per_one_launch) + seed_offset_to_use)],  # 0 4 8 12
                stdout=subprocess.PIPE
            )
            job_id = subprocess.check_output(('cut',  '-d', ' ', '-f', '4'), stdin=sbatch_pipe.stdout).decode("utf_8").rstrip()
            sbatch_pipe.wait()

            # out = subprocess.run(
            #     ['sbatch', 'campain_logs/scripts/{}.sh'.format(exp_name), str((i * one_launch_per_n_seeds) + seed_offset_to_use)],  # 0 4 8 12
            #     capture_output=True
            # ).stdout.decode("utf-8")

            # continue jobs
            for cont_job_i in range(num_chained_jobs-1):
                # write continue script
                cont_script_name = "{}_continue_{}.sh".format(exp_name, job_id)
                continue_slurm_script_fullname = cur_path + "/campain_logs/scripts/"+cont_script_name

                current_script_frames = min(one_script_frames*(2+cont_job_i), frames)
                if n_seeds_per_one_launch == 1:
                    write_script_one_seed(continue_slurm_script_fullname, exp_name, PYTHON_INTERP, n_cpu_cores_per_seed,
                                 slurm_conf_name, run_args, current_script_frames,
                                 is_continue=True, dependecy_jobid=job_id)
                else:
                    write_script(continue_slurm_script_fullname, exp_name, PYTHON_INTERP, n_cpu_cores_per_seed, slurm_conf_name, run_args, current_script_frames,
                                 is_continue=True, dependecy_jobid=job_id)

                # run continue job
                sbatch_pipe = subprocess.Popen(
                    ['sbatch', 'campain_logs/scripts/{}'.format(cont_script_name), str((i * n_seeds_per_one_launch) + seed_offset_to_use)],  # 0 4 8 12
                    stdout=subprocess.PIPE
                )
                job_id = subprocess.check_output(('cut',  '-d', ' ', '-f', '4'), stdin=sbatch_pipe.stdout).decode("utf_8").rstrip()
                sbatch_pipe.wait()

    if incremental:
        global_seed_offset += int(nb_seeds)

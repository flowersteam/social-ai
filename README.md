---
title: SocialAI School Demo
emoji: 🧙🏻‍♂️
colorFrom: gray
colorTo: indigo
sdk: docker
app_port: 7860
---

# SocialAI

[comment]: <> (This repository is the official implementation of [My Paper Title]&#40;https://arxiv.org/abs/2030.12345&#41;. )

[comment]: <> (TODO: add arxiv link later)
This repository is the official implementation of SocialAI: Benchmarking Socio-Cognitive Abilities inDeep Reinforcement Learning Agents.

The website of the project is [here](https://sites.google.com/view/socialai)

The code is based on:
[minigrid](https://github.com/maximecb/gym-minigrid)

Additional repositories used:
[BabyAI](https://github.com/mila-iqia/babyai)
[RIDE](https://github.com/facebookresearch/impact-driven-exploration)
[astar](https://github.com/jrialland/python-astar)


## Installation

[comment]: <> (Clone the repo)

[comment]: <> (```)

[comment]: <> (git clone https://gitlab.inria.fr/gkovac/act-and-speak.git)

[comment]: <> (```)

Create and activate your conda env
```
conda create --name social_ai python=3.7
conda activate social_ai
conda install -c anaconda graphviz 
```

Install the required packages
```
pip install -r requirements.txt
pip install -e torch-ac
pip install -e gym-minigrid 
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

## Interactive policy

To run an enviroment in the interactive mode run:
```
python -m scripts.manual_control.py 
```

You can test different enviroments with the ```--env``` parameter.




# RL experiments

## Training

### Minimal example

To train a policy, run:
```train
python -m scripts.train --model test_model_name --seed 1  --compact-save --algo ppo --env SocialAI-AsocialBoxInformationSeekingParamEnv-v1 --dialogue --save-interval 1 --log-interval 1 --frames 5000000 --multi-modal-babyai11-agent --arch original_endpool_res --custom-ppo-2
`````

The policy should be above 0.95 success rate after the first 2M environment interactions.

### Recreating all the experiments 

See ```run_SAI_final_case_studies.txt``` for the experiments in the paper.

#### Regular machine

To run the experiments on a regular machine `run_SAI_final_case_studies.txt` contains all the bash commands running the RL experiments.



#### Slurm based cluster (todo:)

To recreate all the experiments from the paper on a slurm based server configure the `campaign_launcher.py` script and run:

```
python campaign_launcher.py run_NeurIPS.txt
```

[//]: # (The list of all the experiments and their parameters can be seen in run_NeurIPS.txt)

[//]: # ()
[//]: # (For example the bash equivalent of the following configuration:)

[//]: # (```)

[//]: # (--slurm_conf jz_long_2gpus_32g --nb_seeds 16 --model NeurIPS_Help_NoSocial_NO_BONUS_ABL  --compact-save --algo ppo --*env MiniGrid-AblationExiter-8x8-v0 --*env_args hidden_npc True --dialogue --save-interval 10 --frames 5000000 --*multi-modal-babyai11-agent --*arch original_endpool_res --*custom-ppo-2)

[//]: # (```)

[//]: # (is:)

[//]: # (```)

[//]: # (for SEED in {1..16})

[//]: # (do)

[//]: # (    python -m scripts.train --model NeurIPS_Help_NoSocial_NO_BONUS_ABL  --compact-save --algo ppo --*env MiniGrid-AblationExiter-8x8-v0 --*env_args hidden_npc True --dialogue --save-interval 10 --frames 5000000 --*multi-modal-babyai11-agent --*arch original_endpool_res --*custom-ppo-2 --seed $SEED & )

[//]: # (done)

[//]: # (```)



## Evaluation

To evaluate a policy, run:

```eval
python -m scripts.evaluate_new --episodes 500  --test-set-seed 1  --model-label test_model --eval-env SocialAI-TestLanguageFeedbackSwitchesInformationSeekingParamEnv-v1  --model-to-evaluate storage/test/ --n-seeds 8
````

To visualize a policy, run:
```
python -m scripts.visualize --model storage/test_model_name/1/ --pause 0.1 --seed $RANDOM --episodes 20 --gif viz/test
```


# LLM experiments

For LLMs set your ```OPENAI_API_KEY``` (and ```HF_TOKEN```) variable in ```~/.bashrc``` or wherever you want.

### Creating in-context examples
To create in_context examples you can use the ```create_LLM_examples.py``` script.

This script will open an interactive window, where you can manually control the agent.
By default, nothing is saved.
The general procedure is to press 'enter' to skip over environments which you don't like.
When you see a wanted enviroment, move the agent in the wanted position and start recording (press 'r'). The current and the following steps in the episode will be recorded.
Then control the agent and finish the episode. The new episode will start and recording will be turned off again.

If you already like some of the previously collected examples and want to append to them you can use the ```--load``` argument.

### Evaluating LLM-based agents

The script ```eval_LLMs.sh``` contains the bash commands to run all the experiments in the paper.

Here is an example of running evaluation on the ```text-ada-001``` model on the AsocialBox environment:
```
python -m scripts.LLM_test  --episodes 10 --max-steps 15 --model text-ada-001 --env-args size 7 --env-name SocialAI-AsocialBoxInformationSeekingParamEnv-v1 --in-context-path llm_data/in_context_examples/in_context_asocialbox_SocialAI-AsocialBoxInformationSeekingParamEnv-v1_2023_07_19_19_28_48/episodes.pkl
```

If you want to control the agent yourself you can set the model to ```interactive```.
```dummy``` agent just executes the move forward action, and ```random``` executes a random action. These agent are usefull for testing.



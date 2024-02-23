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

## Jupyter Notebook

Install the jupyter:
```commandline
pip install jupyter
```

Start the jupyter notebook with examples of usage with:
```
jupyter notebook SocialAI_playground.ipynb 
```

You can also play with our [google colab notebook](https://colab.research.google.com/drive/1LrbcRzIJwptZ9OdFko4pIFw72joTyW5q?usp=sharing)

## Interactive policy

To run an enviroment in the interactive mode run:
```
python -m scripts.manual_control.py 
```

You can test different enviroments with the ```--env``` parameter.

## Interactive demo
You can test our [interactive hugginface spaces demo](https://huggingface.co/spaces/flowers-team/SocialAISchool)

There you can create different enviroments and control the agent inside them.



# RL experiments

## Training

### Minimal example

To train a policy, run:
```train
python -m scripts.train --model test_model_name/1 --seed 1  --compact-save --algo ppo --env SocialAI-AsocialBoxInformationSeekingParamEnv-v1 --dialogue --save-interval 1 --log-interval 1 --frames 5000000 --multi-modal-babyai11-agent --arch original_endpool_res --custom-ppo-2
`````

The policy should be above 0.95 success rate after the first 2M environment interactions.


To plot the curve run:
```commandline
python data_visualize.py test_model_name
```

To visualize the policy, run:
```
python -m scripts.visualize --model storage/test_model_name/1/ --pause 0.1 --seed $RANDOM --episodes 20 --gif viz/test --env-name SocialAI-AsocialBoxInformationSeekingParamEnv-v1 ```
```

To evaluate a on a different environment, run:

```
python -m scripts.evaluate_new --episodes 500  --test-set-seed 1  --model-label test_model --eval-env SocialAI-TestLanguageFeedbackSwitchesInformationSeekingParamEnv-v1  --model-to-evaluate storage/test/ --n-seeds 8
````

## Recreating all the experiments 

### Regular machine

To run the experiments on a regular machine `run_SAI_final_case_studies.txt` contains all the bash commands to run the RL experiments.

### Slurm based cluster (todo:)

To recreate all the experiments from the paper on a slurm based server configure the `campaign_launcher.py` script and run:

```
python campaign_launcher.py run_SAI_final_case_studies.txt
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



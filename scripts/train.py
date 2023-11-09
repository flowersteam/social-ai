import argparse
import random
import warnings
import numpy as np
import time
import datetime
import torch

import gym_minigrid.social_ai_envs
import torch_ac
import sys
import json
import utils
from pathlib import Path
from distutils.dir_util import copy_tree
from utils.env import env_args_str_to_dict
from models import *


# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=10,
                    help="number of updates between two logs (default: 10)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--continue-train", default=None,
                    help="path to the model to finetune", type=str)
parser.add_argument("--finetune-train", default=None,
                    help="path to the model to finetune", type=str)
parser.add_argument("--compact-save", "-cs", action="store_true", default=False,
                    help="Keep only last model save")
parser.add_argument("--lr-schedule-end-frames", type=int, default=0,
                    help="Learning rate will be diminished from --lr to 0 linearly over the period of --lr-schedule-end-frames (default: 0 - no diminsh)")
parser.add_argument("--lr-end", type=float, default=0,
                    help="the final lr that will be reached at 'lr-schedule-end-frames' (default = 0)")

## Periodic test parameters
parser.add_argument("--test-set-name", required=False,
                    help="name of the environment to test on, default use the train env", default="SocialAITestSet")
# parser.add_argument("--test-env", required=False,
#                     help="name of the environment to test on, default use the train env")
# parser.add_argument("--no-test", "-nt", action="store_true", default=False,
#                     help="don't perform periodic testing")
parser.add_argument("--test-seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--test-episodes", type=int, default=50,
                    help="number of episodes to test")
parser.add_argument("--test-interval", type=int, default=-1,
                    help="number of updates between two tests (default: -1, no testing)")
parser.add_argument("--test-env-args", nargs='*', default="like_train_no_acl")

## Parameters for main algorithm
parser.add_argument("--acl", action="store_true", default=False,
                    help="use acl")
parser.add_argument("--acl-type", type=str, default=None,
                    help="acl type")
parser.add_argument("--acl-thresholds", nargs="+", type=float, default=(0.75, 0.75),
                    help="per phase thresholds for expert CL")
parser.add_argument("--acl-minimum-episodes", type=int, default=1000,
                    help="Never go to second phase before this.")
parser.add_argument("--acl-average-interval", type=int, default=500,
                    help="Average the perfromance estimate over this many last episodes")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--exploration-bonus", action="store_true", default=False,
                    help="Use a count based exploration bonus")
parser.add_argument("--exploration-bonus-type", nargs="+", default=["lang"],
                    help="modality on which to use the bonus (lang/grid)")
parser.add_argument("--exploration-bonus-params", nargs="+", type=float, default=(30., 50.),
                    help="parameters for a count based exploration bonus (C, M)")
parser.add_argument("--exploration-bonus-tanh", nargs="+", type=float, default=None,
                    help="tanh expl bonus scale, None means no tanh")
parser.add_argument("--expert-exploration-bonus", action="store_true", default=False,
                    help="Use an expert exploration bonus")
parser.add_argument("--episodic-exploration-bonus", action="store_true", default=False,
                    help="Use the exploration bonus in a episodic setting")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--dialogue", action="store_true", default=False,
                    help="add a GRU to the model to handle the history of dialogue input")
parser.add_argument("--current-dialogue-only", action="store_true", default=False,
                    help="add a GRU to the model to handle only the current dialogue input")
parser.add_argument("--multi-headed-agent", action="store_true", default=False,
                    help="add a talking head")
parser.add_argument("--babyai11_agent", action="store_true", default=False,
                    help="use the babyAI 1.1 agent architecture")
parser.add_argument("--multi-headed-babyai11-agent", action="store_true", default=False,
                    help="use the multi headed babyAI 1.1 agent architecture")
parser.add_argument("--custom-ppo", action="store_true", default=False,
                    help="use BabyAI original PPO hyperparameters")
parser.add_argument("--custom-ppo-2", action="store_true", default=False,
                    help="use BabyAI original PPO hyperparameters but with smaller memory")
parser.add_argument("--custom-ppo-3", action="store_true", default=False,
                    help="use BabyAI original PPO hyperparameters but with no memory")
parser.add_argument("--custom-ppo-rnd", action="store_true", default=False,
                    help="rnd reconstruct")
parser.add_argument("--custom-ppo-rnd-reference", action="store_true", default=False,
                    help="rnd reconstruct")
parser.add_argument("--custom-ppo-ride", action="store_true", default=False,
                    help="rnd reconstruct")
parser.add_argument("--custom-ppo-ride-reference", action="store_true", default=False,
                    help="rnd reconstruct")
parser.add_argument("--ppo-hp-tuning", action="store_true", default=False,
                    help="use PPO hyperparameters selected from our HP tuning")
parser.add_argument("--multi-modal-babyai11-agent", action="store_true", default=False,
                    help="use the multi headed babyAI 1.1 agent architecture")

# ride ref
parser.add_argument("--ride-ref-agent", action="store_true", default=False,
                    help="Model from the ride paper")
parser.add_argument("--ride-ref-preprocessor", action="store_true", default=False,
                    help="use ride reference preprocessor (3D images)")

parser.add_argument("--bAI-lang-model", help="lang model type for babyAI models", default="gru")
parser.add_argument("--memory-dim", type=int, help="memory dim (128 is small 2048 is big", default=128)
parser.add_argument("--clipped-rewards", action="store_true", default=False,
                    help="add a talking head")
parser.add_argument("--intrinsic-reward-epochs", type=int, default=0,
                    help="")
parser.add_argument("--balance-moa-training", action="store_true", default=False,
                    help="balance moa training to handle class imbalance.")
parser.add_argument("--moa-memory-dim", type=int, help="memory dim (default=128)", default=128)

# rnd + ride
parser.add_argument("--intrinsic-reward-coef", type=float, default=0.1,
                    help="")
parser.add_argument("--intrinsic-reward-learning-rate", type=float, default=0.0001,
                    help="")
parser.add_argument("--intrinsic-reward-momentum", type=float, default=0,
                    help="")
parser.add_argument("--intrinsic-reward-epsilon", type=float, default=0.01,
                    help="")
parser.add_argument("--intrinsic-reward-alpha", type=float, default=0.99,
                    help="")
parser.add_argument("--intrinsic-reward-max-grad-norm", type=float, default=40,
                    help="")
# rnd + soc_inf
parser.add_argument("--intrinsic-reward-loss-coef", type=float, default=0.1,
                    help="")
# ride
parser.add_argument("--intrinsic-reward-forward-loss-coef", type=float, default=10,
                    help="")
parser.add_argument("--intrinsic-reward-inverse-loss-coef", type=float, default=0.1,
                    help="")

parser.add_argument("--reset-rnd-ride-at-phase", action="store_true", default=False,
                    help="expert knowledge resets rnd ride at acl phase change")

# babyAI1.1 related
parser.add_argument("--arch", default="original_endpool_res",
                  help="image embedding architecture")
parser.add_argument("--num-films", type=int, default=2,
                    help="")

# Put all env related arguments after --env_args, e.g. --env_args nb_foo 1 is_bar True
parser.add_argument("--env-args", nargs='*', default=None)

args = parser.parse_args()

if args.compact_save:
    print("Compact save is deprecated. Don't use it. It doesn't do anything now.")

if args.save_interval != args.log_interval:
    print(f"save_interval ({args.save_interval}) and log_interval ({args.log_interval}) are not the same. This is not ideal for train continuation.")

if args.seed == -1:
    args.seed = np.random.randint(424242)

if args.custom_ppo:
    print("babyAI's ppo config")

    assert not args.custom_ppo_2
    assert not args.custom_ppo_3
    args.frames_per_proc = 40
    args.lr = 1e-4
    args.gae_lambda = 0.99
    args.recurrence = 20
    args.optim_eps = 1e-05
    args.clip_eps = 0.2
    args.batch_size = 1280

elif args.custom_ppo_2:
    print("babyAI's ppo config with smaller memory")

    assert not args.custom_ppo
    assert not args.custom_ppo_3
    args.frames_per_proc = 40
    args.lr = 1e-4
    args.gae_lambda = 0.99
    args.recurrence = 10
    args.optim_eps = 1e-05
    args.clip_eps = 0.2
    args.batch_size = 1280

elif args.custom_ppo_3:
    print("babyAI's ppo config with no memory")

    assert not args.custom_ppo
    assert not args.custom_ppo_2
    args.frames_per_proc = 40
    args.lr = 1e-4
    args.gae_lambda = 0.99
    args.recurrence = 1
    args.optim_eps = 1e-05
    args.clip_eps = 0.2
    args.batch_size = 1280

elif args.custom_ppo_rnd:
    print("RND reconstruct")

    assert not args.custom_ppo
    assert not args.custom_ppo_2
    assert not args.custom_ppo_3
    args.frames_per_proc = 40
    args.lr = 1e-4
    args.recurrence = 1
    # args.recurrence = 5  # use 5 for SocialAI envs
    args.batch_size = 640
    args.epochs = 4

    # args.optim_eps = 1e-05
    # args.entropy_coef = 0.0001
    args.clipped_rewards = True

elif args.custom_ppo_ride:
    print("RIDE reconstruct")

    assert not args.custom_ppo
    assert not args.custom_ppo_2
    assert not args.custom_ppo_3
    assert not args.custom_ppo_rnd

    args.frames_per_proc = 40
    args.lr = 1e-4
    args.recurrence = 1
    # args.recurrence = 5  # use 5 for SocialAI envs
    args.batch_size = 640
    args.epochs = 4

    # args.optim_eps = 1e-05
    # args.entropy_coef = 0.0005
    args.clipped_rewards = True

elif args.custom_ppo_rnd_reference:
    print("RND reconstruct")

    assert not args.custom_ppo
    assert not args.custom_ppo_2
    assert not args.custom_ppo_3

    args.frames_per_proc = 128  # 128 for PPO
    args.lr = 1e-4
    args.recurrence = 64

    args.gae_lambda = 0.99
    args.batch_size = 1280
    args.epochs = 4

    args.optim_eps = 1e-05
    args.clip_eps = 0.2
    args.entropy_coef = 0.0001
    args.clipped_rewards = True


elif args.custom_ppo_ride_reference:
    print("RIDE reference")

    assert not args.custom_ppo
    assert not args.custom_ppo_2
    assert not args.custom_ppo_3
    assert not args.custom_ppo_rnd

    args.frames_per_proc = 128  # 128 for PPO
    args.lr = 1e-4
    args.recurrence = 64

    args.gae_lambda = 0.99
    args.batch_size = 1280
    args.epochs = 4

    args.optim_eps = 1e-05
    args.clip_eps = 0.2
    args.entropy_coef = 0.0005
    args.clipped_rewards = True

elif args.ppo_hp_tuning:

    args.frames_per_proc = 40
    args.lr = 1e-4
    args.recurrence = 5
    args.batch_size = 640
    args.epochs = 4

if args.env not in [
    "MiniGrid-KeyCorridorS3R3-v0",
    "MiniGrid-MultiRoom-N2-S4-v0",
    "MiniGrid-MultiRoom-N4-S5-v0",
    "MiniGrid-MultiRoom-N7-S4-v0",
    "MiniGrid-MultiRoomNoisyTV-N7-S4-v0"
]:
    if args.recurrence <= 1:
        print("You are using recurrence {} with {} env. This is probably unintentional.".format(args.recurrence, args.env))
        # warnings.warn("You are using recurrence {} with {} env. This is probably unintentional.".format(args.recurrence, args.env))


args.mem = args.recurrence > 1

# Set run dir
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

if Path(model_dir).exists() and args.continue_train is None:
    raise ValueError(f"Dir {model_dir} already exists and continue train is None.")

# Load loggers and Tensorboard writer
txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)


# Log command and all script arguments
txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Create env_args dict
env_args = env_args_str_to_dict(args.env_args)

if args.acl:
    # expert_acl = "three_stage_expert"
    expert_acl = args.acl_type
    print(f"Using curriculum: {expert_acl}.")
else:
    expert_acl = None

env_args_no_acl = env_args.copy()
env_args["curriculum"] = expert_acl
env_args["expert_curriculum_thresholds"] = args.acl_thresholds
env_args["expert_curriculum_average_interval"] = args.acl_average_interval
env_args["expert_curriculum_minimum_episodes"] = args.acl_minimum_episodes
env_args["egocentric_observation"] = True

# test env args
if not args.test_env_args:
    test_env_args = {}
elif args.test_env_args == "like_train_no_acl":
    test_env_args = env_args_no_acl
elif args.test_env_args == "like_train":
    test_env_args = env_args
else:
    test_env_args = env_args_str_to_dict(args.test_env_args)


if "SocialAI-" not in args.env:
    env_args = {}
    test_env_args = {}

print("train_env_args:", env_args)
print("test_env_args:", test_env_args)

# Load train environments

envs = []
for i in range(args.procs):
    envs.append(utils.make_env(args.env, args.seed + 10000 * i, env_args=env_args))

txt_logger.info("Environments loaded\n")

if args.continue_train and args.finetune_train:
    raise ValueError(f"Continue path ({args.continue_train}) and finetune path ({args.finetune_train}) can't both be set.")

# Load training status
if args.continue_train:
    if args.continue_train == "auto":
        status_continue_path = Path(model_dir)
        args.continue_train = status_continue_path  # just in case
    else:
        status_continue_path = Path(args.continue_train)

    if status_continue_path.is_dir():
        # if dir, assume experiment dir so append the seed
        # status_continue_path = Path(status_continue_path) / str(args.seed)
        status_continue_path = utils.get_status_path(status_continue_path)

    else:
        if not status_continue_path.is_file():
            raise ValueError(f"{status_continue_path} is not a file")

        if "status" not in status_continue_path.name:
            raise UserWarning(f"{status_continue_path} is does not contain status, is this the correct file? ")

    status = utils.load_status(status_continue_path)

    txt_logger.info("Training status loaded\n")
    txt_logger.info(f"{model_name} continued from {status_continue_path}")

    # copy everything from model_dir to backup_dir
    assert Path(status_continue_path).is_file()

elif args.finetune_train:

    status_finetune_path = Path(args.finetune_train)

    if status_finetune_path.is_dir():
        # if dir, assume experiment dir so append the seed
        status_finetune_seed_path = Path(status_finetune_path) / str(args.seed)
        if status_finetune_seed_path.exists():
            # if a seed folder exists assume that you use that one
            status_finetune_path = utils.get_status_path(status_finetune_seed_path)

        else:
            # if not assume that no seed folder exists
            status_finetune_path = utils.get_status_path(status_finetune_path)

    else:
        if not status_finetune_path.is_file():
            raise ValueError(f"{status_finetune_path} is not dir or a file")

        if "status" not in status_finetune_path.name:
            raise UserWarning(f"{status_finetune_path} is does not contain status, is this the correct file? ")

    status = utils.load_status(status_finetune_path)

    txt_logger.info("Training status loaded\n")
    txt_logger.info(f"{model_name} finetuning from {status_finetune_path}")

    # copy everything from model_dir to backup_dir
    assert Path(status_finetune_path).is_file()

    # reset parameters for finetuning
    status["num_frames"] = 0
    status["update"] = 0
    del status["optimizer_state"]
    del status["lr_scheduler_state"]
    del status["env_args"]

else:
    status = {"num_frames": 0, "update": 0}

# Parameter sanity checks
if args.dialogue and args.current_dialogue_only:
        raise ValueError("Either use dialogue or current-dialogue-only")

if not args.dialogue and not args.current_dialogue_only:
    warnings.warn("Not using dialogue")

if args.text:
    raise ValueError("Text should not be used. Use dialogue instead.")


# Load observations preprocessor
obs_space, preprocess_obss = utils.get_obss_preprocessor(
    obs_space=envs[0].observation_space,
    text=args.text,
    dialogue_current=args.current_dialogue_only,
    dialogue_history=args.dialogue,
    custom_image_preprocessor=utils.ride_ref_image_preprocessor if args.ride_ref_preprocessor else None,
    custom_image_space_preprocessor=utils.ride_ref_image_space_preprocessor if args.ride_ref_preprocessor else None,
)

if args.continue_train is not None or args.finetune_train is not None:
    assert "vocab" in status
    preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

if args.exploration_bonus:
    if args.expert_exploration_bonus:
        warnings.warn("You are using expert exploration bonus.")

# Load model
assert sum(map(int, [
    args.multi_modal_babyai11_agent,
    args.multi_headed_babyai11_agent,
    args.babyai11_agent,
    args.multi_headed_agent,
])) <= 1

if args.multi_modal_babyai11_agent:
    acmodel = MultiModalBaby11ACModel(
        obs_space=obs_space,
        action_space=envs[0].action_space,
        arch=args.arch,
        use_text=args.text,
        use_dialogue=args.dialogue,
        use_current_dialogue_only=args.current_dialogue_only,
        use_memory=args.mem,
        lang_model=args.bAI_lang_model,
        memory_dim=args.memory_dim,
        num_films=args.num_films
    )
elif args.ride_ref_agent:
    assert args.mem
    assert not args.text
    assert not args.dialogue

    acmodel = RefACModel(
        obs_space=obs_space,
        action_space=envs[0].action_space,
        use_memory=args.mem,
        use_text=args.text,
        use_dialogue=args.dialogue,
        input_size=obs_space['image'][-1],
    )
    if args.current_dialogue_only: raise NotImplementedError("current dialogue only")

else:
    acmodel = ACModel(
        obs_space=obs_space,
        action_space=envs[0].action_space,
        use_memory=args.mem,
        use_text=args.text,
        use_dialogue=args.dialogue,
        input_size=obs_space['image'][-1],
    )
    if args.current_dialogue_only: raise NotImplementedError("current dialogue only")

# if args.continue_train is not None:
#     assert "model_state" in status
#     acmodel.load_state_dict(status["model_state"])

acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

# Load algo
assert args.algo == "ppo"
algo = torch_ac.PPOAlgo(
    envs=envs,
    acmodel=acmodel,
    device=device,
    num_frames_per_proc=args.frames_per_proc,
    discount=args.discount,
    lr=args.lr,
    gae_lambda=args.gae_lambda,
    entropy_coef=args.entropy_coef,
    value_loss_coef=args.value_loss_coef,
    max_grad_norm=args.max_grad_norm,
    recurrence=args.recurrence,
    adam_eps=args.optim_eps,
    clip_eps=args.clip_eps,
    epochs=args.epochs,
    batch_size=args.batch_size,
    preprocess_obss=preprocess_obss,
    exploration_bonus=args.exploration_bonus,
    exploration_bonus_tanh=args.exploration_bonus_tanh,
    exploration_bonus_type=args.exploration_bonus_type,
    exploration_bonus_params=args.exploration_bonus_params,
    expert_exploration_bonus=args.expert_exploration_bonus,
    episodic_exploration_bonus=args.episodic_exploration_bonus,
    clipped_rewards=args.clipped_rewards,
    # for rnd, ride, and social influence
    intrinsic_reward_coef=args.intrinsic_reward_coef,
    # for rnd and ride
    intrinsic_reward_epochs=args.intrinsic_reward_epochs,
    intrinsic_reward_learning_rate=args.intrinsic_reward_learning_rate,
    intrinsic_reward_momentum=args.intrinsic_reward_momentum,
    intrinsic_reward_epsilon=args.intrinsic_reward_epsilon,
    intrinsic_reward_alpha=args.intrinsic_reward_alpha,
    intrinsic_reward_max_grad_norm=args.intrinsic_reward_max_grad_norm,
    # for rnd and social influence
    intrinsic_reward_loss_coef=args.intrinsic_reward_loss_coef,
    # for ride
    intrinsic_reward_forward_loss_coef=args.intrinsic_reward_forward_loss_coef,
    intrinsic_reward_inverse_loss_coef=args.intrinsic_reward_inverse_loss_coef,
    # for social influence
    balance_moa_training=args.balance_moa_training,
    moa_memory_dim=args.moa_memory_dim,
    lr_schedule_end_frames=args.lr_schedule_end_frames,
    end_lr=args.lr_end,
    reset_rnd_ride_at_phase=args.reset_rnd_ride_at_phase,
)

if args.continue_train or args.finetune_train:
    algo.load_status_dict(status)
    # txt_logger.info(f"Model + Algo loaded from {args.continue_train or args.finetune_train}\n")
    if args.continue_train:
        txt_logger.info(f"Model + Algo loaded from {status_continue_path} \n")
    elif args.finetune_train:
        txt_logger.info(f"Model + Algo loaded from {status_finetune_path} \n")


# todo: make nicer
# Set and load test environment
if args.test_set_name:
    if args.test_set_name == "SocialAITestSet":
        # "SocialAI-AskEyeContactLanguageBoxesInformationSeekingParamEnv-v1",
        # "SocialAI-NoIntroPointingBoxesInformationSeekingParamEnv-v1"
        test_env_names = [
            "SocialAI-TestLanguageColorBoxesInformationSeekingEnv-v1",
            "SocialAI-TestLanguageFeedbackBoxesInformationSeekingEnv-v1",
            "SocialAI-TestPointingBoxesInformationSeekingEnv-v1",
            "SocialAI-TestEmulationBoxesInformationSeekingEnv-v1",
            "SocialAI-TestLanguageColorSwitchesInformationSeekingEnv-v1",
            "SocialAI-TestLanguageFeedbackSwitchesInformationSeekingEnv-v1",
            "SocialAI-TestPointingSwitchesInformationSeekingEnv-v1",
            "SocialAI-TestEmulationSwitchesInformationSeekingEnv-v1",
            "SocialAI-TestLanguageColorMarbleInformationSeekingEnv-v1",
            "SocialAI-TestLanguageFeedbackMarbleInformationSeekingEnv-v1",
            "SocialAI-TestPointingMarbleInformationSeekingEnv-v1",
            "SocialAI-TestEmulationMarbleInformationSeekingEnv-v1",
            "SocialAI-TestLanguageColorGeneratorsInformationSeekingEnv-v1",
            "SocialAI-TestLanguageFeedbackGeneratorsInformationSeekingEnv-v1",
            "SocialAI-TestPointingGeneratorsInformationSeekingEnv-v1",
            "SocialAI-TestEmulationGeneratorsInformationSeekingEnv-v1",
            "SocialAI-TestLanguageColorLeversInformationSeekingEnv-v1",
            "SocialAI-TestLanguageFeedbackLeversInformationSeekingEnv-v1",
            "SocialAI-TestPointingLeversInformationSeekingEnv-v1",
            "SocialAI-TestEmulationLeversInformationSeekingEnv-v1",
            "SocialAI-TestLanguageColorDoorsInformationSeekingEnv-v1",
            "SocialAI-TestLanguageFeedbackDoorsInformationSeekingEnv-v1",
            "SocialAI-TestPointingDoorsInformationSeekingEnv-v1",
            "SocialAI-TestEmulationDoorsInformationSeekingEnv-v1",

            "SocialAI-TestLeverDoorCollaborationEnv-v1",
            "SocialAI-TestMarblePushCollaborationEnv-v1",
            "SocialAI-TestMarblePassCollaborationEnv-v1",

            "SocialAI-TestAppleStealingPerspectiveTakingEnv-v1"
        ]
    elif args.test_set_name == "SocialAIGSTestSet":
        test_env_names = [
            "SocialAI-GridSearchParamEnv-v1",
            "SocialAI-GridSearchPointingParamEnv-v1",
            "SocialAI-GridSearchLangColorParamEnv-v1",
            "SocialAI-GridSearchLangFeedbackParamEnv-v1",
        ]
    elif args.test_set_name == "SocialAICuesGSTestSet":
        test_env_names = [
            "SocialAI-CuesGridSearchParamEnv-v1",
            "SocialAI-CuesGridSearchPointingParamEnv-v1",
            "SocialAI-CuesGridSearchLangColorParamEnv-v1",
            "SocialAI-CuesGridSearchLangFeedbackParamEnv-v1",
        ]
    elif args.test_set_name == "BoxesPointingTestSet":
        test_env_names = [
            "SocialAI-TestPointingBoxesInformationSeekingParamEnv-v1",
        ]
    elif args.test_set_name == "PointingTestSet":
        test_env_names = gym_minigrid.social_ai_envs.pointing_test_set
    elif args.test_set_name == "LangColorTestSet":
        test_env_names = gym_minigrid.social_ai_envs.language_color_test_set
    elif args.test_set_name == "LangFeedbackTestSet":
        test_env_names = gym_minigrid.social_ai_envs.language_feedback_test_set
    # joint attention
    elif args.test_set_name == "JAPointingTestSet":
        test_env_names = gym_minigrid.social_ai_envs.ja_pointing_test_set
    elif args.test_set_name == "JALangColorTestSet":
        test_env_names = gym_minigrid.social_ai_envs.ja_language_color_test_set
    elif args.test_set_name == "JALangFeedbackTestSet":
        test_env_names = gym_minigrid.social_ai_envs.ja_language_feedback_test_set
    # emulation
    elif args.test_set_name == "DistrEmulationTestSet":
        test_env_names = gym_minigrid.social_ai_envs.distr_emulation_test_set
    elif args.test_set_name == "NoDistrEmulationTestSet":
        test_env_names = gym_minigrid.social_ai_envs.no_distr_emulation_test_set
    # formats
    elif args.test_set_name == "NFormatsTestSet":
        test_env_names = gym_minigrid.social_ai_envs.N_formats_test_set
    elif args.test_set_name == "EFormatsTestSet":
        test_env_names = gym_minigrid.social_ai_envs.E_formats_test_set
    elif args.test_set_name == "AFormatsTestSet":
        test_env_names = gym_minigrid.social_ai_envs.A_formats_test_set
    elif args.test_set_name == "AEFormatsTestSet":
        test_env_names = gym_minigrid.social_ai_envs.AE_formats_test_set

    elif args.test_set_name == "RoleReversalTestSet":
        test_env_names = gym_minigrid.social_ai_envs.role_reversal_test_set

    else:
        raise ValueError("Undefined test set name.")


else:
    test_env_names = [args.env]

# test_envs = []
testers = []
if args.test_interval > 0:
    for test_env_name in test_env_names:
        make_env_args = {
            "env_key": test_env_name,
            "seed": args.test_seed,
            "env_args": test_env_args,
        }
        testers.append(utils.Tester(
            make_env_args, args.test_seed, args.test_episodes, model_dir, acmodel, preprocess_obss, device)
        )

        # test_env = utils.make_env(test_env_name, args.test_seed, env_args=test_env_args)
        # test_envs.append(test_env)

        # init tester
        # testers.append(utils.Tester(test_env, args.test_seed, args.test_episodes, model_dir, acmodel, preprocess_obss, device))

if args.continue_train:
    for tester in testers:
        tester.load()


# Save config
env_args_ = {k: v.__repr__() if k == "curriculum" else v for k, v in env_args.items()}
test_env_args_ = {k: v.__repr__() if k == "curriculum" else v for k, v in test_env_args.items()}
config_dict = {
    "seed": args.seed,
    "env": args.env,
    "env_args": env_args_,
    "test_seed": args.test_seed,
    "test_env": args.test_set_name,
    "test_env_args": test_env_args_
}
config_dict.update(algo.get_config_dict())
config_dict.update(acmodel.get_config_dict())
with open(model_dir+'/config.json', 'w') as fp:
    json.dump(config_dict, fp)


# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

log_add_headers = num_frames == 0 or not args.continue_train

long_term_save_interval = 5000000

if args.continue_train:
    # set next long term save interval
    next_long_term_save = (1 + num_frames // long_term_save_interval) * long_term_save_interval

else:
    next_long_term_save = 0  # for long term logging


while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    # print("current_seed_pre_train:", np.random.get_state()[1][0])
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    NPC_intro = np.mean(logs["NPC_introduced_to"])

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        extrinsic_return_per_episode = utils.synthesize(logs["extrinsic_return_per_episode"])
        exploration_bonus_per_episode = utils.synthesize(logs["exploration_bonus_per_episode"])
        success_rate = utils.synthesize(logs["success_rate_per_episode"])
        curriculum_max_success_rate = utils.synthesize(logs["curriculum_max_mean_perf_per_episode"])
        curriculum_param = utils.synthesize(logs["curriculum_param_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        # intrinsic_reward_perf = utils.synthesize(logs["intr_reward_perf"])
        # intrinsic_reward_perf_ = utils.synthesize(logs["intr_reward_perf_"])

        intrinsic_reward_perf = logs["intr_reward_perf"]
        intrinsic_reward_perf_ = logs["intr_reward_perf_"]

        lr_ = logs["lr"]

        time_now = int(datetime.datetime.now().strftime("%d%m%Y%H%M%S"))

        header = ["update", "frames", "FPS", "duration", "time"]
        data = [update, num_frames, fps, duration, time_now]
        data_to_print = [update, num_frames, fps, duration, time_now]

        header += ["success_rate_" + key for key in success_rate.keys()]
        data += success_rate.values()
        data_to_print += success_rate.values()

        header += ["curriculum_max_success_rate_" + key for key in curriculum_max_success_rate.keys()]
        data += curriculum_max_success_rate.values()
        if args.acl:
            data_to_print += curriculum_max_success_rate.values()

        header += ["curriculum_param_" + key for key in curriculum_param.keys()]
        data += curriculum_param.values()
        if args.acl:
            data_to_print += curriculum_param.values()

        header += ["extrinsic_return_" + key for key in extrinsic_return_per_episode.keys()]
        data += extrinsic_return_per_episode.values()
        data_to_print += extrinsic_return_per_episode.values()

        # turn on
        header += ["exploration_bonus_" + key for key in exploration_bonus_per_episode.keys()]
        data += exploration_bonus_per_episode.values()
        data_to_print += exploration_bonus_per_episode.values()

        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        data_to_print += rreturn_per_episode.values()


        header += ["intrinsic_reward_perf_"]
        data += [intrinsic_reward_perf]
        # data_to_print += [intrinsic_reward_perf]

        header += ["intrinsic_reward_perf2_"]
        data += [intrinsic_reward_perf_]
        # data_to_print += [intrinsic_reward_perf_]

        # header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        # data += num_frames_per_episode.values()

        header += ["NPC_intro"]
        data += [NPC_intro]
        data_to_print += [NPC_intro]

        header += ["lr"]
        data += [lr_]
        data_to_print += [lr_]

        # header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        # data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        # curr_history_len = len(algo.env.envs[0].curriculum.performance_history)
        # header += ["curr_history_len"]
        # data += [curr_history_len]

        txt_logger.info("".join([
            "U {} | F {:06} | FPS {:04.0f} | D {} | T {} ",
            "| SR:μσmM {:.2f} {:.1f} {:.1f} {:.1f} ",
            "| CurMaxSR:μσmM {:.2f} {:.1f} {:.1f} {:.1f} " if args.acl else "",
            "| CurPhase:μσmM {:.2f} {:.1f} {:.1f} {:.1f} " if args.acl else "",
            "| ExR:μσmM {:.2f} {:.1f} {:.1f} {:.1f} ",
            "| InR:μσmM {:.2f} {:.1f} {:.1f} {:.1f} ",
            "| rR:μσmM {:.6f} {:.1f} {:.1f} {:.1f} ",
            # "| irp:μσmM {:.6f} {:.2f} {:.2f} {:.2f} ",
            # "| irp_:μσmM {:.6f} {:.2f} {:.2f} {:.2f} ",
            # "| F:μσmM {:.1f} {:.1f} {} {} ",
            "| NPC_intro: {:.3f}",
            "| lr: {:.5f}",
            # "| cur_his_len: {:.5f}" if args.acl else "",
            # "| H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
        ]).format(*data_to_print))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if log_add_headers:
            csv_logger.writerow(header)
            log_add_headers = False
        csv_logger.writerow(data)
        csv_file.flush()

    # Save status
    long_term_save = False
    if num_frames >= next_long_term_save:
        next_long_term_save += long_term_save_interval
        long_term_save = True

    if (args.save_interval > 0 and update % args.save_interval == 0) or long_term_save:
        # continuing train works best when save_interval == log_interval, the csv is cleaner wo redundancies
        status = {"num_frames": num_frames, "update": update}

        algo_status = algo.get_status_dict()
        status = {**status, **algo_status}

        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
            status["env_args"] = env_args

        if long_term_save:
            utils.save_status(status, model_dir, num_frames=num_frames)
            utils.save_model(acmodel, model_dir, num_frames=num_frames)
            txt_logger.info("Status and Model saved for {} frames".format(num_frames))

        else:
            utils.save_status(status, model_dir)
            utils.save_model(acmodel, model_dir)
            txt_logger.info("Status and Model saved")

    if args.test_interval > 0 and (update % args.test_interval == 0 or update == 1):
        txt_logger.info(f"Testing at update {update}.")
        test_success_rates = []
        for tester in testers:
            mean_success_rate, mean_rewards = tester.test_agent(num_frames)
            test_success_rates.append(mean_success_rate)
            txt_logger.info(f"\t{tester.envs[0].spec.id} -> {mean_success_rate} (SR)")
            tester.dump()

        if len(testers):
            txt_logger.info(f"Test set SR: {np.array(test_success_rates).mean()}")


# save at the end
status = {"num_frames": num_frames, "update": update}
algo_status = algo.get_status_dict()
status = {**status, **algo_status}

if hasattr(preprocess_obss, "vocab"):
    status["vocab"] = preprocess_obss.vocab.vocab
    status["env_args"] = env_args

utils.save_status(status, model_dir)
utils.save_model(acmodel, model_dir)
txt_logger.info("Status and Model saved at the end")

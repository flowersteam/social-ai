import sys
import itertools

if __name__ == '__main__':
    '''
        Generate scripts to perform grid search on agents's hyperparameters.

        Defines the values to test for each hyperparameter.
    '''

    tuning_dict = {
        # "PPO": {
        #     "frames-per-proc": [20, 40, 80],
        #     "lr": [1e-4, 1e-3],
        #     "entropy-coef": [0, 0.01, 0.05],
        #     "recurrence": [5, 10],
        #     "epochs": [4, 8, 12],
        #     "batch-size": [640, 1280, 2560],
        #     "env": ["MiniGrid-CoinThief-8x8-v0 --env_args few_actions True", "MiniGrid-TalkItOutPolite-8x8-v0"]
        # },
        "PPO-RND": {
            # rnd and ride
            "optim-eps": [1e-5, 1e-7],
            "entropy-coef": [0.01, 0.0001, 0.0005],
            "intrinsic-reward-learning-rate": [0.0001, 0.0004, 0.001],
            # "intrinsic-reward-coef": [0.1],
            # "intrinisc-reward-momentum": [0],
            "intrinsic-reward-epsilon": [0.01, 0.001, 0.0001],
            # "intrinsic-reward-alpha": [0.99],
            "intrinsic-reward-max-grad-norm": [1000, 40, 20, 1],
            # rnd
            # "intrinsic-reward-loss-coef": [0.1],
            "env": ["MiniGrid-CoinThief-8x8-v0 --env_args few_actions True", "MiniGrid-TalkItOutPolite-8x8-v0"]
        }
    }

    with open("hp_tuning_agent.txt", 'w') as f:
        for agent in tuning_dict:
            f.write('## {}\n'.format(agent))
            current_agent_parameters = list(tuning_dict[agent].keys())
            current_agent_hyperparams = tuning_dict[agent].values()
            for point in itertools.product(*current_agent_hyperparams):
                current_arguments = ''
                for i in range(len(current_agent_parameters)):
                    current_arguments += ' --*' + current_agent_parameters[i]
                    current_arguments += ' ' + str(point[i]) if point[i] is not None else ''

                if agent == "PPO-RND":
                    f.write(
                        '--slurm_conf jz_short_2gpus_32g --nb_seeds 8 --model PPO_RND_tuning --algo ppo -cs --frames 10000000 '
                        '--save-interval 100 --log-interval 100 '
                        '--dialogue --multi-modal-babyai11-agent '
                        '--exploration-bonus --exploration-bonus-type rnd --clipped-rewards '
                        '--arch original_endpool_res {}\n'.format(current_arguments))
                else:
                    f.write(
                        '--slurm_conf jz_short_2gpus_32g --nb_seeds 8 --model PPO_RND_tuning --algo ppo -cs --frames 10000000 '
                        '--save-interval 100 --log-interval 100 '
                        '--dialogue --multi-modal-babyai11-agent '
                        '--arch original_endpool_res {}\n'.format(current_arguments))


import numpy as np
import utils
import os
import pickle
import torch

class AgentWrap:
    """ Handles action selection without gradient updates for proper testing """

    def __init__(self, acmodel, preprocess_obss, device, num_envs=1, argmax=False):

        self.preprocess_obss = preprocess_obss
        self.acmodel = acmodel

        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=self.device)

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _ = self.acmodel(preprocessed_obss)

        if isinstance(dist, torch.distributions.Distribution):
            if self.argmax:
                actions = dist.probs.max(1, keepdim=True)[1]
            else:
                actions = dist.sample()
        else:
            if self.argmax:
                actions = torch.stack([d.probs.max(1)[1] for d in dist], dim=1)
            else:
                actions = torch.stack([d.sample() for d in dist], dim=1)
        return self.acmodel.construct_final_action(actions.cpu().numpy())

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])


class Tester:

    def __init__(self, env_args, seed, episodes, save_path, acmodel, preprocess_obss, device):

        self.envs = [utils.make_env(
            **env_args
        ) for _ in range(episodes)]
        self.seed = seed
        self.episodes = episodes
        self.ep_counter = 0
        self.savefile = save_path + "/testing_{}.pkl".format(self.envs[0].spec.id)
        print("Testing log: ", self.savefile)

        self.stats_dict = {"test_rewards": [], "test_success_rates": [], "test_step_nb": []}
        self.agent = AgentWrap(acmodel, preprocess_obss, device)

    def test_agent(self, num_frames):
        self.agent.acmodel.eval()

        # set seed
        # self.env.seed(self.seed)
        # save test time (nb training steps)
        self.stats_dict['test_step_nb'].append(num_frames)

        rewards = []
        success_rates = []


        # cols = []
        # s = "-".join([e.current_env.marble.color for e in self.envs])
        # print("s:", s)

        for episode in range(self.episodes):
            # self.envs[episode].seed(self.seed)
            self.envs[episode].seed(episode)
            # print("current_seed", np.random.get_state()[1][0])

            obs = self.envs[episode].reset()

            # cols.append(self.envs[episode].current_env.marble.color)
            # cols.append(str(self.envs[episode].current_env.marble.cur_pos))

            done = False
            while not done:
                action = self.agent.get_action(obs)

                obs, reward, done, info = self.envs[episode].step(action)
                self.agent.analyze_feedback(reward, done)

                if done:
                    rewards.append(reward)
                    success_rates.append(info['success'])
                    break

        # from hashlib import md5
        # hash_string = "-".join(cols).encode()

        # print('hs:', hash_string[:20])
        # print("hash test envs:", md5(hash_string).hexdigest())

        mean_rewards = np.array(rewards).mean()
        mean_success_rates = np.array(success_rates).mean()

        self.stats_dict["test_rewards"].append(mean_rewards)
        self.stats_dict["test_success_rates"].append(mean_success_rates)

        self.agent.acmodel.train()
        return mean_success_rates, mean_rewards

    def load(self):
        if os.path.isfile(self.savefile):
            with open(self.savefile, 'rb') as f:
                stats_dict_loaded = pickle.load(f)

                for k, v in stats_dict_loaded.items():
                    self.stats_dict[k] = v
        else:
            raise ValueError(f"Save file {self.savefile} doesn't exist.")

    def dump(self):
        with open(self.savefile, 'wb') as f:
            pickle.dump(self.stats_dict, f)


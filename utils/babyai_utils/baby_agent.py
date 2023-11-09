from abc import ABC, abstractmethod
import json
import torch
from .. import utils
#from random import Random


class Agent(ABC):
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def on_reset(self):
        pass

    @abstractmethod
    def get_action(self, obs):
        """Propose an action based on observation.

        Returns a dict, with 'action` entry containing the proposed action,
        and optionaly other entries containing auxiliary information
        (e.g. value function).

        """
        pass

    @abstractmethod
    def analyze_feedback(self, reward, done):
        pass


class ModelAgent(Agent):
    """A model-based agent. This agent behaves using a model."""

    def __init__(self, model_dir, obss_preprocessor, argmax, num_frames=None):
        if obss_preprocessor is None:
            assert isinstance(model_dir, str)
            obss_preprocessor = utils.ObssPreprocessor(model_dir, num_frames)
        self.obss_preprocessor = obss_preprocessor
        if isinstance(model_dir, str):
            self.model = utils.load_model(model_dir, num_frames)
            if torch.cuda.is_available():
                self.model.cuda()
        else:
            self.model = model_dir
        self.device = next(self.model.parameters()).device
        self.argmax = argmax
        self.memory = None

    def random_act_batch(self, many_obs):
        if self.memory is None:
            self.memory = torch.zeros(
                len(many_obs), self.model.memory_size, device=self.device)
        elif self.memory.shape[0] != len(many_obs):
            raise ValueError("stick to one batch size for the lifetime of an agent")
        preprocessed_obs = self.obss_preprocessor(many_obs, device=self.device)

        with torch.no_grad():
            raw_action = self.model.model_raw_action_space.sample()
            action = self.model.construct_final_action(raw_action[None, :])

        return action[0]

    def act_batch(self, many_obs):
        if self.memory is None:
            self.memory = torch.zeros(
                len(many_obs), self.model.memory_size, device=self.device)
        elif self.memory.shape[0] != len(many_obs):
            raise ValueError("stick to one batch size for the lifetime of an agent")
        preprocessed_obs = self.obss_preprocessor(many_obs, device=self.device)

        with torch.no_grad():
            dist, value, self.memory = self.model(preprocessed_obs, self.memory)
            if self.argmax:
                action = torch.stack([d.probs.argmax() for d in dist])[None, :]
            else:
                action = self.model.sample_action(dist)

            action = self.model.construct_final_action(action.cpu().numpy())

        return action[0]

    def get_action(self, obs):
        return self.act_batch([obs])

    def get_random_action(self, obs):
        return self.random_act_batch([obs])

    def analyze_feedback(self, reward, done):
        if isinstance(done, tuple):
            for i in range(len(done)):
                if done[i]:
                    self.memory[i, :] *= 0.
        else:
            self.memory *= (1 - done)

def load_agent(env, model_name, argmax=False, num_frames=None):
    # env_name needs to be specified for demo agents
    if model_name is not None:

        with open(model_name + "/config.json") as f:
            conf = json.load(f)
            text = conf['use_text']
            curr_dial = conf.get('use_current_dialogue_only', False)
            dial_hist = conf['use_dialogue']

        _, preprocess_obss = utils.get_obss_preprocessor(
            obs_space=env.observation_space,
            text=text,
            dialogue_current=curr_dial,
            dialogue_history=dial_hist
        )
        vocab = utils.get_status(model_name, num_frames)["vocab"]
        preprocess_obss.vocab.load_vocab(vocab)
        print("loaded vocabulary:", vocab.keys())
        return ModelAgent(model_name, preprocess_obss, argmax, num_frames)

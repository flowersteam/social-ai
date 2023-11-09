import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from utils.other import init_params

class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, use_dialogue=False, input_size=3):
        super().__init__()

        # store config
        self.config = locals()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.env_action_space = action_space
        self.model_raw_action_space = action_space
        self.input_size = input_size

        if use_dialogue:
            raise NotImplementedError("This model does not support dialogue inputs yet")

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(self.input_size, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.nvec[0])
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory, return_embeddings=False):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        if return_embeddings:
            return [dist], value, memory, None
        else:
            return [dist], value, memory

    # def sample_action(self, dist):
    #     return dist.sample()
    #
    # def calculate_log_probs(self, dist, action):
    #     return dist.log_prob(action)

    def calculate_action_gradient_masks(self, action):
        """Always train"""
        mask = torch.ones_like(action).detach()
        assert action.shape == mask.shape

        return mask

    def sample_action(self, dist):
        return torch.stack([d.sample() for d in dist], dim=1)

    def calculate_log_probs(self, dist, action):
        return torch.stack([d.log_prob(action[:, i]) for i, d in enumerate(dist)], dim=1)

    def calculate_action_masks(self, action):
        mask = torch.ones_like(action)
        assert action.shape == mask.shape

        return mask

    def construct_final_action(self, action):
        return action

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def get_config_dict(self):
        del self.config['__class__']
        self.config['self'] = str(self.config['self'])
        self.config['action_space'] = self.config['action_space'].nvec.tolist()
        return self.config


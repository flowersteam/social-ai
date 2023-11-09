import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac

from utils.other import init_params




class BlindTalkingMultiHeadedACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, use_dialogue=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_dialogue = use_dialogue
        self.use_memory = use_memory

        # multi dim
        if action_space.shape == ():
            raise ValueError("The action space is not multi modal. Use ACModel instead.")

        self.n_primitive_actions = action_space.nvec[0] + 1  # for talk
        self.talk_action = int(self.n_primitive_actions) - 1

        self.n_utterance_actions = action_space.nvec[1:]

        # in this model the talking is just finding one right thing to say
        self.utterance_actions_params = [
            torch.nn.Parameter(torch.ones(n)) for n in self.n_utterance_actions
        ]
        for i, p in enumerate(self.utterance_actions_params):
            self.register_parameter(
                name="utterance_p_{}".format(i),
                param=p
            )

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
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

        if self.use_text or self.use_dialogue:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)

        # Define text embedding
        if self.use_text:
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Define dialogue embedding
        if self.use_dialogue:
            self.dialogue_embedding_size = 128
            self.dialogue_rnn = nn.GRU(self.word_embedding_size, self.dialogue_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size

        if self.use_text:
            self.embedding_size += self.text_embedding_size

        if self.use_dialogue:
            self.embedding_size += self.dialogue_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.n_primitive_actions)
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

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)

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

        if self.use_dialogue:
            embed_dial = self._get_embed_dialogue(obs.dialogue)
            embedding = torch.cat((embedding, embed_dial), dim=1)

        x = self.actor(embedding)
        primtive_actions_dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        # construct utterance action distributions, for this model they are radndom
        utterance_actions_dists = [Categorical(logits=p.repeat(batch_size, 1)) for p in self.utterance_actions_params]
        # print("utterance params argmax: ", list(map(lambda x: int(x.argmax()), self.utterance_actions_params)))
        # print("utterance params", self.utterance_actions_params)

        dist = [primtive_actions_dist] + utterance_actions_dists

        return dist, value, memory

    def sample_action(self, dist):
        return torch.stack([d.sample() for d in dist], dim=1)

    def calculate_log_probs(self, dist, action):
        return torch.stack([d.log_prob(action[:, i]) for i, d in enumerate(dist)], dim=1)

    def calculate_action_masks(self, action):
        talk_mask = action[:, 0] == self.talk_action
        mask = torch.stack(
            (torch.ones_like(talk_mask), talk_mask, talk_mask),
            dim=1).detach()

        assert action.shape == mask.shape

        return mask
        # return torch.ones_like(mask).detach()

    def construct_final_action(self, action):
        act_mask = action[:, 0] != self.n_primitive_actions - 1

        nan_mask = np.array([
            np.array([1, np.nan, np.nan]) if t else np.array([np.nan, 1, 1]) for t in act_mask
        ])

        action = nan_mask*action

        return action

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def _get_embed_dialogue(self, dial):
        _, hidden = self.dialogue_rnn(self.word_embedding(dial))
        return hidden[-1]



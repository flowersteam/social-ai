import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch_ac

from utils.babyai_utils.supervised_losses import required_heads
import gym.spaces as spaces


 

def safe_relu(x):
    return torch.maximum(x, torch.zeros_like(x))

# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias

        # return F.relu(self.bn2(out)) # this causes an error in the new version of pytorch -> replaced by safe_relu
        return safe_relu(self.bn2(out))

class ImageBOWEmbedding(nn.Module):
   def __init__(self, space, embedding_dim):
       super().__init__()
       # self.max_value = max(space)
       self.max_value = 255  # 255, because of "no_point" encoding, which is encoded as 255
       self.space = space
       self.embedding_dim = embedding_dim
       self.embedding = nn.Embedding(self.space[-1] * self.max_value, embedding_dim)
       self.apply(initialize_parameters)

   def forward(self, inputs):
       offsets = torch.Tensor([x * self.max_value for x in range(self.space[-1])]).to(inputs.device)
       inputs = (inputs + offsets[None, :, None, None]).long()
       return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)

#notes: what they call instr is what we call text

#class ACModel(nn.Module, babyai.rl.RecurrentACModel):

# instr (them) == text (us)
class MultiModalBaby11ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space,
                 image_dim=128, memory_dim=128, text_dim=128, dialog_dim=128,
                 use_text=False, use_dialogue=False, use_current_dialogue_only=False, lang_model="gru", use_memory=False,
                 arch="bow_endpool_res", aux_info=None, num_films=2):
        super().__init__()

        # store config
        self.config = locals()

        # multi dim
        if action_space.shape == ():
            raise ValueError("The action space is not multi modal. Use ACModel instead.")

        if use_text:  # for now we do not consider goal conditioned policies
            raise ValueError("You should not use text but dialogue. --text is cheating.")

        endpool = 'endpool' in arch
        use_bow = 'bow' in arch
        pixel = 'pixel' in arch
        self.res = 'res' in arch

        # Decide which components are enabled
        self.use_text = use_text
        self.use_dialogue = use_dialogue
        self.use_current_dialogue_only = use_current_dialogue_only
        self.use_memory = use_memory
        self.arch = arch
        self.lang_model = lang_model
        self.aux_info = aux_info
        if self.res and image_dim != 128:
            raise ValueError(f"image_dim is {image_dim}, expected 128")
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.text_dim = text_dim
        self.dialog_dim = dialog_dim

        self.num_module = num_films
        self.n_primitive_actions = action_space.nvec[0] + 1  # not move action added
        self.move_switch_action = int(self.n_primitive_actions) - 1

        self.n_utterance_actions = np.concatenate(([2], action_space.nvec[1:]))  # binary to not speak
        self.talk_switch_subhead = 0

        self.env_action_space = action_space
        self.model_raw_action_space = spaces.MultiDiscrete([self.n_primitive_actions, *self.n_utterance_actions])

        self.obs_space = obs_space

        # transform given 3d obs_space into what babyai11 baseline uses, i.e. 1d embedding size
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        nb_img_channels = self.obs_space['image'][2]
        self.obs_space = ((n-1)//2-2)*((m-1)//2-2)*64

        for part in self.arch.split('_'):
            if part not in ['original', 'bow', 'pixels', 'endpool', 'res']:
                raise ValueError("Incorrect architecture name: {}".format(self.arch))

        # if not self.use_text:
        #     raise ValueError("FiLM architecture can be used when textuctions are enabled")
        self.image_conv = nn.Sequential(*[
            *([ImageBOWEmbedding(obs_space['image'], 128)] if use_bow else []),
            *([nn.Conv2d(
                in_channels=nb_img_channels, out_channels=128, kernel_size=(8, 8),
                stride=8, padding=0)] if pixel else []),
            nn.Conv2d(
                in_channels=128 if use_bow or pixel else nb_img_channels, out_channels=128,
                kernel_size=(3, 3) if endpool else (2, 2), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)]),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            *([] if endpool else [nn.MaxPool2d(kernel_size=(2, 2), stride=2)])
        ])
        self.film_pool = nn.MaxPool2d(kernel_size=(7, 7) if endpool else (2, 2), stride=2)

        # Define DIALOGUE embedding
        if self.use_dialogue or self.use_current_dialogue_only:
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                #self.word_embedding = nn.Embedding(obs_space["instr"], self.dialog_dim)
                self.word_embedding = nn.Embedding(obs_space["text"], self.dialog_dim)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.dialog_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.dialog_rnn = nn.GRU(
                        self.dialog_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    self.final_dialog_dim = self.dialog_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.dialog_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.dialog_dim)) for K in kernel_sizes])
                    self.final_dialog_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == 'attgru':
                self.memory2key = nn.Linear(self.memory_size, self.final_dialog_dim)

            self.controllers = []
            for ni in range(self.num_module):
                mod = FiLM(
                    in_features=self.final_dialog_dim,
                    out_features=128 if ni < self.num_module-1 else self.image_dim,
                    in_channels=128, imm_channels=128)
                self.controllers.append(mod)
                self.add_module('FiLM_' + str(ni), mod)

        # Define memory and resize image embedding
        self.embedding_size = self.image_dim
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)
            self.embedding_size = self.semi_memory_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.n_primitive_actions)
        )

        self.talker = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, n)
            ) for n in self.n_utterance_actions])

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

    def add_heads(self):
        '''
        When using auxiliary tasks, the environment yields at each step some binary, continous, or multiclass
        information. The agent needs to predict those information. This function add extra heads to the model
        that output the predictions. There is a head per extra information (the head type depends on the extra
        information type).
        '''
        self.extra_heads = nn.ModuleDict()
        for info in self.aux_info:
            if required_heads[info] == 'binary':
                self.extra_heads[info] = nn.Linear(self.embedding_size, 1)
            elif required_heads[info].startswith('multiclass'):
                n_classes = int(required_heads[info].split('multiclass')[-1])
                self.extra_heads[info] = nn.Linear(self.embedding_size, n_classes)
            elif required_heads[info].startswith('continuous'):
                if required_heads[info].endswith('01'):
                    self.extra_heads[info] = nn.Sequential(nn.Linear(self.embedding_size, 1), nn.Sigmoid())
                else:
                    raise ValueError('Only continous01 is implemented')
            else:
                raise ValueError('Type not supported')
            # initializing these parameters independently is done in order to have consistency of results when using
            # supervised-loss-coef = 0 and when not using any extra binary information
            self.extra_heads[info].apply(initialize_parameters)

    def add_extra_heads_if_necessary(self, aux_info):
        '''
        This function allows using a pre-trained model without aux_info and add aux_info to it and still make
        it possible to finetune.
        '''
        try:
            if not hasattr(self, 'aux_info') or not set(self.aux_info) == set(aux_info):
                self.aux_info = aux_info
                self.add_heads()
        except Exception:
            raise ValueError('Could not add extra heads')

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, dialog_embedding=None, return_embeddings=False):
        if self.use_dialogue and dialog_embedding is None:
            if not hasattr(obs, "utterance_history"):
                raise ValueError("The environment need's to be updated to 'utterance' and 'utterance_history' keys'")

            dialog_embedding = self._get_dialog_embedding(obs.utterance_history)

        elif self.use_current_dialogue_only and dialog_embedding is None:
            if not hasattr(obs, "utterance"):
                raise ValueError("The environment need's to be updated to 'utterance' and 'utterance_history' keys'")

            dialog_embedding = self._get_dialog_embedding(obs.utterance)

        if (self.use_dialogue or self.use_current_dialogue_only) and self.lang_model == "attgru":
            # outputs: B x L x D
            # memory: B x M
            #mask = (obs.instr != 0).float()
            mask = (obs.utterance_history != 0).float()
            # The mask tensor has the same length as obs.instr, and
            # thus can be both shorter and longer than instr_embedding.
            # It can be longer if instr_embedding is computed
            # for a subbatch of obs.instr.
            # It can be shorter if obs.instr is a subbatch of
            # the batch that instr_embeddings was computed for.
            # Here, we make sure that mask and instr_embeddings
            # have equal length along dimension 1.
            mask = mask[:, :dialog_embedding.shape[1]]
            dialog_embedding = dialog_embedding[:, :mask.shape[1]]

            keys = self.memory2key(memory)
            pre_softmax = (keys[:, None, :] * dialog_embedding).sum(2) + 1000 * mask
            attention = F.softmax(pre_softmax, dim=1)
            dialog_embedding = (dialog_embedding * attention[:, :, None]).sum(1)

        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)

        if 'pixel' in self.arch:
            x /= 256.0
        x = self.image_conv(x)
        if (self.use_dialogue or self.use_current_dialogue_only):
            for controller in self.controllers:
                out = controller(x, dialog_embedding)
                if self.res:
                    out += x
                x = out
        x = F.relu(self.film_pool(x))
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if hasattr(self, 'aux_info') and self.aux_info:
            extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
        else:
            extra_predictions = dict()

        # x = self.actor(embedding)
        # dist = Categorical(logits=F.log_softmax(x, dim=1))
        x = self.actor(embedding)
        primitive_actions_dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)
        utterance_actions_dists = [
            Categorical(logits=F.log_softmax(
                tal(embedding),
                dim=1,
            )) for tal in self.talker
        ]

        dist = [primitive_actions_dist] + utterance_actions_dists
        #return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}

        if return_embeddings:
            return dist, value, memory, embedding
        else:
            return dist, value, memory

    def _get_dialog_embedding(self, dialog):
        lengths = (dialog != 0).sum(1).long()
        if self.lang_model == 'gru':
            out, _ = self.dialog_rnn(self.word_embedding(dialog))
            hidden = out[range(len(lengths)), lengths-1, :]
            return hidden

        elif self.lang_model in ['bigru', 'attgru']:
            masks = (dialog != 0).float()

            if lengths.shape[0] > 1:
                seq_lengths, perm_idx = lengths.sort(0, descending=True)
                iperm_idx = torch.LongTensor(perm_idx.shape).fill_(0)
                if dialog.is_cuda: iperm_idx = iperm_idx.cuda()
                for i, v in enumerate(perm_idx):
                    iperm_idx[v.data] = i

                inputs = self.word_embedding(dialog)
                inputs = inputs[perm_idx]

                inputs = pack_padded_sequence(inputs, seq_lengths.data.cpu().numpy(), batch_first=True)

                outputs, final_states = self.dialog_rnn(inputs)
            else:
                dialog = dialog[:, 0:lengths[0]]
                outputs, final_states = self.dialog_rnn(self.word_embedding(dialog))
                iperm_idx = None
            final_states = final_states.transpose(0, 1).contiguous()
            final_states = final_states.view(final_states.shape[0], -1)
            if iperm_idx is not None:
                outputs, _ = pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[iperm_idx]
                final_states = final_states[iperm_idx]

            return outputs if self.lang_model == 'attgru' else final_states

        else:
            ValueError("Undefined lang_model architecture: {}".format(self.lang_model))

    # add action sampling to fit our interaction pipeline
    ## baby ai [[Categorical(logits: torch.Size([16, 8])), Categorical(logits: torch.Size([16, 2])), Categorical(logits: torch.Size([16, 2]))]]
    ## mh ac [Categorical(logits: torch.Size([16, 8])), Categorical(logits: torch.Size([16, 2])), Categorical(logits: torch.Size([16, 2]))]

    def det_action(self, dist):
        return torch.stack([d.probs.argmax(dim=-1) for d in dist], dim=1)

    def sample_action(self, dist):
        return torch.stack([d.sample() for d in dist], dim=1)


    def is_raw_action_speaking(self, action):
        is_speaking = action[:, 1:][:, self.talk_switch_subhead] == 1  # talking heads are [1:]
        return is_speaking

    def no_speak_to_speak_action(self, action):
        action[:, 1] = 1  # set speaking action to speak (1)

        assert all(self.is_raw_action_speaking(action))

        return action

    def raw_action_to_act_speak_mask(self, action):
        """
        Defines how the final action to be sent to the environment is computed
        Does NOT define how gradients are propagated, see calculate_action_gradient_masks() for that
        """

        assert action.shape[-1] == 4
        assert self.model_raw_action_space.shape[0] == action.shape[-1]

        act_mask = action[:, 0] != self.move_switch_action  # acting head is [0]
        # speak_mask = action[:, 1:][:, self.talk_switch_subhead] == 1  # talking heads are [1:]
        speak_mask = self.is_raw_action_speaking(action)
        return act_mask, speak_mask

    def construct_final_action(self, action):
        act_mask, speak_mask = self.raw_action_to_act_speak_mask(action)

        nan_mask = np.stack((act_mask, speak_mask, speak_mask), axis=1).astype(float)
        nan_mask[nan_mask == 0] = np.nan

        assert self.talk_switch_subhead == 0
        final_action = action[:, [True, False, True, True]]  # we drop the talk_switch_subhead
        final_action = nan_mask*final_action

        assert self.env_action_space.shape[0] == final_action.shape[-1]

        return final_action

    # add calculate log probs to fit our interaction pipeline
    def calculate_log_probs(self, dist, action):
        return torch.stack([d.log_prob(action[:, i]) for i, d in enumerate(dist)], dim=1)

    # add calculate action masks to fit our interaction pipeline
    def calculate_action_gradient_masks(self, action):
        """
        Defines how the gradients are propagated.
        Moving head is always trained.
        Speak switch is always trained.
        Grammar heads are trained only when speak switch is ON
        """
        _, speak_mask = self.raw_action_to_act_speak_mask(action)

        mask = torch.stack(
            (
                torch.ones_like(speak_mask),    # always train
                torch.ones_like(speak_mask),    # always train
                speak_mask,                     # train only when speaking
                speak_mask,                     # train only when speaking
            ), dim=1).detach()
        assert action.shape == mask.shape

        return mask

    def get_config_dict(self):
        del self.config['__class__']
        self.config['self'] = str(self.config['self'])
        self.config['action_space'] = self.config['action_space'].nvec.tolist()
        return self.config

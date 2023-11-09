from torch import nn
import torch
from torch.nn import functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class MinigridInverseDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MinigridInverseDynamicsNet, self).__init__()
        self.num_actions = num_actions

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * 128, 256)),
            nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(256, self.num_actions))

    def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=2)
        action_logits = self.id_out(self.inverse_dynamics(inputs))
        return action_logits

class MinigridForwardDynamicsNet(nn.Module):
    def __init__(self, num_actions):
        super(MinigridForwardDynamicsNet, self).__init__()
        self.num_actions = num_actions

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(128 + self.num_actions, 256)),
            nn.ReLU(),
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.fd_out = init_(nn.Linear(256, 128))

    def forward(self, state_embedding, action):
        action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        inputs = torch.cat((state_embedding, action_one_hot), dim=2)
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb


class MinigridStateEmbeddingNet(nn.Module):
    def __init__(self, observation_shape):
        super(MinigridStateEmbeddingNet, self).__init__()
        self.observation_shape = observation_shape

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.observation_shape[2], out_channels=32, kernel_size=(3, 3),
                            stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)),
            nn.ELU(),
        )

    def forward(self, inputs):
        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs
        T, B, *_ = x.shape

        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.

        x = x.float() / 255.0

        # -- [unroll_length*batch_size x channels x width x height]
        x = x.transpose(1, 3)
        x = self.feat_extract(x)

        state_embedding = x.view(T, B, -1)

        return state_embedding

def compute_forward_dynamics_loss(pred_next_emb, next_emb):
    forward_dynamics_loss = torch.norm(pred_next_emb - next_emb, dim=2, p=2)
    return torch.sum(torch.mean(forward_dynamics_loss, dim=1))

def compute_inverse_dynamics_loss(pred_actions, true_actions):
    inverse_dynamics_loss = F.nll_loss(
        F.log_softmax(torch.flatten(pred_actions, 0, 1), dim=-1),
        target=torch.flatten(true_actions, 0, 1),
        reduction='none')
    inverse_dynamics_loss = inverse_dynamics_loss.view_as(true_actions)
    return torch.sum(torch.mean(inverse_dynamics_loss, dim=1))

class LSTMMoaNet(nn.Module):
    def __init__(self, input_size, num_npc_prim_actions, acmodel, num_npc_utterance_actions=None, memory_dim=128):
        super(LSTMMoaNet, self).__init__()
        self.num_npc_prim_actions = num_npc_prim_actions
        self.num_npc_utterance_actions = num_npc_utterance_actions
        self.utterance_moa = num_npc_utterance_actions is not None
        self.input_size = input_size
        self.acmodel = acmodel


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.hidden_size = 128 # 256 in the original paper
        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(self.input_size, self.hidden_size)),
            nn.ReLU(),
        )

        self.memory_dim = memory_dim
        self.memory_rnn = nn.LSTMCell(self.hidden_size, self.memory_dim)
        self.embedding_size = self.semi_memory_size

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.fd_out_prim = init_(nn.Linear(self.embedding_size, self.num_npc_prim_actions))

        if self.utterance_moa:
            self.fd_out_utt = init_(nn.Linear(self.embedding_size, self.num_npc_utterance_actions))

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, embeddings, npc_previous_prim_actions, agent_actions, memory, npc_previous_utterance_actions=None):


        npc_previous_prim_actions_OH = F.one_hot(npc_previous_prim_actions, self.num_npc_prim_actions)

        if self.utterance_moa:
            npc_previous_utterance_actions_OH = F.one_hot(
                npc_previous_utterance_actions,
                self.num_npc_utterance_actions
            )

        # is_agent_speaking = self.acmodel.is_raw_action_speaking(agent_action[None, :])
        # assert len(is_agent_speaking) == 1
        # is_agent_speaking = is_agent_speaking[0]
        # enocde agents' action

        is_agent_speaking = self.acmodel.is_raw_action_speaking(agent_actions)

        # prim_action_OH_ = prim_action_OH[None, :].repeat([len(npc_previous_actions_OH), 1])
        # template_OH_ = template_OH[None, :].repeat([len(npc_previous_actions_OH), 1])
        # word_OH_ = word_OH[None, :].repeat([len(npc_previous_actions_OH), 1])

        prim_action_OH = F.one_hot(agent_actions[:, 0], self.acmodel.model_raw_action_space.nvec[0])
        template_OH = F.one_hot(agent_actions[:, 2], self.acmodel.model_raw_action_space.nvec[2])
        word_OH = F.one_hot(agent_actions[:, 3], self.acmodel.model_raw_action_space.nvec[3])

        # if not speaking make the templates 0
        template_OH = template_OH * is_agent_speaking[:, None]
        word_OH = word_OH * is_agent_speaking[:, None]

        if self.utterance_moa:
            inputs = torch.cat((
                embeddings,  # obs
                npc_previous_prim_actions_OH,  # npc
                npc_previous_utterance_actions_OH,
                prim_action_OH, template_OH, word_OH  # agent
            ), dim=1).float()

        else:
            inputs = torch.cat((
                embeddings,  # obs
                npc_previous_prim_actions_OH,  # npc
                prim_action_OH, template_OH, word_OH  # agent
            ), dim=1).float()

        outs_1 = self.forward_dynamics(inputs)

        # LSTM
        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(outs_1, hidden)

        embedding = hidden[0]

        memory = torch.cat(hidden, dim=1)

        outs_prim = self.fd_out_prim(embedding)

        if self.num_npc_utterance_actions:
            outs_utt = self.fd_out_utt(embedding)

            # cartesian product
            # outs = torch.bmm(outs_prim.unsqueeze(2), outs_utt.unsqueeze(1)).reshape(-1, self.num_npc_prim_actions*self.num_npc_utterance_actions)

            # outer sum
            outs = (outs_prim[..., None] + outs_utt[..., None, :]).reshape(-1, self.num_npc_prim_actions*self.num_npc_utterance_actions)
        else:
            outs = outs_prim

        return outs, memory

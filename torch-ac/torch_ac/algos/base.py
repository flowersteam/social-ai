from abc import ABC, abstractmethod
import numpy as np
import torch

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
from torch_ac.intrinsic_reward_models import *

from collections import Counter


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self,
                 envs,
                 acmodel,
                 device,
                 num_frames_per_proc,
                 discount,
                 lr,
                 gae_lambda,
                 entropy_coef,
                 value_loss_coef,
                 max_grad_norm,
                 recurrence,
                 preprocess_obss,
                 reshape_reward,
                 exploration_bonus=False,
                 exploration_bonus_params=None,
                 exploration_bonus_tanh=None,
                 expert_exploration_bonus=False,
                 exploration_bonus_type="lang",
                 episodic_exploration_bonus=True,
                 utterance_moa_net=True,  # used for social influence
                 clipped_rewards=False,
                 # default is set to fit RND
                 intrinsic_reward_loss_coef=0.1,  # also used for social influence
                 intrinsic_reward_coef=0.1,  # also used for social influence
                 intrinsic_reward_learning_rate=0.0001,
                 intrinsic_reward_momentum=0,
                 intrinsic_reward_epsilon=0.01,
                 intrinsic_reward_alpha=0.99,
                 intrinsic_reward_max_grad_norm=40,
                 intrinsic_reward_forward_loss_coef=10,
                 intrinsic_reward_inverse_loss_coef=0.1,
                 reset_rnd_ride_at_phase=False,
                 # social_influence
                 balance_moa_training=False,
                 moa_memory_dim=128,
    ):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.exploration_bonus = exploration_bonus
        self.expert_exploration_bonus = expert_exploration_bonus
        self.exploration_bonus_type = exploration_bonus_type
        self.episodic_exploration_bonus = episodic_exploration_bonus
        self.clipped_rewards = clipped_rewards
        self.update_epoch = 0
        self.utterance_moa_net = utterance_moa_net  # todo: as parameter

        self.reset_rnd_ride_at_phase = reset_rnd_ride_at_phase
        self.was_reset = False

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])

        self.info = [{}]*(shape[0])
        self.infos = [None]*(shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.next_masks = torch.zeros(*shape, device=self.device)

        self.values = torch.zeros(*shape, device=self.device)
        self.next_values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.extrinsic_rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)

        # as_shape = self.env.envs[0].action_space.shape
        as_shape = self.acmodel.model_raw_action_space.shape
        self.actions = torch.zeros(*(shape+as_shape), device=self.device, dtype=torch.int)
        self.log_probs = torch.zeros(*(shape+as_shape), device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_extrinsic_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_exploration_bonus = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_success_rate = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_curriculum_mean_perf = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_mission_string_observed = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_NPC_introduced_to = np.zeros(self.num_procs).astype(bool)
        self.log_episode_curriculum_param = torch.zeros(self.num_procs, device=self.device)

        self.intrinsic_reward_loss_coef = intrinsic_reward_loss_coef
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.intrinsic_reward_learning_rate = intrinsic_reward_learning_rate
        self.intrinsic_reward_momentum = intrinsic_reward_momentum
        self.intrinsic_reward_epsilon = intrinsic_reward_epsilon
        self.intrinsic_reward_alpha = intrinsic_reward_alpha
        self.intrinsic_reward_max_grad_norm = intrinsic_reward_max_grad_norm
        self.intrinsic_reward_forward_loss_coef = intrinsic_reward_forward_loss_coef
        self.intrinsic_reward_inverse_loss_coef = intrinsic_reward_inverse_loss_coef
        self.balance_moa_training = balance_moa_training
        self.moa_memory_dim = moa_memory_dim

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_extrinsic_return = [0] * self.num_procs
        self.log_exploration_bonus = [0] * self.num_procs
        self.log_success_rate = [0] * self.num_procs
        self.log_curriculum_max_mean_perf = [0] * self.num_procs
        self.log_curriculum_param = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs
        self.log_mission_string_observed = [0] * self.num_procs
        self.log_NPC_introduced_to = [False] * self.num_procs
        self.images_counter = [Counter() for _ in range(self.num_procs)]

        if self.exploration_bonus:
            self.visitation_counter = {}
            self.exploration_bonus_params = {}
            self.exploration_bonus_tanh = {}

            for i, bonus_type in enumerate(self.exploration_bonus_type):
                if bonus_type == "rnd":
                    assert not self.episodic_exploration_bonus
                    self.init_rnd_networks_and_optimizer()

                elif bonus_type == "ride":
                    self.init_ride_networks_and_optimizer()


                elif bonus_type == "soc_inf":

                    # npc actions
                    self.fn_name_to_npc_prim_act = self.env.envs[0].npc_prim_actions_dict

                    self.num_npc_prim_actions = len(self.fn_name_to_npc_prim_act)

                    self.npc_utterance_to_id = {a: i for i, a in enumerate(self.env.envs[0].all_npc_utterance_actions)}
                    self.num_npc_utterance_actions = len(self.npc_utterance_to_id)

                    if self.utterance_moa_net:
                        self.num_npc_all_actions = self.num_npc_prim_actions * self.num_npc_utterance_actions
                    else:
                        self.num_npc_all_actions = self.num_npc_prim_actions

                    # construct possible agent_action's list
                    self.all_possible_agent_actions, self.act_to_ind_dict = self.construct_all_possible_agent_actions()
                    self.agent_actions_tiled_all = None

                    im_shape = self.env.observation_space['image'].shape

                    embedding_size = self.acmodel.semi_memory_size

                    input_size = embedding_size \
                                    + self.num_npc_prim_actions \
                                    + self.acmodel.model_raw_action_space.nvec[0] \
                                    + self.acmodel.model_raw_action_space.nvec[2] \
                                    + self.acmodel.model_raw_action_space.nvec[3]

                    if self.utterance_moa_net:
                        input_size += self.num_npc_utterance_actions  # todo: feed as index or as text?

                    self.moa_net = LSTMMoaNet(
                        input_size=input_size,
                        num_npc_prim_actions=self.num_npc_prim_actions,
                        num_npc_utterance_actions=self.num_npc_utterance_actions if self.utterance_moa_net else None,
                        acmodel=self.acmodel,
                        memory_dim=self.moa_memory_dim
                    ).to(device=self.device)

                    # memory
                    assert shape == (self.num_frames_per_proc, self.num_procs)
                    self.moa_memory = torch.zeros(shape[1], self.moa_net.memory_size, device=self.device)
                    self.moa_memories = torch.zeros(*shape, self.moa_net.memory_size, device=self.device)

                elif bonus_type in ["cell", "grid", "lang"]:
                    if self.episodic_exploration_bonus:
                        self.visitation_counter[bonus_type] = [Counter() for _ in range(self.num_procs)]
                    else:
                        self.visitation_counter[bonus_type] = Counter()

                    if exploration_bonus_params:
                        self.exploration_bonus_params[bonus_type] = exploration_bonus_params[2*i:2*i+2]
                    else:
                        self.exploration_bonus_params[bonus_type] = (100, 50.)

                    if exploration_bonus_tanh is None:
                        self.exploration_bonus_tanh[bonus_type] = None
                    else:
                        self.exploration_bonus_tanh[bonus_type] = exploration_bonus_tanh[i]
                else:
                    raise ValueError(f"bonus type: {bonus_type} unknown.")

    def load_status_dict(self, status):

        self.acmodel.load_state_dict(status["model_state"])

        if hasattr(self.env, "curriculum") and self.env.curriculum is not None:
            self.env.curriculum.load_status_dict(status)
            self.env.broadcast_curriculum_parameters(self.env.curriculum.get_parameters())

        # self.optimizer.load_state_dict(status["optimizer_state"])

        if self.exploration_bonus:
            for i, bonus_type in enumerate(self.exploration_bonus_type):

                if bonus_type == "rnd":
                    self.random_target_network.load_state_dict(status["random_target_network"])
                    self.predictor_network.load_state_dict(status["predictor_network"])
                    self.intrinsic_reward_optimizer.load_state_dict(status["intrinsic_reward_optimizer"])

                elif bonus_type == "ride":
                    self.forward_dynamics_model.load_state_dict(status["forward_dynamics_model"])
                    self.inverse_dynamics_model.load_state_dict(status["inverse_dynamics_model"])
                    self.state_embedding_model.load_state_dict(status["state_embedding_model"])

                    self.state_embedding_optimizer.load_state_dict(status["state_embedding_optimizer"])
                    self.inverse_dynamics_optimizer.load_state_dict(status["inverse_dynamics_optimizer"])
                    self.forward_dynamics_optimizer.load_state_dict(status["forward_dynamics_optimizer"])

                elif bonus_type == "soc_inf":
                    self.moa_net.load_state_dict(status["moa_net"])

    def get_status_dict(self):

        algo_status_dict = {
            "model_state": self.acmodel.state_dict(),
        }

        if hasattr(self.env, "curriculum") and self.env.curriculum is not None:
            algo_status_dict = {
                **algo_status_dict,
                **self.env.curriculum.get_status_dict()
            }

        if self.exploration_bonus:
            for i, bonus_type in enumerate(self.exploration_bonus_type):

                if bonus_type == "rnd":
                    algo_status_dict["random_target_network"] = self.random_target_network.state_dict()
                    algo_status_dict["predictor_network"] = self.predictor_network.state_dict()
                    algo_status_dict["intrinsic_reward_optimizer"] = self.intrinsic_reward_optimizer.state_dict()

                elif bonus_type == "ride":
                    algo_status_dict["forward_dynamics_model"] = self.forward_dynamics_model.state_dict()
                    algo_status_dict["inverse_dynamics_model"] = self.inverse_dynamics_model.state_dict()
                    algo_status_dict["state_embedding_model"] = self.state_embedding_model.state_dict()

                    algo_status_dict["state_embedding_optimizer"] = self.state_embedding_optimizer.state_dict()
                    algo_status_dict["inverse_dynamics_optimizer"] = self.inverse_dynamics_optimizer.state_dict()
                    algo_status_dict["forward_dynamics_optimizer"] = self.forward_dynamics_optimizer.state_dict()

                elif bonus_type == "soc_inf":
                    algo_status_dict["moa_net"] = self.moa_net.state_dict()

        return algo_status_dict

    def construct_all_possible_agent_actions(self):

        if self.acmodel is None:
            raise ValueError("This should be called after the model has been set")

        # add non-speaking actions

        # a non-speaking actions look like (?, 0, 0, 0)
        # the last two zeros would normally mean the frst template and first word, but here they are to be
        # ignored because of the second 0 (which means to not speak)
        non_speaking_action_subspace = (self.acmodel.model_raw_action_space.nvec[0], 1, 1, 1)
        non_speaking_actions = np.array(list(np.ndindex(non_speaking_action_subspace)))

        # add speaking actions
        speaking_action_subspace = (
            self.acmodel.model_raw_action_space.nvec[0],
            1,  # one action,
            self.acmodel.model_raw_action_space.nvec[2],
            self.acmodel.model_raw_action_space.nvec[3],
        )

        speaking_actions = np.array(list(np.ndindex(speaking_action_subspace)))
        speaking_actions = self.acmodel.no_speak_to_speak_action(speaking_actions)

        # all actions
        all_possible_agent_actions = np.concatenate([non_speaking_actions, speaking_actions])

        # create the action -> index dict
        act_to_ind_dict = {tuple(act): ind for ind, act in enumerate(all_possible_agent_actions)}

        # map other non-speaking actions to the (?, 0, 0, 0), ex. (3, 0, 4, 12) -> (3, 0, 0, 0)
        other_non_speaking_action_subspace = (
            self.acmodel.model_raw_action_space.nvec[0],
            1,
            self.acmodel.model_raw_action_space.nvec[2],
            self.acmodel.model_raw_action_space.nvec[3]
        )
        for action in np.ndindex(other_non_speaking_action_subspace):
            assert action[1] == 0  # non-speaking
            act_to_ind_dict[tuple(action)] = act_to_ind_dict[(action[0], 0, 0, 0)]

        return all_possible_agent_actions, act_to_ind_dict

    def step_to_n_frames(self, step):
        return step * self.num_frames_per_proc * self.num_procs

    def calculate_exploration_bonus(self, obs=None, done=None, prev_obs=None, info=None, prev_info=None, agent_actions=None, dist=None,
                                    i_step=None, embeddings=None):

        def state_hashes(observation, exploration_bonus_type):
            if exploration_bonus_type == "lang":
                hashes = [observation['utterance']]
                assert len(hashes) == 1
            elif exploration_bonus_type == "cell":
                # for all new cells
                im = observation["image"]
                hashes = np.unique(im.reshape(-1, im.shape[-1]), axis=0)
                hashes = np.apply_along_axis(lambda a: a.data.tobytes(), 1, hashes)

            elif exploration_bonus_type == "grid":
                # for seeing new grid configurations
                im = observation["image"]
                hashes = [im.data.tobytes()]
                assert len(hashes) == 1
            else:
                raise ValueError(f"Unknown exploration bonus type {bonus_type}")

            return hashes

        total_bonus = [0]*len(obs)
        for bonus_type in self.exploration_bonus_type:
            if bonus_type == "rnd":
                # -- [unroll_length x batch_size x height x width x channels] == [1, n_proc, 7, 7, 4]
                batch = torch.tensor(np.array([[o['image'] for o in obs]])).to(self.device)

                with torch.no_grad():
                    random_embedding = self.random_target_network(batch).reshape(len(obs), 128)
                    predicted_embedding = self.predictor_network(batch).reshape(len(obs), 128)
                    intrinsic_rewards = torch.norm(predicted_embedding.detach() - random_embedding.detach(), dim=1, p=2)
                    intrinsic_reward_coef = self.intrinsic_reward_coef
                    intrinsic_rewards *= intrinsic_reward_coef

                    # is this the best way? should we somehow extract the next_state?
                    bonus = [0.0 if d else float(r) for d, r in zip(done, intrinsic_rewards)]

            elif bonus_type == "ride":
                with torch.no_grad():
                    _obs = torch.tensor(np.array([[o['image'] for o in prev_obs]])).to(self.device)
                    _next_obs = torch.tensor(np.array([[o['image'] for o in obs]])).to(self.device)

                    # counts - number of times a state was seen during the SAME episode -> can be computed here
                    count_rewards = torch.tensor([1/np.sqrt(self.images_counter[p_i][np.array(o.to("cpu")).tobytes()]) for p_i, o in enumerate(_next_obs[0])]).to(self.device)
                    assert not any(torch.isinf(count_rewards))

                    state_emb = self.state_embedding_model(_obs.to(device=self.device)).reshape(len(obs), 128)
                    next_state_emb = self.state_embedding_model(_next_obs.to(device=self.device)).reshape(len(obs), 128)

                    control_rewards = torch.norm(next_state_emb - state_emb, dim=1, p=2)

                    intrinsic_rewards = self.intrinsic_reward_coef*(count_rewards * control_rewards)

                    # is this the best way? should we somehow extract the next_state?
                    bonus = [0.0 if d else float(r) for d, r in zip(done, intrinsic_rewards)]

            elif bonus_type == "soc_inf":
                if prev_info == [{}] * len(prev_info):
                    # this is the first step, info is not given during reset

                    # first step in the episode no influence can be estimated as there is no previous action
                    # todo: padd with zeros, and estimate anyway?
                    bonus = [0.0 for _ in done]
                else:
                    # social influence
                    n_procs = len(obs)

                    _prev_NPC_prim_actions = torch.tensor(
                        [self.fn_name_to_npc_prim_act[o["NPC_prim_action"]] for o in prev_info]
                    ).to(self.device)

                    # todo: what is the best way to feed utt action?
                    _prev_NPC_utt_actions = torch.tensor(
                        [self.npc_utterance_to_id[o["NPC_utterance"]] for o in prev_info]
                    ).to(self.device)

                    # new
                    # calculate counterfactuals
                    npc_previous_prim_actions_all = _prev_NPC_prim_actions.repeat(len(self.all_possible_agent_actions)) # [A_ag*n_procs, ...]
                    npc_previous_utt_actions_all = _prev_NPC_utt_actions.repeat(len(self.all_possible_agent_actions)) # [A_ag*n_procs, ...]

                    # agent actions tiled
                    if self.agent_actions_tiled_all is not None:
                        agent_actions_tiled_all = self.agent_actions_tiled_all

                    else:
                        # only first time, we can't do it in init because we need len(im_obs)
                        agent_actions_tiled_all = []
                        for pot_agent_action in self.all_possible_agent_actions:
                            pot_agent_action_tiled = torch.from_numpy(np.tile(pot_agent_action, (n_procs, 1))) # [n_procs,...]
                            agent_actions_tiled_all.append(pot_agent_action_tiled.to(self.device))

                        agent_actions_tiled_all = torch.concat(agent_actions_tiled_all)  # [A_ag*n_procs,....]

                        self.agent_actions_tiled_all = agent_actions_tiled_all

                    with torch.no_grad():
                        # todo: move this tiling above?
                        masked_memory = self.moa_memory * self.mask.unsqueeze(1)
                        masked_memory_tiled_all = masked_memory.repeat([len(self.all_possible_agent_actions), 1])
                        embedding_tiled_all = embeddings.repeat([len(self.all_possible_agent_actions), 1])

                        # use current memory for every action

                        counterfactuals_logits, moa_memory = self.moa_net(
                            embeddings=embedding_tiled_all,
                            # observations=observations_all,
                            npc_previous_prim_actions=npc_previous_prim_actions_all,
                            npc_previous_utterance_actions=npc_previous_utt_actions_all if self.utterance_moa_net else None,
                            agent_actions=agent_actions_tiled_all,
                            memory=masked_memory_tiled_all
                        )  # logits : [A_ag * n_procs, A_npc]

                        counterfactuals_logits = counterfactuals_logits.reshape(
                            [len(self.all_possible_agent_actions), n_procs, self.num_npc_all_actions])

                        counterfactuals_logits = counterfactuals_logits.swapaxes(0, 1) # [n_procs, A_ag, A_npc]

                    assert counterfactuals_logits.shape == (len(obs), len(self.all_possible_agent_actions), self.num_npc_all_actions)

                    # compute npc logits p(A_npc|A_ag, s)

                    # note: ex (5,0,5,2) is mapped to (5,0,0,0), todo: is this ok everywhere?
                    agent_action_indices = [self.act_to_ind_dict[tuple(act.cpu().numpy())] for act in agent_actions]
                    # ~ p(a_npc| a_ag, ...)

                    predicted_logits = torch.stack([ctr[ind] for ctr, ind in zip(counterfactuals_logits, agent_action_indices)])

                    assert i_step is not None
                    self.moa_memories[i_step] = self.moa_memory

                    # only save for the actions actually taken
                    self.moa_memory = moa_memory[agent_action_indices]

                    assert predicted_logits.shape == (len(obs), self.num_npc_all_actions)

                    predicted_probs = torch.softmax(predicted_logits, dim=1)  # use exp_softmax or something?


                    # compute marginal npc logits p(A_npc|s) = sum( p(A_NPC|A_ag,s), for every A_ag )
                    # compute agent logits for all possible agent actions
                    per_non_speaking_action_log_probs = dist[0].logits + dist[1].logits[:, :1]

                    per_speaking_action_log_probs = []
                    for p in range(n_procs):

                        log_probs_for_proc_p = [d.logits[p].cpu().numpy() for d in dist]

                        # speaking actions
                        speaking_log_probs = log_probs_for_proc_p
                        speaking_log_probs[1] = speaking_log_probs[1][1:]  # only the speak action

                        # sum everybody with everybody
                        out = np.add.outer(speaking_log_probs[0], speaking_log_probs[1]).reshape(-1)
                        out = np.add.outer(out, speaking_log_probs[2]).reshape(-1)
                        out = np.add.outer(out, speaking_log_probs[3]).reshape(-1)
                        per_speaking_action_log_probs_proc_p = out

                        per_speaking_action_log_probs.append(per_speaking_action_log_probs_proc_p)

                    per_speaking_action_log_probs = np.stack(per_speaking_action_log_probs)

                    agent_log_probs = torch.concat([
                        per_non_speaking_action_log_probs,
                        torch.tensor(per_speaking_action_log_probs).to(device=self.device),
                    ], dim=1)

                    # assert
                    for p in range(n_procs):
                        log_probs_for_proc_p = [d.logits[p].cpu().numpy() for d in dist]

                        assert torch.abs(agent_log_probs[p][self.act_to_ind_dict[(0, 1, 3, 1)]] - sum([p[a] for p, a in list(zip(log_probs_for_proc_p, (0, 1, 3, 1)))])) < 1e-5
                        assert torch.abs(agent_log_probs[p][self.act_to_ind_dict[(0, 1, 1, 10)]] - sum([p[a] for p, a in list(zip(log_probs_for_proc_p, (0, 1, 1, 10)))])) < 1e-5


                    agent_probs = agent_log_probs.exp()

                    counterfactuals_probs = counterfactuals_logits.softmax(dim=-1)  # [n_procs, A_ag, A_npc]
                    counterfactuals_perm = counterfactuals_probs.permute(0, 2, 1)  # [n_procs, A_npc, A_agent]

                    # compute marginal distributions
                    marginals = (counterfactuals_perm * agent_probs[:, None, :]).sum(-1)

                    # this already sums to one, so the following normalization is not needed
                    marginal_probs = marginals / marginals.sum(1, keepdims=True)  # sum over npc_actions
                    assert marginal_probs.shape == (n_procs, self.num_npc_all_actions)  # [batch, A_npc]

                    KL_loss = (predicted_probs * (predicted_probs.log() - marginal_probs.log())).sum(axis=-1)


                    intrinsic_rewards = self.intrinsic_reward_coef * KL_loss

                    # is the NPC observed in the image that is fed as input in this step
                    # (returned by the previous step() call )
                    NPC_observed = torch.tensor([pi["NPC_observed"] for pi in prev_info]).to(self.device)

                    intrinsic_rewards = intrinsic_rewards * NPC_observed

                    bonus = [0.0 if d else float(r) for d, r in zip(done, intrinsic_rewards)]

            elif bonus_type in ["cell", "grid", "lang"]:
                C, M = self.exploration_bonus_params[bonus_type]
                C_ = C / self.num_frames_per_proc

                if self.expert_exploration_bonus:
                    # expert
                    raise DeprecationWarning("Deprecated exploration bonus type")

                elif self.episodic_exploration_bonus:

                    hashes = [state_hashes(o, bonus_type) for o in obs]
                    bonus = [
                        0 if d else  # no bonus if done
                        np.sum([
                            C_ / ((self.visitation_counter[bonus_type][i_p][h] + 1) ** M) for h in hs
                        ])
                        for i_p, (hs, d) in enumerate(zip(hashes, done))
                    ]

                    # update the counters
                    for i_p, (o, d, hs) in enumerate(zip(obs, done, hashes)):
                        if not d:
                            for h in hs:
                                self.visitation_counter[bonus_type][i_p][h] += 1

                else:
                    raise DeprecationWarning("Use episodic exploration bonus.")
                    # non-episodic exploration bonus

                    bonus = [
                        0 if d else  # no bonus if done
                        np.sum([
                        C_ / ((self.visitation_counter[bonus_type][h] + 1) ** M) for h in state_hashes(o. bonus_type)
                        ]) for o, d in zip(obs, done)
                    ]

                    # update the counters
                    for o, d in zip(obs, done):
                        if not d:
                            for h in state_hashes(o, self.exploration_bonus_type):
                                self.visitation_counter[bonus_type][h] += 1

                if self.exploration_bonus_tanh[bonus_type] is not None:
                    bonus = [np.tanh(b)*self.exploration_bonus_tanh[bonus_type] for b in bonus]
            else:
                raise ValueError(f"Unknown exploration bonus type {bonus_type}")

            assert len(total_bonus) == len(bonus)
            total_bonus = [tb+b for tb, b in zip(total_bonus, bonus)]

        return total_bonus

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i_step in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():

                if self.acmodel.recurrent:
                    dist, value, memory, policy_embedding = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1), return_embeddings=True)
                else:
                    dist, value, policy_embedding = self.acmodel(preprocessed_obs, return_embeddings=True)

            action = self.acmodel.sample_action(dist)

            obs, reward, done, info = self.env.step(
                self.acmodel.construct_final_action(
                    action.cpu().numpy()
                )
            )

            if hasattr(self.env, "curriculum") and self.env.curriculum is not None:
                curriculum_params = self.env.curriculum.update_parameters({
                    "obs": obs,
                    "reward": reward,
                    "done": done,
                    "info": info,
                })
                # broadcast new parameters to all parallel environments
                self.env.broadcast_curriculum_parameters(curriculum_params)

                if self.reset_rnd_ride_at_phase and curriculum_params['phase'] == 2 and not self.was_reset:
                    self.was_reset = True
                    assert not self.episodic_exploration_bonus

                    for i, bonus_type in enumerate(self.exploration_bonus_type):
                        if bonus_type == "rnd":
                            self.init_rnd_networks_and_optimizer()

                        elif bonus_type == "ride":
                            self.init_ride_networks_and_optimizer()

            for p_i, o in enumerate(obs):
                self.images_counter[p_i][o['image'].tobytes()] += 1

            extrinsic_reward = reward
            exploration_bonus = (0,) * len(reward)

            if self.exploration_bonus:
                bonus = self.calculate_exploration_bonus(
                    obs=obs, done=done, prev_obs=self.obs, info=info, prev_info=self.info, agent_actions=action, dist=dist,
                    i_step=i_step, embeddings=policy_embedding,
                )
                exploration_bonus = bonus
                reward = [r + b for r, b in zip(reward, bonus)]

            if self.clipped_rewards:
                # this should not be used with classic count-based rewards as they often,
                # when combined with extr. rew go past 1.0
                reward = list(map(float, torch.clamp(torch.tensor(reward), -1, 1)))

            # Update experiences values
            self.obss[i_step] = self.obs
            self.obs = obs
            self.infos[i_step] = info  # info of this step is the current info
            self.info = info  # save as previous info

            if self.acmodel.recurrent:
                self.memories[i_step] = self.memory
                self.memory = memory
            self.masks[i_step] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)

            self.actions[i_step] = action
            self.values[i_step] = value

            if self.reshape_reward is not None:
                self.rewards[i_step] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i_step] = torch.tensor(reward, device=self.device)

            self.log_probs[i_step] = self.acmodel.calculate_log_probs(dist, action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_extrinsic_return += torch.tensor(extrinsic_reward, device=self.device, dtype=torch.float)
            self.log_episode_exploration_bonus += torch.tensor(exploration_bonus, device=self.device, dtype=torch.float)
            self.log_episode_success_rate = torch.tensor([i["success"] for i in info]).float().to(self.device)
            self.log_episode_curriculum_mean_perf = torch.tensor([i.get("curriculum_info_max_mean_perf", 0) for i in info]).float().to(self.device)
            self.log_episode_reshaped_return += self.rewards[i_step]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)
            self.log_episode_curriculum_param = torch.tensor([i.get("curriculum_info_param", 0.0) for i in info]).float().to(self.device)
            # self.log_episode_curriculum_param = torch.tensor([i.get("curriculum_info_mean_perf", 0.0) for i in info]).float().to(self.device)
            assert self.log_episode_curriculum_param.var() == 0

            log_episode_NPC_introduced_to_current = np.array([i.get('NPC_was_introduced_to', False) for i in info])
            assert all((self.log_episode_NPC_introduced_to | log_episode_NPC_introduced_to_current) == log_episode_NPC_introduced_to_current)

            self.log_episode_NPC_introduced_to = self.log_episode_NPC_introduced_to | log_episode_NPC_introduced_to_current

            self.log_episode_mission_string_observed += torch.tensor([
                float(m in o.get("utterance", ''))
                for m, o in zip(self.env.get_mission(), self.obs)
            ], device=self.device, dtype=torch.float)

            for p, done_ in enumerate(done):
                if done_:
                    self.log_mission_string_observed.append(
                        torch.clamp(self.log_episode_mission_string_observed[p], 0, 1).item()
                    )
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[p].item())
                    self.log_extrinsic_return.append(self.log_episode_extrinsic_return[p].item())
                    self.log_exploration_bonus.append(self.log_episode_exploration_bonus[p].item())
                    self.log_success_rate.append(self.log_episode_success_rate[p].item())
                    self.log_curriculum_max_mean_perf.append(self.log_episode_curriculum_mean_perf[p].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[p].item())
                    self.log_num_frames.append(self.log_episode_num_frames[p].item())
                    self.log_curriculum_param.append(self.log_episode_curriculum_param[p].item())
                    if self.episodic_exploration_bonus:
                        for v in self.visitation_counter.values():
                            v[p] = Counter()
                    self.images_counter[p] = Counter()
                    self.log_NPC_introduced_to.append(self.log_episode_NPC_introduced_to[p])
                    # print("log history:", self.log_success_rate)
                    # print("log history len:", len(self.log_success_rate)-16)

            self.log_episode_mission_string_observed *= self.mask
            self.log_episode_return *= self.mask
            self.log_episode_extrinsic_return *= self.mask
            self.log_episode_exploration_bonus *= self.mask
            self.log_episode_success_rate *= self.mask
            self.log_episode_curriculum_mean_perf *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask
            self.log_episode_NPC_introduced_to *= self.mask.cpu().numpy().astype(bool)
            self.log_episode_curriculum_param *= self.mask
        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)
        for f in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[f+1] if f < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[f+1] if f < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[f+1] if f < self.num_frames_per_proc - 1 else 0

            self.next_masks[f] = next_mask
            self.next_values[f] = next_value

            delta = self.rewards[f] + self.discount * next_value * next_mask - self.values[f]
            self.advantages[f] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[f][p]
                    for p in range(self.num_procs)
                    for f in range(self.num_frames_per_proc)]

        exps.infos = np.array([self.infos[f][p]
                    for p in range(self.num_procs)
                    for f in range(self.num_frames_per_proc)])

        # obs: (p1 (f1,f2,f3) ; p2 (f1,f2,f3); p3 (f1,f2,f3)

        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
            exps.next_mask = self.next_masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        if self.exploration_bonus and "soc_inf" in self.exploration_bonus_type:
            exps.moa_memory = self.moa_memories.transpose(0, 1).reshape(-1, *self.moa_memories.shape[2:])

        # for all tensors below, T x P -> P x T -> P * T

        exps.action = self.actions.transpose(0, 1).reshape((-1, self.actions.shape[-1]))
        exps.log_prob = self.log_probs.transpose(0, 1).reshape((-1, self.actions.shape[-1]))

        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.next_value = self.next_values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        flat_actions = self.actions.reshape(-1, self.actions.shape[-1])
        action_modalities = {
            "action_modality_{}".format(m): flat_actions[:, m].cpu().numpy() for m in range(self.actions.shape[-1])
        }

        if not self.exploration_bonus:
            assert self.log_return == self.log_extrinsic_return

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "mission_string_observed": self.log_mission_string_observed[-keep:],
            "extrinsic_return_per_episode": self.log_extrinsic_return[-keep:],
            "exploration_bonus_per_episode": self.log_exploration_bonus[-keep:],
            "success_rate_per_episode": self.log_success_rate[-keep:],
            "curriculum_max_mean_perf_per_episode": self.log_curriculum_max_mean_perf[-keep:],
            "curriculum_param_per_episode": self.log_curriculum_param[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "NPC_introduced_to": self.log_NPC_introduced_to[-keep:],
            **action_modalities
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_extrinsic_return = self.log_extrinsic_return[-self.num_procs:]
        self.log_exploration_bonus = self.log_exploration_bonus[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    def compute_advantages_and_returnn(self, exps):
        """
        This function can be used for algorithms which reuse old data (not online RL) to
        recompute non episodic intrinsic rewards on old experience.
        This method is not used in PPO training.

        Example usage from update_parameters
        advs, retnn = self.compute_advantages_and_returnn(exps)

        # if you want to do a sanity check
        assert torch.equal(exps.advantage, advs)
        assert torch.equal(exps.returnn, retnn)

        exps.advantages, exps.returnn = advs, retnn
        """
        shape = (self.num_frames_per_proc, self.num_procs)
        advs = torch.zeros(*shape, device=self.device)

        rewards = exps.reward.reshape(self.num_procs, self.num_frames_per_proc).transpose(0, 1)
        values = exps.value.reshape(self.num_procs, self.num_frames_per_proc).transpose(0, 1)
        next_values = exps.next_value.reshape(self.num_procs, self.num_frames_per_proc).transpose(0, 1)
        next_masks = exps.next_mask.reshape(self.num_procs, self.num_frames_per_proc).transpose(0, 1)

        for f in reversed(range(self.num_frames_per_proc)):
            next_advantage = advs[f+1] if f < self.num_frames_per_proc - 1 else 0

            delta = rewards[f] + self.discount * next_values[f] * next_masks[f] - values[f]
            advs[f] = delta + self.discount * self.gae_lambda * next_advantage * next_masks[f]

        advantage = advs.transpose(0, 1).reshape(-1)
        returnn = exps.value + advantage
        return advantage, returnn

    @abstractmethod
    def update_parameters(self):
        pass

    def init_rnd_networks_and_optimizer(self):
        self.random_target_network = MinigridStateEmbeddingNet(self.env.observation_space['image'].shape).to(
            device=self.device)
        self.predictor_network = MinigridStateEmbeddingNet(self.env.observation_space['image'].shape).to(device=self.device)

        self.intrinsic_reward_optimizer = torch.optim.RMSprop(
            self.predictor_network.parameters(),
            lr=self.intrinsic_reward_learning_rate,
            momentum=self.intrinsic_reward_momentum,
            eps=self.intrinsic_reward_epsilon,
            alpha=self.intrinsic_reward_alpha,
        )

    def init_ride_networks_and_optimizer(self):
        self.state_embedding_model = MinigridStateEmbeddingNet(self.env.observation_space['image'].shape).to(
            device=self.device)
        # linquistic actions
        # n_actions = self.acmodel.model_raw_action_space.nvec.prod

        # we only use primitive actions for ride
        n_actions = self.acmodel.model_raw_action_space.nvec[0]

        self.forward_dynamics_model = MinigridForwardDynamicsNet(n_actions).to(device=self.device)
        self.inverse_dynamics_model = MinigridInverseDynamicsNet(n_actions).to(device=self.device)

        self.state_embedding_optimizer = torch.optim.RMSprop(
            self.state_embedding_model.parameters(),
            lr=self.intrinsic_reward_learning_rate,
            momentum=self.intrinsic_reward_momentum,
            eps=self.intrinsic_reward_epsilon,
            alpha=self.intrinsic_reward_alpha)

        self.inverse_dynamics_optimizer = torch.optim.RMSprop(
            self.inverse_dynamics_model.parameters(),
            lr=self.intrinsic_reward_learning_rate,
            momentum=self.intrinsic_reward_momentum,
            eps=self.intrinsic_reward_epsilon,
            alpha=self.intrinsic_reward_alpha)

        self.forward_dynamics_optimizer = torch.optim.RMSprop(
            self.forward_dynamics_model.parameters(),
            lr=self.intrinsic_reward_learning_rate,
            momentum=self.intrinsic_reward_momentum,
            eps=self.intrinsic_reward_epsilon,
            alpha=self.intrinsic_reward_alpha)

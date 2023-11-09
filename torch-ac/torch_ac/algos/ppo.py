import numpy
import torch
import torch.nn.functional as F
from torch_ac.intrinsic_reward_models import compute_forward_dynamics_loss, compute_inverse_dynamics_loss
from sklearn.metrics import f1_score

from torch_ac.algos.base import BaseAlgo

def compute_balance_mask(target, n_classes):
    if target.float().var() == 0:
        # all the same class, don't train at all
        return torch.zeros_like(target).detach()

    # compute the balance mask
    per_class_n = torch.bincount(target, minlength=n_classes)

    # number of times the least common class (that appeared) appeared
    n_for_each_class = per_class_n[torch.nonzero(per_class_n)].min()

    # undersample other classes
    per_class_n = n_for_each_class  # sample each class that many times

    balanced_indexes_ = []

    for c in range(n_classes):
        c_indexes = torch.where(target == c)[0]
        if len(c_indexes) == 0:
            continue

        # c_sampled_indexes = c_indexes[torch.randint(len(c_indexes), (per_class_n,))]
        c_sampled_indexes = c_indexes[torch.randperm(len(c_indexes))[:per_class_n]]
        balanced_indexes_.append(c_sampled_indexes)

    balanced_indexes = torch.concat(balanced_indexes_)
    balance_mask = torch.zeros_like(target)
    balance_mask[balanced_indexes] = 1.0

    return balance_mask.detach()


class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None, exploration_bonus=False, exploration_bonus_params=None,
                 expert_exploration_bonus=False, episodic_exploration_bonus=True, exploration_bonus_type="lang",
                 exploration_bonus_tanh=None, clipped_rewards=False, intrinsic_reward_epochs=0,
                 # default is set to fit RND
                 intrinsic_reward_coef=0.1,
                 intrinsic_reward_learning_rate=0.0001,
                 intrinsic_reward_momentum=0,
                 intrinsic_reward_epsilon=0.01,
                 intrinsic_reward_alpha=0.99,
                 intrinsic_reward_max_grad_norm=40,
                 intrinsic_reward_loss_coef=0.1,
                 intrinsic_reward_forward_loss_coef=10,
                 intrinsic_reward_inverse_loss_coef=0.1,
                 reset_rnd_ride_at_phase=False,
                 balance_moa_training=False,
                 moa_memory_dim=128,
                 schedule_lr=False,
                 lr_schedule_end_frames=0,
                 end_lr=0.0,
    ):
        num_frames_per_proc = num_frames_per_proc or 128

        # save config
        self.config = locals()

        super().__init__(
            envs=envs,
            acmodel=acmodel,
            device=device,
            num_frames_per_proc=num_frames_per_proc,
            discount=discount,
            lr=lr,
            gae_lambda=gae_lambda,
            entropy_coef=entropy_coef,
            value_loss_coef=value_loss_coef,
            max_grad_norm=max_grad_norm,
            recurrence=recurrence,
            preprocess_obss=preprocess_obss,
            reshape_reward=reshape_reward,
            exploration_bonus=exploration_bonus,
            expert_exploration_bonus=expert_exploration_bonus,
            episodic_exploration_bonus=episodic_exploration_bonus,
            exploration_bonus_params=exploration_bonus_params,
            exploration_bonus_tanh=exploration_bonus_tanh,
            exploration_bonus_type=exploration_bonus_type,
            clipped_rewards=clipped_rewards,
            intrinsic_reward_loss_coef=intrinsic_reward_loss_coef,
            intrinsic_reward_coef=intrinsic_reward_coef,
            intrinsic_reward_learning_rate=intrinsic_reward_learning_rate,
            intrinsic_reward_momentum=intrinsic_reward_momentum,
            intrinsic_reward_epsilon=intrinsic_reward_epsilon,
            intrinsic_reward_alpha=intrinsic_reward_alpha,
            intrinsic_reward_max_grad_norm=intrinsic_reward_max_grad_norm,
            intrinsic_reward_forward_loss_coef=intrinsic_reward_forward_loss_coef,
            intrinsic_reward_inverse_loss_coef=intrinsic_reward_inverse_loss_coef,
            balance_moa_training=balance_moa_training,
            moa_memory_dim=moa_memory_dim,
            reset_rnd_ride_at_phase=reset_rnd_ride_at_phase,
        )

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.intrinsic_reward_epochs = intrinsic_reward_epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        if self.exploration_bonus and "soc_inf" in self.exploration_bonus_type:
            adam_params = list(dict.fromkeys(list(self.acmodel.parameters()) + list(self.moa_net.parameters())))
            self.optimizer = torch.optim.Adam(adam_params, lr, eps=adam_eps)

        else:
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)

        self.schedule_lr = schedule_lr

        self.lr_schedule_end_frames = lr_schedule_end_frames

        assert end_lr <= lr
        def lr_lambda(step):
            if self.lr_schedule_end_frames == 0:
                # no schedule
                return 1

            end_factor = end_lr/lr
            final_diminished_factor = 1-end_factor
            n_frames = self.step_to_n_frames(step)
            return 1 - (min(n_frames, self.lr_schedule_end_frames) / self.lr_schedule_end_frames) * final_diminished_factor

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.batch_num = 0

    def load_status_dict(self, status):
        super().load_status_dict(status)

        if "optimizer_state" in status:
            self.optimizer.load_state_dict(status["optimizer_state"])

        if "lr_scheduler_state" in status:
            self.lr_scheduler.load_state_dict(status["lr_scheduler_state"])

    def get_status_dict(self):

        status_dict = super().get_status_dict()

        status_dict["optimizer_state"] = self.optimizer.state_dict()

        status_dict["lr_scheduler_state"] = self.lr_scheduler.state_dict()

        return status_dict

    def update_parameters(self, exps):
        # Collect experiences

        self.acmodel.train()

        self.update_epoch += 1

        intr_rew_perf = torch.tensor(0.0)
        intr_rew_perf_ = 0.0

        social_influence = False

        if self.exploration_bonus:
            if "rnd" in self.exploration_bonus_type:
                imgs = exps.obs.image.reshape(
                    self.num_procs, self.num_frames_per_proc, *exps.obs.image.shape[1:]
                ).transpose(0, 1)
                mask = exps.mask.reshape(
                    self.num_procs, self.num_frames_per_proc, 1,
                ).transpose(0, 1)

                self.random_target_network.train()
                self.predictor_network.train()

                random_embedding = self.random_target_network(imgs).reshape(self.num_frames_per_proc, self.num_procs, 128)
                predicted_embedding = self.predictor_network(imgs).reshape(self.num_frames_per_proc, self.num_procs, 128)
                intr_rew_loss = self.intrinsic_reward_loss_coef * compute_forward_dynamics_loss(mask*predicted_embedding, mask*random_embedding.detach())

                # update the intr rew models
                self.intrinsic_reward_optimizer.zero_grad()
                intr_rew_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.predictor_network.parameters(), self.intrinsic_reward_max_grad_norm)
                self.intrinsic_reward_optimizer.step()

                intr_rew_perf = intr_rew_loss

            elif "ride" in self.exploration_bonus_type:
                imgs = exps.obs.image.reshape(
                    self.num_procs, self.num_frames_per_proc, *exps.obs.image.shape[1:]
                ).transpose(0, 1)

                mask = exps.mask.reshape(
                    self.num_procs, self.num_frames_per_proc
                ).transpose(0, 1).to(torch.int64)

                # we only take the first (primitive) action
                action = exps.action[:, 0].reshape(
                    self.num_procs, self.num_frames_per_proc
                ).transpose(0, 1).to(torch.int64)

                _mask = mask[:-1]
                _obs = imgs[:-1]
                _actions = action[:-1]
                _next_obs = imgs[1:]

                self.state_embedding_model.train()
                self.forward_dynamics_model.train()
                self.inverse_dynamics_model.train()

                state_emb = self.state_embedding_model(_obs.to(device=self.device))
                next_state_emb = self.state_embedding_model(_next_obs.to(device=self.device))

                pred_next_state_emb = self.forward_dynamics_model(state_emb, _actions.to(device=self.device))

                pred_actions = self.inverse_dynamics_model(state_emb, next_state_emb)

                forward_dynamics_loss = self.intrinsic_reward_forward_loss_coef * \
                                        compute_forward_dynamics_loss(_mask[:,:,None]*pred_next_state_emb, _mask[:,:,None]*next_state_emb)

                inverse_dynamics_loss = self.intrinsic_reward_inverse_loss_coef * \
                                        compute_inverse_dynamics_loss(_mask[:,:,None]*pred_actions, _mask*_actions)

                # update the intr rew models
                self.state_embedding_optimizer.zero_grad()
                self.forward_dynamics_optimizer.zero_grad()
                self.inverse_dynamics_optimizer.zero_grad()

                intr_rew_loss = forward_dynamics_loss + inverse_dynamics_loss
                intr_rew_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.state_embedding_model.parameters(), self.intrinsic_reward_max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.forward_dynamics_model.parameters(), self.intrinsic_reward_max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.inverse_dynamics_model.parameters(), self.intrinsic_reward_max_grad_norm)

                self.state_embedding_optimizer.step()
                self.forward_dynamics_optimizer.step()
                self.inverse_dynamics_optimizer.step()

                intr_rew_perf = intr_rew_loss

            elif "soc_inf" in self.exploration_bonus_type:

                # trained together with the policy
                social_influence = True
                self.moa_net.train()
                if self.intrinsic_reward_epochs > 0:
                    raise DeprecationWarning(f"Moa must be trained with the agent. intrinsic_reward_epochs must be 0 but is {self.intrinsic_reward_epochs}")

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_lrs = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # intr reward metrics
                batch_intr_rew_loss = 0
                batch_intr_rew_acc = 0
                batch_intr_rew_f1 = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                if social_influence:
                    # Initialize moa memory
                    moa_memory = exps.moa_memory[inds]
                    prev_npc_prim_action = None

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss
                    if self.acmodel.recurrent:
                        dist, value, memory, policy_embeddings = self.acmodel(sb.obs, memory * sb.mask, return_embeddings=True)
                    else:
                        dist, value, policy_embeddings = self.acmodel(sb.obs, return_embeddings=True)

                    losses = []

                    for head_i, d in enumerate(dist):
                        action_masks = self.acmodel.calculate_action_gradient_masks(sb.action).type(sb.log_prob.type())

                        entropy = (d.entropy() * action_masks[:, head_i]).mean()
                        ratio = torch.exp(d.log_prob(sb.action[:, head_i]) - sb.log_prob[:, head_i])
                        surr1 = ratio * sb.advantage
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                        policy_loss = (
                            -torch.min(surr1, surr2) * action_masks[:, head_i]
                        ).mean()

                        value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                        surr1 = (value - sb.returnn).pow(2)
                        surr2 = (value_clipped - sb.returnn).pow(2)
                        value_loss = (
                            torch.max(surr1, surr2) * action_masks[:, head_i]
                        ).mean()

                        head_loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss
                        losses.append(head_loss)

                    if social_influence:
                        # moa loss
                        imgs = sb.obs.image
                        mask = sb.mask.to(torch.int64)
                        # we only take the first (primitive) action
                        agent_action = sb.action.to(torch.int64)
                        infos = numpy.array(sb.infos)
                        npc_prim_action = torch.tensor(
                            numpy.array([self.fn_name_to_npc_prim_act[info["NPC_prim_action"]] for info in infos]))
                        npc_utt_action = torch.tensor(
                            numpy.array([self.npc_utterance_to_id[info["NPC_utterance"]] for info in infos]))

                        assert infos.shape == imgs.shape[:1] == agent_action.shape[:1]  # [bs]

                        if i == 0:
                            prev_npc_prim_action = npc_prim_action
                            prev_npc_utt_action = npc_utt_action

                        else:
                            # compute loss and train moa net
                            if self.utterance_moa_net:
                                # transform to long logits
                                target = npc_prim_action.detach().to(self.device) * self.num_npc_utterance_actions + npc_utt_action.detach().to(self.device)
                            else:
                                target = npc_prim_action.detach().to(self.device)

                            if self.balance_moa_training:
                                balance_mask = compute_balance_mask(target, n_classes=self.num_npc_all_actions)
                            else:
                                balance_mask = torch.ones_like(target)

                            moa_predictions_logs, moa_memory = self.moa_net(
                                embeddings=policy_embeddings,
                                npc_previous_prim_actions=prev_npc_prim_action.detach().to(self.device),
                                npc_previous_utterance_actions=prev_npc_utt_action.detach().to(self.device) if self.utterance_moa_net else None,
                                agent_actions=agent_action.detach().to(self.device),
                                memory=moa_memory * mask,
                            )

                            # moa_predictions_logs = moa_predictions_logs.reshape([*prev_shape, -1])  # is this needed

                            # loss
                            ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
                            intr_rew_loss = (
                                balance_mask * mask * ce_loss(
                                input=moa_predictions_logs,
                                target=target,
                            )).mean() * self.intrinsic_reward_loss_coef

                            preds = moa_predictions_logs.detach().argmax(dim=-1)
                            intr_rew_f1 = f1_score(
                                y_pred=preds.detach().cpu().numpy(),
                                y_true=target.detach().cpu().numpy(),
                                average="macro"
                            )

                            intr_rew_acc = (
                                    torch.argmax(moa_predictions_logs.to(self.device), dim=-1) == target
                            ).to(float).mean()

                            batch_intr_rew_loss += intr_rew_loss
                            batch_intr_rew_acc += intr_rew_acc.detach().cpu().numpy().mean()
                            batch_intr_rew_f1 += intr_rew_f1

                            losses.append(intr_rew_loss)  # trained with the policy optimizer

                    loss = torch.stack(losses).mean()

                    # Update batch values
                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch
                    # assert self.acmodel.recurrent == (self.recurrence > 1)
                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                    if social_influence and i < self.recurrence - 1:
                        exps.moa_memory[inds + i + 1] = moa_memory.detach()


                # Update batch values
                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.lr_scheduler.step()

                if social_influence:
                    # recurrence-1 because we skipped the first step
                    batch_intr_rew_loss /= self.recurrence - 1
                    batch_intr_rew_acc /= self.recurrence - 1
                    batch_intr_rew_f1 /= self.recurrence - 1

                intr_rew_perf_ = batch_intr_rew_f1
                intr_rew_perf = batch_intr_rew_acc

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)
                log_lrs.append(self.optimizer.param_groups[0]['lr'])

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "intr_reward_perf": intr_rew_perf,
            "intr_reward_perf_": intr_rew_perf_,
            "lr": numpy.mean(log_lrs),
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

    def get_config_dict(self):

        del self.config['envs']
        del self.config['acmodel']
        del self.config['__class__']
        del self.config['self']
        del self.config['preprocess_obss']
        del self.config['device']
        return self.config

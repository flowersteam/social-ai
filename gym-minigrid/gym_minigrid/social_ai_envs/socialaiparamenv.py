import warnings
from itertools import chain
from gym_minigrid.minigrid import *
from gym_minigrid.parametric_env import *
from gym_minigrid.register import register
from gym_minigrid.social_ai_envs import InformationSeekingEnv, MarblePassEnv, LeverDoorEnv, MarblePushEnv, AppleStealingEnv, ObjectsCollaborationEnv
from gym_minigrid.social_ai_envs.socialaigrammar import SocialAIGrammar, SocialAIActions, SocialAIActionSpace
from gym_minigrid.curriculums import *

import inspect, importlib

# for used for automatic registration of environments
defined_classes = [name for name, _ in inspect.getmembers(importlib.import_module(__name__), inspect.isclass)]


class SocialAIParamEnv(gym.Env):
    """
    Meta-Environment containing all other environment (multi-task learning)
    """

    def __init__(
            self,
            size=10,
            hidden_npc=False,
            see_through_walls=False,
            max_steps=80,  # before it was 50, 80 is maybe better because of emulation ?
            switch_no_light=True,
            lever_active_steps=10,
            curriculum=None,
            expert_curriculum_thresholds=(0.9, 0.8),
            expert_curriculum_average_interval=100,
            expert_curriculum_minimum_episodes=1000,
            n_colors=3,
            egocentric_observation=True,
    ):
        if n_colors != 3:
            warnings.warn(f"You are ussing {n_colors} instead of the usual 3.")

        self.lever_active_steps = lever_active_steps
        self.egocentric_observation = egocentric_observation

        # Number of cells (width and height) in the agent view
        self.agent_view_size = 7

        # Number of object dimensions (i.e. number of channels in symbolic image)
        # if egocentric is not used absolute coordiantes are added to the encoding
        self.encoding_size = 6 + 2*bool(not egocentric_observation)

        self.max_steps = max_steps

        self.switch_no_light = switch_no_light

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, self.encoding_size),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        self.hidden_npc = hidden_npc

        # construct the tree
        self.parameter_tree = self.construct_tree()

        # print tree for logging purposes
        # self.parameter_tree.print_tree()

        if curriculum in ["intro_seq", "intro_seq_scaf"]:
            print("Scaffolding Expert")
            self.expert_curriculum_thresholds = expert_curriculum_thresholds
            self.expert_curriculum_average_interval = expert_curriculum_average_interval
            self.expert_curriculum_minimum_episodes = expert_curriculum_minimum_episodes
            self.curriculum = ScaffoldingExpertCurriculum(
                phase_thresholds=self.expert_curriculum_thresholds,
                average_interval=self.expert_curriculum_average_interval,
                minimum_episodes=self.expert_curriculum_minimum_episodes,
                type=curriculum,
            )

        else:
            self.curriculum = curriculum

        self.current_env = None

        self.envs = {}

        if self.parameter_tree.root.label == "Env_type":
            for env_type in self.parameter_tree.root.children:
                if env_type.label == "Information_seeking":
                    e = InformationSeekingEnv(
                            max_steps=max_steps,
                            size=size,
                            switch_no_light=self.switch_no_light,
                            see_through_walls=see_through_walls,
                            n_colors=n_colors,
                            hidden_npc=self.hidden_npc,
                            egocentric_observation=self.egocentric_observation,
                    )
                    self.envs["Info"] = e

                elif env_type.label == "Collaboration":
                    e = MarblePassEnv(max_steps=max_steps, size=size, hidden_npc=self.hidden_npc, egocentric_observation=egocentric_observation)
                    self.envs["Collaboration_Marble_Pass"] = e

                    e = LeverDoorEnv(max_steps=max_steps, size=size, lever_active_steps=self.lever_active_steps, hidden_npc=self.hidden_npc, egocentric_observation=egocentric_observation)
                    self.envs["Collaboration_Lever_Door"] = e

                    e = MarblePushEnv(max_steps=max_steps, size=size, lever_active_steps=self.lever_active_steps, hidden_npc=self.hidden_npc, egocentric_observation=egocentric_observation)
                    self.envs["Collaboration_Marble_Push"] = e

                    e = ObjectsCollaborationEnv(max_steps=max_steps, size=size, hidden_npc=self.hidden_npc, switch_no_light=self.switch_no_light, egocentric_observation=egocentric_observation)
                    self.envs["Collaboration_Objects"] = e

                elif env_type.label == "AppleStealing":
                    e = AppleStealingEnv(max_steps=max_steps, size=size, see_through_walls=see_through_walls,
                                     hidden_npc=self.hidden_npc, egocentric_observation=egocentric_observation)
                    self.envs["OthersPerceptionInference"] = e

                else:
                    raise ValueError(f"Undefined env type {env_type.label}.")

        else:
            raise ValueError("Env_type should be the root node")

        self.all_npc_utterance_actions = sorted(list(set(chain(*[e.all_npc_utterance_actions for e in self.envs.values()]))))

        self.grammar = SocialAIGrammar()

        # set up the action space
        self.action_space = SocialAIActionSpace
        self.actions = SocialAIActions
        self.npc_prim_actions_dict = SocialAINPCActionsDict

        # all envs must have the same grammar
        for env in self.envs.values():
            assert isinstance(env.grammar, type(self.grammar))
            assert env.actions is self.actions
            assert env.action_space is self.action_space

            # suggestion: encoding size is automatically set to max?
            assert env.encoding_size is self.encoding_size
            assert env.observation_space == self.observation_space
            assert env.prim_actions_dict == self.npc_prim_actions_dict

        self.reset()

    def draw_tree(self, ignore_labels=[], savedir="viz"):
        self.parameter_tree.draw_tree("{}/param_tree_{}".format(savedir, self.spec.id), ignore_labels=ignore_labels)

    def print_tree(self):
        self.parameter_tree.print_tree()

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
        tree.add_node("No", parent=prag_fr_compl_nd, type="value")
        tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
        tree.add_node("Ask", parent=prag_fr_compl_nd, type="value")
        tree.add_node("Ask_Eye_contact", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
        scaffolding_Y_nd = tree.add_node("Y", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        tree.add_node("Language_Color", parent=cue_type_nd, type="value")
        tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
        tree.add_node("Pointing", parent=cue_type_nd, type="value")
        tree.add_node("Emulation", parent=cue_type_nd, type="value")


        N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
        tree.add_node("2", parent=N_bo_nd, type="value")
        tree.add_node("1", parent=N_bo_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

        doors_nd = tree.add_node("Doors", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=doors_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        boxes_nd = tree.add_node("Boxes", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        switches_nd = tree.add_node("Switches", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=switches_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=switches_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        generators_nd = tree.add_node("Generators", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=generators_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=generators_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        levers_nd = tree.add_node("Levers", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=levers_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=levers_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        doors_nd = tree.add_node("Marble", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=doors_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        # Collaboration
        collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")

        colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")

        problem_nd = tree.add_node("Boxes", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")
        role_nd = tree.add_node("Version", parent=problem_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        problem_nd = tree.add_node("Switches", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")
        role_nd = tree.add_node("Version", parent=problem_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        problem_nd = tree.add_node("Generators", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")
        role_nd = tree.add_node("Version", parent=problem_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        problem_nd = tree.add_node("Marble", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")
        role_nd = tree.add_node("Version", parent=problem_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        problem_nd = tree.add_node("MarblePass", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")
        role_nd = tree.add_node("Version", parent=problem_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")
        tree.add_node("Asocial", parent=role_nd, type="value")

        problem_nd = tree.add_node("MarblePush", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")
        role_nd = tree.add_node("Version", parent=problem_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        problem_nd = tree.add_node("LeverDoor", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")
        role_nd = tree.add_node("Version", parent=problem_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        # Perspective taking
        collab_nd = tree.add_node("AppleStealing", parent=env_type_nd, type="value")

        role_nd = tree.add_node("Version", parent=collab_nd, type="param")
        tree.add_node("Asocial", parent=role_nd, type="value")
        social_nd = tree.add_node("Social", parent=role_nd, type="value")

        move_nd = tree.add_node("NPC_movement", parent=social_nd, type="param")
        tree.add_node("Walking", parent=move_nd, type="value")
        tree.add_node("Rotating", parent=move_nd, type="value")

        obstacles_nd = tree.add_node("Obstacles", parent=collab_nd, type="param")
        tree.add_node("No", parent=obstacles_nd, type="value")
        tree.add_node("A_bit", parent=obstacles_nd, type="value")
        tree.add_node("Medium", parent=obstacles_nd, type="value")
        tree.add_node("A_lot", parent=obstacles_nd, type="value")

        return tree

    def construct_env_from_params(self, params):
        params_labels = {k.label: v.label for k, v in params.items()}
        if params_labels['Env_type'] == "Collaboration":

            if params_labels["Problem"] == "MarblePass":
                env = self.envs["Collaboration_Marble_Pass"]

            elif params_labels["Problem"] == "LeverDoor":
                env = self.envs["Collaboration_Lever_Door"]

            elif params_labels["Problem"] == "MarblePush":
                env = self.envs["Collaboration_Marble_Push"]

            elif params_labels["Problem"] in ["Boxes", "Switches", "Generators", "Marble"]:
                env = self.envs["Collaboration_Objects"]

            else:
                raise ValueError("params badly defined.")

        elif params_labels['Env_type'] == "Information_seeking":
            env = self.envs["Info"]

        elif params_labels['Env_type'] == "AppleStealing":
            env = self.envs["OthersPerceptionInference"]

        else:
            raise ValueError("params badly defined.")

        reset_kwargs = params_labels

        return env, reset_kwargs

    def reset(self, with_info=False):
        # select a new social environment at random, for each new episode

        old_window = None
        if self.current_env:  # a previous env exists, save old window
            old_window = self.current_env.window

        self.current_params = self.parameter_tree.sample_env_params(ACL=self.curriculum)

        self.current_env, reset_kwargs = self.construct_env_from_params(self.current_params)
        assert reset_kwargs is not {}
        assert reset_kwargs is not None

        # print("Sampled parameters:")
        # for k, v in reset_kwargs.items():
        #     print(f'\t{k}:{v}')

        if with_info:
            obs, info = self.current_env.reset_with_info(**reset_kwargs)
        else:
            obs = self.current_env.reset(**reset_kwargs)

        # carry on window if this env is not the first
        if old_window:
            self.current_env.window = old_window

        if with_info:
            return obs, info
        else:
            return obs

    def reset_with_info(self):
        return self.reset(with_info=True)


    def seed(self, seed=1337):
        # Seed the random number generator
        for env in self.envs.values():
            env.seed(seed)

        return [seed]

    def set_curriculum_parameters(self, params):
        if self.curriculum is not None:
            self.curriculum.set_parameters(params)

    def step(self, action):
        assert self.current_env
        assert self.current_env.parameters is not None

        obs, reward, done, info = self.current_env.step(action)

        info["parameters"] = self.current_params

        if done:
            if info["success"]:
                # self.current_env.outcome_info = "SUCCESS: agent got {} reward \n".format(np.round(reward, 1))
                self.current_env.outcome_info = "SUCCESS\n"
            else:
                self.current_env.outcome_info = "FAILURE\n"

        if self.curriculum is not None:
            for k, v in self.curriculum.get_info().items():
                info["curriculum_info_"+k] = v

        return obs, reward, done, info


    @property
    def window(self):
        assert self.current_env
        return self.current_env.window

    @window.setter
    def window(self, value):
        self.current_env.window = value

    def render(self, *args, **kwargs):
        assert self.current_env
        return self.current_env.render(*args, **kwargs)

    @property
    def step_count(self):
        return self.current_env.step_count

    def get_mission(self):
        return self.current_env.get_mission()


defined_classes_ = [name for name, _ in inspect.getmembers(importlib.import_module(__name__), inspect.isclass)]

envs = list(set(defined_classes_) - set(defined_classes))
assert all([e.endswith("Env") for e in envs])

for env in envs:
    register(
        id='SocialAI-{}-v1'.format(env),
        entry_point='gym_minigrid.social_ai_envs:{}'.format(env)
    )

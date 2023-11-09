from gym_minigrid.social_ai_envs.socialaiparamenv import SocialAIParamEnv
from gym_minigrid.parametric_env import *
from gym_minigrid.register import register

import inspect, importlib
# for used for automatic registration of environments
defined_classes = [name for name, _ in inspect.getmembers(importlib.import_module(__name__), inspect.isclass)]

class EAsocialInformationSeekingParamEnv(SocialAIParamEnv):

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
        tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        tree.add_node("Language_Color", parent=cue_type_nd, type="value")
        tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
        tree.add_node("Pointing", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

        boxes_nd = tree.add_node("Boxes", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        tree.add_node("N", parent=peer_nd, type="value")

        switches_nd = tree.add_node("Switches", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=switches_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=switches_nd, type="param")
        tree.add_node("N", parent=peer_nd, type="value")

        generators_nd = tree.add_node("Generators", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=generators_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=generators_nd, type="param")
        tree.add_node("N", parent=peer_nd, type="value")

        levers_nd = tree.add_node("Levers", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=levers_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=levers_nd, type="param")
        tree.add_node("N", parent=peer_nd, type="value")

        doors_nd = tree.add_node("Doors", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=doors_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
        tree.add_node("N", parent=peer_nd, type="value")

        marble_nd = tree.add_node("Marble", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=marble_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=marble_nd, type="param")
        tree.add_node("N", parent=peer_nd, type="value")

        return tree

# Pointing case study

# training
class EPointingInformationSeekingParamEnv(SocialAIParamEnv):


    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
        tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        # tree.add_node("Language_Color", parent=cue_type_nd, type="value")
        # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
        tree.add_node("Pointing", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

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

        doors_nd = tree.add_node("Doors", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=doors_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        marble_nd = tree.add_node("Marble", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=marble_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=marble_nd, type="param")
        tree.add_node("N", parent=peer_nd, type="value")

        return tree

# testing
class EPointingTestingInformationSeekingParamEnv(SocialAIParamEnv):

    def __init__(self, problem, asocial, **kwargs):

        self.problem = problem
        if self.problem not in ["Boxes", "Switches", "Generators", "Levers", "Doors", "Marble"]:
            raise ValueError(f"Problem {self.problem} undefined.")

        self.asocial = asocial

        super(EPointingTestingInformationSeekingParamEnv, self).__init__(**kwargs)

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
        tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        tree.add_node("Pointing", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

        boxes_nd = tree.add_node(self.problem, parent=problem_nd, type="value")

        N = "1" if self.asocial else "2"
        version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        tree.add_node(N, parent=version_nd, type="value")

        peer = "N" if self.asocial else "Y"
        peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        tree.add_node(peer, parent=peer_nd, type="value")

        return tree


# Joint Attention

# training
class JAEPointingInformationSeekingParamEnv(SocialAIParamEnv):

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
        tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        # tree.add_node("Language_Color", parent=cue_type_nd, type="value")
        # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
        tree.add_node("Pointing", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

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

        doors_nd = tree.add_node("Doors", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=doors_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        marble_nd = tree.add_node("Marble", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=marble_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=marble_nd, type="param")
        tree.add_node("N", parent=peer_nd, type="value")

        N_bo_nd = tree.add_node("JA_recursive", parent=inf_seeking_nd, type="param")
        tree.add_node("Y", parent=N_bo_nd, type="value")

        return tree


# testing
class JAEPointingTestingInformationSeekingParamEnv(SocialAIParamEnv):

    def __init__(self, problem, asocial, **kwargs):

        self.problem = problem
        if self.problem not in ["Boxes", "Switches", "Generators", "Levers", "Doors", "Marble"]:
            raise ValueError(f"Problem {self.problem} undefined.")

        self.asocial = asocial

        super(JAEPointingTestingInformationSeekingParamEnv, self).__init__(**kwargs)

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
        tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        tree.add_node("Pointing", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

        boxes_nd = tree.add_node(self.problem, parent=problem_nd, type="value")

        N = "1" if self.asocial else "2"
        version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        tree.add_node(N, parent=version_nd, type="value")

        peer = "N" if self.asocial else "Y"
        peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        tree.add_node(peer, parent=peer_nd, type="value")

        N_bo_nd = tree.add_node("JA_recursive", parent=inf_seeking_nd, type="param")
        tree.add_node("Y", parent=N_bo_nd, type="value")

        return tree


# automatic registration of environments
defined_classes_ = [name for name, _ in inspect.getmembers(importlib.import_module(__name__), inspect.isclass)]

envs = list(set(defined_classes_) - set(defined_classes))
assert all([e.endswith("Env") for e in envs])

for env in envs:
    try:
        register(
            id='SocialAI-{}-v1'.format(env),
            entry_point='gym_minigrid.social_ai_envs:{}'.format(env)
        )
    except:
        print(f"Env : {env} registratoin failed.")
        exit()

# register testing envs
problems = ["Boxes", "Switches", "Generators", "Levers", "Doors", "Marble"]

for problem in problems:
    for asocial in [True, False]:

        if asocial:
            env_name = f'EPointing{problem}AsocialInformationSeekingParamEnv'
        else:
            env_name = f'EPointing{problem}InformationSeekingParamEnv'

        print("env name:", env_name)

        register(
            id='SocialAI-{}-v1'.format(env_name),
            entry_point='gym_minigrid.social_ai_envs:EPointingTestingInformationSeekingParamEnv',
            kwargs={
                'asocial': asocial,
                'problem': problem,
            }
        )

        if asocial:
            env_name = f'JAEPointing{problem}AsocialInformationSeekingParamEnv'
        else:
            env_name = f'JAEPointing{problem}InformationSeekingParamEnv'

        register(
            id='SocialAI-{}-v1'.format(env_name),
            entry_point='gym_minigrid.social_ai_envs:JAEPointingTestingInformationSeekingParamEnv',
            kwargs={
                'asocial': asocial,
                'problem': problem,
            }
        )


pointing_test_set = [
    "SocialAI-EPointingMarbleInformationSeekingParamEnv-v1",
]


ja_pointing_test_set = [
    "SocialAI-JAEPointingMarbleInformationSeekingParamEnv-v1",
]

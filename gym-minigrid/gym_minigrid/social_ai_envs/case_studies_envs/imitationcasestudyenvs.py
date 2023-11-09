from gym_minigrid.social_ai_envs.socialaiparamenv import SocialAIParamEnv
from gym_minigrid.parametric_env import *
from gym_minigrid.register import register

import inspect, importlib

# for used for automatic registration of environments
defined_classes = [name for name, _ in inspect.getmembers(importlib.import_module(__name__), inspect.isclass)]


# Emulation case study (table 2)

# emulation without distractor
# training
class EEmulationNoDistrInformationSeekingParamEnv(SocialAIParamEnv):

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
        tree.add_node("Emulation", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

        boxes_nd = tree.add_node("Boxes", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        switches_nd = tree.add_node("Switches", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=switches_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=switches_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        generators_nd = tree.add_node("Generators", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=generators_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=generators_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        levers_nd = tree.add_node("Levers", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=levers_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=levers_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        doors_nd = tree.add_node("Marble", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=doors_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        return tree

# testing
class EEmulationNoDistrDoorsInformationSeekingParamEnv(SocialAIParamEnv):

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
        tree.add_node("Emulation", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

        marble_nd = tree.add_node("Doors", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=marble_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=marble_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        return tree



# emulation with a distractor

# training
class EEmulationDistrInformationSeekingParamEnv(SocialAIParamEnv):

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
        tree.add_node("Emulation", parent=cue_type_nd, type="value")

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

        doors_nd = tree.add_node("Marble", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=doors_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        return tree

# testing
class EEmulationDistrDoorsInformationSeekingParamEnv(SocialAIParamEnv):

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
        tree.add_node("Emulation", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

        doors_nd = tree.add_node("Doors", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=doors_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

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


distr_emulation_test_set = [
    # "SocialAI-EEmulationDistrBoxesInformationSeekingParamEnv-v1",
    # "SocialAI-EEmulationDistrSwitchesInformationSeekingParamEnv-v1",
    # "SocialAI-EEmulationDistrMarbleInformationSeekingParamEnv-v1",
    # "SocialAI-EEmulationDistrGeneratorsInformationSeekingParamEnv-v1",
    # "SocialAI-EEmulationDistrLeversInformationSeekingParamEnv-v1",
    "SocialAI-EEmulationDistrDoorsInformationSeekingParamEnv-v1",
]

no_distr_emulation_test_set = [
    # "SocialAI-EEmulationNoDistrBoxesInformationSeekingParamEnv-v1",
    # "SocialAI-EEmulationNoDistrSwitchesInformationSeekingParamEnv-v1",
    # "SocialAI-EEmulationNoDistrMarbleInformationSeekingParamEnv-v1",
    # "SocialAI-EEmulationNoDistrGeneratorsInformationSeekingParamEnv-v1",
    # "SocialAI-EEmulationNoDistrLeversInformationSeekingParamEnv-v1",
    "SocialAI-EEmulationNoDistrDoorsInformationSeekingParamEnv-v1",
]

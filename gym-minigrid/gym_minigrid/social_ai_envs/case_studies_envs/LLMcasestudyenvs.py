from gym_minigrid.social_ai_envs.socialaiparamenv import SocialAIParamEnv
from gym_minigrid.parametric_env import *
from gym_minigrid.register import register

'''
These are the environments for case studies 1-3: Pointing, Language (Color and Feedback), and Joint Attention.

Intro sequence is always eye contact (E) in both the training and testing envs

The Training environments have the 5 problems and Marbles in the Asocial version (no distractor, no peer)
registered training envs : cues x {joint attention, no}

The Testing e environments are always one problem per env - i.e no testing on two problems at the same time
registered testing envs : cues x problems x {social, asocial} x {joint attention, no}
'''

PROBLEMS = ["Boxes", "Switches", "Generators", "Levers", "Doors", "Marble"]
CUES = ["Pointing", "LangFeedback", "LangColor"]
INTRO_SEC = ["E"]
# INTRO_SEC = ["N", "E", "A", "AE"]


class AsocialBoxInformationSeekingParamEnv(SocialAIParamEnv):
    '''
    Env with all problems in the asocial version -> just for testing
    '''

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        # prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
        # tree.add_node("No", parent=prag_fr_compl_nd, type="value")
        #
        # # scaffolding
        # scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        # scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
        #
        # cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        # tree.add_node("Language_Color", parent=cue_type_nd, type="value")
        # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
        # tree.add_node("Pointing", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

        boxes_nd = tree.add_node("Boxes", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        tree.add_node("N", parent=peer_nd, type="value")

        return tree


class ColorBoxesLLMCSParamEnv(SocialAIParamEnv):


    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
        tree.add_node("No", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        tree.add_node("Language_Color", parent=cue_type_nd, type="value")
        # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
        # tree.add_node("Pointing", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

        boxes_nd = tree.add_node("Boxes", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        return tree


class ColorLLMCSParamEnv(SocialAIParamEnv):

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
        tree.add_node("No", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        tree.add_node("Language_Color", parent=cue_type_nd, type="value")
        # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
        # tree.add_node("Pointing", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

        # boxes_nd = tree.add_node("Boxes", parent=problem_nd, type="value")
        # version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        # tree.add_node("2", parent=version_nd, type="value")
        # peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        # tree.add_node("Y", parent=peer_nd, type="value")

        boxes_nd = tree.add_node("Boxes", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        tree.add_node("N", parent=peer_nd, type="value")

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
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=marble_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        return tree

# register dummy env
register(
    id='SocialAI-AsocialBoxInformationSeekingParamEnv-v1',
    entry_point='gym_minigrid.social_ai_envs:AsocialBoxInformationSeekingParamEnv',
)


register(
    id='SocialAI-ColorBoxesLLMCSParamEnv-v1',
    entry_point='gym_minigrid.social_ai_envs:ColorBoxesLLMCSParamEnv',
)

register(
    id='SocialAI-ColorLLMCSParamEnv-v1',
    entry_point='gym_minigrid.social_ai_envs:ColorLLMCSParamEnv',
)

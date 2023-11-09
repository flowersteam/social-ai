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


class EAsocialInformationSeekingParamEnv(SocialAIParamEnv):
    '''
    Env with all problems in the asocial version -> just for testing
    '''

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
class TrainingInformationSeekingParamEnv(SocialAIParamEnv):

    def __init__(self, cue, intro_sec, ja, heldout="Doors", **kwargs):

        if cue not in CUES:
            raise ValueError(f"Cue {cue} undefined.")
        self.cue = cue

        if intro_sec not in INTRO_SEC:
            raise ValueError(f"Cue {intro_sec} undefined.")
        self.intro_sec = intro_sec

        self.heldout=heldout

        if ja not in [True, False]:
            raise ValueError(f"JA {ja} undefined.")
        self.ja = ja

        super(TrainingInformationSeekingParamEnv, self).__init__(**kwargs)

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")

        if self.intro_sec == "N":
            tree.add_node("No", parent=prag_fr_compl_nd, type="value")
        elif self.intro_sec == "E":
            tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
        elif self.intro_sec == "A":
            tree.add_node("Ask", parent=prag_fr_compl_nd, type="value")
        elif self.intro_sec == "AE":
            tree.add_node("Ask_Eye_contact", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")

        if self.cue == "Pointing":
            tree.add_node("Pointing", parent=cue_type_nd, type="value")
        elif self.cue == "LangColor":
            tree.add_node("Language_Color", parent=cue_type_nd, type="value")
        elif self.cue == "LangFeedback":
            tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")

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

        if self.heldout == "Doors":
            tree.add_node("1", parent=version_nd, type="value")
            peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
            tree.add_node("N", parent=peer_nd, type="value")
        else:
            tree.add_node("2", parent=version_nd, type="value")
            peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
            tree.add_node("Y", parent=peer_nd, type="value")

        marble_nd = tree.add_node("Marble", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=marble_nd, type="param")

        if self.heldout == "Marble":
            tree.add_node("1", parent=version_nd, type="value")
            peer_nd = tree.add_node("Peer", parent=marble_nd, type="param")
            tree.add_node("N", parent=peer_nd, type="value")
        else:
            tree.add_node("2", parent=version_nd, type="value")
            peer_nd = tree.add_node("Peer", parent=marble_nd, type="param")
            tree.add_node("Y", parent=peer_nd, type="value")

        if self.ja:
            N_bo_nd = tree.add_node("JA_recursive", parent=inf_seeking_nd, type="param")
            tree.add_node("Y", parent=N_bo_nd, type="value")
            tree.add_node("N", parent=N_bo_nd, type="value")

        return tree

# testing
class TestingInformationSeekingParamEnv(SocialAIParamEnv):

    def __init__(self, cue, intro_sec, ja, problem, asocial, **kwargs):

        if cue not in CUES:
            raise ValueError(f"Cue {cue} undefined.")
        self.cue = cue

        if intro_sec not in INTRO_SEC:
            raise ValueError(f"Cue {intro_sec} undefined.")
        self.intro_sec = intro_sec

        if ja not in [True, False]:
            raise ValueError(f"JA {ja} undefined.")
        self.ja = ja

        self.problem = problem
        if self.problem not in PROBLEMS:
            raise ValueError(f"Problem {self.problem} undefined.")

        self.asocial = asocial

        super(TestingInformationSeekingParamEnv, self).__init__(**kwargs)

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")

        if self.intro_sec == "N":
            tree.add_node("No", parent=prag_fr_compl_nd, type="value")
        elif self.intro_sec == "E":
            tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
        elif self.intro_sec == "A":
            tree.add_node("Ask", parent=prag_fr_compl_nd, type="value")
        elif self.intro_sec == "AE":
            tree.add_node("Ask_Eye_contact", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        if self.cue == "Pointing":
            tree.add_node("Pointing", parent=cue_type_nd, type="value")
        elif self.cue == "LangColor":
            tree.add_node("Language_Color", parent=cue_type_nd, type="value")
        elif self.cue == "LangFeedback":
            tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")

        boxes_nd = tree.add_node(self.problem, parent=problem_nd, type="value")

        N = "1" if self.asocial else "2"
        version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        tree.add_node(N, parent=version_nd, type="value")

        peer = "N" if self.asocial else "Y"
        peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        tree.add_node(peer, parent=peer_nd, type="value")

        if self.ja:
            N_bo_nd = tree.add_node("JA_recursive", parent=inf_seeking_nd, type="param")
            tree.add_node("Y", parent=N_bo_nd, type="value")

        return tree


class DrawingEnv(SocialAIParamEnv):

    def __init__(self, **kwargs):

        self.cue = "Pointing"

        self.intro_sec = "E"

        self.heldout= "Doors"

        self.ja = False

        super(DrawingEnv, self).__init__(**kwargs)

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")
        # colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")
        # tree.add_node("MarblePush", parent=colab_type_nd, type="value")
        #
        # role_nd = tree.add_node("Role", parent=collab_nd, type="param")
        # tree.add_node("A", parent=role_nd, type="value")
        #
        # role_nd = tree.add_node("Version", parent=collab_nd, type="param")
        # tree.add_node("Social", parent=role_nd, type="value")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")


        # as_nd = tree.add_node("AppleStealing", parent=env_type_nd, type="value")
        # ver_nd = tree.add_node("Version", parent=as_nd, type="param")
        # tree.add_node("Asocial", parent=ver_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")

        if self.intro_sec == "N":
            tree.add_node("No", parent=prag_fr_compl_nd, type="value")
        elif self.intro_sec == "E":
            tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
        elif self.intro_sec == "A":
            tree.add_node("Ask", parent=prag_fr_compl_nd, type="value")
        elif self.intro_sec == "AE":
            tree.add_node("Ask_Eye_contact", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Peer_help", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")

        if self.cue == "Pointing":
            tree.add_node("Pointing", parent=cue_type_nd, type="value")
        elif self.cue == "LangColor":
            tree.add_node("Language_Color", parent=cue_type_nd, type="value")
        elif self.cue == "LangFeedback":
            tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")

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

        if self.heldout == "Doors":
            tree.add_node("1", parent=version_nd, type="value")
            peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
            tree.add_node("N", parent=peer_nd, type="value")
        else:
            tree.add_node("2", parent=version_nd, type="value")
            peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
            tree.add_node("Y", parent=peer_nd, type="value")

        marble_nd = tree.add_node("Marble", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=marble_nd, type="param")

        if self.heldout == "Marble":
            tree.add_node("1", parent=version_nd, type="value")
            peer_nd = tree.add_node("Peer", parent=marble_nd, type="param")
            tree.add_node("N", parent=peer_nd, type="value")
        else:
            tree.add_node("2", parent=version_nd, type="value")
            peer_nd = tree.add_node("Peer", parent=marble_nd, type="param")
            tree.add_node("Y", parent=peer_nd, type="value")

        if self.ja:
            N_bo_nd = tree.add_node("JA_recursive", parent=inf_seeking_nd, type="param")
            tree.add_node("Y", parent=N_bo_nd, type="value")
            tree.add_node("N", parent=N_bo_nd, type="value")

        return tree

# register drawing env
register(
    id='SocialAI-DrawingEnv-v1',
    entry_point='gym_minigrid.social_ai_envs:DrawingEnv',
)

# register dummy env
register(
        id='SocialAI-EAsocialInformationSeekingParamEnv-v1',
        entry_point='gym_minigrid.social_ai_envs:EAsocialInformationSeekingParamEnv',
)


# register training envs
for cue in CUES:
    for ja in [True, False]:
        for intro_sec in INTRO_SEC:
            env_name = f'{"JA" if ja else ""}{intro_sec}{cue}TrainInformationSeekingParamEnv'

            register(
                id='SocialAI-{}-v1'.format(env_name),
                entry_point='gym_minigrid.social_ai_envs:TrainingInformationSeekingParamEnv',
                kwargs={
                    'intro_sec': intro_sec,
                    'cue': cue,
                    'ja': ja,
                }
            )

# register training envs: heldout generators
for cue in CUES:
    for ja in [True, False]:
        for intro_sec in INTRO_SEC:
            env_name = f'{"JA" if ja else ""}{intro_sec}{cue}HeldoutDoorsTrainInformationSeekingParamEnv'

            register(
                id='SocialAI-{}-v1'.format(env_name),
                entry_point='gym_minigrid.social_ai_envs:TrainingInformationSeekingParamEnv',
                kwargs={
                    'intro_sec': intro_sec,
                    'cue': cue,
                    'ja': ja,
                    'heldout': "Doors",
                }
            )


# register testing envs : cues x problems x {social, asocial} x {joint attention, no}
for cue in CUES:
    for problem in PROBLEMS:
        for asocial in [True, False]:
            for ja in [True, False]:
                for intro_sec in INTRO_SEC:
                    env_name = f'{"JA" if ja else ""}{intro_sec}{cue}{problem}{"Asocial" if asocial else ""}TestInformationSeekingParamEnv'

                    register(
                        id='SocialAI-{}-v1'.format(env_name),
                        entry_point='gym_minigrid.social_ai_envs:TestingInformationSeekingParamEnv',
                        kwargs={
                            'asocial': asocial,
                            'problem': problem,
                            'cue': cue,
                            'ja': ja,
                            'intro_sec': intro_sec
                        }
                    )

pointing_test_set = [
    f"SocialAI-EPointing{problem}TestInformationSeekingParamEnv-v1" for problem in PROBLEMS
]+["SocialAI-EPointingDoorsAsocialTestInformationSeekingParamEnv-v1"]

language_feedback_test_set = [
    f"SocialAI-ELangFeedback{problem}TestInformationSeekingParamEnv-v1" for problem in PROBLEMS
]+["SocialAI-ELangFeedbackDoorsAsocialTestInformationSeekingParamEnv-v1"]

language_color_test_set = [
    f"SocialAI-ELangColor{problem}TestInformationSeekingParamEnv-v1" for problem in PROBLEMS
]+["SocialAI-ELangColorDoorsAsocialTestInformationSeekingParamEnv-v1"]

ja_pointing_test_set = [
    f"SocialAI-JAEPointing{problem}TestInformationSeekingParamEnv-v1" for problem in PROBLEMS
]+["SocialAI-JAEPointingDoorsAsocialTestInformationSeekingParamEnv-v1"]

ja_language_feedback_test_set = [
    f"SocialAI-JAELangFeedback{problem}TestInformationSeekingParamEnv-v1" for problem in PROBLEMS
]+["SocialAI-JAELangFeedbackDoorsAsocialTestInformationSeekingParamEnv-v1"]

ja_language_color_test_set = [
    f"SocialAI-JAELangColor{problem}TestInformationSeekingParamEnv-v1" for problem in PROBLEMS
]+["SocialAI-JAELangColorDoorsAsocialTestInformationSeekingParamEnv-v1"]

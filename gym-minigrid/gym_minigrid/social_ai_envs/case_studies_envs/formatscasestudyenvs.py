from gym_minigrid.social_ai_envs.socialaiparamenv import SocialAIParamEnv
from gym_minigrid.parametric_env import *
from gym_minigrid.register import register

'''
These are the environments for the formats case study: 

All the environments use the Language Feedback cue type.

There are four Intro sequences:
- no (N)
- eye contact (E)
- ask (A)
- ask during eye contact (AE)

The Training environments all have the 6 problems.
There will be four registered training corresponding to the 4 different introductory sequences.

The Testing environments are always one problem per env - i.e no testing on two problems at the same time
registered testing envs : problems x {N, A, E, AE}
'''

PROBLEMS = ["Boxes", "Switches", "Generators", "Levers", "Doors", "Marble"]
CUES = ["LangFeedback"]
INTRO_SEC = ["N", "E", "A", "AE"]

# training
class TrainingFormatsCSParamEnv(SocialAIParamEnv):

    def __init__(self, cue, intro_sec, scaffolding=False, **kwargs):

        if cue not in CUES:
            raise ValueError(f"Cue {cue} undefined.")
        self.cue = cue

        if intro_sec not in INTRO_SEC:
            raise ValueError(f"Cue {intro_sec} undefined.")
        self.intro_sec = intro_sec

        self.scaffolding = scaffolding

        super(TrainingFormatsCSParamEnv, self).__init__(**kwargs)

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")

        if self.scaffolding:
            tree.add_node("No", parent=prag_fr_compl_nd, type="value")
            tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
            tree.add_node("Ask", parent=prag_fr_compl_nd, type="value")
            tree.add_node("Ask_Eye_contact", parent=prag_fr_compl_nd, type="value")

        else:
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
        if self.scaffolding:
            scaffolding_Y_nd = tree.add_node("Y", parent=scaffolding_nd, type="value")

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
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=doors_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        marble_nd = tree.add_node("Marble", parent=problem_nd, type="value")
        version_nd = tree.add_node("N", parent=marble_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")
        peer_nd = tree.add_node("Peer", parent=marble_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        return tree

# testing
class TestingFormatsCSParamEnv(SocialAIParamEnv):

    def __init__(self, cue, intro_sec, problem, **kwargs):

        if cue not in CUES:
            raise ValueError(f"Cue {cue} undefined.")
        self.cue = cue

        if intro_sec not in INTRO_SEC:
            raise ValueError(f"Cue {intro_sec} undefined.")
        self.intro_sec = intro_sec

        self.problem = problem
        if self.problem not in PROBLEMS:
            raise ValueError(f"Problem {self.problem} undefined.")

        super(TestingFormatsCSParamEnv, self).__init__(**kwargs)

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

        version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        tree.add_node("2", parent=version_nd, type="value")

        peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        tree.add_node("Y", parent=peer_nd, type="value")

        return tree



# register training envs
for cue in CUES:
    for intro_sec in INTRO_SEC:
        env_name = f'{intro_sec}{cue}TrainFormatsCSParamEnv'

        assert cue == "LangFeedback"

        register(
            id='SocialAI-{}-v1'.format(env_name),
            entry_point='gym_minigrid.social_ai_envs:TrainingFormatsCSParamEnv',
            kwargs={
                'intro_sec': intro_sec,
                'cue': cue,
            }
        )

for intro_sec in INTRO_SEC:
    # scaffolding train env
    for cue in CUES:
        # intro_sec = "AE"
        env_name = f'{intro_sec}{cue}TrainScaffoldingCSParamEnv'

        assert cue == "LangFeedback"

        register(
            id='SocialAI-{}-v1'.format(env_name),
            entry_point='gym_minigrid.social_ai_envs:TrainingFormatsCSParamEnv',
            kwargs={
                'intro_sec': intro_sec,
                'cue': cue,
                'scaffolding': True
            }
        )


# register testing envs : cues x problems x {social, asocial} x {joint attention, no}
for cue in CUES:
    for problem in PROBLEMS:
        for intro_sec in INTRO_SEC:
            env_name = f'{intro_sec}{cue}{problem}TestFormatsCSParamEnv'

            assert cue == "LangFeedback"

            register(
                id='SocialAI-{}-v1'.format(env_name),
                entry_point='gym_minigrid.social_ai_envs:TestingFormatsCSParamEnv',
                kwargs={
                    'problem': problem,
                    'cue': cue,
                    'intro_sec': intro_sec
                }
            )

N_formats_test_set = [
    f"SocialAI-NLangFeedback{problem}TestFormatsCSParamEnv-v1" for problem in PROBLEMS
]
E_formats_test_set = [
    f"SocialAI-ELangFeedback{problem}TestFormatsCSParamEnv-v1" for problem in PROBLEMS
]
A_formats_test_set = [
    f"SocialAI-ALangFeedback{problem}TestFormatsCSParamEnv-v1" for problem in PROBLEMS
]
AE_formats_test_set = [
    f"SocialAI-AELangFeedback{problem}TestFormatsCSParamEnv-v1" for problem in PROBLEMS
]

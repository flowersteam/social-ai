from gym_minigrid.social_ai_envs.socialaiparamenv import SocialAIParamEnv
from gym_minigrid.parametric_env import *
from gym_minigrid.register import register

import inspect, importlib

# for used for automatic registration of environments
defined_classes = [name for name, _ in inspect.getmembers(importlib.import_module(__name__), inspect.isclass)]


# # Pointing case study (table 1)
#
# # training
# class EPointingInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         # tree.add_node("Language_Color", parent=cue_type_nd, type="value")
#         # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#         tree.add_node("Marbles", parent=problem_nd, type="value")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
# # testing
# class EPointingBoxesInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#
#         return tree
#
# class EPointingSwitchesInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#
#         return tree
#
# class EPointingMarbleInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Marble", parent=problem_nd, type="value")
#
#         return tree
#
# class EPointingGeneratorsInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#
#         return tree
#
# class EPointingLeversInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#
#         return tree
#
# class EPointingDoorsInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
#
#
# # Lang Color case study (table 1)
# # training
# class EPointingInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         # tree.add_node("Language_Color", parent=cue_type_nd, type="value")
#         # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#         tree.add_node("Marbles", parent=problem_nd, type="value")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
# # testing
# class EPointingBoxesInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#
#         return tree
#
# class EPointingSwitchesInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#
#         return tree
#
# class EPointingMarbleInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Marble", parent=problem_nd, type="value")
#
#         return tree
#
# class EPointingGeneratorsInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#
#         return tree
#
# class EPointingLeversInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#
#         return tree
#
# class EPointingDoorsInformationSeekingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree




# grid searches envs
# Doors
# class ELanguageColorDoorsInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Color", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
#
# class ELanguageFeedbackDoorsInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Feedback", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
#
#
#
#
# # Levers
# class ELanguageColorLeversInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Color", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#
#         return tree
#
#
# class ELanguageFeedbackLeversInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Feedback", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#
#         return tree
# # Switches
# class ELanguageColorSwitchesInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Color", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#
#         return tree
#
#
# class ELanguageFeedbackSwitchesInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Feedback", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#
#         return tree
#
#
# # Marble
# class ELanguageColorMarbleInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Color", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Marble", parent=problem_nd, type="value")
#
#         return tree
#
#
# class ELanguageFeedbackMarbleInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Feedback", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Marble", parent=problem_nd, type="value")
#
#         return tree
#
#
#
# # Generators
# class ELanguageColorGeneratorsInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Color", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#
#         return tree
#
#
# class ELanguageFeedbackGeneratorsInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Feedback", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#
#         return tree
#
#
#
#
# class CuesGridSearchParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Language_Color", parent=cue_type_nd, type="value")
#         tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#         tree.add_node("Marbles", parent=problem_nd, type="value")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
# class EmulationGridSearchParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Emulation", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#         tree.add_node("Marbles", parent=problem_nd, type="value")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
# class CuesGridSearchPointingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         # tree.add_node("Language_Color", parent=cue_type_nd, type="value")
#         # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#         tree.add_node("Marbles", parent=problem_nd, type="value")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
# class CuesGridSearchLangColorParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Language_Color", parent=cue_type_nd, type="value")
#         # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
#         # tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#         tree.add_node("Marbles", parent=problem_nd, type="value")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
#
# class CuesGridSearchLangFeedbackParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         # tree.add_node("Language_Color", parent=cue_type_nd, type="value")
#         tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
#         # tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#         tree.add_node("Marbles", parent=problem_nd, type="value")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
# class GridSearchParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Language_Color", parent=cue_type_nd, type="value")
#         tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#
#         return tree
#
# class GridSearchPointingParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         # tree.add_node("Language_Color", parent=cue_type_nd, type="value")
#         # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
#         tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#
#         return tree
#
# class GridSearchLangColorParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Language_Color", parent=cue_type_nd, type="value")
#         # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
#         # tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#
#         return tree
#
#
# class GridSearchLangFeedbackParamEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
#         # tree.add_node("Language_Color", parent=cue_type_nd, type="value")
#         tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
#         # tree.add_node("Pointing", parent=cue_type_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#
#         return tree
#
#
# # Boxes
# class ELanguageColorBoxesInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Color", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#
#         return tree
#
#
# class ELanguageFeedbackBoxesInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Feedback", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#
#         return tree
#
#
# class EPointingBoxesInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         pointing_nd = tree.add_node("Pointing", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Direct", parent=pointing_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Boxes", parent=problem_nd, type="value")
#
#         return tree
#
#
#
# # Levers
# class ELanguageColorLeversInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Color", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#
#         return tree
#
#
# class ELanguageFeedbackLeversInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Feedback", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#
#         return tree
#
#
# class EPointingLeversInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         pointing_nd = tree.add_node("Pointing", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Direct", parent=pointing_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Levers", parent=problem_nd, type="value")
#
#         return tree
#
#
#
#
# # Doors
# class ELanguageColorDoorsInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Color", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
#
# class ELanguageFeedbackDoorsInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Feedback", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
#
# class EPointingDoorsInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         pointing_nd = tree.add_node("Pointing", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Direct", parent=pointing_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Doors", parent=problem_nd, type="value")
#
#         return tree
#
#
#
# # Switches
# class ELanguageColorSwitchesInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Color", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#
#         return tree
#
#
# class ELanguageFeedbackSwitchesInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Feedback", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#
#         return tree
#
#
# class EPointingSwitchesInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         pointing_nd = tree.add_node("Pointing", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Direct", parent=pointing_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Switches", parent=problem_nd, type="value")
#
#         return tree
#
#
#
#
# # Marble
# class ELanguageColorMarbleInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Color", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Marble", parent=problem_nd, type="value")
#
#         return tree
#
#
# class ELanguageFeedbackMarbleInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Feedback", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Marble", parent=problem_nd, type="value")
#
#         return tree
#
#
# class EPointingMarbleInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         pointing_nd = tree.add_node("Pointing", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Direct", parent=pointing_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Marble", parent=problem_nd, type="value")
#
#         return tree
#
#
# # Generators
# class ELanguageColorGeneratorsInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Color", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#
#         return tree
#
#
# class ELanguageFeedbackGeneratorsInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         language_grounding_nd = tree.add_node("Language_grounding", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Feedback", parent=language_grounding_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#
#         return tree
#
#
# class EPointingGeneratorsInformationSeekingEnv(SocialAIParamEnv):
#
#     def construct_tree(self):
#         tree = ParameterTree()
#
#         env_type_nd = tree.add_node("Env_type", type="param")
#
#         # Information seeking
#         inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
#
#         prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
#         tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")
#
#         # scaffolding
#         scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
#         scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")
#
#         pointing_nd = tree.add_node("Pointing", parent=scaffolding_N_nd, type="param")
#         tree.add_node("Direct", parent=pointing_nd, type="value")
#
#         N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
#         tree.add_node("2", parent=N_bo_nd, type="value")
#
#         problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
#         tree.add_node("Generators", parent=problem_nd, type="value")
#
#         return tree


# Collaboration
class LeverDoorCollaborationParamEnv(SocialAIParamEnv):
    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Collaboration
        collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")

        colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")
        tree.add_node("LeverDoor", parent=colab_type_nd, type="value")

        role_nd = tree.add_node("Version", parent=collab_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        role_nd = tree.add_node("Role", parent=collab_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        obstacles_nd = tree.add_node("Obstacles", parent=collab_nd, type="param")
        tree.add_node("No", parent=obstacles_nd, type="value")

        return tree


class MarblePushCollaborationParamEnv(SocialAIParamEnv):
    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Collaboration
        collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")

        colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")
        tree.add_node("MarblePush", parent=colab_type_nd, type="value")

        role_nd = tree.add_node("Version", parent=collab_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        role_nd = tree.add_node("Role", parent=collab_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        return tree


class MarblePassCollaborationParamEnv(SocialAIParamEnv):
    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Collaboration
        collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")

        colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")
        tree.add_node("MarblePass", parent=colab_type_nd, type="value")

        role_nd = tree.add_node("Version", parent=collab_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        role_nd = tree.add_node("Role", parent=collab_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        return tree

class MarblePassACollaborationParamEnv(SocialAIParamEnv):
    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Collaboration
        collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")

        colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")
        tree.add_node("MarblePass", parent=colab_type_nd, type="value")

        role_nd = tree.add_node("Version", parent=collab_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        role_nd = tree.add_node("Role", parent=collab_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")

        return tree

class MarblePassBCollaborationParamEnv(SocialAIParamEnv):
    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Collaboration
        collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")

        colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")
        tree.add_node("MarblePass", parent=colab_type_nd, type="value")

        role_nd = tree.add_node("Version", parent=collab_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        role_nd = tree.add_node("Role", parent=collab_nd, type="param")
        tree.add_node("B", parent=role_nd, type="value")

        return tree

class ObjectsCollaborationParamEnv(SocialAIParamEnv):
    def __init__(self, problem=None, **kwargs):

        self.problem = problem

        super(ObjectsCollaborationParamEnv, self).__init__(**kwargs)
    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Collaboration
        collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")

        colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")
        if self.problem is None:
            tree.add_node("Boxes", parent=colab_type_nd, type="value")
            tree.add_node("Switches", parent=colab_type_nd, type="value")
            tree.add_node("Generators", parent=colab_type_nd, type="value")
            tree.add_node("Marble", parent=colab_type_nd, type="value")
        else:
            tree.add_node(self.problem, parent=colab_type_nd, type="value")

        role_nd = tree.add_node("Version", parent=collab_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        role_nd = tree.add_node("Role", parent=collab_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        return tree



class RoleReversalCollaborationParamEnv(SocialAIParamEnv):
    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Collaboration
        collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")

        colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")
        tree.add_node("Boxes", parent=colab_type_nd, type="value")
        tree.add_node("Switches", parent=colab_type_nd, type="value")
        tree.add_node("Generators", parent=colab_type_nd, type="value")
        tree.add_node("Marble", parent=colab_type_nd, type="value")
        tree.add_node("MarblePass", parent=colab_type_nd, type="value")
        tree.add_node("MarblePush", parent=colab_type_nd, type="value")
        tree.add_node("LeverDoor", parent=colab_type_nd, type="value")

        role_nd = tree.add_node("Version", parent=collab_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")

        role_nd = tree.add_node("Role", parent=collab_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        # obstacles_nd = tree.add_node("Obstacles", parent=collab_nd, type="param")
        # tree.add_node("No", parent=obstacles_nd, type="value")

        return tree

class RoleReversalGroupExperimentalCollaborationParamEnv(SocialAIParamEnv):
    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Collaboration
        collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")

        colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")

        problem_nd = tree.add_node("Boxes", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        problem_nd = tree.add_node("Switches", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        problem_nd = tree.add_node("Generators", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        problem_nd = tree.add_node("Marble", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        problem_nd = tree.add_node("MarblePass", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        # tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        problem_nd = tree.add_node("MarblePush", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        problem_nd = tree.add_node("LeverDoor", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        role_nd = tree.add_node("Version", parent=collab_nd, type="param")
        tree.add_node("Social", parent=role_nd, type="value")


        # obstacles_nd = tree.add_node("Obstacles", parent=collab_nd, type="param")
        # tree.add_node("No", parent=obstacles_nd, type="value")

        return tree

class RoleReversalGroupControlCollaborationParamEnv(SocialAIParamEnv):
    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

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
        tree.add_node("B", parent=role_nd, type="value")

        role_nd = tree.add_node("Version", parent=problem_nd, type="param")
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


        # obstacles_nd = tree.add_node("Obstacles", parent=collab_nd, type="param")
        # tree.add_node("No", parent=obstacles_nd, type="value")

        return tree

class AsocialMarbleCollaborationParamEnv(SocialAIParamEnv):

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Collaboration
        collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")

        colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")

        problem_nd = tree.add_node("MarblePass", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Role", parent=problem_nd, type="param")
        # tree.add_node("A", parent=role_nd, type="value")
        tree.add_node("B", parent=role_nd, type="value")

        role_nd = tree.add_node("Version", parent=problem_nd, type="param")
        tree.add_node("Asocial", parent=role_nd, type="value")

        return tree

class AsocialMarbleInformationSeekingParamEnv(SocialAIParamEnv):

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        # irrelevant because no peer: todo: remove?
        prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")

        tree.add_node("No", parent=prag_fr_compl_nd, type="value")

        # irrelevant because no peer, todo: remove?
        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        tree.add_node("Language_Color", parent=cue_type_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
        boxes_nd = tree.add_node("Marble", parent=problem_nd, type="value")

        version_nd = tree.add_node("N", parent=boxes_nd, type="param")
        tree.add_node("1", parent=version_nd, type="value")

        peer_nd = tree.add_node("Peer", parent=boxes_nd, type="param")
        tree.add_node("N", parent=peer_nd, type="value")

        return tree

# automatic registration of environments
defined_classes_ = [name for name, _ in inspect.getmembers(importlib.import_module(__name__), inspect.isclass)]

envs = list(set(defined_classes_) - set(defined_classes))
assert all([e.endswith("Env") for e in envs])

for env in envs:
    register(
        id='SocialAI-{}-v1'.format(env),
        entry_point='gym_minigrid.social_ai_envs:{}'.format(env)
    )

PROBLEMS = ["Boxes", "Switches", "Generators", "Marble"]
for problem in PROBLEMS:
    env_name = f'Objects{problem}CollaborationParamEnv'

    register(
        id='SocialAI-{}-v1'.format(env_name),
        entry_point='gym_minigrid.social_ai_envs:ObjectsCollaborationParamEnv',
        kwargs={
            'problem': problem,
        }
    )


role_reversal_test_set = [
    "SocialAI-LeverDoorCollaborationParamEnv-v1",
    "SocialAI-MarblePushCollaborationParamEnv-v1",
    "SocialAI-MarblePassACollaborationParamEnv-v1",
    "SocialAI-MarblePassBCollaborationParamEnv-v1",
    "SocialAI-AsocialMarbleCollaborationParamEnv-v1",
    "SocialAI-ObjectsBoxesCollaborationParamEnv-v1",
    "SocialAI-ObjectsSwitchesCollaborationParamEnv-v1",
    "SocialAI-ObjectsGeneratorsCollaborationParamEnv-v1",
    "SocialAI-ObjectsMarbleCollaborationParamEnv-v1",
]
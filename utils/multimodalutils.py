from enum import IntEnum
import numpy as np
import gym.spaces as spaces
import torch

raise DeprecationWarning("Do not use this. Grammar is defined in the env class; SocialAIGrammar is socialaigrammar.py")

# class Grammar(object):
#
#     templates = ["Where is ", "Who is"]
#     things = ["me", "exit", "you", "him", "task"]
#
#     grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])
#
#     @classmethod
#     def construct_utterance(cls, action):
#         return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + ". "

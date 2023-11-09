import gym.spaces as spaces
from enum import IntEnum

# Enumeration of possible actions
class SocialAIActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2

    # no pickup-drop
    # # Pick up an object
    # pickup = 3
    # # Drop an object
    # drop = 4

    # Toggle/activate an object
    toggle = 3

    # Done completing task
    done = 4


class SocialAIGrammar(object):

    templates = ["Where is", "Help", "Close", "How are"]
    things = [
        "please", "the exit", "the wall", "you", "the ceiling", "the window", "the entrance", "the closet",
        "the drawer", "the fridge", "the floor", "the lamp", "the trash can", "the chair", "the bed", "the sofa"
    ]
    assert len(templates)*len(things) == 64
    print("language complexity {}:".format(len(templates)*len(things)))

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def get_action(cls, template, thing):
        return [cls.templates.index(template), cls.things.index(thing)]
    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "

    @classmethod
    def contains_utterance(cls, utterance):
        for t in range(len(cls.templates)):
            for th in range(len(cls.things)):
                if utterance == cls.construct_utterance([t, th]):
                    return True
        return False

SocialAIActionSpace = spaces.MultiDiscrete([len(SocialAIActions),
                                                  *SocialAIGrammar.grammar_action_space.nvec])

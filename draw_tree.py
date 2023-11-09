#!/usr/bin/env python3
import sys

from utils import *
from gym_minigrid.parametric_env import *

class DummyTreeParamEnv(gym.Env):
    """
    Meta-Environment containing all other environment (multi-task learning)
    """

    def __init__(
            self,
    ):

        # construct the tree
        self.parameter_tree = self.construct_tree()
        self.parameter_tree.print_tree()

    def draw_tree(self, ignore_labels=[], folded_nodes=[]):
        self.parameter_tree.draw_tree("viz/param_tree_{}".format(self.spec.id), ignore_labels=ignore_labels, folded_nodes=folded_nodes)

    def print_tree(self):
        self.parameter_tree.print_tree()

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Information seeking
        inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")

        prag_fr_compl_nd = tree.add_node("Introductory_sequence", parent=inf_seeking_nd, type="param")
        tree.add_node("Eye_contact", parent=prag_fr_compl_nd, type="value")

        # scaffolding
        scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
        scaffolding_N_nd = tree.add_node("N", parent=scaffolding_nd, type="value")

        cue_type_nd = tree.add_node("Cue_type", parent=scaffolding_N_nd, type="param")
        # tree.add_node("Language_Color", parent=cue_type_nd, type="value")
        # tree.add_node("Language_Feedback", parent=cue_type_nd, type="value")
        tree.add_node("Pointing", parent=cue_type_nd, type="value")

        # N_bo_nd = tree.add_node("N", parent=inf_seeking_nd, type="param")
        # tree.add_node("2", parent=N_bo_nd, type="value")

        problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
        tree.add_node("Boxes", parent=problem_nd, type="value")
        tree.add_node("Switches", parent=problem_nd, type="value")
        tree.add_node("Marbles", parent=problem_nd, type="value")
        tree.add_node("Generators", parent=problem_nd, type="value")
        tree.add_node("Doors", parent=problem_nd, type="value")
        tree.add_node("Levers", parent=problem_nd, type="value")

        return tree



filename = sys.argv[1]

if len(sys.argv) > 2:
    env_name = sys.argv[2]
    env = gym.make(env_name)

else:
    env = DummyTreeParamEnv()

# draw tree

folded_nodes = [
    # "Information_Seeking",
    # "Perspective_Inference",
]


# selected_parameters_labels = {
#     "Env_type": "Information_Seeking",
#     "Distractor": "Yes",
#     "Problem": "Boxes",
# }

env.parameter_tree.draw_tree(
    filename=f"viz/{filename}",
    ignore_labels=["Num_of_colors"],
    # selected_parameters=selected_parameters_labels,
    folded_nodes=folded_nodes,
    label_parser={
        "Scaffolding": "Help"
    }
)

# for i in range(3):
#     params = env.parameter_tree.sample_env_params()
#     selected_parameters_labels = {k.label: v.label for k, v in params.items()}
#
#     env.parameter_tree.draw_tree(
#         filename=f"viz/{filename}_{i}",
#         ignore_labels=["Num_of_colors"],
#         selected_parameters=selected_parameters_labels,
#         folded_nodes=folded_nodes,
#     )
#

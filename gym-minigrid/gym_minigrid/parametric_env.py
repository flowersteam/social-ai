from abc import ABC
from graphviz import Digraph
import re
import random
from termcolor import cprint
from collections import defaultdict


class Node:
    def __init__(self, id, label, type, parent=None):
        """
        type: type must be "param" or "value"
        for type "param" one of the children must be chosen
        for type "value" all of the children must be set
        """
        self.id = id
        self.label = label
        self.parent = parent
        self.children = []
        self.type = type

        # calculate node's level
        parent_ = self.parent
        self.level = 1
        while parent_ is not None:
            self.level += 1
            parent_ = parent_.parent

    def __repr__(self):
        return f"{self.id}({self.type})-'{self.label}'"


class ParameterTree(ABC):

    def __init__(self):

        self.last_node_id = 0

        self.create_digraph()

        self.nodes = {}
        self.root = None

    def create_digraph(self):
        self.tree = Digraph("unix", format='svg')
        self.tree.attr(size='30,100')

    def get_node_for_id(self, id):
        return self.nodes[id]

    def add_node(self, label, parent=None, type="param"):
        """
        All children of this node must be set
        """
        if type not in ["param", "value"]:
            raise ValueError('Node type must be "param" or "value"')

        if parent is None and self.root is not None:
            raise ValueError("Root already set: {}. parent cannot be None. ".format(self.root.id))

        # add to graph
        node_id = self.new_node_id()
        self.nodes[node_id] = Node(id=node_id, label=label, parent=parent, type=type)

        if parent is None:
            self.root = self.nodes[node_id]
        else:
            self.nodes[parent.id].children.append(self.nodes[node_id])

        return self.nodes[node_id]

    def sample_env_params(self, ACL=None):
        parameters = {}

        nodes = [self.root]

        # BFS
        while nodes:
            node = nodes.pop(0)

            if node.type == "param":

                if len(node.children) == 0:
                    raise ValueError("Node {} doesn't have any children.".format(node.label))

                if ACL is None:
                    # choose randomly
                    chosen = random.choice(node.children)
                else:
                    # let the ACL choose
                    chosen = ACL.choose(node, parameters)

                assert chosen.type == "value"
                nodes.append(chosen)

                parameters[node] = chosen

            elif node.type == "value":
                nodes.extend(node.children)

            else:
                raise ValueError('Node type must be "param" or "value" and is {}'.format(node.type))

        return parameters

    def new_node_id(self):
        new_id = self.last_node_id + 1
        self.last_node_id = new_id
        return str("node_"+str(new_id))

    def print_tree(self, selected_parameters={}):
        print("Parameter tree")

        nodes = [self.root]
        color = None

        # BFS
        while nodes:
            node = nodes.pop(0)

            if node.type == "param":
                if node in selected_parameters.keys():
                    color = "blue"
                else:
                    color = None

            if node.parent is not None:

                cprint("{}: {} ({}) -----> {}: {} ({})".format(
                    node.parent.type, node.parent.label, node.parent.id,
                    node.type, node.label, node.id
                ), color)

            else:
                cprint("{}: {} ({})".format(node.type, node.label, node.id), color)

            nodes.extend(node.children)

    def get_all_params(self):
        all_params = defaultdict(list)

        nodes = [self.root]
        while nodes:
            node = nodes.pop(0)

            if node.type == "value":
                all_params[node.parent].append(node)

            nodes.extend(node.children)

        return all_params

    def draw_tree(self, filename, selected_parameters={}, ignore_labels=[], folded_nodes=[], label_parser={}, save=True):

        self.create_digraph()

        nodes = [self.root]

        color_param = "grey60"
        color_value = "lightgray"
        fontcolor = "black"
        fontsize = "18"

        dots_fontsize = "30"
        folded_param = "grey95"
        folded_value = "grey95"
        folded_fontcolor = "gray70"

        def add_fold_symbol(label, folded=False):
            return label
            # return label + " ❯" if folded else label

        # BFS - construct vizgraph
        while nodes:

            node = nodes.pop(0)

            while node.label in ignore_labels:
                node = nodes.pop(0)

            if node.label in folded_nodes:

                n_label = label_parser.get(node.label, node.label)
                n_label = add_fold_symbol(n_label, folded=True)

                if node.type == "param":
                    color = folded_param
                    self.tree.attr('node', shape='box', style="filled", color=color, fontcolor=folded_fontcolor, fontsize=fontsize)
                    self.tree.node(name=node.id, label=n_label, type="parameter")

                elif node.type == "value":
                    color = folded_value
                    self.tree.attr('node', shape='ellipse', style='filled', color=color, fontcolor=folded_fontcolor, fontsize=fontsize)
                    self.tree.node(name=node.id, label=n_label, type="value")


                else:
                    raise ValueError(f"Undefined node type {node.type}")

                # add folded node sign
                folded_node_id = node.id+"_fold"
                # self.tree.attr('node', shape='ellipse', style='filled', color="white", fontcolor=folded_fontcolor, fontsize=fontsize)
                # self.tree.attr('node', shape='none', style='filled', color="gray", fontcolor=folded_fontcolor, fontsize=dots_fontsize)
                self.tree.attr('node', shape='none', color="white",fontcolor=folded_fontcolor, fontsize=dots_fontsize)
                self.tree.node(name=folded_node_id, label="...", type="value")
                self.tree.edge(node.id, folded_node_id, color=folded_fontcolor)

            elif node.type == "param":

                if node.label in selected_parameters.keys() and (node == self.root or node.parent.selected):
                    color = "lightblue3"
                    node.selected=True
                else:
                    color = color_param
                    node.selected=False

                n_label = label_parser.get(node.label, node.label)
                n_label = add_fold_symbol(n_label, folded=False)

                self.tree.attr('node', shape='box', style="filled", color=color, fontcolor=fontcolor, fontsize=fontsize)
                self.tree.node(name=node.id, label=n_label, type="parameter")

                nodes.extend(node.children)

            elif node.type == "value":

                if (selected_parameters.get(node.parent.label, "Not existent") == node.label) and (node == self.root or node.parent.selected):
                # if node.label in selected_parameters.values() and (node == self.root or node.parent.selected):
                    color = "lightblue2"
                    node.selected = True
                else:
                    color = color_value
                    node.selected = False

                n_label = label_parser.get(node.label, node.label)
                n_label = add_fold_symbol(n_label, folded=False)

                # add to vizgraph
                self.tree.attr('node', shape='ellipse', style='filled', color=color, fontcolor=fontcolor, fontsize=fontsize)
                self.tree.node(name=node.id, label=n_label, type="value")

                nodes.extend(node.children)
            else:
                raise ValueError(f"Undefined node type {node.type}")

            if node.parent is not None:
                self.tree.edge(node.parent.id, node.id)


        # draw image
        if save:
            self.tree.render(filename)
            print("Tree image saved in : {}".format(filename))


if __name__ == '__main__':
    # demo of how to use the ParameterTree class

    tree = ParameterTree()
    env_type_nd = tree.add_node("Env_type", type="param")

    inf_seeking_nd = tree.add_node("Information_seeking", parent=env_type_nd, type="value")
    collab_nd = tree.add_node("Collaboration", parent=env_type_nd, type="value")
    perc_inf_nd = tree.add_node("Perception_inference", parent=env_type_nd, type="value")
    raise DeprecationWarning("deprecated parameters")

    # Information seeking
    scaffolding_nd = tree.add_node("Scaffolding", parent=inf_seeking_nd, type="param")
    tree.add_node("lot", parent=scaffolding_nd, type="value")
    tree.add_node("medium", parent=scaffolding_nd, type="value")
    tree.add_node("little", parent=scaffolding_nd, type="value")
    tree.add_node("none", parent=scaffolding_nd, type="value")

    prag_fr_compl_nd = tree.add_node("Pragmatic_frame_complexity", parent=inf_seeking_nd, type="param")
    tree.add_node("Eye contact", parent=prag_fr_compl_nd, type="value")
    tree.add_node("Hello", parent=prag_fr_compl_nd, type="value")

    emulation_nd = tree.add_node("Emulation", parent=inf_seeking_nd, type="param")
    tree.add_node("N", parent=emulation_nd, type="value")
    tree.add_node("Y", parent=emulation_nd, type="value")

    pointing_nd = tree.add_node("Pointing", parent=inf_seeking_nd, type="param")
    tree.add_node("No", parent=pointing_nd, type="value")
    tree.add_node("Direct", parent=pointing_nd, type="value")
    tree.add_node("Indirect", parent=pointing_nd, type="value")

    language_graounding_nd = tree.add_node("Language_grounding", parent=inf_seeking_nd, type="param")
    tree.add_node("No", parent=language_graounding_nd, type="value")
    tree.add_node("Color", parent=language_graounding_nd, type="value")
    tree.add_node("Feedback", parent=language_graounding_nd, type="value")

    problem_nd = tree.add_node("Problem", parent=inf_seeking_nd, type="param")
    tree.add_node("Boxes", parent=problem_nd, type="value")
    tree.add_node("Switches", parent=problem_nd, type="value")
    tree.add_node("Corridors", parent=problem_nd, type="value")

    obstacles_nd = tree.add_node("Obstacles", parent=inf_seeking_nd, type="param")
    tree.add_node("no", parent=obstacles_nd, type="value")
    tree.add_node("lava", parent=obstacles_nd, type="value")
    tree.add_node("walls", parent=obstacles_nd, type="value")

    # Collaboration
    colab_type_nd = tree.add_node("Collaboration type", parent=collab_nd, type="param")
    tree.add_node("Door Lever", parent=colab_type_nd, type="value")
    tree.add_node("Door Button", parent=colab_type_nd, type="value")
    tree.add_node("Marble Run", parent=colab_type_nd, type="value")
    tree.add_node("Marble Pass", parent=colab_type_nd, type="value")

    role_nd = tree.add_node("Role", parent=collab_nd, type="param")
    tree.add_node("A", parent=role_nd, type="value")
    tree.add_node("B", parent=role_nd, type="value")
    tree.add_node("asocial", parent=role_nd, type="value")

    # Perception inference
    NPC_movement_nd = tree.add_node("NPC movement", parent=perc_inf_nd, type="param")
    tree.add_node("can't turn; can't move", parent=NPC_movement_nd, type="value")
    tree.add_node("can turn; can't move", parent=NPC_movement_nd, type="value")
    tree.add_node("can turn; can move", parent=NPC_movement_nd, type="value")

    occlusion_nd = tree.add_node("Occlusions", parent=perc_inf_nd, type="param")
    tree.add_node("no", parent=occlusion_nd, type="value")
    tree.add_node("walls", parent=occlusion_nd, type="value")

    params = tree.sample_env_params()
    tree.draw_tree("viz/demotree", params)


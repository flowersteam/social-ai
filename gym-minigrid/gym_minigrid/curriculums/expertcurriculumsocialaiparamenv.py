import warnings

import numpy as np
import random

class ScaffoldingExpertCurriculum:

    def __init__(self, type, minimum_episodes=1000, average_interval=500, phase_thresholds=(0.75, 0.75)):
        self.phase = 1
        self.performance_history = []
        self.phase_two_current_type = None
        self.minimum_episodes = minimum_episodes
        self.phase_thresholds = phase_thresholds # how many episodes to wait for before starting to compute the estimate
        self.average_interval = average_interval  # number of episodes to use to estimate current performance (100 ~ 10 updated)
        self.mean_perf = 0
        self.max_mean_perf = 0
        self.type = type

    def get_status_dict(self):
        return {
            "curriculum_phase": self.phase,
            "curriculum_performance_history": self.performance_history,
        }

    def load_status_dict(self, status):
        self.phase = status["curriculum_phase"]
        self.performance_history = status["curriculum_performance_history"]

    @staticmethod
    def select(children, label):
        ch = list(filter(lambda c: c.label == label, children))

        if len(ch) == 0:
            raise ValueError(f"Label {label} not found in children {children}.")
        elif len(ch) > 1:
            raise ValueError(f"Multiple labels {label} found in children {children}.")

        selected = ch[0]
        assert selected is not None
        return selected

    def choose(self, node, chosen_parameters):
        """
        Choose a child of the parameter node.
        All the parameters used here should be updated by set_curriculum_parameters.
        """
        assert node.type == 'param'

        # E + scaf
        # E + full
        # AE + full

        # N cs -> N full -> A/E/N/AE full -> AE full

        # A/E/N/AE scaf/full -> AE full
        if len(self.phase_thresholds) < 2:
            warnings.WarningMessage(f"Num of thresholds ({len(self.phase_thresholds)}) is less than the num of phases.")

        if node.label == "Scaffolding":

            if self.type == "intro_seq":
                return ScaffoldingExpertCurriculum.select(node.children, "N")

            elif self.type == "intro_seq_scaf":
                if self.phase in [1]:
                    return random.choice(node.children)

                elif self.phase in [2]:
                    return ScaffoldingExpertCurriculum.select(node.children, "N")

                else:
                    raise ValueError(f"Undefined phase {self.phase}.")

            else:
                raise ValueError(f"Curriculum type {self.type} unknown.")

        elif node.label == "Pragmatic_frame_complexity":

            if self.type not in ["intro_seq", "intro_seq_scaf"]:
                raise ValueError(f"Undefined type {self.type}.")

            if self.phase in [1]:
                # return random.choice(node.children)
                return random.choice([
                    ScaffoldingExpertCurriculum.select(node.children, "No"),
                    ScaffoldingExpertCurriculum.select(node.children, "Ask"),
                    ScaffoldingExpertCurriculum.select(node.children, "Eye_contact"),
                    ScaffoldingExpertCurriculum.select(node.children, "Ask_Eye_contact"),
                ])

            elif self.phase in [2]:
                return ScaffoldingExpertCurriculum.select(node.children, "Ask_Eye_contact")

            else:
                raise ValueError(f"Undefined phase {self.phase}")

        else:
            return random.choice(node.children)

    def set_parameters(self, params):
        """
        Set ALL the parameters used in choose.
        This is important for parallel environments. This function is called by broadcast_curriculum_parameters()
        """
        self.phase = params["phase"]
        self.mean_perf = params["mean_perf"]
        self.max_mean_perf = params["max_mean_perf"]

    def get_parameters(self):
        """
        Get ALL the parameters used in choose. Used when restoring the curriculum.
        """
        return {
            "phase": self.phase,
            "mean_perf": self.mean_perf,
            "max_mean_perf": self.max_mean_perf,
        }

    def update_parameters(self, data):
        """
        Updates the parameters of the ACL used in choose().
        If using parallel processes these parameters should be broadcasted with broadcast_curriculum_parameters()
        """
        for obs, reward, done, info in zip(data["obs"], data["reward"], data["done"], data["info"]):
            if not done:
                continue

            self.performance_history.append(info["success"])
            self.mean_perf = np.mean(self.performance_history[-self.average_interval:])
            self.max_mean_perf = max(self.mean_perf, self.max_mean_perf)

            if self.phase in [1]:
                if len(self.performance_history) > self.minimum_episodes and self.mean_perf >= self.phase_thresholds[self.phase-1]:
                    # next phase
                    self.phase = self.phase + 1
                    self.performance_history = []
                    self.max_mean_perf = 0

        return self.get_parameters()

    def get_info(self):
        return {"param": self.phase, "mean_perf": self.mean_perf, "max_mean_perf": self.max_mean_perf}


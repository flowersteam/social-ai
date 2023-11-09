from gym_minigrid.social_ai_envs.socialaiparamenv import SocialAIParamEnv
from gym_minigrid.parametric_env import *
from gym_minigrid.register import register

import inspect, importlib

# for used for automatic registration of environments
defined_classes = [name for name, _ in inspect.getmembers(importlib.import_module(__name__), inspect.isclass)]

class AppleStealingParamEnv(SocialAIParamEnv):

    def __init__(self, obstacles, asocial, walk, **kwargs):

        self.asocial = asocial
        self.obstacles = obstacles
        self.walk = walk

        super(AppleStealingParamEnv, self).__init__(**kwargs)

    def construct_tree(self):
        tree = ParameterTree()

        env_type_nd = tree.add_node("Env_type", type="param")

        # Collaboration
        collab_nd = tree.add_node("AppleStealing", parent=env_type_nd, type="value")

        # colab_type_nd = tree.add_node("Problem", parent=collab_nd, type="param")
        # tree.add_node("AppleStealing", parent=colab_type_nd, type="value")
        role_nd = tree.add_node("Version", parent=collab_nd, type="param")
        if self.asocial:
            tree.add_node("Asocial", parent=role_nd, type="value")
        else:
            social_nd = tree.add_node("Social", parent=role_nd, type="value")

            role_nd = tree.add_node("NPC_movement", parent=social_nd, type="param")
            if self.walk:
                tree.add_node("Walking", parent=role_nd, type="value")
            else:
                tree.add_node("Rotating", parent=role_nd, type="value")

        obstacles_nd = tree.add_node("Obstacles", parent=collab_nd, type="param")

        if self.obstacles not in ["No", "A_bit", "Medium", "A_lot"]:
            raise ValueError("Undefined obstacle amount.")

        tree.add_node(self.obstacles, parent=obstacles_nd, type="value")

        return tree


# automatic registration of environments
defined_classes_ = [name for name, _ in inspect.getmembers(importlib.import_module(__name__), inspect.isclass)]

envs = list(set(defined_classes_) - set(defined_classes))
assert all([e.endswith("Env") for e in envs])


# register testing envs : cues x problems x {social, asocial} x {joint attention, no}
for asocial in [True, False]:
    for obst in ["No", "A_bit", "Medium", "A_lot"]:
        if asocial:
            env_name = f'{"Asocial" if asocial else ""}AppleStealingObst_{obst}ParamEnv'

            register(
                id='SocialAI-{}-v1'.format(env_name),
                entry_point='gym_minigrid.social_ai_envs:AppleStealingParamEnv',
                kwargs={
                    'asocial': asocial,
                    'obstacles': obst,
                    'walk': False,
                }
            )

        else:
            for walk in [True, False]:
                env_name = f'{"Asocial" if asocial else ""}AppleStealing{"Walk" if walk and not asocial else ""}Obst_{obst}ParamEnv'

                register(
                    id='SocialAI-{}-v1'.format(env_name),
                    entry_point='gym_minigrid.social_ai_envs:AppleStealingParamEnv',
                    kwargs={
                        'asocial': asocial,
                        'obstacles': obst,
                        'walk': walk,
                    }
                )

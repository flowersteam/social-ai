from itertools import chain
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 
from gym_minigrid.envs import DanceWithOneNPC8x8Env, CoinThief8x8Env, TalkItOutPolite8x8Env, ShowMe8x8Env, \
    DiverseExit8x8Env, Exiter8x8Env, Helper8x8Env
from gym_minigrid.envs import DanceWithOneNPCGrammar, CoinThiefGrammar, TalkItOutPoliteGrammar, DemonstrationGrammar, \
    EasyTeachingGamesGrammar, ExiterGrammar
import time
from collections import deque


class SocialEnvMetaGrammar(object):

    def __init__(self, grammar_list, env_list):
        self.templates = []
        self.things = []
        self.original_template_idx = []
        self.original_thing_idx = []

        self.meta_template_idx_to_env_name = {}
        self.meta_thing_idx_to_env_name = {}
        self.template_idx, self.thing_idx = 0, 0
        env_names = [e.__class__.__name__ for e in env_list]

        for g, env_name in zip(grammar_list, env_names):
            # add templates
            self.templates += g.templates
            # add things
            self.things += g.things

            # save original idx for both
            self.original_template_idx += list(range(0, len(g.templates)))
            self.original_thing_idx += list(range(0, len(g.things)))

            # update meta_idx to env_names dictionaries
            self.meta_template_idx_to_env_name.update(dict.fromkeys(list(range(self.template_idx,
                                                                               self.template_idx + len(g.templates))),
                                                                    env_name))
            self.template_idx += len(g.templates)

            self.meta_thing_idx_to_env_name.update(dict.fromkeys(list(range(self.thing_idx,
                                                                            self.thing_idx + len(g.things))),
                                                                 env_name))
            self.thing_idx += len(g.things)

        self.grammar_action_space = spaces.MultiDiscrete([len(self.templates), len(self.things)])

    @classmethod
    def construct_utterance(self, action):
        return self.templates[int(action[0])] + " " + self.things[int(action[1])] + " "

    @classmethod
    def random_utterance(self):
        return np.random.choice(self.templates) + " " + np.random.choice(self.things) + " "

    def construct_original_action(self, action, current_env_name):
        template_env_name = self.meta_template_idx_to_env_name[int(action[0])]
        thing_env_name = self.meta_thing_idx_to_env_name[int(action[1])]

        if template_env_name == current_env_name and thing_env_name == current_env_name:
            original_action = [self.original_template_idx[int(action[0])], self.original_thing_idx[int(action[1])]]
        else:
            original_action = [np.nan, np.nan]
        return original_action


class SocialEnv(gym.Env):
    """
    Meta-Environment containing all other environment (multi-task learning)
    """

    def __init__(
            self,
            size=8,
            hidden_npc=False,
            is_test_env=False

    ):

        # Number of cells (width and height) in the agent view
        self.agent_view_size = 7

        # Number of object dimensions (i.e. number of channels in symbolic image)
        self.nb_obj_dims = 4

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, self.nb_obj_dims),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        self.hidden_npc = hidden_npc  # TODO: implement hidden npc

        # TODO get max step from env list

        self.env_list = [DanceWithOneNPC8x8Env, CoinThief8x8Env, TalkItOutPolite8x8Env, ShowMe8x8Env, DiverseExit8x8Env,
                         Exiter8x8Env]
        self.all_npc_utterance_actions = sorted(list(set(chain(*[e.all_npc_utterance_actions for e in self.env_list]))))
        self.grammar_list = [DanceWithOneNPCGrammar, CoinThiefGrammar, TalkItOutPoliteGrammar, DemonstrationGrammar,
                             EasyTeachingGamesGrammar, ExiterGrammar]

        if is_test_env:
            self.env_list[-1] = Helper8x8Env

        # instanciate all envs
        self.env_list = [env() for env in self.env_list]

        self.current_env = None

        self.metaGrammar = SocialEnvMetaGrammar(self.grammar_list, self.env_list)

        # Actions are discrete integer values
        self.action_space = spaces.MultiDiscrete([len(MiniGridEnv.Actions),
                                                  *self.metaGrammar.grammar_action_space.nvec])
        self.actions = MiniGridEnv.Actions

        self._window = None

    def reset(self):
        # select a new social environment at random, for each new episode

        old_window = None
        if self.current_env:  # a previous env exists, save old window
            old_window = self.current_env.window

        # sample new environment
        self.current_env = np.random.choice(self.env_list)
        obs = self.current_env.reset()

        # carry on window if this env is not the first
        if old_window:
            self.current_env.window = old_window
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        for env in self.env_list:
            env.seed(seed)
        np.random.seed(seed)
        return [seed]

    def step(self, action):
        assert (self.current_env)
        if len(action) == 1:  # agent cannot speak
            utterance_action = [np.nan, np.nan]
        else:
            utterance_action = action[1:]

        if len(action) >= 1 and not all(np.isnan(utterance_action)):  # if agent speaks, contruct env-specific action
            action[1:] = self.metaGrammar.construct_original_action(action[1:], self.current_env.__class__.__name__)

        return self.current_env.step(action)

    @property
    def window(self):
        return self.current_env.window

    @window.setter
    def window(self, value):
        self.current_env.window = value

    def render(self, *args, **kwargs):
        assert self.current_env
        return self.current_env.render(*args, **kwargs)

    @property
    def step_count(self):
        return self.current_env.step_count

    def get_mission(self):
        return self.current_env.get_mission()


class SocialEnv8x8Env(SocialEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, **kwargs)


register(
    id='MiniGrid-SocialEnv-5x5-v0',
    entry_point='gym_minigrid.envs:SocialEnvEnv'
)

register(
    id='MiniGrid-SocialEnv-8x8-v0',
    entry_point='gym_minigrid.envs:SocialEnv8x8Env'
)

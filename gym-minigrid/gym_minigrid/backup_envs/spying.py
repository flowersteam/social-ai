import numpy as np

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 
import time
from collections import deque


class Peer(NPC):
    """
    A dancing NPC that the agent has to copy
    """

    def __init__(self, color, name, env, knowledgeable=False):
        super().__init__(color)
        self.name = name
        self.npc_dir = 1  # NPC initially looks downward
        self.npc_type = 0
        self.env = env
        self.knowledgeable = knowledgeable 
        self.npc_actions = []
        self.dancing_step_idx = 0
        self.actions = MiniGridEnv.Actions
        self.add_npc_direction = True
        self.available_moves = [self.rotate_left, self.rotate_right, self.go_forward, self.toggle_action]
        self.exited = False

    def step(self):
        if self.exited:
            return

        if all(np.array(self.cur_pos) == np.array(self.env.door_pos)):
            # disappear
            self.env.grid.set(*self.cur_pos, self.env.object)
            self.cur_pos = np.array([np.nan, np.nan])

            # close door
            self.env.object.toggle(self.env, self.cur_pos)

            # reset switches door
            for s in self.env.switches:
                s.is_on = False

            # update door
            self.env.update_door_lock()

            self.exited = True

        elif self.knowledgeable:

            if self.env.object.is_locked:
                first_wrong_id = np.where(self.env.get_selected_password() != self.env.password)[0][0]
                print("first_wrong_id:", first_wrong_id)
                goal_pos = self.env.switches_pos[first_wrong_id]
                act = self.path_to_toggle_pos(goal_pos)
                act()

            else:
                if all(self.front_pos == self.env.door_pos) and self.env.object.is_open:
                    self.go_forward()

                else:
                    act = self.path_to_toggle_pos(self.env.door_pos)
                    act()

        else:
            self.env._rand_elem(self.available_moves)()

        self.env.update_door_lock()


class SpyingGrammar(object):

    templates = ["Move your", "Shake your"]
    things = ["body", "head"]

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "


class SpyingEnv(MultiModalMiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self,
        size=5,
        diminished_reward=True,
        step_penalty=False,
        knowledgeable=False,
        hard_password=False,
        max_steps=None,
        n_switches=3
    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.knowledgeable = knowledgeable
        self.hard_password = hard_password
        self.n_switches = n_switches

        super().__init__(
            grid_size=size,
            max_steps=max_steps or 5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True,
            actions=MiniGridEnv.Actions,
            action_space=spaces.MultiDiscrete([
                len(MiniGridEnv.Actions),
                *SpyingGrammar.grammar_action_space.nvec
            ]),
            add_npc_direction=True
        )

        print({
            "size": size,
            "diminished_reward": diminished_reward,
            "step_penalty": step_penalty,
        })

    def get_selected_password(self):
        return np.array([int(s.is_on) for s in self.switches])

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height, nb_obj_dims=4)

        # Randomly vary the room width and height
        width = self._rand_int(5, width+1)
        height = self._rand_int(5, height+1)

        self.wall_x = width - 1
        self.wall_y = height - 1

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        door_color = self._rand_elem(COLOR_NAMES)

        wall_for_door = self._rand_int(1, 4)

        if wall_for_door < 2:
            w = self._rand_int(1, width-1)
            h = height-1 if wall_for_door == 0 else 0
        else:
            w = width-1 if wall_for_door == 3 else 0
            h = self._rand_int(1, height-1)

        assert h != height-1  # door mustn't be on the bottom wall

        self.door_pos = (w, h)
        self.door = Door(door_color, is_locked=True)
        self.grid.set(*self.door_pos, self.door)

        # add the switches
        self.switches = []
        self.switches_pos = []
        for i in range(self.n_switches):
            c = COLOR_NAMES[i]
            pos = np.array([i+1, height-1])
            sw = Switch(c)
            self.grid.set(*pos, sw)
            self.switches.append(sw)
            self.switches_pos.append(pos)

        # sample password
        if self.hard_password:
            self.password = np.array([self._rand_int(0, 2) for _ in range(self.n_switches)])

        else:
            idx = self._rand_int(0, self.n_switches)
            self.password = np.zeros(self.n_switches)
            self.password[idx] = 1.0

        # Set a randomly coloured Dancer NPC
        color = self._rand_elem(COLOR_NAMES)
        self.peer = Peer(color, "Jim", self, knowledgeable=self.knowledgeable)

        # Place it on the middle left side of the room
        peer_pos = np.array((self._rand_int(1, width - 1), self._rand_int(1, height - 1)))

        self.grid.set(*peer_pos, self.peer)
        self.peer.init_pos = peer_pos
        self.peer.cur_pos = peer_pos

        # Randomize the agent's start position and orientation
        self.place_agent(size=(width, height))

        # Generate the mission string
        self.mission = 'exit the room'

        # Dummy beginning string
        self.beginning_string = "This is what you hear. \n"
        self.utterance = self.beginning_string

        # utterance appended at the end of each step
        self.utterance_history = ""

        # used for rendering
        self.conversation = self.utterance

    def update_door_lock(self):
        if np.array_equal(self.get_selected_password(), self.password):
            self.door.is_locked = False
        else:
            self.door.is_locked = True
            self.door.is_open = False

    def step(self, action):
        p_action = action[0]
        utterance_action = action[1:]

        obs, reward, done, info = super().step(p_action)
        self.update_door_lock()

        print("pass:", self.password)

        if p_action == self.actions.done:
            done = True

        self.peer.step()

        if all(self.agent_pos == self.door_pos):
            done = True
            if self.peer.exited:
                # only give reward of both exited
                reward = self._reward()

        # discount
        if self.step_penalty:
            reward = reward - 0.01

        # fill observation with text
        self.append_existing_utterance_to_history()
        obs = self.add_utterance_to_observation(obs)
        self.reset_utterance()
        return obs, reward, done, info

    def _reward(self):
        if self.diminished_reward:
            return super()._reward()
        else:
            return 1.0

    def render(self, *args, **kwargs):
        obs = super().render(*args, **kwargs)
        print("conversation:\n", self.conversation)
        print("utterance_history:\n", self.utterance_history)
        self.window.set_caption(self.conversation, [self.peer.name])
        return obs


class Spying8x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8)


class Spying6x6Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=6)


# knowledgeable
class SpyingKnowledgeableEnv(SpyingEnv):
    def __init__(self):
        super().__init__(size=5, knowledgeable=True)

class SpyingKnowledgeable6x6Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=6, knowledgeable=True)

class SpyingKnowledgeable8x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8, knowledgeable=True)

class SpyingKnowledgeableHardPassword8x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8, knowledgeable=True, hard_password=True)

class Spying508x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8, max_steps=50)

class SpyingKnowledgeable508x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8, knowledgeable=True, max_steps=50)

class SpyingKnowledgeableHardPassword508x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8, knowledgeable=True, hard_password=True, max_steps=50)

class SpyingKnowledgeable1008x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8, knowledgeable=True, max_steps=100)

class SpyingKnowledgeable100OneSwitch8x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8, knowledgeable=True, max_steps=100, n_switches=1)

class SpyingKnowledgeable50OneSwitch5x5Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=5, knowledgeable=True, max_steps=50, n_switches=1)


class SpyingKnowledgeable505x5Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=5, knowledgeable=True, max_steps=50, n_switches=3)

class SpyingKnowledgeable50TwoSwitches8x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8, knowledgeable=True, max_steps=50, n_switches=2)

class SpyingKnowledgeable50TwoSwitchesHard8x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8, knowledgeable=True, max_steps=50, n_switches=2, hard_password=True)


class SpyingKnowledgeable100TwoSwitches8x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8, knowledgeable=True, max_steps=100, n_switches=2)

class SpyingKnowledgeable100TwoSwitchesHard8x8Env(SpyingEnv):
    def __init__(self):
        super().__init__(size=8, knowledgeable=True, max_steps=100, n_switches=2, hard_password=True)




register(
    id='MiniGrid-Spying-5x5-v0',
    entry_point='gym_minigrid.envs:SpyingEnv'
)

register(
    id='MiniGrid-Spying-6x6-v0',
    entry_point='gym_minigrid.envs:Spying6x6Env'
)

register(
    id='MiniGrid-Spying-8x8-v0',
    entry_point='gym_minigrid.envs:Spying8x8Env'
)

register(
    id='MiniGrid-SpyingKnowledgeable-5x5-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeableEnv'
)

register(
    id='MiniGrid-SpyingKnowledgeable-6x6-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeable6x6Env'
)

register(
    id='MiniGrid-SpyingKnowledgeable-8x8-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeable8x8Env'
)

register(
    id='MiniGrid-SpyingKnowledgeableHardPassword-8x8-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeableHardPassword8x8Env'
)

# max len 50
register(
    id='MiniGrid-Spying50-8x8-v0',
    entry_point='gym_minigrid.envs:Spying508x8Env'
)

register(
    id='MiniGrid-SpyingKnowledgeable50-8x8-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeable508x8Env'
)

register(
    id='MiniGrid-SpyingKnowledgeableHardPassword50-8x8-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeableHardPassword508x8Env'
)

# max len 100
register(
    id='MiniGrid-SpyingKnowledgeable100-8x8-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeable1008x8Env'
)

# max len OneSwitch
register(
    id='MiniGrid-SpyingKnowledgeable100OneSwitch-8x8-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeable100OneSwitch8x8Env'
)

register(
    id='MiniGrid-SpyingKnowledgeable50OneSwitch-5x5-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeable50OneSwitch5x5Env'
)

register(
    id='MiniGrid-SpyingUnknowledgeable50OneSwitch-5x5-v0',
    entry_point='gym_minigrid.envs:SpyingUnknowledgeable50OneSwitch5x5Env'
)

register(
    id='MiniGrid-SpyingKnowledgeable50-5x5-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeable505x5Env'
)

register(
    id='MiniGrid-SpyingKnowledgeable50TwoSwitches-8x8-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeable50TwoSwitches8x8Env'
)
register(
    id='MiniGrid-SpyingKnowledgeable50TwoSwitchesHard-8x8-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeable50TwoSwitchesHard8x8Env'
)
register(
    id='MiniGrid-SpyingKnowledgeable100TwoSwitches-8x8-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeable100TwoSwitches8x8Env'
)
register(
    id='MiniGrid-SpyingKnowledgeable100TwoSwitchesHard-8x8-v0',
    entry_point='gym_minigrid.envs:SpyingKnowledgeable100TwoSwitchesHard8x8Env'
)

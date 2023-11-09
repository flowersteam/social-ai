import numpy as np

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 
import time
from collections import deque


class Peer(NPC):
    """
    A dancing NPC that the agent has to copy
    """

    def __init__(self, color, name, env):
        super().__init__(color)
        self.name = name
        self.npc_dir = 1  # NPC initially looks downward
        self.npc_type = 0
        self.env = env
        self.npc_actions = []
        self.dancing_step_idx = 0
        self.actions = MiniGridEnv.Actions
        self.add_npc_direction = True
        self.available_moves = [self.rotate_left, self.rotate_right, self.go_forward, self.toggle_action]

        selected_door_id = self.env._rand_elem([0, 1])
        self.selected_door_pos = [self.env.door_pos_top, self.env.door_pos_bottom][selected_door_id]
        self.selected_door = [self.env.door_top, self.env.door_bottom][selected_door_id]

    def step(self):

        if all(self.front_pos == self.selected_door_pos):
            # in front of door
            if self.selected_door.is_open:
                self.go_forward()

        else:
            if (self.cur_pos[0] == self.selected_door_pos[0]) or (self.cur_pos[1] == self.selected_door_pos[1]):
                # is either in the correct row on in the correct column
                next_wanted_position = self.selected_door_pos
            else:
                # choose the midpoint
                for cand_x, cand_y in [
                    (self.cur_pos[0], self.selected_door_pos[1]),
                    (self.selected_door_pos[0], self.cur_pos[1])
                ]:
                    print("wX:", self.env.wall_x)
                    print("wY:", self.env.wall_y)
                    if (
                            cand_x > 0 and cand_x < self.env.wall_x
                    ) and (
                            cand_y > 0 and cand_y < self.env.wall_y
                    ):
                        next_wanted_position = (cand_x, cand_y)
                    print("wanted_pos:", next_wanted_position)

            if self.cur_pos[1] == next_wanted_position[1]:
                # same y
                if self.cur_pos[0] < next_wanted_position[0]:
                    wanted_dir = 0
                else:
                    wanted_dir = 2
                if self.npc_dir == wanted_dir:
                    self.go_forward()

                else:
                    self.rotate_left()

            elif self.cur_pos[0] == next_wanted_position[0]:
                # same x
                if self.cur_pos[1] < next_wanted_position[1]:
                    wanted_dir = 1
                else:
                    wanted_dir = 3

                if self.npc_dir == wanted_dir:
                    self.go_forward()

                else:
                    self.rotate_left()
            else:
                raise ValueError("Something is wrong.")


class TwoDoorsIntentGrammar(object):

    templates = ["Move your", "Shake your"]
    things = ["body", "head"]

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "


class TwoDoorsIntentEnv(MultiModalMiniGridEnv):
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
    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.knowledgeable = knowledgeable

        super().__init__(
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True,
            actions=MiniGridEnv.Actions,
            action_space=spaces.MultiDiscrete([
                len(MiniGridEnv.Actions),
                *TwoDoorsIntentGrammar.grammar_action_space.nvec
            ]),
            add_npc_direction=True
        )

        print({
            "size": size,
            "diminished_reward": diminished_reward,
            "step_penalty": step_penalty,
        })

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height, nb_obj_dims=4)

        # Randomly vary the room width and height
        width = self._rand_int(5, width+1)
        height = self._rand_int(5, height+1)

        self.wall_x = width-1
        self.wall_y = height-1

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # door top
        door_color_top = self._rand_elem(COLOR_NAMES)
        self.door_pos_top = (width-1, 1)
        self.door_top = Door(door_color_top)
        self.grid.set(*self.door_pos_top, self.door_top)

        # switch top
        self.switch_pos_top = (0, 1)
        self.switch_top = Switch(door_color_top, lockable_object=self.door_top)
        self.grid.set(*self.switch_pos_top, self.switch_top)

        # door bottom
        door_color_bottom = self._rand_elem(COLOR_NAMES)
        self.door_pos_bottom = (width-1, height-2)
        self.door_bottom = Door(door_color_bottom)
        self.grid.set(*self.door_pos_bottom, self.door_bottom)

        # switch bottom
        self.switch_pos_bottom = (0, height-2)
        self.switch_bottom = Switch(door_color_bottom, lockable_object=self.door_bottom)
        self.grid.set(*self.switch_pos_bottom, self.switch_bottom)

        # Set a randomly coloured Dancer NPC
        color = self._rand_elem(COLOR_NAMES)
        self.peer = Peer(color, "Jill", self)

        # Place it on the middle left side of the room
        peer_pos = np.array((self._rand_int(1, width - 1), self._rand_int(1, height - 1)))

        self.grid.set(*peer_pos, self.peer)
        self.peer.init_pos = peer_pos
        self.peer.cur_pos = peer_pos

        # Randomize the agent's start position and orientation
        self.place_agent(size=(width, height))

        # Generate the mission string
        self.mission = 'watch dancer and repeat his moves afterwards'

        # Dummy beginning string
        self.beginning_string = "This is what you hear. \n"
        self.utterance = self.beginning_string

        # utterance appended at the end of each step
        self.utterance_history = ""

        # used for rendering
        self.conversation = self.utterance

    def step(self, action):
        p_action = action[0]
        utterance_action = action[1:]

        obs, reward, done, info = super().step(p_action)
        self.peer.step()

        if np.isnan(p_action):
            pass

        if p_action == self.actions.done:
            done = True

        elif all(self.agent_pos == self.door_pos_top):
            done = True

        elif all(self.agent_pos == self.door_pos_bottom):
            done = True

        elif all([self.switch_top.is_on, self.switch_bottom.is_on]):
            # if both switches are on no reward is given and episode ends
            done = True

        elif all(self.peer.cur_pos == self.peer.selected_door_pos):
            reward = self._reward()
            done = True

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


class TwoDoorsIntent8x8Env(TwoDoorsIntentEnv):
    def __init__(self):
        super().__init__(size=8)


class TwoDoorsIntent6x6Env(TwoDoorsIntentEnv):
    def __init__(self):
        super().__init__(size=6)



register(
    id='MiniGrid-TwoDoorsIntent-5x5-v0',
    entry_point='gym_minigrid.envs:TwoDoorsIntentEnv'
)

register(
    id='MiniGrid-TwoDoorsIntent-6x6-v0',
    entry_point='gym_minigrid.envs:TwoDoorsIntent6x6Env'
)

register(
    id='MiniGrid-TwoDoorsIntent-8x8-v0',
    entry_point='gym_minigrid.envs:TwoDoorsIntent8x8Env'
)

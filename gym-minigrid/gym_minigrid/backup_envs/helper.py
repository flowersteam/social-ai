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
        self.joint_attention_achieved = False

    def can_overlap(self):
        # If the NPC is hidden, agent can overlap on it
        return self.env.hidden_npc

    def encode(self, nb_dims=3):
        if self.env.hidden_npc:
            if nb_dims == 3:
                return (1, 0, 0)
            elif nb_dims == 4:
                return (1, 0, 0, 0)
        else:
            return super().encode(nb_dims=nb_dims)

    def step(self):

        distance_to_door = np.abs(self.selected_door_pos - self.cur_pos).sum(-1)

        if all(self.front_pos == self.selected_door_pos) and self.selected_door.is_open:
            # in front of door
            self.go_forward()

        elif distance_to_door == 1 and not self.joint_attention_achieved:
            # before turning to the door look at the agent
            wanted_dir = self.compute_wanted_dir(self.env.agent_pos)
            act = self.compute_turn_action(wanted_dir)
            act()
            if self.is_eye_contact():
                self.joint_attention_achieved = True

        else:
            act = self.path_to_toggle_pos(self.selected_door_pos)
            act()

        # not really important as the NPC doesn't speak
        if self.env.hidden_npc:
            return None



class HelperGrammar(object):

    templates = ["Move your", "Shake your"]
    things = ["body", "head"]

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "


class HelperEnv(MultiModalMiniGridEnv):
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
        max_steps=20,
        hidden_npc=False,
    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.knowledgeable = knowledgeable
        self.hidden_npc = hidden_npc

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            actions=MiniGridEnv.Actions,
            action_space=spaces.MultiDiscrete([
                len(MiniGridEnv.Actions),
                *HelperGrammar.grammar_action_space.nvec
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

        # add lava
        self.grid.vert_wall(width//2, 1, height - 2, Lava)

        # door top
        door_color_top = self._rand_elem(COLOR_NAMES)
        self.door_pos_top = (width-1, 1)
        self.door_top = Door(door_color_top, is_locked=True)
        self.grid.set(*self.door_pos_top, self.door_top)

        # switch top
        self.switch_pos_top = (0, 1)
        self.switch_top = Switch(door_color_top, lockable_object=self.door_top, locker_switch=True)
        self.grid.set(*self.switch_pos_top, self.switch_top)

        # door bottom
        door_color_bottom = self._rand_elem(COLOR_NAMES)
        self.door_pos_bottom = (width-1, height-2)
        self.door_bottom = Door(door_color_bottom, is_locked=True)
        self.grid.set(*self.door_pos_bottom, self.door_bottom)

        # switch bottom
        self.switch_pos_bottom = (0, height-2)
        self.switch_bottom = Switch(door_color_bottom, lockable_object=self.door_bottom, locker_switch=True)
        self.grid.set(*self.switch_pos_bottom, self.switch_bottom)

        # save to variables
        self.switches = [self.switch_top, self.switch_bottom]
        self.switches_pos = [self.switch_pos_top, self.switch_pos_bottom]
        self.door = [self.door_top, self.door_bottom]
        self.door_pos = [self.door_pos_top, self.door_pos_bottom]

        # Set a randomly coloured Dancer NPC
        color = self._rand_elem(COLOR_NAMES)
        self.peer = Peer(color, "Jill", self)

        # Place it on the middle right side of the room
        peer_pos = np.array((self._rand_int(width//2+1, width - 1), self._rand_int(1, height - 1)))

        self.grid.set(*peer_pos, self.peer)
        self.peer.init_pos = peer_pos
        self.peer.cur_pos = peer_pos

        # Randomize the agent's start position and orientation
        self.place_agent(size=(width//2, height))

        # Generate the mission string
        self.mission = 'watch dancer and repeat his moves afterwards'

        # Dummy beginning string
        self.beginning_string = "This is what you hear. \n"
        self.utterance = self.beginning_string

        # utterance appended at the end of each step
        self.utterance_history = ""

        # used for rendering
        self.conversation = self.utterance
        self.outcome_info = None

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

        if self.hidden_npc:
            # all npc are hidden
            assert np.argwhere(obs['image'][:,:,0] == OBJECT_TO_IDX['npc']).size == 0
            assert "{}:".format(self.peer.name) not in self.utterance

        # fill observation with text
        self.append_existing_utterance_to_history()
        obs = self.add_utterance_to_observation(obs)
        self.reset_utterance()

        if done:
            if reward > 0:
                self.outcome_info = "SUCCESS: agent got {} reward \n".format(np.round(reward, 1))
            else:
                self.outcome_info = "FAILURE: agent got {} reward \n".format(reward)

        return obs, reward, done, info

    def _reward(self):
        if self.diminished_reward:
            return super()._reward()
        else:
            return 1.0

    def render(self, *args, **kwargs):
        obs = super().render(*args, **kwargs)
        self.window.clear_text()  # erase previous text

        # self.window.set_caption(self.conversation, [self.peer.name])
        # self.window.ax.set_title("correct door: {}".format(self.true_guide.target_color), loc="left", fontsize=10)
        if self.outcome_info:
            color = None
            if "SUCCESS" in self.outcome_info:
                color = "lime"
            elif "FAILURE" in self.outcome_info:
                color = "red"
            self.window.add_text(*(0.01, 0.85, self.outcome_info),
                                 **{'fontsize':15, 'color':color, 'weight':"bold"})

        self.window.show_img(obs)  # re-draw image to add changes to window
        return obs


class Helper8x8Env(HelperEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, max_steps=20, **kwargs)


class Helper6x6Env(HelperEnv):
    def __init__(self):
        super().__init__(size=6, max_steps=20)



register(
    id='MiniGrid-Helper-5x5-v0',
    entry_point='gym_minigrid.envs:HelperEnv'
)

register(
    id='MiniGrid-Helper-6x6-v0',
    entry_point='gym_minigrid.envs:Helper6x6Env'
)

register(
    id='MiniGrid-Helper-8x8-v0',
    entry_point='gym_minigrid.envs:Helper8x8Env'
)

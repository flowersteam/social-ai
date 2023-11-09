import numpy as np

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 
import time
from collections import deque

class DemonstratingPeer(NPC):
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
        super().step()
        reply = None
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

            if self.joint_attention_achieved:
                if self.env.object.is_locked:
                    first_wrong_id = np.where(self.env.get_selected_password() != self.env.password)[0][0]
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
                wanted_dir = self.compute_wanted_dir(self.env.agent_pos)
                action = self.compute_turn_action(wanted_dir)
                action()

                if self.is_eye_contact():
                    self.joint_attention_achieved = True
                    reply = "Look at me"

        else:
            self.env._rand_elem(self.available_moves)()

        self.env.update_door_lock()

        if self.env.hidden_npc:
            reply = None

        return reply


class DemonstrationGrammar(object):

    templates = ["Move your", "Shake your"]
    things = ["body", "head"]

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "


class DemonstrationEnv(MultiModalMiniGridEnv):
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
        max_steps=100,
        n_switches=3,
        augmentation=False,
        stump=False,
        no_turn_off=False,
        no_light=False,
        hidden_npc=False
    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.knowledgeable = knowledgeable
        self.hard_password = hard_password
        self.n_switches = n_switches
        self.augmentation = augmentation
        self.stump = stump
        self.no_turn_off=no_turn_off
        self.hidden_npc = hidden_npc

        if self.augmentation:
           assert not no_light

        self.no_light = no_light


        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=False if self.stump else True,
            actions=MiniGridEnv.Actions,
            action_space=spaces.MultiDiscrete([
                len(MiniGridEnv.Actions),
                *DemonstrationGrammar.grammar_action_space.nvec
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

        if self.stump:
            wall_for_door = 1
        else:
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

        if self.stump:
            self.stump_pos = (w, h+2)
            self.stump_obj = Wall()
            self.grid.set(*self.stump_pos, self.stump_obj)

        # sample password
        if self.hard_password:
            self.password = np.array([self._rand_int(0, 2) for _ in range(self.n_switches)])

        else:
            idx = self._rand_int(0, self.n_switches)
            self.password = np.zeros(self.n_switches)
            self.password[idx] = 1.0

        # add the switches
        self.switches = []
        self.switches_pos = []
        for i in range(self.n_switches):
            c = COLOR_NAMES[i]
            pos = np.array([i+1, height-1])
            sw = Switch(c, is_on=bool(self.password[i]) if self.augmentation else False, no_light=self.no_light)
            self.grid.set(*pos, sw)
            self.switches.append(sw)
            self.switches_pos.append(pos)

        # Set a randomly coloured Dancer NPC
        color = self._rand_elem(COLOR_NAMES)

        if not self.augmentation:
            self.peer = DemonstratingPeer(color, "Jim", self, knowledgeable=self.knowledgeable)

            # height -2 so its not in front of the buttons in the way
            peer_pos = np.array((self._rand_int(1, width - 1), self._rand_int(1, height - 2)))

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
        self.outcome_info = None

    def update_door_lock(self):
        if self.augmentation and self.step_count <= 10:
                self.door.is_locked = True
                self.door.is_open = False
        else:
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
        # print("pass:", self.password)
        # print("selected pass:", self.get_selected_password())

        if self.augmentation and self.step_count == 10:
            # reset switches door
            for s in self.switches:
                s.is_on = False

            # update door
            self.update_door_lock()

        if p_action == self.actions.done:
            done = True

        if not self.augmentation:
            peer_reply = self.peer.step()

            if peer_reply is not None:
                self.utterance += "{}: {} \n".format(self.peer.name, peer_reply)
                self.conversation += "{}: {} \n".format(self.peer.name, peer_reply)

        if all(self.agent_pos == self.door_pos):
            done = True
            if not self.augmentation:
                if self.peer.exited:
                    # only give reward if both exited
                    reward = self._reward()
            else:
                reward = self._reward()

        # discount
        if self.step_penalty:
            reward = reward - 0.01

        if self.hidden_npc:
            # all npc are hidden
            assert np.argwhere(obs['image'][:,:,0] == OBJECT_TO_IDX['npc']).size == 0
            if not self.augmentation:
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
        self.window.set_caption(self.conversation)
        sw_color = self.switches[np.argmax(self.password)].color
        self.window.ax.set_title("correct switch: {}".format(sw_color), loc="left", fontsize=10)
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


## 100 Demonstrating
# register(
#     id='MiniGrid-DemonstrationNoLightNoTurnOff100-8x8-v0',
#     entry_point='gym_minigrid.envs:DemonstrationNoLightNoTurnOff1008x8Env'
# )
#class Demonstration100TwoSwitches8x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100, n_switches=2)
#
#class Demonstration100TwoSwitchesHard8x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100, n_switches=2, hard_password=True)
#
## 100 AUG Demonstrating
#class AugmentationDemonstration100TwoSwitches8x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100, n_switches=2, augmentation=True)
#
#class AugmentationDemonstration100TwoSwitchesHard8x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100, n_switches=2, hard_password=True, augmentation=True)
#
#
## Three switches
## 100 Demonstrating
#class Demonstration1008x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100)
#
#class Demonstration100Hard8x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100, hard_password=True)
#
## 100 AUG Demonstrating
#class AugmentationDemonstration1008x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100, augmentation=True)
#
#class AugmentationDemonstration100Hard8x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100, hard_password=True, augmentation=True)
#
## No turn off
## 100 Demonstrating:  No light, no turn off
#
#class DemonstrationNoLightNoTurnOff100Hard8x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100, no_turn_off=True, hard_password=True, no_light=True)
#
## 100 no turn off
#class DemonstrationNoTurnOff1008x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100, no_turn_off=True)
#
#class DemonstrationNoTurnOff100Hard8x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100, no_turn_off=True, hard_password=True)
#
## 100 AUG Demonstrating
#
#class AugmentationDemonstrationNoTurnOff100Hard8x8Env(DemonstrationEnv):
#    def __init__(self):
#        super().__init__(size=8, knowledgeable=True, max_steps=100, no_turn_off=True, hard_password=True, augmentation=True)


## demonstrating 100 steps
#register(
#    id='MiniGrid-Demonstration100TwoSwitches-8x8-v0',
#    entry_point='gym_minigrid.envs:Demonstration100TwoSwitches8x8Env'
#)
#register(
#    id='MiniGrid-Demonstration100TwoSwitchesHard-8x8-v0',
#    entry_point='gym_minigrid.envs:Demonstration100TwoSwitchesHard8x8Env'
#)
#
## AUG demonstrating 100 steps
#register(
#    id='MiniGrid-AugmentationDemonstration100TwoSwitches-8x8-v0',
#    entry_point='gym_minigrid.envs:AugmentationDemonstration100TwoSwitches8x8Env'
#)
#register(
#    id='MiniGrid-AugmentationDemonstration100TwoSwitchesHard-8x8-v0',
#    entry_point='gym_minigrid.envs:AugmentationDemonstration100TwoSwitchesHard8x8Env'
#)
#
## three switches
#
## demonstrating 100 steps
#register(
#    id='MiniGrid-Demonstration100-8x8-v0',
#    entry_point='gym_minigrid.envs:Demonstration1008x8Env'
#)
#register(
#    id='MiniGrid-Demonstration100Hard-8x8-v0',
#    entry_point='gym_minigrid.envs:Demonstration100Hard8x8Env'
#)
#
## AUG demonstrating 100 steps
#register(
#    id='MiniGrid-AugmentationDemonstration100-8x8-v0',
#    entry_point='gym_minigrid.envs:AugmentationDemonstration1008x8Env'
#)
#register(
#    id='MiniGrid-AugmentationDemonstration100Hard-8x8-v0',
#    entry_point='gym_minigrid.envs:AugmentationDemonstration100Hard8x8Env'
#)
#
## no turn off three switches
#
## demonstrating 100 steps
#register(
#    id='MiniGrid-DemonstrationNoTurnOff100-8x8-v0',
#    entry_point='gym_minigrid.envs:DemonstrationNoTurnOff1008x8Env'
#)
#register(
#    id='MiniGrid-DemonstrationNoTurnOff100Hard-8x8-v0',
#    entry_point='gym_minigrid.envs:DemonstrationNoTurnOff100Hard8x8Env'
#)
#
## demonstrating 100 steps no light
#register(
#    id='MiniGrid-DemonstrationNoLightNoTurnOff100-8x8-v0',
#    entry_point='gym_minigrid.envs:DemonstrationNoLightNoTurnOff1008x8Env'
#)
#register(
#    id='MiniGrid-DemonstrationNoLightNoTurnOff100Hard-8x8-v0',
#    entry_point='gym_minigrid.envs:DemonstrationNoLightNoTurnOff100Hard8x8Env'
#)
#
## AUG demonstrating 100 steps
#register(
#    id='MiniGrid-AugmentationDemonstrationNoTurnOff100-8x8-v0',
#    entry_point='gym_minigrid.envs:AugmentationDemonstrationNoTurnOff1008x8Env'
#)
#register(
#    id='MiniGrid-AugmentationDemonstrationNoTurnOff100Hard-8x8-v0',
#    entry_point='gym_minigrid.envs:AugmentationDemonstrationNoTurnOff100Hard8x8Env'
#)
# register(
#     id='MiniGrid-AugmentationDemonstrationNoTurnOff100-8x8-v0',
#     entry_point='gym_minigrid.envs:AugmentationDemonstrationNoTurnOff1008x8Env'
# )
#
# class DemonstrationNoLightNoTurnOff1008x8Env(DemonstrationEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=100, no_turn_off=True, no_light=True)
#
# class AugmentationDemonstrationNoTurnOff1008x8Env(DemonstrationEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=100, no_turn_off=True, augmentation=True)

class ShowMe8x8Env(DemonstrationEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, knowledgeable=True, max_steps=100, no_turn_off=True, no_light=True, **kwargs)

class ShowMeNoSocial8x8Env(DemonstrationEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, knowledgeable=True, max_steps=100, no_turn_off=True, augmentation=True, **kwargs)


# AUG demonstrating 100 steps
register(
    id='MiniGrid-ShowMeNoSocial-8x8-v0',
    entry_point='gym_minigrid.envs:ShowMeNoSocial8x8Env'
)
register(
    id='MiniGrid-ShowMe-8x8-v0',
    entry_point='gym_minigrid.envs:ShowMe8x8Env'
)

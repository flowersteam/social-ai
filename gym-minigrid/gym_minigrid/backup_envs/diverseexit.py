import numpy as np

from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 
import time
from collections import deque

class TeacherPeer(NPC):
    """
    A dancing NPC that the agent has to copy
    """

    def __init__(self, color, name, env, npc_type=0, knowledgeable=False, easier=False, idl=False):
        super().__init__(color)
        self.name = name
        self.npc_dir = 1  # NPC initially looks downward
        self.npc_type = npc_type
        self.env = env
        self.knowledgeable = knowledgeable
        self.npc_actions = []
        self.dancing_step_idx = 0
        self.actions = MiniGridEnv.Actions
        self.add_npc_direction = True
        self.available_moves = [self.rotate_left, self.rotate_right, self.go_forward, self.toggle_action]
        self.was_introduced_to = False
        self.easier = easier
        assert not self.easier
        self.idl = idl

        self.must_eye_contact = True if (self.npc_type // 3) % 2 == 0 else False
        self.wanted_intro_utterances = [
            EasyTeachingGamesGrammar.construct_utterance([2, 2]),
            EasyTeachingGamesGrammar.construct_utterance([0, 1])
        ]
        self.wanted_intro_utterance = self.wanted_intro_utterances[0] if (self.npc_type // 3) // 2 == 0 else self.wanted_intro_utterances[1]
        if self.npc_type % 3 == 0:
            # must be far, must not poke
            self.must_be_poked = False
            self.must_be_close = False

        elif self.npc_type % 3 == 1:
            # must be close, must not poke
            self.must_be_poked = False
            self.must_be_close = True

        elif self.npc_type % 3 == 2:
            # must be close, must poke
            self.must_be_poked = True
            self.must_be_close = True

        else:
            raise ValueError("npc tyep {} unknown". format(self.npc_type))

        # print("Peer type: ", self.npc_type)
        # print("Peer conf: ", self.wanted_intro_utterance, self.must_eye_contact, self.must_be_close, self.must_be_poked)


        if self.must_be_poked and not self.must_be_close:
            raise ValueError("Must be poked means it must be close also.")

        self.poked = False

        self.exited = False
        self.joint_attention_achieved = False

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        self.poked = True
        return True

    def is_introduction_state_ok(self):
        if (self.must_be_poked and self.introduction_state["poked"]) or (
                not self.must_be_poked and not self.introduction_state["poked"]):
            if (self.must_be_close and self.introduction_state["close"]) or (
                    not self.must_be_close and not self.introduction_state["close"]):
                if (self.must_eye_contact and self.introduction_state["eye_contact"]) or (
                    not self.must_eye_contact and not self.introduction_state["eye_contact"]
                ):
                    if self.introduction_state["intro_utterance"] == self.wanted_intro_utterance:
                        return True

        return False

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

    def step(self, agent_utterance):
        super().step()

        if self.knowledgeable:
            if self.easier:
                raise DeprecationWarning()
                # wanted_dir = self.compute_wanted_dir(self.env.agent_pos)
                # action = self.compute_turn_action(wanted_dir)
                # action()
                # if not self.was_introduced_to and (agent_utterance in self.wanted_intro_utterances):
                #     self.was_introduced_to = True
                #     self.introduction_state = {
                #         "poked": self.poked,
                #         "close": self.is_near_agent(),
                #         "eye_contact": self.is_eye_contact(),
                #         "correct_intro_utterance": agent_utterance == self.wanted_intro_utterance
                #     }
                #     if self.is_introduction_state_ok():
                #         utterance = "Go to the {} door \n".format(self.env.target_color)
                #         return utterance

            else:
                wanted_dir = self.compute_wanted_dir(self.env.agent_pos)
                action = self.compute_turn_action(wanted_dir)
                action()
                if not self.was_introduced_to and (agent_utterance in self.wanted_intro_utterances):
                    self.was_introduced_to = True
                    self.introduction_state = {
                        "poked": self.poked,
                        "close": self.is_near_agent(),
                        "eye_contact": self.is_eye_contact(),
                        "intro_utterance": agent_utterance,
                    }
                    if not self.is_introduction_state_ok():
                        if self.idl:
                            if self.env.hidden_npc:
                                return None
                            else:
                                return "I don't like that \n"
                        else:
                            return None

                if self.is_eye_contact() and self.was_introduced_to:

                    if self.is_introduction_state_ok():
                        utterance = "Go to the {} door \n".format(self.env.target_color)
                        if self.env.hidden_npc:
                            return None
                        else:
                            return utterance
                    else:
                        # no utterance
                        return None

        else:
            self.env._rand_elem(self.available_moves)()
            return None


    def render(self, img):
        c = COLORS[self.color]

        npc_shapes = []
        # Draw eyes

        if self.npc_type % 3 == 0:
            npc_shapes.append(point_in_circle(cx=0.70, cy=0.50, r=0.10))
            npc_shapes.append(point_in_circle(cx=0.30, cy=0.50, r=0.10))
            # Draw mouth
            npc_shapes.append(point_in_rect(0.20, 0.80, 0.72, 0.81))
            # Draw top hat
            npc_shapes.append(point_in_rect(0.30, 0.70, 0.05, 0.28))

        elif self.npc_type % 3 == 1:
            npc_shapes.append(point_in_circle(cx=0.70, cy=0.50, r=0.10))
            npc_shapes.append(point_in_circle(cx=0.30, cy=0.50, r=0.10))
            # Draw mouth
            npc_shapes.append(point_in_rect(0.20, 0.80, 0.72, 0.81))
            # Draw bottom hat
            npc_shapes.append(point_in_triangle((0.15, 0.28),
                                                (0.85, 0.28),
                                                (0.50, 0.05)))
        elif self.npc_type % 3 == 2:
            npc_shapes.append(point_in_circle(cx=0.70, cy=0.50, r=0.10))
            npc_shapes.append(point_in_circle(cx=0.30, cy=0.50, r=0.10))
            # Draw mouth
            npc_shapes.append(point_in_rect(0.20, 0.80, 0.72, 0.81))
            # Draw bottom hat
            npc_shapes.append(point_in_triangle((0.15, 0.28),
                                                (0.85, 0.28),
                                                (0.50, 0.05)))
            # Draw top hat
            npc_shapes.append(point_in_rect(0.30, 0.70, 0.05, 0.28))


        # todo: move this to super function
        # todo: super.render should be able to take the npc_shapes and then rotate them

        if hasattr(self, "npc_dir"):
            # Pre-rotation to ensure npc_dir = 1 means NPC looks downwards
            npc_shapes = [rotate_fn(v, cx=0.5, cy=0.5, theta=-1 * (math.pi / 2)) for v in npc_shapes]
            # Rotate npc based on its direction
            npc_shapes = [rotate_fn(v, cx=0.5, cy=0.5, theta=(math.pi / 2) * self.npc_dir) for v in npc_shapes]

        # Draw shapes
        for v in npc_shapes:
            fill_coords(img, v, c)

# class EasyTeachingGamesSmallGrammar(object):
#
#     templates = ["Where is", "Open", "What is"]
#     things = ["sesame", "the exit", "the password"]
#
#     grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])
#
#     @classmethod
#     def construct_utterance(cls, action):
#         if all(np.isnan(action)):
#             return ""
#         return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "


class EasyTeachingGamesGrammar(object):

    templates = ["Where is", "Open", "Which is", "How are"]
    things = [
        "sesame", "the exit", "the correct door", "you", "the ceiling", "the window", "the entrance", "the closet",
        "the drawer", "the fridge", "the floor", "the lamp", "the trash can", "the chair", "the bed", "the sofa"
    ]

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        if all(np.isnan(action)):
            return ""
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "


class EasyTeachingGamesEnv(MultiModalMiniGridEnv):
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
        max_steps=50,
        n_switches=3,
        peer_type=None,
        no_turn_off=False,
        easier=False,
        idl=False,
        hidden_npc = False,
    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.knowledgeable = knowledgeable
        self.hard_password = hard_password
        self.n_switches = n_switches
        self.peer_type = peer_type
        self.no_turn_off = no_turn_off
        self.easier = easier
        self.idl = idl
        self.hidden_npc = hidden_npc

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            actions=MiniGridEnv.Actions,
            action_space=spaces.MultiDiscrete([
                len(MiniGridEnv.Actions),
                *EasyTeachingGamesGrammar.grammar_action_space.nvec
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

        self.wall_x = width - 1
        self.wall_y = height - 1

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        self.door_pos = []
        self.door_front_pos = []  # Remembers positions in front of door to avoid setting wizard here

        self.door_pos.append((self._rand_int(2, width-2), 0))
        self.door_front_pos.append((self.door_pos[-1][0], self.door_pos[-1][1]+1))

        self.door_pos.append((self._rand_int(2, width-2), height-1))
        self.door_front_pos.append((self.door_pos[-1][0], self.door_pos[-1][1] - 1))

        self.door_pos.append((0, self._rand_int(2, height-2)))
        self.door_front_pos.append((self.door_pos[-1][0] + 1, self.door_pos[-1][1]))

        self.door_pos.append((width-1, self._rand_int(2, height-2)))
        self.door_front_pos.append((self.door_pos[-1][0] - 1, self.door_pos[-1][1]))

        # Generate the door colors
        self.door_colors = []
        while len(self.door_colors) < len(self.door_pos):
            color = self._rand_elem(COLOR_NAMES)
            if color in self.door_colors:
                continue
            self.door_colors.append(color)

        # Place the doors in the grid
        for idx, pos in enumerate(self.door_pos):
            color = self.door_colors[idx]
            self.grid.set(*pos, Door(color))

        # Select a random target door
        self.doorIdx = self._rand_int(0, len(self.door_pos))
        self.target_pos = self.door_pos[self.doorIdx]
        self.target_color = self.door_colors[self.doorIdx]

        # Set a randomly coloured Dancer NPC
        color = self._rand_elem(COLOR_NAMES)

        if self.peer_type is None:
            self.current_peer_type = self._rand_int(0, 12)
        else:
            self.current_peer_type = self.peer_type

        self.peer = TeacherPeer(
            color,
            ["Bobby", "Robby", "Toby"][self.current_peer_type % 3],
            self,
            knowledgeable=self.knowledgeable,
            npc_type=self.current_peer_type,
            easier=self.easier,
            idl=self.idl
        )

        # height -2 so its not in front of the buttons in the way
        while True:
            peer_pos = np.array((self._rand_int(1, width - 1), self._rand_int(1, height - 2)))

            if (
                # not in front of any door
                not tuple(peer_pos) in self.door_front_pos
            ) and (
                # no_close npc is not in the middle of the 5x5 env
                not (not self.peer.must_be_close and (width == 5 and height == 5) and all(peer_pos == (2, 2)))
            ):
                break

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


    def step(self, action):
        p_action = action[0]
        utterance_action = action[1:]

        obs, reward, done, info = super().step(p_action)

        if p_action == self.actions.done:
            done = True

        peer_utterance = EasyTeachingGamesGrammar.construct_utterance(utterance_action)
        peer_reply = self.peer.step(peer_utterance)

        if peer_reply is not None:
            self.utterance += "{}: {} \n".format(self.peer.name, peer_reply)
            self.conversation += "{}: {} \n".format(self.peer.name, peer_reply)

        if all(self.agent_pos == self.target_pos):
            done = True
            reward = self._reward()

        elif tuple(self.agent_pos) in self.door_pos:
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

        self.window.set_caption(self.conversation, self.peer.name)

        self.window.ax.set_title("correct door: {}".format(self.target_color), loc="left", fontsize=10)
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


# # must be far, must not poke
# class EasyTeachingGames8x8Env(EasyTeachingGamesEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=50, peer_type=0)
#
# # must be close, must not poke
# class EasyTeachingGamesClose8x8Env(EasyTeachingGamesEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=50, peer_type=1)
#
# # must be close, must poke
# class EasyTeachingGamesPoke8x8Env(EasyTeachingGamesEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=50, peer_type=2)
#
# # 100 multi
# class EasyTeachingGamesMulti8x8Env(EasyTeachingGamesEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=50, peer_type=None)
#
#
#
# # speaking 50 steps
# register(
#     id='MiniGrid-EasyTeachingGames-8x8-v0',
#     entry_point='gym_minigrid.envs:EasyTeachingGames8x8Env'
# )
#
# # demonstrating 50 steps
# register(
#     id='MiniGrid-EasyTeachingGamesPoke-8x8-v0',
#     entry_point='gym_minigrid.envs:EasyTeachingGamesPoke8x8Env'
# )
#
# # demonstrating 50 steps
# register(
#     id='MiniGrid-EasyTeachingGamesClose-8x8-v0',
#     entry_point='gym_minigrid.envs:EasyTeachingGamesClose8x8Env'
# )
#
# # speaking 50 steps
# register(
#     id='MiniGrid-EasyTeachingGamesMulti-8x8-v0',
#     entry_point='gym_minigrid.envs:EasyTeachingGamesMulti8x8Env'
# )

# # must be far, must not poke
# class EasierTeachingGames8x8Env(EasyTeachingGamesEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=50, peer_type=0, easier=True)
#
# # must be close, must not poke
# class EasierTeachingGamesClose8x8Env(EasyTeachingGamesEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=50, peer_type=1, easier=True)
#
# # must be close, must poke
# class EasierTeachingGamesPoke8x8Env(EasyTeachingGamesEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=50, peer_type=2, easier=True)
#
# # 100 multi
# class EasierTeachingGamesMulti8x8Env(EasyTeachingGamesEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=50, peer_type=None, easier=True)
#
# # Multi Many
# class ManyTeachingGamesMulti8x8Env(EasyTeachingGamesEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=50, peer_type=None, easier=False, many=True)
#
# class ManyTeachingGamesMultiIDL8x8Env(EasyTeachingGamesEnv):
#     def __init__(self):
#         super().__init__(size=8, knowledgeable=True, max_steps=50, peer_type=None, easier=False, many=True, idl=True)


# # speaking 50 steps
# register(
#     id='MiniGrid-EasierTeachingGames-8x8-v0',
#     entry_point='gym_minigrid.envs:EasierTeachingGames8x8Env'
# )
#
# # demonstrating 50 steps
# register(
#     id='MiniGrid-EasierTeachingGamesPoke-8x8-v0',
#     entry_point='gym_minigrid.envs:EasierTeachingGamesPoke8x8Env'
# )
#
# # demonstrating 50 steps
# register(
#     id='MiniGrid-EasierTeachingGamesClose-8x8-v0',
#     entry_point='gym_minigrid.envs:EasierTeachingGamesClose8x8Env'
# )
#
# # speaking 50 steps
# register(
#     id='MiniGrid-EasierTeachingGamesMulti-8x8-v0',
#     entry_point='gym_minigrid.envs:EasierTeachingGamesMulti8x8Env'
# )
#
# # speaking 50 steps
# register(
#     id='MiniGrid-ManyTeachingGamesMulti-8x8-v0',
#     entry_point='gym_minigrid.envs:ManyTeachingGamesMulti8x8Env'
# )
#
# # speaking 50 steps
# register(
#     id='MiniGrid-ManyTeachingGamesMultiIDL-8x8-v0',
#     entry_point='gym_minigrid.envs:ManyTeachingGamesMultiIDL8x8Env'
# )

# Multi Many
class DiverseExit8x8Env(EasyTeachingGamesEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, knowledgeable=True, max_steps=50, peer_type=None, easier=False, **kwargs)

# speaking 50 steps
register(
    id='MiniGrid-DiverseExit-8x8-v0',
    entry_point='gym_minigrid.envs:DiverseExit8x8Env'
)


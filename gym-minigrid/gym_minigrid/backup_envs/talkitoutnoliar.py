from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 

class Wizard(NPC):
    """
    A simple NPC that knows who is telling the truth
    """

    def __init__(self, color, name, env):
        super().__init__(color)
        self.name = name
        self.env = env
        self.npc_dir = 1  # NPC initially looks downward
        # todo: this should be id == name
        self.npc_type = 0  # this will be put into the encoding

    def listen(self, utterance):
        if utterance == TalkItOutNoLiarGrammar.construct_utterance([0, 1]):
            if self.env.nameless:
                return "Ask the {} guide.".format(self.env.true_guide.color)
            else:
                return "Ask {}.".format(self.env.true_guide.name)

        return None

    def is_near_agent(self):
        ax, ay = self.env.agent_pos
        wx, wy = self.cur_pos
        if (ax == wx and abs(ay - wy) == 1) or (ay == wy and abs(ax - wx) == 1):
            return True
        return False


class Guide(NPC):
    """
    A simple NPC that knows the correct door.
    """

    def __init__(self, color, name, env, liar=False):
        super().__init__(color)
        self.name = name
        self.env = env
        self.liar = liar
        self.npc_dir = 1  # NPC initially looks downward
        assert not self.liar  # in this env the guide is always good
        # todo: this should be id == name
        self.npc_type = 1  # this will be put into the encoding

        # Select a random target object as mission
        obj_idx = self.env._rand_int(0, len(self.env.door_pos))
        self.target_pos = self.env.door_pos[obj_idx]
        self.target_color = self.env.door_colors[obj_idx]

    def listen(self, utterance):
        if utterance == TalkItOutNoLiarGrammar.construct_utterance([0, 1]):
                return self.env.mission

        return None

    def render(self, img):
        c = COLORS[self.color]

        npc_shapes = []
        # Draw eyes
        npc_shapes.append(point_in_circle(cx=0.70, cy=0.50, r=0.10))
        npc_shapes.append(point_in_circle(cx=0.30, cy=0.50, r=0.10))

        # Draw mouth
        npc_shapes.append(point_in_rect(0.20, 0.80, 0.72, 0.81))

        # todo: move this to super function
        # todo: super.render should be able to take the npc_shapes and then rotate them

        if hasattr(self, "npc_dir"):
            # Pre-rotation to ensure npc_dir = 1 means NPC looks downwards
            npc_shapes = [rotate_fn(v, cx=0.5, cy=0.5, theta=-1*(math.pi / 2)) for v in npc_shapes]
            # Rotate npc based on its direction
            npc_shapes = [rotate_fn(v, cx=0.5, cy=0.5, theta=(math.pi/2) * self.npc_dir) for v in npc_shapes]

        # Draw shapes
        for v in npc_shapes:
            fill_coords(img, v, c)

    def is_near_agent(self):
        ax, ay = self.env.agent_pos
        wx, wy = self.cur_pos
        if (ax == wx and abs(ay - wy) == 1) or (ay == wy and abs(ax - wx) == 1):
            return True
        return False


class TalkItOutNoLiarGrammar(object):

    templates = ["Where is", "Open", "Close", "What is"]
    things = [
        "sesame", "the exit", "the wall", "the floor", "the ceiling", "the window", "the entrance", "the closet",
        "the drawer", "the fridge", "oven", "the lamp", "the trash can", "the chair", "the bed", "the sofa"
    ]

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "


class TalkItOutNoLiarEnv(MultiModalMiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self,
        size=5,
        hear_yourself=False,
        diminished_reward=True,
        step_penalty=False,
        nameless=False,
    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.hear_yourself = hear_yourself
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.nameless = nameless

        super().__init__(
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True,
            actions=MiniGridEnv.Actions,
            action_space=spaces.MultiDiscrete([
                len(MiniGridEnv.Actions),
                *TalkItOutNoLiarGrammar.grammar_action_space.nvec
            ]),
            add_npc_direction=True
        )

        print({
            "size": size,
            "hear_yourself": hear_yourself,
            "diminished_reward": diminished_reward,
            "step_penalty": step_penalty,
        })

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height, nb_obj_dims=4)

        # Randomly vary the room width and height
        width = self._rand_int(5, width+1)
        height = self._rand_int(5, height+1)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the 4 doors at random positions
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


        # Set a randomly coloured WIZARD at a random position
        color = self._rand_elem(COLOR_NAMES)
        self.wizard = Wizard(color, "Gandalf", self)

        # Place it randomly, omitting front of door positions
        self.place_obj(self.wizard,
                       size=(width, height),
                       reject_fn=lambda _, p: tuple(p) in self.door_front_pos)

        # Set a randomly coloured TRUE GUIDE at a random position
        color = self._rand_elem(COLOR_NAMES)
        self.true_guide = Guide(color, "Jack", self, liar=False)

        # Place it randomly, omitting invalid positions
        self.place_obj(self.true_guide,
                       size=(width, height),
                       # reject_fn=lambda _, p: tuple(p) in self.door_front_pos)
                       reject_fn=lambda _, p: tuple(p) in [*self.door_front_pos, tuple(self.wizard.cur_pos)])

        # Randomize the agent's start position and orientation
        self.place_agent(size=(width, height))

        # Select a random target door
        self.doorIdx = self._rand_int(0, len(self.door_pos))
        self.target_pos = self.door_pos[self.doorIdx]
        self.target_color = self.door_colors[self.doorIdx]

        # Generate the mission string
        self.mission = 'go to the %s door' % self.target_color

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

        # assert all nan or neither nan
        assert len(set(np.isnan(utterance_action))) == 1

        speak_flag = not all(np.isnan(utterance_action))

        obs, reward, done, info = super().step(p_action)

        if speak_flag:
            utterance = TalkItOutNoLiarGrammar.construct_utterance(utterance_action)
            if self.hear_yourself:
                if self.nameless:
                    self.utterance += "{} \n".format(utterance)
                else:
                    self.utterance += "YOU: {} \n".format(utterance)
                
            self.conversation += "YOU: {} \n".format(utterance)

            # check if near wizard
            if self.wizard.is_near_agent():
                reply = self.wizard.listen(utterance)

                if reply:
                    if self.nameless:
                        self.utterance += "{} \n".format(reply)
                    else:
                        self.utterance += "{}: {} \n".format(self.wizard.name, reply)

                    self.conversation += "{}: {} \n".format(self.wizard.name, reply)

            if self.true_guide.is_near_agent():
                reply = self.true_guide.listen(utterance)

                if reply:
                    if self.nameless:
                        self.utterance += "{} \n".format(reply)
                    else:
                        self.utterance += "{}: {} \n".format(self.true_guide.name, reply)

                    self.conversation += "{}: {} \n".format(self.true_guide.name, reply)

            if utterance == TalkItOutNoLiarGrammar.construct_utterance([1, 0]):
                ax, ay = self.agent_pos
                tx, ty = self.target_pos

                if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
                    reward = self._reward()

                for dx, dy in self.door_pos:
                    if (ax == dx and abs(ay - dy) == 1) or (ay == dy and abs(ax - dx) == 1):
                        # agent has chosen some door episode, regardless of if the door is correct the episode is over
                        done = True

        # Don't let the agent open any of the doors
        if p_action == self.actions.toggle:
            done = True

        if p_action == self.actions.done:
            done = True

        # discount
        if self.step_penalty:
            reward = reward - 0.01

        # fill observation with text
        self.append_existing_utterance_to_history()
        obs = self.add_utterance_to_observation(obs)
        self.reset_utterance()

        return obs, reward, done, info

    # def reset(self):
    #     obs = super().reset()
    #     self.append_existing_utterance_to_history()
    #     obs = self.add_utterance_to_observation(obs)
    #     self.reset_utterance()
    #     return obs
    #
    # def append_existing_utterance_to_history(self):
    #     if self.utterance != self.empty_symbol:
    #         if self.utterance.startswith(self.empty_symbol):
    #             self.utterance_history += self.utterance[len(self.empty_symbol):]
    #         else:
    #             assert self.utterance == self.beginning_string
    #             self.utterance_history += self.utterance
    #
    # def add_utterance_to_observation(self, obs):
    #     obs["utterance"] = self.utterance
    #     obs["utterance_history"] = self.utterance_history
    #     obs["mission"] = "Hidden"
    #     return obs
    #
    # def reset_utterance(self):
    #     # set utterance to empty indicator
    #     self.utterance = self.empty_symbol

    def _reward(self):
        if self.diminished_reward:
            return super()._reward()
        else:
            return 1.0

    def render(self, *args, **kwargs):
        obs = super().render(*args, **kwargs)
        print("conversation:\n", self.conversation)
        print("utterance_history:\n", self.utterance_history)
        self.window.set_caption(self.conversation, [
            "Gandalf:",
            "Jack:",
            "John:",
            "Where is the exit",
            "Open sesame",
        ])
        return obs


class TalkItOutNoLiar8x8Env(TalkItOutNoLiarEnv):
    def __init__(self):
        super().__init__(size=8)


class TalkItOutNoLiar6x6Env(TalkItOutNoLiarEnv):
    def __init__(self):
        super().__init__(size=6)


class TalkItOutNoLiarNameless8x8Env(TalkItOutNoLiarEnv):
    def __init__(self):
        super().__init__(size=8, nameless=True)

register(
    id='MiniGrid-TalkItOutNoLiar-5x5-v0',
    entry_point='gym_minigrid.envs:TalkItOutNoLiarEnv'
)

register(
    id='MiniGrid-TalkItOutNoLiar-6x6-v0',
    entry_point='gym_minigrid.envs:TalkItOutNoLiar6x6Env'
)

register(
    id='MiniGrid-TalkItOutNoLiar-8x8-v0',
    entry_point='gym_minigrid.envs:TalkItOutNoLiar8x8Env'
)

register(
    id='MiniGrid-TalkItOutNoLiarNameless-8x8-v0',
    entry_point='gym_minigrid.envs:TalkItOutNoLiarNameless8x8Env'
)
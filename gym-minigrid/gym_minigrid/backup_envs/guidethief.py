from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class Guide(NPC):
    """
    A simple NPC that knows the correct door.
    """

    def __init__(self, color, name, id, env, liar=False):
        super().__init__(color)
        self.name = name
        self.env = env
        self.liar = liar
        self.npc_dir = 1  # NPC initially looks downward
        self.npc_type = id  # this will be put into the encoding

        # Select a random target object as mission
        obj_idx = self.env._rand_int(0, len(self.env.door_pos))
        self.target_pos = self.env.door_pos[obj_idx]
        self.target_color = self.env.door_colors[obj_idx]

    def listen(self, utterance):
        if utterance == GuideThiefGrammar.construct_utterance([0, 1]):
            if self.liar:
                fake_colors = [c for c in self.env.door_colors if c != self.env.target_color]
                fake_color = self.env._rand_elem(fake_colors)

                # Generate the mission string
                assert fake_color != self.env.target_color
                if self.env.one_word:
                    return '%s' % fake_color
                elif self.env.very_diff:
                    return 'you want the %s door' % fake_color
                else:
                    return 'go to the %s door' % fake_color

            else:
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


class GuideThiefGrammar(object):

    templates = ["Where is", "Open", "Close", "What is"]
    things = [
        "sesame", "the exit", "the wall", "the floor", "the ceiling", "the window", "the entrance", "the closet",
        "the drawer", "the fridge", "oven", "the lamp", "the trash can", "the chair", "the bed", "the sofa"
    ]

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "


class GuideThiefEnv(MultiModalMiniGridEnv):
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
        max_steps=None,
        very_diff=False,
        one_word=False,
    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.hear_yourself = hear_yourself
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.nameless = nameless
        self.very_diff = very_diff
        self.one_word = one_word

        super().__init__(
            grid_size=size,
            max_steps=max_steps or 5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True,
            actions=MiniGridEnv.Actions,
            action_space=spaces.MultiDiscrete([
                len(MiniGridEnv.Actions),
                *GuideThiefGrammar.grammar_action_space.nvec
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
        self.door_front_pos = []

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

        # Place it randomly, omitting front of door positions

        # add guides
        GUIDE_NAMES = ["John", "Jack"]
        name_2_id = {name: id for id, name in enumerate(GUIDE_NAMES)}

        # Set a randomly coloured TRUE GUIDE at a random position

        true_guide_name = GUIDE_NAMES[0]
        color = self._rand_elem(COLOR_NAMES)
        self.true_guide = Guide(
            color=color,
            name=true_guide_name,
            id=name_2_id[true_guide_name],
            env=self,
            liar=False
        )

        # Place it randomly, omitting invalid positions
        self.place_obj(self.true_guide,
                       size=(width, height),
                       # reject_fn=lambda _, p: tuple(p) in self.door_front_pos)
                       reject_fn=lambda _, p: tuple(p) in self.door_front_pos)

        # Set a randomly coloured FALSE GUIDE at a random position
        false_guide_name = GUIDE_NAMES[1]
        if self.nameless:
            color = self._rand_elem([c for c in COLOR_NAMES if c != self.true_guide.color])
        else:
            color = self._rand_elem(COLOR_NAMES)

        self.false_guide = Guide(
            color=color,
            name=false_guide_name,
            id=name_2_id[false_guide_name],
            env=self,
            liar=True
        )

        # Place it randomly, omitting invalid positions
        self.place_obj(self.false_guide,
                       size=(width, height),
                       reject_fn=lambda _, p: tuple(p) in [
                           *self.door_front_pos, tuple(self.true_guide.cur_pos)])
        assert self.true_guide.name != self.false_guide.name

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
            utterance = GuideThiefGrammar.construct_utterance(utterance_action)
            if self.hear_yourself:
                if self.nameless:
                    self.utterance += "{} \n".format(utterance)
                else:
                    self.utterance += "YOU: {} \n".format(utterance)
                
            self.conversation += "YOU: {} \n".format(utterance)

            if self.true_guide.is_near_agent():
                reply = self.true_guide.listen(utterance)

                if reply:
                    if self.nameless:
                        self.utterance += "{} \n".format(reply)
                    else:
                        self.utterance += "{}: {} \n".format(self.true_guide.name, reply)

                    self.conversation += "{}: {} \n".format(self.true_guide.name, reply)

            if self.false_guide.is_near_agent():
                reply = self.false_guide.listen(utterance)

                if reply:
                    if self.nameless:
                        self.utterance += "{} \n".format(reply)
                    else:
                        self.utterance += "{}: {} \n".format(self.false_guide.name, reply)

                    self.conversation += "{}: {} \n".format(self.false_guide.name, reply)

            if utterance == GuideThiefGrammar.construct_utterance([1, 0]):
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


class GuideThief8x8Env(GuideThiefEnv):
    def __init__(self):
        super().__init__(size=8)


class GuideThief6x6Env(GuideThiefEnv):
    def __init__(self):
        super().__init__(size=6)


class GuideThiefNameless8x8Env(GuideThiefEnv):
    def __init__(self):
        super().__init__(size=8, nameless=True)


class GuideThiefTestEnv(GuideThiefEnv):
    def __init__(self):
        super().__init__(
            size=5,
            nameless=False,
            max_steps=20,
        )

class GuideThiefVeryDiff(GuideThiefEnv):
    def __init__(self):
        super().__init__(
            size=5,
            nameless=False,
            max_steps=20,
            very_diff=True,
        )

class GuideThiefOneWord(GuideThiefEnv):
    def __init__(self):
        super().__init__(
            size=5,
            nameless=False,
            max_steps=20,
            very_diff=False,
            one_word=True
        )

register(
    id='MiniGrid-GuideThief-5x5-v0',
    entry_point='gym_minigrid.envs:GuideThiefEnv'
)

register(
    id='MiniGrid-GuideThief-6x6-v0',
    entry_point='gym_minigrid.envs:GuideThief6x6Env'
)

register(
    id='MiniGrid-GuideThief-8x8-v0',
    entry_point='gym_minigrid.envs:GuideThief8x8Env'
)

register(
    id='MiniGrid-GuideThiefNameless-8x8-v0',
    entry_point='gym_minigrid.envs:GuideThiefNameless8x8Env'
)

register(
    id='MiniGrid-GuideThiefTest-v0',
    entry_point='gym_minigrid.envs:GuideThiefTestEnv'
)

register(
    id='MiniGrid-GuideThiefVeryDiff-v0',
    entry_point='gym_minigrid.envs:GuideThiefVeryDiff'
)
register(
    id='MiniGrid-GuideThiefOneWord-v0',
    entry_point='gym_minigrid.envs:GuideThiefOneWord'
)
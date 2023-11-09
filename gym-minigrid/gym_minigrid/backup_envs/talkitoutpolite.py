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
        self.was_introduced_to = False

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

    def listen(self, utterance):
        if self.env.hidden_npc:
            return None

        if self.was_introduced_to:
            if utterance == TalkItOutPoliteGrammar.construct_utterance([0, 1]):
                if self.env.nameless:
                    return "Ask the {} guide.".format(self.env.true_guide.color)
                else:
                    return "Ask {}.".format(self.env.true_guide.name)
        else:
            if utterance == TalkItOutPoliteGrammar.construct_utterance([3, 3]):
                self.was_introduced_to = True
                return "I am well."

        return None



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
        # todo: this should be id == name
        self.npc_type = 1  # this will be put into the encoding
        self.was_introduced_to = False

        # Select a random target object as mission
        obj_idx = self.env._rand_int(0, len(self.env.door_pos))
        self.target_pos = self.env.door_pos[obj_idx]
        self.target_color = self.env.door_colors[obj_idx]

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

    def listen(self, utterance):
        if self.env.hidden_npc:
            return None

        if self.was_introduced_to:
            if utterance == TalkItOutPoliteGrammar.construct_utterance([0, 1]):
                if self.liar:
                    fake_colors = [c for c in self.env.door_colors if c != self.env.target_color]
                    fake_color = self.env._rand_elem(fake_colors)

                    # Generate the mission string
                    assert fake_color != self.env.target_color
                    return 'go to the %s door' % fake_color

                else:
                    return self.env.mission
        else:
            if utterance == TalkItOutPoliteGrammar.construct_utterance([3, 3]):
                self.was_introduced_to = True
                return "I am well."


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


class TalkItOutPoliteGrammar(object):

    templates = ["Where is", "Open", "Close", "How are"]
    things = [
        "sesame", "the exit", "the wall", "you", "the ceiling", "the window", "the entrance", "the closet",
        "the drawer", "the fridge", "the floor", "the lamp", "the trash can", "the chair", "the bed", "the sofa"
    ]
    assert len(templates)*len(things) == 64
    print("language complexity {}:".format(len(templates)*len(things)))

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "


class TalkItOutPoliteEnv(MultiModalMiniGridEnv):
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
        max_steps=100,
        hidden_npc=False,

    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.hear_yourself = hear_yourself
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.nameless = nameless
        self.hidden_npc = hidden_npc

        if max_steps is None:
            max_steps = 5*size**2

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            actions=MiniGridEnv.Actions,
            action_space=spaces.MultiDiscrete([
                len(MiniGridEnv.Actions),
                *TalkItOutPoliteGrammar.grammar_action_space.nvec
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

        # add guides
        GUIDE_NAMES = ["John", "Jack"]

        # Set a randomly coloured TRUE GUIDE at a random position
        name = self._rand_elem(GUIDE_NAMES)
        color = self._rand_elem(COLOR_NAMES)
        self.true_guide = Guide(color, name, self, liar=False)

        # Place it randomly, omitting invalid positions
        self.place_obj(self.true_guide,
                       size=(width, height),
                       # reject_fn=lambda _, p: tuple(p) in self.door_front_pos)
                       reject_fn=lambda _, p: tuple(p) in [*self.door_front_pos, tuple(self.wizard.cur_pos)])

        # Set a randomly coloured FALSE GUIDE at a random position
        name = self._rand_elem([n for n in GUIDE_NAMES if n != self.true_guide.name])

        color = self._rand_elem([c for c in COLOR_NAMES if c != self.true_guide.color])

        self.false_guide = Guide(color, name, self, liar=True)

        # Place it randomly, omitting invalid positions
        self.place_obj(self.false_guide,
                       size=(width, height),
                       reject_fn=lambda _, p: tuple(p) in [
                           *self.door_front_pos, tuple(self.wizard.cur_pos), tuple(self.true_guide.cur_pos)])

        assert self.true_guide.name != self.false_guide.name
        assert self.true_guide.color != self.false_guide.color

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
        self.outcome_info = None

    def step(self, action):
        p_action = action[0]
        utterance_action = action[1:]

        # assert all nan or neither nan
        assert len(set(np.isnan(utterance_action))) == 1

        speak_flag = not all(np.isnan(utterance_action))

        obs, reward, done, info = super().step(p_action)

        if speak_flag:
            utterance = TalkItOutPoliteGrammar.construct_utterance(utterance_action)
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

            if self.false_guide.is_near_agent():
                reply = self.false_guide.listen(utterance)

                if reply:
                    if self.nameless:
                        self.utterance += "{} \n".format(reply)
                    else:
                        self.utterance += "{}: {} \n".format(self.false_guide.name, reply)

                    self.conversation += "{}: {} \n".format(self.false_guide.name, reply)

            if utterance == TalkItOutPoliteGrammar.construct_utterance([1, 0]):
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

        if self.hidden_npc:
            # all npc are hidden
            assert np.argwhere(obs['image'][:,:,0] == OBJECT_TO_IDX['npc']).size == 0
            assert "{}:".format(self.wizard.name) not in self.utterance
            # assert "{}:".format(self.true_guide.name) not in self.utterance
            # assert "{}:".format(self.false_guide.name) not in self.utterance

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

        self.window.set_caption(self.conversation, [
            "Gandalf:",
            "Jack:",
            "John:",
            "Where is the exit",
            "Open sesame",
        ])

        self.window.ax.set_title("correct door: {}".format(self.true_guide.target_color), loc="left", fontsize=10)
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


class TalkItOutPolite8x8Env(TalkItOutPoliteEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, max_steps=100, **kwargs)


class TalkItOutPolite6x6Env(TalkItOutPoliteEnv):
    def __init__(self):
        super().__init__(size=6, max_steps=100)


class TalkItOutPoliteNameless8x8Env(TalkItOutPoliteEnv):
    def __init__(self):
        super().__init__(size=8, max_steps=100, nameless=True)

register(
    id='MiniGrid-TalkItOutPolite-5x5-v0',
    entry_point='gym_minigrid.envs:TalkItOutPoliteEnv'
)

register(
    id='MiniGrid-TalkItOutPolite-6x6-v0',
    entry_point='gym_minigrid.envs:TalkItOutPolite6x6Env'
)

register(
    id='MiniGrid-TalkItOutPolite-8x8-v0',
    entry_point='gym_minigrid.envs:TalkItOutPolite8x8Env'
)

register(
    id='MiniGrid-TalkItOutPoliteNameless-8x8-v0',
    entry_point='gym_minigrid.envs:TalkItOutPoliteNameless8x8Env'
)
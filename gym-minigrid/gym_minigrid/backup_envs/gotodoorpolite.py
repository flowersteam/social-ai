from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 

class Guide(NPC):
    """
    A simple NPC that wants an agent to go to an object (randomly chosen among object_pos list)
    """

    def __init__(self, color, name, env):
        super().__init__(color)
        self.name = name
        self.env = env
        self.introduced = False

        # Select a random target object as mission
        obj_idx = self.env._rand_int(0, len(self.env.door_pos))
        self.target_pos = self.env.door_pos[obj_idx]
        self.target_color = self.env.door_colors[obj_idx]

    def listen(self, utterance):
        if utterance == PoliteGrammar.construct_utterance([0, 2]):
            self.introduced = True
            return "I am good. Thank you."
        elif utterance == PoliteGrammar.construct_utterance([1, 1]):
            if self.introduced:
                return self.env.mission

        return None

    # def is_near_agent(self):
    #     ax, ay = self.env.agent_pos
    #     wx, wy = self.cur_pos
    #     if (ax == wx and abs(ay - wy) == 1) or (ay == wy and abs(ax - wx) == 1):
    #         return True
    #     return False


class PoliteGrammar(object):

    templates = ["How are", "Where is", "Open"]
    things = ["sesame", "the exit", 'you']

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "


class GoToDoorPoliteEnv(MultiModalMiniGridEnv):
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
        max_steps=100,
    ):
        assert size >= 5

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            actions=MiniGridEnv.Actions,
            action_space=spaces.MultiDiscrete([
                len(MiniGridEnv.Actions),
                *PoliteGrammar.grammar_action_space.nvec
            ])
        )
        self.hear_yourself = hear_yourself
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty

        self.empty_symbol = "NA \n"

        print({
            "size": size,
            "hear_yourself": hear_yourself,
            "diminished_reward": diminished_reward,
            "step_penalty": step_penalty,
        })


    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

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

        # Set a randomly coloured NPC at a random position
        color = self._rand_elem(COLOR_NAMES)
        self.wizard = Guide(color, "Gandalf", self)

        # Place it randomly, omitting front of door positions
        self.place_obj(self.wizard,
                       size=(width, height),
                       reject_fn=lambda _, p: tuple(p) in self.door_front_pos)

        # Randomize the agent start position and orientation
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

    def step(self, action):
        p_action = action[0]
        utterance_action = action[1:]

        assert len(set(np.isnan(utterance_action))) == 1

        speak_flag = not all(np.isnan(utterance_action))

        obs, reward, done, info = super().step(p_action)

        if speak_flag:
            agent_utterance = PoliteGrammar.construct_utterance(utterance_action)
            if self.hear_yourself:
                self.utterance += "YOU: {} \n".format(agent_utterance)

            # check if near wizard
            if self.wizard.is_near_agent():
                reply = self.wizard.listen(agent_utterance)

                if reply:
                    self.utterance += "{}: {} \n".format(self.wizard.name, reply)

        # Don't let the agent open any of the doors
        if p_action == self.actions.toggle:
            done = True

        if p_action == self.actions.done:
            ax, ay = self.agent_pos
            tx, ty = self.target_pos

            if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
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
        self.window.set_caption(self.utterance_history, [
            "Gandalf:",
            "Jack:",
            "John:",
            "Where is the exit",
            "Open sesame",
        ])
        return obs


class GoToDoorPoliteTesting(GoToDoorPoliteEnv):
    def __init__(self):
        super().__init__(
            size=5,
            hear_yourself=False,
            diminished_reward=False,
            step_penalty=True,
            max_steps=100
        )

class GoToDoorPolite8x8Env(GoToDoorPoliteEnv):
    def __init__(self):
        super().__init__(size=8, max_steps=100)


class GoToDoorPolite6x6Env(GoToDoorPoliteEnv):
    def __init__(self):
        super().__init__(size=6, max_steps=100)


# hear yourself
class GoToDoorPoliteHY8x8Env(GoToDoorPoliteEnv):
    def __init__(self):
        super().__init__(size=8, hear_yourself=True, max_steps=100)


class GoToDoorPoliteHY6x6Env(GoToDoorPoliteEnv):
    def __init__(self):
        super().__init__(size=6, hear_yourself=True, max_steps=100)


class GoToDoorPoliteHY5x5Env(GoToDoorPoliteEnv):
    def __init__(self):
        super().__init__(size=5, hear_yourself=True, max_steps=100)

register(
    id='MiniGrid-GoToDoorPolite-Testing-v0',
    entry_point='gym_minigrid.envs:GoToDoorPoliteTesting'
)

register(
    id='MiniGrid-GoToDoorPolite-5x5-v0',
    entry_point='gym_minigrid.envs:GoToDoorPoliteEnv'
)

register(
    id='MiniGrid-GoToDoorPolite-6x6-v0',
    entry_point='gym_minigrid.envs:GoToDoorPolite6x6Env'
)

register(
    id='MiniGrid-GoToDoorPolite-8x8-v0',
    entry_point='gym_minigrid.envs:GoToDoorPolite8x8Env'
)
register(
    id='MiniGrid-GoToDoorPoliteHY-5x5-v0',
    entry_point='gym_minigrid.envs:GoToDoorPoliteHY5x5Env'
)

register(
    id='MiniGrid-GoToDoorPoliteHY-6x6-v0',
    entry_point='gym_minigrid.envs:GoToDoorPoliteHY6x6Env'
)

register(
    id='MiniGrid-GoToDoorPoliteHY-8x8-v0',
    entry_point='gym_minigrid.envs:GoToDoorPoliteHY8x8Env'
)

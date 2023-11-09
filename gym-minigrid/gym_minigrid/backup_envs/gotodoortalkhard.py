from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 


class TalkHardGrammar(object):

    templates = ["Where is", "What is"]
    things = ["the exit", "the chair"]

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + "."


class GoToDoorTalkHardEnv(MultiModalMiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self,
        size=5,
        hear_yourself=False,
    ):
        assert size >= 5

        super().__init__(
            grid_size=size,
            max_steps=5*size**2,
            # Set this to True for maximum speed
            see_through_walls=True,
            actions=MiniGridEnv.Actions,
            action_space=spaces.MultiDiscrete([
                len(MiniGridEnv.Actions),
                *TalkHardGrammar.grammar_action_space.nvec
            ])
        )
        self.hear_yourself = hear_yourself

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Randomly vary the room width and height
        width = self._rand_int(5, width+1)
        height = self._rand_int(5, height+1)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the 4 doors at random positions
        doorPos = []
        doorPos.append((self._rand_int(2, width-2), 0))
        doorPos.append((self._rand_int(2, width-2), height-1))
        doorPos.append((0, self._rand_int(2, height-2)))
        doorPos.append((width-1, self._rand_int(2, height-2)))

        # Generate the door colors
        doorColors = []
        while len(doorColors) < len(doorPos):
            color = self._rand_elem(COLOR_NAMES)
            if color in doorColors:
                continue
            doorColors.append(color)

        # Place the doors in the grid
        for idx, pos in enumerate(doorPos):
            color = doorColors[idx]
            self.grid.set(*pos, Door(color))

        # Randomize the agent start position and orientation
        self.place_agent(size=(width, height))

        # Select a random target door
        doorIdx = self._rand_int(0, len(doorPos))
        self.target_pos = doorPos[doorIdx]
        self.target_color = doorColors[doorIdx]

        # Generate the mission string
        self.mission = 'go to the %s door' % self.target_color

        # Initialize the dialogue string
        self.dialogue = "This is what you hear. "

    def gen_obs(self):
        obs = super().gen_obs()

        # add dialogue to obs
        obs["dialogue"] = self.dialogue

        return obs

    def step(self, action):
        p_action = action[0]
        utterance_action = action[1:]

        # assert all nan or neither nan
        assert len(set(np.isnan(utterance_action))) == 1

        speak_flag = not all(np.isnan(utterance_action))

        if speak_flag:
            utterance = TalkHardGrammar.construct_utterance(utterance_action)

            reply = self.mission
            NPC_name = "Wizard"

            if self.hear_yourself:
                self.dialogue += "YOU: {} \n".format(utterance)

            if utterance == TalkHardGrammar.construct_utterance([0, 0]):
                self.dialogue += "{}: {} \n".format(NPC_name, reply)  # dummy reply gives mission

        obs, reward, done, info = super().step(p_action)

        ax, ay = self.agent_pos
        tx, ty = self.target_pos

        # Don't let the agent open any of the doors
        if p_action == self.actions.toggle:
            done = True

        # Reward performing done action in front of the target door
        if p_action == self.actions.done:
            if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
                reward = self._reward()
            done = True

        return obs, reward, done, info

    def render(self, *args, **kwargs):
        obs = super().render(*args, **kwargs)
        self.window.set_caption(self.dialogue, [
            "Gandalf:",
            "Jack:",
            "John:",
            "Where is the exit",
            "Open sesame",
        ])
        return obs


class GoToDoorTalkHard8x8Env(GoToDoorTalkHardEnv):
    def __init__(self):
        super().__init__(size=8)


class GoToDoorTalkHard6x6Env(GoToDoorTalkHardEnv):
    def __init__(self):
        super().__init__(size=6)


# hear yourself
class GoToDoorTalkHardHY8x8Env(GoToDoorTalkHardEnv):
    def __init__(self):
        super().__init__(size=8, hear_yourself=True)


class GoToDoorTalkHardHY6x6Env(GoToDoorTalkHardEnv):
    def __init__(self):
        super().__init__(size=6, hear_yourself=True)


class GoToDoorTalkHardHY5x5Env(GoToDoorTalkHardEnv):
    def __init__(self):
        super().__init__(size=5, hear_yourself=True)

register(
    id='MiniGrid-GoToDoorTalkHard-5x5-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHardEnv'
)

register(
    id='MiniGrid-GoToDoorTalkHard-6x6-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHard6x6Env'
)

register(
    id='MiniGrid-GoToDoorTalkHard-8x8-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHard8x8Env'
)
register(
    id='MiniGrid-GoToDoorTalkHardHY-5x5-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHardHY5x5Env'
)

register(
    id='MiniGrid-GoToDoorTalkHardHY-6x6-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHardHY6x6Env'
)

register(
    id='MiniGrid-GoToDoorTalkHardHY-8x8-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHardHY8x8Env'
)

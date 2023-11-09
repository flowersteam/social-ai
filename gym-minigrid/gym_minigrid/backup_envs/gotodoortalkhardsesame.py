from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 


class TalkHardSesameGrammar(object):

    templates = ["Where is", "Open"]
    things = ["sesame", "the exit"]

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "


class GoToDoorTalkHardSesameEnv(MultiModalMiniGridEnv):
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
                *TalkHardSesameGrammar.grammar_action_space.nvec
            ])
        )
        self.hear_yourself = hear_yourself
        
        self.empty_symbol = "NA \n"

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Randomly vary the room width and height
        width = self._rand_int(5, width+1)
        height = self._rand_int(5, height+1)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the 4 doors at random positions
        self.doorPos = []
        self.doorPos.append((self._rand_int(2, width-2), 0))
        self.doorPos.append((self._rand_int(2, width-2), height-1))
        self.doorPos.append((0, self._rand_int(2, height-2)))
        self.doorPos.append((width-1, self._rand_int(2, height-2)))

        # Generate the door colors
        doorColors = []
        while len(doorColors) < len(self.doorPos):
            color = self._rand_elem(COLOR_NAMES)
            if color in doorColors:
                continue
            doorColors.append(color)

        # Place the doors in the grid
        for idx, pos in enumerate(self.doorPos):
            color = doorColors[idx]
            self.grid.set(*pos, Door(color))

        # Randomize the agent start position and orientation
        self.place_agent(size=(width, height))

        # Select a random target door
        doorIdx = self._rand_int(0, len(self.doorPos))
        self.target_pos = self.doorPos[doorIdx]
        self.target_color = doorColors[doorIdx]

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

        # assert all nan or neither nan
        assert len(set(np.isnan(utterance_action))) == 1

        speak_flag = not all(np.isnan(utterance_action))

        obs, reward, done, info = super().step(p_action)

        if speak_flag:
            utterance = TalkHardSesameGrammar.construct_utterance(utterance_action)

            if self.hear_yourself:
                self.utterance += "YOU: {} \n".format(utterance)

            if utterance == TalkHardSesameGrammar.construct_utterance([0, 1]):
                reply = self.mission
                NPC_name = "Wizard"
                self.utterance += "{}: {} \n".format(NPC_name, reply)  # dummy reply gives mission

            elif utterance == TalkHardSesameGrammar.construct_utterance([1, 0]):
                ax, ay = self.agent_pos
                tx, ty = self.target_pos

                if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
                    reward = self._reward()
                    
                for dx, dy in self.doorPos:
                    if (ax == dx and abs(ay - dy) == 1) or (ay == dy and abs(ax - dx) == 1):
                        # agent has chosen some door episode, regardless of if the door is correct the episode is over
                        done = True

        # Don't let the agent open any of the doors
        if p_action == self.actions.toggle:
            done = True
            
        # fill observation with text
        self.append_existing_utterance_to_history()
        obs = self.add_utterance_to_observation(obs)
        self.reset_utterance()

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


class GoToDoorTalkHardSesame8x8Env(GoToDoorTalkHardSesameEnv):
    def __init__(self):
        super().__init__(size=8)


class GoToDoorTalkHardSesame6x6Env(GoToDoorTalkHardSesameEnv):
    def __init__(self):
        super().__init__(size=6)


# hear yourself
class GoToDoorTalkHardSesameHY8x8Env(GoToDoorTalkHardSesameEnv):
    def __init__(self):
        super().__init__(size=8, hear_yourself=True)


class GoToDoorTalkHardSesameHY6x6Env(GoToDoorTalkHardSesameEnv):
    def __init__(self):
        super().__init__(size=6, hear_yourself=True)


class GoToDoorTalkHardSesameHY5x5Env(GoToDoorTalkHardSesameEnv):
    def __init__(self):
        super().__init__(size=5, hear_yourself=True)

register(
    id='MiniGrid-GoToDoorTalkHardSesame-5x5-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHardSesameEnv'
)

register(
    id='MiniGrid-GoToDoorTalkHardSesame-6x6-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHardSesame6x6Env'
)

register(
    id='MiniGrid-GoToDoorTalkHardSesame-8x8-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHardSesame8x8Env'
)
register(
    id='MiniGrid-GoToDoorTalkHardSesameHY-5x5-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHardSesameHY5x5Env'
)

register(
    id='MiniGrid-GoToDoorTalkHardSesameHY-6x6-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHardSesameHY6x6Env'
)

register(
    id='MiniGrid-GoToDoorTalkHardSesameHY-8x8-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHardSesameHY8x8Env'
)

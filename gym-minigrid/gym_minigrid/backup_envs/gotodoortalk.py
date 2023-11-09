from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 

# these two classes should maybe be extracted to a utils file so they can be used all over our envs


class GoToDoorTalkEnv(MultiModalMiniGridEnv):
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
                *Grammar.grammar_action_space.nvec
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

        if speak_flag:
            agent_utterance = Grammar.construct_utterance(utterance_action)

            reply = self.mission
            NPC_name = "Wizard"

            if self.hear_yourself:
                self.utterance += "YOU: {} \n".format(agent_utterance)

            self.utterance += "{}: {} \n".format(NPC_name, reply)

        obs, reward, done, info = super().step(p_action)

        # Don't let the agent open any of the doors
        if p_action == self.actions.toggle:
            done = True

        # Reward performing done action in front of the target door
        if p_action == self.actions.done:
            ax, ay = self.agent_pos
            tx, ty = self.target_pos
            
            if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
                reward = self._reward()
            done = True

        # fill observation with text
        self.append_existing_utterance_to_history()
        obs = self.add_utterance_to_observation(obs)
        self.reset_utterance()

        return obs, reward, done, info

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


class GoToDoorTalk8x8Env(GoToDoorTalkEnv):
    def __init__(self):
        super().__init__(size=8)

class GoToDoorTalk6x6Env(GoToDoorTalkEnv):
    def __init__(self):
        super().__init__(size=6)

# hear yourself
class GoToDoorTalkHY8x8Env(GoToDoorTalkEnv):
    def __init__(self):
        super().__init__(size=8, hear_yourself=True)

class GoToDoorTalkHY6x6Env(GoToDoorTalkEnv):
    def __init__(self):
        super().__init__(size=6, hear_yourself=True)

class GoToDoorTalkHYEnv(GoToDoorTalkEnv):
    def __init__(self):
        super().__init__(size=5, hear_yourself=True)


register(
    id='MiniGrid-GoToDoorTalk-5x5-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkEnv'
)

register(
    id='MiniGrid-GoToDoorTalk-6x6-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalk6x6Env'
)

register(
    id='MiniGrid-GoToDoorTalk-8x8-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalk8x8Env'
)

# hear yourself
register(
    id='MiniGrid-GoToDoorTalkHY-5x5-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHYEnv'
)

register(
    id='MiniGrid-GoToDoorTalkHY-6x6-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHY6x6Env'
)

register(
    id='MiniGrid-GoToDoorTalkHY-8x8-v0',
    entry_point='gym_minigrid.envs:GoToDoorTalkHY8x8Env'
)

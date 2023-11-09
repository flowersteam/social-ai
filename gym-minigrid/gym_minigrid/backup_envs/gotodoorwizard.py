from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 


class simpleWizard(NPC):
    """
    A simple NPC that wants an agent to go to an object (randomly chosen among object_pos list)
    """
    def __init__(self, color, name, env):
        super().__init__(color)
        self.name = name
        self.env = env
        self.has_spoken = False  # wizards only speak once

        # Select a random target object as mission
        obj_idx = self.env._rand_int(0, len(self.env.door_pos))
        self.target_pos = self.env.door_pos[obj_idx]
        self.target_color = self.env.door_colors[obj_idx]

        # Generate the mission string
        self.wizard_mission = 'go to the %s door' % self.target_color

    def listen(self, utterance):
        if not self.has_spoken:
            self.has_spoken = True
            return self.wizard_mission
        return None

    def is_satisfied(self):
        ax, ay = self.env.agent_pos
        tx, ty = self.target_pos
        if (ax == tx and abs(ay - ty) == 1) or (ay == ty and abs(ax - tx) == 1):
            return True
        return False

    def is_near_agent(self):
        ax, ay = self.env.agent_pos
        wx, wy = self.cur_pos
        if (ax == wx and abs(ay - wy) == 1) or (ay == wy and abs(ax - wx) == 1):
            return True
        return False


class GoToDoorWizard(MiniGridEnv):
    """
    Environment in which the agent is instructed to "please the wizard",
    i.e. to go ask him for a quest (which is goto door)
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

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Randomly vary the room width and height
        width = self._rand_int(5, width+1)
        height = self._rand_int(5, height+1)

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
        self.wizard = simpleWizard(color, "Gandalf", self)

        # Place it randomly, omitting front of door positions
        self.place_obj(self.wizard,
                       size=(width, height),
                       reject_fn=lambda _, p: tuple(p) in self.door_front_pos)

        # Randomize the agent start position and orientation
        self.place_agent(size=(width, height))

        # Generate the mission string
        self.mission = 'please the wizard'

        # Initialize the dialogue string
        self.dialogue = "This is what you hear. "

    def gen_obs(self):
        obs = super().gen_obs()

        # add dialogue to obs
        obs["dialogue"] = self.dialogue

        return obs

    def step(self, action):

        # dirty handle of action provided by manual_control todo improve
        if type(action) == MiniGridEnv.Actions:
            action = [action, None]

        p_action = action[0]
        utterance_action = action[1:]

        obs, reward, done, info = super().step(p_action)

        # check if near wizard
        if self.wizard.is_near_agent():#p_action == self.actions.talk and self.near_wizard:
            #utterance = Grammar.construct_utterance(utterance_action)
            reply = self.wizard.listen("")
            # if self.hear_yourself:
            #     self.dialogue += "YOU: " + utterance
            if reply:
                self.dialogue += "{}: {}".format(self.wizard.name, reply)

        # Don't let the agent open any of the doors
        if p_action == self.actions.toggle:
            done = True

        # Reward performing done action if pleasing the wizard
        if p_action == self.actions.done:
            if self.wizard.is_satisfied():
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
        self.window.fig.gca().set_title("goal: "+self.mission)
        return obs


class GoToDoorWizard5x5Env(GoToDoorWizard):
    def __init__(self):
        super().__init__(size=5)


class GoToDoorWizard7x7Env(GoToDoorWizard):
    def __init__(self):
        super().__init__(size=7)

class GoToDoorWizard8x8Env(GoToDoorWizard):
    def __init__(self):
        super().__init__(size=8)



register(
    id='MiniGrid-GoToDoorWizard-5x5-v0',
    entry_point='gym_minigrid.envs:GoToDoorWizard5x5Env'
)

register(
    id='MiniGrid-GoToDoorWizard-7x7-v0',
    entry_point='gym_minigrid.envs:GoToDoorWizard7x7Env'
)

register(
    id='MiniGrid-GoToDoorWizard-8x8-v0',
    entry_point='gym_minigrid.envs:GoToDoorWizard8x8Env'
)
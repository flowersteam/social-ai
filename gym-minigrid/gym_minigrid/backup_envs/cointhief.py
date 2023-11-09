from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import time
from collections import deque


class Thief(NPC):
    """
    A dancing NPC that the agent has to copy
    NPC executes a sequence of movement and utterances
    """

    def __init__(self, color, name, env, thief_pos, hidden_npc=False, tag_visible_coins=False, view_size=5, look_around=True):
        super().__init__(color)
        self.name = name
        self.npc_type = 0
        self.env = env
        self.cur_pos = thief_pos
        self.init_pos = thief_pos
        self.view_size = view_size
        self.npc_dir = self._look_at_agent()  # Thief stares at its victim
        self.init_dir = self.npc_dir
        self.actions = self.env.possible_actions
        self.tag_visible_coins = tag_visible_coins

        self.nb_given_coins = None
        self.look_around = look_around
        if self.look_around:  # randomly select in which direction NPC is looking around
            if np.random.random() > 0.5:  # will look left
                self.look = self.rotate_left
                self.look_back = self.rotate_right
            else:  # will look right
                self.look = self.rotate_right
                self.look_back = self.rotate_left

        self.nb_seen_coins = self._count_coins()  # This is how much coins Thief wants, at least
        self.add_npc_direction = True
        self.nb_steps = 0
        self.hidden_npc = hidden_npc

    def step(self, agent_action, agent_utterance):
        agent_disobeyed = False
        agent_gave_coins = False
        utterance = None

        if self.nb_steps == 0:
            utterance = "Freeze! Give me all the coins you have!"

        if self.nb_steps >= 0 and self.look_around:
            if self.npc_dir == self.init_dir:  # start to look around
                self.look()
            else:  # resume looking to agent
                self.look_back()

        if not(agent_utterance is None):
            self.nb_given_coins = int(agent_utterance[-2])

            if self.nb_given_coins >= self.nb_seen_coins:
                agent_gave_coins = True
            else:  # agent gave not enough coins
                agent_disobeyed = True

        # The thief forbids the agent to move, apart from looking around (rotating)
        if not (np.isnan(agent_action) or agent_action == self.actions.left or agent_action == self.actions.right):
            agent_disobeyed = True

        self.nb_steps += 1
        return agent_disobeyed, agent_gave_coins, utterance

    def get_status_str(self):
        return "thief sees: {} \n agent gives: {}".format(self.nb_seen_coins, self.nb_given_coins)

    def _count_coins(self):
        # get seen coins
        coins_pos = self.get_pos_visible_coins()

        if self.look_around:
            self.look()
            # add coins visible from this new direction
            coins_pos += self.get_pos_visible_coins()
            # remove coins that we already saw
            if len(coins_pos) > 0:
                coins_pos = np.unique(coins_pos, axis=0).tolist()
            self.look_back()

        return len(coins_pos)

    def _look_at_agent(self):
        npc_dir = None
        ax, ay = self.env.agent_pos
        tx, ty = self.cur_pos
        delta_x, delta_y = ax - tx, ay - ty
        if delta_x == 1:
            npc_dir = 0
        elif delta_x == -1:
            npc_dir = 2
        elif delta_y == 1:
            npc_dir = 1
        elif delta_y == -1:
            npc_dir = 3
        else:
            raise NotImplementedError

        return npc_dir

    def gen_npc_obs_grid(self):
        """
                Generate the sub-grid observed by the npc.
                This method also outputs a visibility mask telling us which grid
                cells the npc can actually see.
        """
        view_size = self.view_size

        topX, topY, botX, botY = self.env.get_view_exts(dir=self.npc_dir, view_size=view_size, pos=self.cur_pos)

        grid = self.env.grid.slice(topX, topY, view_size, view_size)

        for i in range(self.npc_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.env.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(view_size // 2, view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        # agent_pos = grid.width // 2, grid.height - 1
        # if self.carrying:
        #     grid.set(*agent_pos, self.carrying)
        # else:
        #     grid.set(*agent_pos, None)

        return grid, vis_mask

    def get_pos_visible_coins(self):
        """
        Generate the npc's view (partially observable, low-resolution encoding)
        return the list of unique visible coins
        """

        grid, vis_mask = self.gen_npc_obs_grid()

        coins_pos = []

        for obj in grid.grid:
            if isinstance(obj, Ball):
                coins_pos.append(obj.cur_pos)
                if self.tag_visible_coins:
                    obj.tag()

        return coins_pos

    def can_overlap(self):
        # If the NPC is hidden, agent can overlap on it
        return self.hidden_npc


class CoinThiefGrammar(object):

    templates = ["Here is"]
    things = ["0","1","2","3","4","5","6"]

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "

    @classmethod
    def random_utterance(cls):
        return np.random.choice(cls.templates) + " " + np.random.choice(cls.things) + " "


class ThiefActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2


class CoinThiefEnv(MultiModalMiniGridEnv):
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
        hidden_npc=False,
        max_steps=20,
        full_obs=False,
        few_actions=False,
        tag_visible_coins=False,
        nb_coins=6,
        npc_view_size=5,
        npc_look_around=True

    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.hear_yourself = hear_yourself
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.hidden_npc = hidden_npc
        self.few_actions = few_actions
        self.possible_actions = ThiefActions if self.few_actions else MiniGridEnv.Actions
        self.nb_coins = nb_coins
        self.tag_visible_coins = tag_visible_coins
        self.npc_view_size = npc_view_size
        self.npc_look_around = npc_look_around
        if max_steps is None:
            max_steps = 5*size**2

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=True,
            full_obs=full_obs,
            actions=MiniGridEnv.Actions,
            action_space=spaces.MultiDiscrete([
                len(self.possible_actions),
                *CoinThiefGrammar.grammar_action_space.nvec
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
        # width = self._rand_int(5, width+1)
        # height = self._rand_int(5, height+1)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Randomize the agent's start position and orientation
        self.place_agent(size=(width, height))

        # Get possible near-agent positions, and place thief in one of them
        ax, ay = self.agent_pos
        near_agent_pos = [[ax, ay + 1], [ax, ay - 1], [ax - 1, ay], [ax + 1, ay]]
        # get empty cells positions
        available_pos = []
        for p in near_agent_pos:
            if self.grid.get(*p) is None:
                available_pos.append(p)
        thief_pos = self._rand_elem(available_pos)

        # Add randomly placed coins
        # Types and colors of objects we can generate
        types = ['ball']
        objs = []
        objPos = []

        # Until we have generated all the objects
        while len(objs) < self.nb_coins:
            objType = self._rand_elem(types)
            objColor = 'yellow'

            if objType == 'ball':
                obj = Ball(objColor)
            else:
                raise NotImplementedError

            pos = self.place_obj(obj, reject_fn=lambda env,pos: pos.tolist() == thief_pos)
            objs.append((objType, objColor))
            objPos.append(pos)

        # Set a randomly coloured Thief NPC next to the agent
        color = self._rand_elem(COLOR_NAMES)

        self.thief = Thief(color, "Eve", self, thief_pos,
                           hidden_npc=self.hidden_npc,
                           tag_visible_coins=self.tag_visible_coins,
                           view_size=self.npc_view_size,
                           look_around=self.npc_look_around)

        self.grid.set(*thief_pos, self.thief)

        # Generate the mission string
        self.mission = 'save as much coins as possible'

        # Dummy beginning string
        self.beginning_string = "This is what you hear. \n"
        self.utterance = self.beginning_string

        # utterance appended at the end of each step
        self.utterance_history = ""

        # used for rendering
        self.conversation = self.utterance
        self.outcome_info = None

    def step(self, action):
        p_action = action[0] if np.isnan(action[0]) else int(action[0])
        if len(action) == 1:  # agent cannot speak
            utterance_action = [np.nan, np.nan]
        else:
            utterance_action = action[1:]

        obs, reward, done, info = super().step(p_action)

        # assert all nan or neither nan
        assert len(set(np.isnan(utterance_action))) == 1
        speak_flag = not all(np.isnan(utterance_action))

        if speak_flag:
            utterance = CoinThiefGrammar.construct_utterance(utterance_action)
            self.conversation += "{}: {} \n".format("Agent", utterance)

        # Don't let the agent open any doors
        if not self.few_actions and p_action == self.actions.toggle:
            done = True

        if not self.few_actions and p_action == self.actions.done:
            done = True

        # npc's turn
        agent_disobeyed, agent_gave_coins, npc_utterance = self.thief.step(p_action, utterance if speak_flag else None)

        if self.hidden_npc:
            npc_utterance = None

        if npc_utterance:
            self.utterance += "{} \n".format(npc_utterance)
            self.conversation += "{}: {} \n".format(self.thief.name, npc_utterance)

        if agent_disobeyed:
            done = True

        if agent_gave_coins:
            done = True
            if self.thief.nb_seen_coins == self.thief.nb_given_coins:
                reward = self._reward()
                self.outcome_info = "SUCCESS: agent got {} reward \n".format(np.round(reward,1))

        if done and reward == 0:
            self.outcome_info = "FAILURE: agent got {} reward \n".format(reward)

        # discount
        if self.step_penalty:
            reward = reward - 0.01

        if self.hidden_npc:
            # remove npc from agent view
            npc_obs_idx = np.argwhere(obs['image'] == 11)
            if npc_obs_idx.size != 0:  # agent sees npc
                obs['image'][npc_obs_idx[0][0], npc_obs_idx[0][1], :] = [1, 0, 0, 0]

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

        self.window.clear_text()  # erase previous text

        self.window.set_caption(self.conversation)  # overwrites super class caption
        self.window.ax.set_title(self.thief.get_status_str(), loc="left")
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


class CoinThief8x8Env(CoinThiefEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, **kwargs)


class CoinThief6x6Env(CoinThiefEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)


register(
    id='MiniGrid-CoinThief-5x5-v0',
    entry_point='gym_minigrid.envs:CoinThiefEnv'
)

register(
    id='MiniGrid-CoinThief-6x6-v0',
    entry_point='gym_minigrid.envs:CoinThief6x6Env'
)

register(
    id='MiniGrid-CoinThief-8x8-v0',
    entry_point='gym_minigrid.envs:CoinThief8x8Env'
)
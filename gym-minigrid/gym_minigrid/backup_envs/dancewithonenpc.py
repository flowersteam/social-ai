from gym_minigrid.minigrid import *
from gym_minigrid.register import register
 
import time
from collections import deque


class Dancer(NPC):
    """
    A dancing NPC that the agent has to copy
    NPC executes a sequence of movement and utterances
    """

    def __init__(self, color, name, env, dancing_pattern=None,
                 dance_len=3, p_sing=.5, hidden_npc=False, sing_only=False):
        super().__init__(color)
        self.name = name
        self.npc_dir = 1  # NPC initially looks downward
        self.npc_type = 0
        self.env = env
        self.actions = self.env.possible_actions
        self.p_sing = p_sing
        self.sing_only = sing_only
        if self.sing_only:
            p_sing = 1
        self.dancing_pattern = dancing_pattern if dancing_pattern else self._gen_dancing_pattern(dance_len, p_sing)
        self.agent_actions = deque(maxlen=len(self.dancing_pattern))
        self.movement_id_to_fun = {self.actions.left: self.rotate_left,
                                   self.actions.right: self.rotate_right,
                                   self.actions.forward: self.go_forward}
        # for vizualisation only
        self.movement_id_to_str = {self.actions.left: "left",
                                   self.actions.right: "right",
                                   self.actions.forward: "forward",
                                   self.actions.pickup: "pickup",
                                   self.actions.drop: "drop",
                                   self.actions.toggle: "toggle",
                                   self.actions.done: "done",
                                   None: "None"}
        self.dancing_step_idx = 0
        self.done_dancing = False
        self.add_npc_direction = True
        self.nb_steps = 0
        self.hidden_npc = hidden_npc

    def step(self, agent_action, agent_utterance):
        agent_matched_moves = False
        utterance = None

        if self.nb_steps == 0:
            utterance = "Look at me!"
        if self.nb_steps >= 2:  # Wait a couple steps before dancing
            if not self.done_dancing:
                if self.dancing_step_idx == len(self.dancing_pattern):
                    self.done_dancing = True
                    utterance = "Now repeat my moves!"
                else:
                    # NPC moves and speaks according to dance step
                    move_id, utterance = self.dancing_pattern[self.dancing_step_idx]
                    self.movement_id_to_fun[move_id]()

                    self.dancing_step_idx += 1
            else:  # record agent dancing pattern
                self.agent_actions.append((agent_action, agent_utterance))

                if not self.sing_only and list(self.agent_actions) == list(self.dancing_pattern):
                    agent_matched_moves = True
                if self.sing_only:  # only compare utterances
                    if [x[1] for x in self.agent_actions] == [x[1] for x in self.dancing_pattern]:
                        agent_matched_moves = True

        self.nb_steps += 1
        return agent_matched_moves, utterance

    def get_status_str(self):
        readable_dancing_pattern = [(self.movement_id_to_str[dp[0]], dp[1]) for dp in self.dancing_pattern]
        readable_agent_actions = [(self.movement_id_to_str[aa[0]], aa[1]) for aa in self.agent_actions]
        return "dance: {} \n agent: {}".format(readable_dancing_pattern, readable_agent_actions)

    def _gen_dancing_pattern(self, dance_len, p_sing):
        available_moves = [self.actions.left, self.actions.right, self.actions.forward]
        dance_pattern = []
        for _ in range(dance_len):
            move = self.env._rand_elem(available_moves)
            sing = None
            if np.random.random() < p_sing:
                sing = DanceWithOneNPCGrammar.random_utterance()
            dance_pattern.append((move, sing))
        return dance_pattern

    def can_overlap(self):
        # If the NPC is hidden, agent can overlap on it
        return self.hidden_npc



class DanceWithOneNPCGrammar(object):

    templates = ["Move your", "Shake your"]
    things = ["body", "head"]

    grammar_action_space = spaces.MultiDiscrete([len(templates), len(things)])

    @classmethod
    def construct_utterance(cls, action):
        return cls.templates[int(action[0])] + " " + cls.things[int(action[1])] + " "

    @classmethod
    def random_utterance(cls):
        return np.random.choice(cls.templates) + " " + np.random.choice(cls.things) + " "



class DanceActions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2


class DanceWithOneNPCEnv(MultiModalMiniGridEnv):
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
        dance_len=3,
        hidden_npc=False,
        p_sing=.5,
        max_steps=20,
        full_obs=False,
        few_actions=False,
        sing_only=False

    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.hear_yourself = hear_yourself
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.dance_len = dance_len
        self.hidden_npc = hidden_npc
        self.p_sing = p_sing
        self.few_actions = few_actions
        self.possible_actions = DanceActions if self.few_actions else MiniGridEnv.Actions
        self.sing_only = sing_only
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
                *DanceWithOneNPCGrammar.grammar_action_space.nvec
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


        # Set a randomly coloured Dancer NPC
        color = self._rand_elem(COLOR_NAMES)
        self.dancer = Dancer(color, "Ren", self, dance_len=self.dance_len,
                             p_sing=self.p_sing, hidden_npc=self.hidden_npc, sing_only=self.sing_only)

        # Place it on the middle left side of the room
        left_pos = (int((width / 2) - 1), int(height / 2))
        #right_pos = [(width / 2) + 1, height / 2]

        self.grid.set(*left_pos, self.dancer)
        self.dancer.init_pos = left_pos
        self.dancer.cur_pos = left_pos

        # Place it randomly left or right
        #self.place_obj(self.dancer,
        #               size=(width, height))

        # Randomize the agent's start position and orientation
        self.place_agent(size=(width, height))

        # Generate the mission string
        self.mission = 'watch dancer and repeat his moves afterwards'

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
            assert self.p_sing == 0, "Non speaking agent used in a dance env requiring to speak"
            utterance_action = [np.nan, np.nan]
        else:
            utterance_action = action[1:]

        obs, reward, done, info = super().step(p_action)

        if np.isnan(p_action):
            pass


        # assert all nan or neither nan
        assert len(set(np.isnan(utterance_action))) == 1
        speak_flag = not all(np.isnan(utterance_action))

        if speak_flag:
            utterance = DanceWithOneNPCGrammar.construct_utterance(utterance_action)
            self.conversation += "{}: {} \n".format("Agent", utterance)

        # Don't let the agent open any of the doors
        if not self.few_actions and p_action == self.actions.toggle:
            done = True

        if not self.few_actions and p_action == self.actions.done:
            done = True

        # npc's turn
        agent_matched_moves, npc_utterance = self.dancer.step(p_action if not np.isnan(p_action) else None,
                                                              utterance if speak_flag else None)
        if self.hidden_npc:
            npc_utterance = None
        if npc_utterance:
            self.utterance += "{} \n".format(npc_utterance)
            self.conversation += "{}: {} \n".format(self.dancer.name, npc_utterance)
        if agent_matched_moves:
            reward = self._reward()
            self.outcome_info = "SUCCESS: agent got {} reward \n".format(np.round(reward, 1))
            done = True

        # discount
        if self.step_penalty:
            reward = reward - 0.01

        if self.hidden_npc:
            # remove npc from agent view
            npc_obs_idx = np.argwhere(obs['image'] == 11)
            if npc_obs_idx.size != 0:  # agent sees npc
                obs['image'][npc_obs_idx[0][0], npc_obs_idx[0][1], :] = [1, 0, 0, 0]

        if done and reward == 0:
            self.outcome_info = "FAILURE: agent got {} reward \n".format(reward)

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
        self.window.ax.set_title(self.dancer.get_status_str(), loc="left", fontsize=10)
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




class DanceWithOneNPC8x8Env(DanceWithOneNPCEnv):
    def __init__(self, **kwargs):
        super().__init__(size=8, **kwargs)

class DanceWithOneNPC6x6Env(DanceWithOneNPCEnv):
    def __init__(self, **kwargs):
        super().__init__(size=6, **kwargs)



register(
    id='MiniGrid-DanceWithOneNPC-5x5-v0',
    entry_point='gym_minigrid.envs:DanceWithOneNPCEnv'
)

register(
    id='MiniGrid-DanceWithOneNPC-6x6-v0',
    entry_point='gym_minigrid.envs:DanceWithOneNPC6x6Env'
)

register(
    id='MiniGrid-DanceWithOneNPC-8x8-v0',
    entry_point='gym_minigrid.envs:DanceWithOneNPC8x8Env'
)
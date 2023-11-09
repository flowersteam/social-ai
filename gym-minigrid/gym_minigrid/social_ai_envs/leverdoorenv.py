import time
import random

import numpy as np
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.social_ai_envs.socialaigrammar import SocialAIGrammar, SocialAIActions, SocialAIActionSpace
import time
from collections import deque


class Partner(NPC):
    """
    A simple NPC that knows who is telling the truth
    """
    def __init__(self, color, name, env):
        super().__init__(color)
        self.name = name
        self.env = env
        self.npc_dir = 1  # NPC initially looks downward
        self.npc_type = 0  # this will be put into the encoding

        # opposite role of the agent
        self.npc_side = "L" if self.env.agent_side == "R" else "R"

        # how many random action at the beginning -> removes trivial solutions
        self.random_to_go = random.randint(self.env.lever_active_steps, 10)

        assert set([self.npc_side, self.env.agent_side]) == {"L", "R"}

        self.was_introduced_to = False

        self.ate_an_apple = False
        self.pressed_the_lever = False
        self.pushed_the_generator = False
        self.toggling = False

        # target obj
        assert self.env.problem == self.env.parameters["Problem"] if self.env.parameters else "MarblePass"

        self.target_obj = self.env.generator

        assert self.env.grammar.contains_utterance(self.introduction_statement)


    def step(self, utterance):
        reply, info = super().step()

        if self.env.hidden_npc:
            return reply, info

        reply, action = self.handle_introduction(utterance)

        if self.was_introduced_to:

            if self.random_to_go > 0:
                action = random.choice([self.go_forward, self.rotate_left, self.rotate_right])
                self.random_to_go -= 1

            elif self.npc_side == "L":

                if not self.pressed_the_lever:
                    # is the NPC next to the lever
                    if np.abs(self.env.lever.cur_pos - self.cur_pos).sum() > 1:
                        # go to the lever
                        action = self.path_to_pos(self.env.lever.cur_pos)
                    else:

                        # look at the agent
                        wanted_dir = self.compute_wanted_dir(self.env.agent_pos)

                        if wanted_dir != self.npc_dir and not self.toggling:
                            # turn to look at the agent
                            action = self.compute_turn_action(wanted_dir)
                        else:
                            # check if the agent is next to the door
                            if np.abs(self.env.door.cur_pos - self.env.agent_pos).sum() <= 1:
                                self.toggling = True
                                action = self.path_to_toggle_pos(self.env.lever.cur_pos)

            elif self.npc_side == "R":
                if not self.pushed_the_generator:
                    # go to generator and push it
                    action = self.path_to_pos(self.env.generator.cur_pos)

                    if action is None:
                        # the door is not open, no paths exist
                        action = self.path_to_pos(self.env.door.cur_pos)

            else:
                raise ValueError("Undefined role")

        eaten_before = self.env.partner_apple.eaten
        lever_on_before = self.env.lever.is_on
        generator_pushed_before = self.env.generator.is_pressed

        if action is not None:
            action()

        # check if the NPC ate the apple
        if not self.ate_an_apple:
            self.ate_an_apple = not eaten_before and self.env.partner_apple.eaten

        if not self.pressed_the_lever:
            self.pressed_the_lever = not lever_on_before and self.env.lever.is_on

        if not self.pushed_the_generator:
            self.pushed_the_generator = not generator_pushed_before and self.env.generator.is_pressed

        info = {
            "prim_action": action.__name__ if action is not None else "no_op",
            "utterance": reply or "no_op",
            "was_introduced_to": self.was_introduced_to
        }

        assert (reply or "no_op") in self.list_of_possible_utterances

        return reply, info


class LeverDoorEnv(MultiModalMiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self,
        size=10,
        diminished_reward=True,
        step_penalty=False,
        knowledgeable=False,
        max_steps=80,
        hidden_npc=False,
        lever_active_steps=10,
        reward_diminish_factor=0.1,
        egocentric_observation=True,
    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.knowledgeable = knowledgeable
        self.hidden_npc = hidden_npc
        self.hear_yourself = False

        self.grammar = SocialAIGrammar()

        self.init_done = False
        # parameters - to be set in reset
        self.parameters = None

        # encoding size should be 5
        self.add_npc_direction = True
        self.add_npc_point_direction = True
        self.add_npc_last_prim_action = True
        self.lever_active_steps = lever_active_steps

        self.reward_diminish_factor = reward_diminish_factor

        self.egocentric_observation = egocentric_observation
        self.encoding_size = 3 + 2*bool(not self.egocentric_observation) + bool(self.add_npc_direction) + bool(self.add_npc_point_direction) + bool(self.add_npc_last_prim_action)

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=False,
            actions=SocialAIActions,  # primitive actions
            action_space=SocialAIActionSpace,
            add_npc_direction=self.add_npc_direction,
            add_npc_point_direction=self.add_npc_point_direction,
            add_npc_last_prim_action=self.add_npc_last_prim_action,
            reward_diminish_factor=self.reward_diminish_factor,
        )

        self.all_npc_utterance_actions = Partner.get_list_of_possible_utterances()
        self.prim_actions_dict = SocialAINPCActionsDict

    def _gen_grid(self, width_, height_):
        # Create the grid
        self.grid = Grid(width_, height_, nb_obj_dims=self.encoding_size)

        min_w = min(9, width_)
        min_h = min(9, height_)
        self.current_width = self._rand_int(min_w, width_+1)
        self.current_height = self._rand_int(min_h, height_+1)
        # print("Room size: {}x{}".format(self.current_width, self.current_height))

        # previous
        # self.current_width = self._rand_int(6, width_+1)
        # self.current_height = self._rand_int(6, height_+1)

        # original
        # self.current_width = self._rand_int(5, width_+1)
        # self.current_height = self._rand_int(5, height_+1)

        # self.current_width = width_
        # self.current_height = height_
        # self.current_width = 8
        # self.current_height = 8
        # warnings.warn("env size fixed: {}x{}".format(self.current_width, self.current_height))

        self.wall_x = self.current_width-1
        self.wall_y = self.current_height-1

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.current_width, self.current_height)

        self.problem = self.parameters["Problem"] if self.parameters else "LeverDoor"
        self.version = self.parameters["Version"] if self.parameters else "Asocial"
        self.role = self.parameters["Role"] if self.parameters else "A"
        assert self.role in ["A", "B", "Meta"]

        if self.role in ["B", "Meta"]:
            self.agent_side = "R"  # starts on the right side
        else:
            self.agent_side = "L"  # starts on the right side

        num_of_colors = self.parameters.get("Num_of_colors", None) if self.parameters else None

        self.add_obstacles()

        # apple
        if num_of_colors is None:
            POSSIBLE_COLORS = COLOR_NAMES

        else:
            POSSIBLE_COLORS = COLOR_NAMES[:int(num_of_colors)]

        self.left_half_size = (self.current_width//2, self.current_height)
        self.left_half_top = (0, 0)

        self.right_half_size = (self.current_width//2, self.current_height)
        self.right_half_top = (self.current_width - self.current_width // 2, 0)

        # generator
        self.generator_pos = (self.current_width//2, self.current_height)
        self.generator_color = self._rand_elem(POSSIBLE_COLORS)

        self.generator_current_pos = self.find_loc(
            # on the right wall
            top=(self.current_width-1, 1),
            size=(1, self.current_height-2),
            reject_agent_pos=True,
            reject_taken_pos=False,  # so that it can be placed on the wall
        )

        # create hole in the wall for the generator
        assert type(self.grid.get(*self.generator_current_pos)) == Wall
        self.grid.set(*self.generator_current_pos, None)

        # add fence to grid
        self.grid.vert_wall(
            x=self.current_width//2,
            y=1,
            length=self.current_height - 2,
            obj_type=Fence
        )

        # door in front of generator
        self.door_current_pos = self.generator_current_pos - np.array([1, 0])
        self.door_color = self._rand_elem(POSSIBLE_COLORS)

        # lever
        self.lever_current_pos = self.find_loc(
            top=self.left_half_top, size=self.left_half_size, reject_agent_pos=True,
            reject_fn=lambda _, pos: tuple(pos) in map(tuple, [
                self.door_current_pos])
        )
        self.lever_color = self._rand_elem(POSSIBLE_COLORS)

        # generator platform

        # find the position for generator_platforms
        self.left_apple_current_pos = self.find_loc(
            top=self.left_half_top, size=self.left_half_size, reject_agent_pos=True,
            reject_fn=lambda _, pos: tuple(pos) in map(tuple, [
                self.generator_current_pos, self.door_current_pos, self.lever_current_pos])
            )

        self.right_apple_current_pos = self.find_loc(
            top=self.right_half_top, size=self.right_half_size, reject_agent_pos=True,
            reject_fn=lambda _, pos: tuple(pos) in map(tuple, [
                self.generator_current_pos, self.door_current_pos, self.lever_current_pos])
        )

        assert all(self.left_apple_current_pos < np.array([self.current_width - 1, self.current_height - 1]))
        assert all(self.right_apple_current_pos < np.array([self.current_width - 1, self.current_height - 1]))

        self.agent_generator_platform_color = self._rand_elem(POSSIBLE_COLORS)
        self.partner_generator_platform_color = self._rand_elem(POSSIBLE_COLORS)

        self.put_objects_in_env()

        # agent
        if self.agent_side == "L":
            self.place_agent(size=self.left_half_size, top=self.left_half_top)
        else:
            self.place_agent(size=self.right_half_size, top=self.right_half_top)

        if self.version == "Social":
            # NPC
            self.npc_color = self._rand_elem(COLOR_NAMES)
            self.caretaker = Partner(self.npc_color, "Partner", self)

            if self.agent_side == "L":
                self.place_obj(self.caretaker, size=self.right_half_size, top=self.right_half_top)
            else:
                self.place_obj(self.caretaker, size=self.left_half_size, top=self.left_half_top)

        # Generate the mission string
        self.mission = 'lets collaborate'

        # Dummy beginning string
        # self.beginning_string = "This is what you hear. \n"
        self.beginning_string = "Conversation: \n"
        self.utterance = self.beginning_string

        # utterance appended at the end of each step
        self.utterance_history = ""

        # used for rendering
        self.full_conversation = self.utterance
        self.outcome_info = None

    def put_objects_in_env(self, remove_objects=False):

        assert self.left_apple_current_pos is not None
        assert self.right_apple_current_pos is not None
        assert self.generator_current_pos is not None
        assert self.agent_generator_platform_color is not None
        assert self.partner_generator_platform_color is not None

        assert self.problem == self.parameters["Problem"] if self.parameters else "MarblePass"

        if remove_objects:
            self.grid.set(*self.agent_generator_platform.cur_pos, None)  # remove apple
            self.grid.set(*self.partner_generator_platform.cur_pos, None)  # remove apple
            self.grid.set(*self.generator.cur_pos, None)  # remove generator
            self.grid.set(*self.door.cur_pos, None)  # remove door
            self.grid.set(*self.lever.cur_pos, None)  # remove lever

        # apple
        self.agent_apple = Apple()
        self.partner_apple = Apple()

        def generate_apples():
            if self.agent_side == "L":
                self.grid.set(*self.left_apple_current_pos, self.agent_apple),
                self.grid.set(*self.right_apple_current_pos, self.partner_apple),
            else:
                self.grid.set(*self.left_apple_current_pos, self.partner_apple),
                self.grid.set(*self.right_apple_current_pos, self.agent_apple),

        # Generator
        self.generator = AppleGenerator(
            self.generator_color,
            on_push=generate_apples,
        )

        door_open = self.version == "Asocial"
        self.door = RemoteDoor(color=self.door_color, is_open=door_open)

        self.lever = Lever(color=self.lever_color, object=self.door, active_steps=self.lever_active_steps)

        self.agent_generator_platform = GeneratorPlatform(self.agent_generator_platform_color)
        self.partner_generator_platform = GeneratorPlatform(self.partner_generator_platform_color)


        self.put_obj_np(self.agent_generator_platform, self.left_apple_current_pos)
        self.put_obj_np(self.partner_generator_platform, self.right_apple_current_pos)

        self.put_obj_np(self.generator, self.generator_current_pos)
        self.put_obj_np(self.door, self.door_current_pos)
        self.put_obj_np(self.lever, self.lever_current_pos)



    def reset(
            self, *args, **kwargs
    ):
        # This env must be used inside the parametric env
        if not kwargs:
            # The only place when kwargs can empty is during the class construction
            # reset should be called again before using the env (paramenv does it in its constructor)
            assert self.parameters is None
            assert not self.init_done
            self.init_done = True

            obs = super().reset()
            return obs

        else:
            assert self.init_done

        self.parameters = dict(kwargs)

        assert self.parameters is not None
        assert len(self.parameters) > 0

        obs = super().reset()

        self.agent_ate_the_apple = False
        self.agent_opened_the_box = False
        self.agent_turned_on_the_switch = False
        self.agent_pressed_the_generator = False

        return obs

    def step(self, action):
        success = False
        p_action = action[0]
        utterance_action = action[1:]

        apple_had_been_eaten = self.agent_apple.eaten
        generator_had_been_pressed = self.generator.is_pressed

        # primitive actions
        _, reward, done, info = super().step(p_action)

        self.lever.step()

        # eaten just now by primitive actions of the agent
        if not self.agent_ate_the_apple:
            self.agent_ate_the_apple = self.agent_apple.eaten and not apple_had_been_eaten

        if not self.agent_pressed_the_generator:
            self.agent_pressed_the_generator = self.generator.is_pressed and not generator_had_been_pressed


        # utterances
        agent_spoke = not all(np.isnan(utterance_action))
        if agent_spoke:
            utterance = self.grammar.construct_utterance(utterance_action)

            if self.hear_yourself:
                self.utterance += "YOU: {} \n".format(utterance)
            self.full_conversation += "YOU: {} \n".format(utterance)
        else:
            utterance = None

        if self.version == "Social":
            reply, npc_info = self.caretaker.step(utterance)

            if reply:
                self.utterance += "{}: {} \n".format(self.caretaker.name, reply)
                self.full_conversation += "{}: {} \n".format(self.caretaker.name, reply)
        else:
            npc_info = {
                "prim_action": "no_op",
                "utterance": "no_op",
                "was_introduced_to": False,
            }

        # aftermath
        if p_action == self.actions.done:
            done = True

        elif self.agent_ate_the_apple:
            # check that it is the agent who ate it
            assert self.actions(p_action) == self.actions.toggle
            assert self.get_cell(*self.front_pos) == self.agent_apple

            if self.version == "Asocial" or self.role in ["A", "B"]:
                reward = self._reward()
                success = True
                done = True

            elif self.role == "Meta":

                if self.agent_side == "L":
                    reward = self._reward() / 2
                    success = True
                    done = True

                elif self.agent_side == "R":
                    reward = self._reward() / 2
                    self.agent_ate_the_apple=False
                    self.agent_side = "L"
                    self.put_objects_in_env(remove_objects=True)

                    # teleport the agent and the NPC
                    self.place_agent(size=self.left_half_size, top=self.left_half_top)

                    self.grid.set(*self.caretaker.cur_pos, None)

                    self.caretaker = Partner(self.npc_color, "Partner", self)
                    self.place_obj(self.caretaker, size=self.right_half_size, top=self.right_half_top)

                else:
                    raise ValueError(f"Side unknown - {self.agent_side}.")

            else:
                raise ValueError(f"Role unknown - {self.role}.")

        # discount
        if self.step_penalty:
            reward = reward - 0.01

        # update obs with NPC movement
        obs = self.gen_obs(full_obs=self.full_obs)

        # fill observation with text
        self.append_existing_utterance_to_history()
        obs = self.add_utterance_to_observation(obs)
        self.reset_utterance()

        if done:
            if reward > 0:
                self.outcome_info = "SUCCESS: agent got {} reward \n".format(np.round(reward, 1))
            else:
                self.outcome_info = "FAILURE: agent got {} reward \n".format(reward)

        if self.version == "Social":
            # is the npc seen by the agent
            ag_view_npc = self.relative_coords(*self.caretaker.cur_pos)

            if ag_view_npc is not None:
                # in the agent's field of view
                ag_view_npc_x, ag_view_npc_y = ag_view_npc

                n_dims = obs['image'].shape[-1]
                npc_encoding = self.caretaker.encode(n_dims)

                # is it occluded
                npc_observed = all(obs['image'][ag_view_npc_x, ag_view_npc_y] == npc_encoding)
            else:
                npc_observed = False
        else:
            npc_observed = False

        info = {**info, **{"NPC_"+k: v for k, v in npc_info.items()}}

        info["NPC_observed"] = npc_observed
        info["success"] = success

        return obs, reward, done, info

    def _reward(self):
        if self.diminished_reward:
            return super()._reward()
        else:
            return 1.0

    # def render(self, *args, **kwargs):
    #     obs = super().render(*args, **kwargs)
    #     self.window.clear_text()  # erase previous text
    #     self.window.set_caption(self.full_conversation)
    #
    #     # self.window.ax.set_title("correct color: {}".format(self.box.target_color), loc="left", fontsize=10)
    #
    #     if self.outcome_info:
    #         color = None
    #         if "SUCCESS" in self.outcome_info:
    #             color = "lime"
    #         elif "FAILURE" in self.outcome_info:
    #             color = "red"
    #         self.window.add_text(*(0.01, 0.85, self.outcome_info),
    #                              **{'fontsize':15, 'color':color, 'weight':"bold"})
    #
    #     self.window.show_img(obs)  # re-draw image to add changes to window
    #     return obs


register(
    id='SocialAI-LeverDoorEnv-v1',
    entry_point='gym_minigrid.social_ai_envs:LeverDoorEnv'
)
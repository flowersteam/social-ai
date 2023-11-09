import time

import numpy as np
from gym_minigrid.social_ai_envs.socialaigrammar import SocialAIGrammar, SocialAIActions, SocialAIActionSpace
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
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
        # todo: this should be id == name
        self.npc_type = 0  # this will be put into the encoding

        self.npc_side = "L" if self.env.agent_side == "R" else "R"
        assert {self.npc_side, self.env.agent_side} == {"L", "R"}

        self.target_obj = None

        self.was_introduced_to = False

        self.ate_an_apple = False
        self.demo_over = False
        self.demo_over_and_position_safe = False
        self.apple_unlocked_for_agent = False

        self.list_of_possible_utterances = [
            *self.list_of_possible_utterances,
            "Hot",  # change to hot -> all with small letters
            "Warm",
            "Medium",
            "Cold",
            *COLOR_NAMES
        ]

        assert self.env.grammar.contains_utterance(self.introduction_statement)

    def step(self, utterance):

        reply, info = super().step()

        if self.env.hidden_npc:
            return reply, info

        if self.npc_side == "L":
            # the npc waits for the agent to open one of the right boxes, and then uses the object of the same color
            action = None
            if self.env.chosen_left_obj is not None:
                self.target_obj = self.env.chosen_left_obj

                if type(self.target_obj) == Switch and self.target_obj.is_on:
                    next_target_position = self.env.box.cur_pos

                elif type(self.target_obj) == AppleGenerator and self.target_obj.is_pressed:
                    next_target_position = self.env.left_generator_platform.cur_pos

                else:
                    next_target_position = self.target_obj.cur_pos

                if type(self.target_obj) == AppleGenerator and not self.target_obj.is_pressed:
                    # we have to activate the generator
                    if not self.env.generator.marble_activation:
                        # push generator
                        action = self.path_to_pos(next_target_position)
                    else:
                        # find angle
                        if self.env.marble.moving_dir is None:
                            distance = (self.env.marble.cur_pos - self.target_obj.cur_pos)

                            diff = np.sign(distance)
                            if sum(abs(diff)) == 1:
                                push_pos = self.env.marble.cur_pos + diff
                                if all(self.cur_pos == push_pos):
                                    next_target_position = self.env.marble.cur_pos
                                else:
                                    next_target_position = push_pos

                                # go to loc in front of
                                # push
                                action = self.path_to_pos(next_target_position)

                        else:
                            action = None

                else:
                    # toggle all other objects
                    action = self.path_to_toggle_pos(next_target_position)
            else:
                action = self.turn_to_see_agent()

        else:
            if self.ate_an_apple:
                action = self.turn_to_see_agent()
            else:
                # toggle the chosen box then the apple
                if self.target_obj is None:
                    self.target_obj = self.env._rand_elem([
                        self.env.right_box1,
                        self.env.right_box2
                    ])

                action = self.path_to_toggle_pos(self.target_obj.cur_pos)

        if self.npc_side == "R":
            eaten_before = self.env.right_apple.eaten
        else:
            eaten_before = self.env.left_apple.eaten

        if action is not None:
            action()

        if not self.ate_an_apple:
            # check if the NPC ate the apple
            if self.npc_side == "R":
                self.ate_an_apple = not eaten_before and self.env.right_apple.eaten
            else:
                self.ate_an_apple = not eaten_before and self.env.left_apple.eaten

        info = {
            "prim_action": action.__name__ if action is not None else "no_op",
            "utterance": "no_op",
            "was_introduced_to": self.was_introduced_to
        }

        reply = None

        return reply, info

    def is_point_from_loc(self, pos):
        target_pos = self.target_obj.cur_pos
        if self.distractor_obj is not None:
            distractor_pos = self.distractor_obj.cur_pos
        else:
            distractor_pos = [None, None]

        if self.env.is_in_marble_way(pos):
            return False

        if any(pos == target_pos):
            same_ind = np.argmax(target_pos == pos)

            if pos[same_ind] != distractor_pos[same_ind]:
                return True

            if pos[same_ind] == distractor_pos[same_ind]:
                # if in between
                if distractor_pos[1-same_ind] < pos[1-same_ind] < target_pos[1-same_ind]:
                    return True

                if distractor_pos[1-same_ind] > pos[1-same_ind] > target_pos[1-same_ind]:
                    return True

        return False

    def find_point_from_loc(self):
        reject_fn = lambda env, p: not self.is_point_from_loc(p)

        point = self.env.find_loc(size=(self.env.wall_x, self.env.wall_y), reject_fn=reject_fn, reject_agent_pos=False)

        assert all(point < np.array([self.env.wall_x, self.env.wall_y]))
        assert all(point > np.array([0, 0]))

        return point


class ObjectsCollaborationEnv(MultiModalMiniGridEnv):
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
        switch_no_light=True,
        reward_diminish_factor=0.1,
        see_through_walls=False,
        egocentric_observation=True,
    ):
        assert size >= 5
        self.empty_symbol = "NA \n"
        self.diminished_reward = diminished_reward
        self.step_penalty = step_penalty
        self.knowledgeable = knowledgeable
        self.hidden_npc = hidden_npc
        self.hear_yourself = False
        self.switch_no_light = switch_no_light

        self.grammar = SocialAIGrammar()

        self.init_done = False
        # parameters - to be set in reset
        self.parameters = None

        # encoding size should be 5
        self.add_npc_direction = True
        self.add_npc_point_direction = True
        self.add_npc_last_prim_action = True

        self.reward_diminish_factor = reward_diminish_factor

        self.egocentric_observation = egocentric_observation
        self.encoding_size = 3 + 2*bool(not self.egocentric_observation) + bool(self.add_npc_direction) + bool(self.add_npc_point_direction) + bool(self.add_npc_last_prim_action)

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            # Set this to True for maximum speed
            see_through_walls=see_through_walls,
            actions=SocialAIActions,  # primitive actions
            action_space=SocialAIActionSpace,
            add_npc_direction=self.add_npc_direction,
            add_npc_point_direction=self.add_npc_point_direction,
            add_npc_last_prim_action=self.add_npc_last_prim_action,
            reward_diminish_factor=self.reward_diminish_factor,
        )
        self.all_npc_utterance_actions = Partner.get_list_of_possible_utterances()
        self.prim_actions_dict = SocialAINPCActionsDict

    def revert(self):
        self.put_objects_in_env(remove_objects=True)

    def is_in_marble_way(self, pos):
        target_pos = self.generator_current_pos

        # generator distractor is in the same row / collumn as the marble and the generator
        # if self.distractor_current_pos is not None:
        #     distractor_pos = self.distractor_current_pos
        # else:
        #     distractor_pos = [None, None]

        if self.problem in ["Marble"]:
            # point can't be in the same row or column as both the marble and the generator
            # all three: marble, generator, loc are in the same row or column
            if any((pos == target_pos) * (pos == self.marble_current_pos)):
                # all three: marble, generator, loc are in the same row or column -> is in its way
                return True

            # is it in the way for the distractor generator
            if any((pos == self.distractor_current_pos) * (pos == self.marble_current_pos)):
                # all three: marble, distractor generator, loc are in the same row or column -> is in its way
                return True

        # all good
        return False

    def _gen_grid(self, width_, height_):
        # Create the grid
        self.grid = Grid(width_, height_, nb_obj_dims=self.encoding_size)

        # new
        min_w = min(9, width_)
        min_h = min(9, height_)
        self.current_width = self._rand_int(min_w, width_+1)
        self.current_height = self._rand_int(min_h, height_+1)

        self.wall_x = self.current_width-1
        self.wall_y = self.current_height-1

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.current_width, self.current_height)

        # problem: Apples/Boxes/Switches/Generators/Marbles
        self.problem = self.parameters["Problem"] if self.parameters else "Apples"
        num_of_colors = self.parameters.get("Num_of_colors", None) if self.parameters else None
        self.version = self.parameters["Version"] if self.parameters else "Asocial"
        self.role = self.parameters["Role"] if self.parameters else "A"
        assert self.role in ["A", "B", "Meta"]

        if self.role in ["B", "Meta"]:
            self.agent_side = "R"  # starts on the right side
        else:
            self.agent_side = "L"  # starts on the right side

        self.add_obstacles()

        # apple

        # box
        locked = self.problem == "Switches"

        if num_of_colors is None:
            POSSIBLE_COLORS = COLOR_NAMES.copy()

        else:
            POSSIBLE_COLORS = COLOR_NAMES[:int(num_of_colors)].copy()

        self.left_half_size = (self.current_width//2, self.current_height)
        self.left_half_top = (0, 0)

        self.right_half_size = (self.current_width//2 - 1, self.current_height)
        self.right_half_top = (self.current_width - self.current_width // 2 + 1, 0)

        # add fence to grid
        self.grid.vert_wall(
            x=self.current_width//2 + 1,  # one collumn to the right of the center
            y=1,
            length=self.current_height - 2,
            obj_type=Fence
        )

        self.right_box1_color = self._rand_elem(POSSIBLE_COLORS)
        POSSIBLE_COLORS.remove(self.right_box1_color)

        self.right_box2_color = self._rand_elem(POSSIBLE_COLORS)

        assert self.right_box1_color != self.right_box2_color

        POSSIBLE_COLORS_LEFT = [self.right_box1_color, self.right_box2_color]

        self.left_color_1 = self._rand_elem(POSSIBLE_COLORS_LEFT)
        POSSIBLE_COLORS_LEFT.remove(self.left_color_1)
        self.left_color_2 = self._rand_elem(POSSIBLE_COLORS_LEFT)


        self.box_color = self.left_color_1
        # find the position for the apple/box/generator_platform
        self.left_apple_current_pos = self.find_loc(
            size=self.left_half_size,
            top=self.left_half_top,
            reject_agent_pos=True
        )

        # right boxes
        self.right_box1_current_pos = self.find_loc(
            size=self.right_half_size,
            top=self.right_half_top,
            reject_agent_pos=True
        )
        self.right_box2_current_pos = self.find_loc(
            size=self.right_half_size,
            top=self.right_half_top,
            reject_agent_pos=True,
            reject_fn=lambda _, pos: tuple(pos) in map(tuple, [self.right_box1_current_pos]),
        )
        assert all(self.left_apple_current_pos < np.array([self.current_width - 1, self.current_height - 1]))

        # switch
        # self.switch_pos = (self.current_width, self.current_height)
        self.switch_color = self.left_color_1
        self.switch_current_pos = self.find_loc(
            top=self.left_half_top,
            size=self.left_half_size,
            reject_agent_pos=True,
            reject_fn=lambda _, pos: tuple(pos) in map(tuple, [self.left_apple_current_pos]),
        )

        # generator
        # self.generator_pos = (self.current_width, self.current_height)
        self.generator_color = self.left_color_1
        self.generator_current_pos = self.find_loc(
            top=self.left_half_top,
            size=self.left_half_size,
            reject_agent_pos=True,
            reject_fn=lambda _, pos: (
                tuple(pos) in map(tuple, [self.left_apple_current_pos])
                or
                (self.problem in ["Marbles", "Marble"] and tuple(pos) in [
                    # not in corners
                    (1, 1),
                    (self.current_width-2, 1),
                    (1, self.current_height-2),
                    (self.current_width-2, self.current_height-2),
                ])
                or
                # not in the same row collumn as the platform
                (self.problem in ["Marbles", "Marble"] and any(pos == self.left_apple_current_pos))
            ),
        )

        # generator platform
        self.left_generator_platform_color = self._rand_elem(POSSIBLE_COLORS)

        # marbles
        # self.marble_pos = (self.current_width, self.current_height)
        self.marble_color = self._rand_elem(POSSIBLE_COLORS)
        self.marble_current_pos = self.find_loc(
            top=self.left_half_top,
            size=self.left_half_size,
            reject_agent_pos=True,
            reject_fn=lambda _, pos: self.problem in ["Marbles", "Marble"] and (
                tuple(pos) in map(tuple, [self.left_apple_current_pos, self.generator_current_pos])
                or
                all(pos != self.generator_current_pos)  # reject if not in row or column as the generator
                or
                any(pos == 1)  # next to a wall
                or
                pos[1] == self.current_height-2
                or
                pos[0] == self.current_width-2
            ),
        )

        self.distractor_color = self.left_color_2
        # self.distractor_pos = (self.current_width, self.current_height)

        if self.problem in ["Apples", "Boxes"]:
            distractor_reject_fn = lambda _, pos: tuple(pos) in map(tuple, [self.left_apple_current_pos])

        elif self.problem in ["Switches"]:
            distractor_reject_fn = lambda _, pos: tuple(pos) in map(tuple, [self.left_apple_current_pos, self.switch_current_pos])

        elif self.problem in ["Generators"]:
            distractor_reject_fn = lambda _, pos: tuple(pos) in map(tuple, [self.left_apple_current_pos, self.generator_current_pos])

        elif self.problem in ["Marbles", "Marble"]:
            # problem is marbles
            same_dim = (self.generator_current_pos == self.marble_current_pos).argmax()
            distactor_same_dim = 1-same_dim
            distractor_reject_fn = lambda _, pos: tuple(pos) in map(tuple, [
                self.left_apple_current_pos,
                self.generator_current_pos,
                self.marble_current_pos
            ]) or pos[distactor_same_dim] != self.marble_current_pos[distactor_same_dim]
            # todo: not in corners -> but it's not that important
            # or tuple(pos) in [
            #     # not in corners
            #     (1, 1),
            #     (self.current_width-2, 1),
            #     (1, self.current_height-2),
            #     (self.current_width-2, self.current_height-2),
            # ])

        else:
            raise ValueError("Problem {} indefined.".format(self.problem))

        self.distractor_current_pos = self.find_loc(
            top=self.left_half_top,
            size=self.left_half_size,
            reject_agent_pos=True,
            # todo: reject based on problem
            reject_fn=distractor_reject_fn
        )

        self.put_objects_in_env()

        # place agent
        if self.agent_side == "L":
            self.place_agent(size=self.left_half_size, top=self.left_half_top)
        else:
            self.place_agent(size=self.right_half_size, top=self.right_half_top)

        # NPC
        if self.version == "Social":
            self.npc_color = self._rand_elem(COLOR_NAMES)
            self.caretaker = Partner(self.npc_color, "Partner", self)

            if self.agent_side == "L":
                self.place_obj(self.caretaker, size=self.right_half_size, top=self.right_half_top, reject_fn=ObjectsCollaborationEnv.is_in_marble_way)
            else:
                self.place_obj(self.caretaker, size=self.left_half_size, top=self.left_half_top, reject_fn=ObjectsCollaborationEnv.is_in_marble_way)

        # Generate the mission string
        self.mission = 'lets collaborate'

        # Dummy beginning string
        # self.beginning_string = "This is what you hear. \n"
        self.beginning_string = "Conversation: \n" # todo: go back to "this what you hear?
        self.utterance = self.beginning_string

        # utterance appended at the end of each step
        self.utterance_history = ""

        # used for rendering
        self.full_conversation = self.utterance
        self.outcome_info = None

    def put_objects_in_env(self, remove_objects=False):

        assert self.left_apple_current_pos is not None
        assert self.right_box1_current_pos is not None
        assert self.right_box2_current_pos is not None
        assert self.switch_current_pos is not None

        self.switches_block_set = []
        self.boxes_block_set = []
        self.right_boxes_block_set = []
        self.generators_block_set = []

        self.other_box = None
        self.other_switch = None
        self.other_generator = None

        # problem: Apples/Boxes/Switches/Generators
        assert self.problem == self.parameters["Problem"] if self.parameters else "Apples"

        # move objects (used only in revert), not in gen_grid
        if remove_objects:
            # remove apple or box
            # assert type(self.grid.get(*self.apple_current_pos)) in [Apple, LockableBox]
            # self.grid.set(*self.apple_current_pos, None)

            # remove apple (after demo it must be an apple)
            assert type(self.grid.get(*self.left_apple_current_pos)) in [Apple]
            self.grid.set(*self.left_apple_current_pos, None)

            self.grid.set(*self.right_apple_current_pos, None)

            if self.problem in ["Switches"]:
                # remove switch
                assert type(self.grid.get(*self.switch_current_pos)) in [Switch]
                self.grid.set(*self.switch.cur_pos, None)

            elif self.problem in ["Generators", "Marbles", "Marble"]:
                # remove generator
                assert type(self.grid.get(*self.generator.cur_pos)) in [AppleGenerator]
                self.grid.set(*self.generator.cur_pos, None)

                if self.problem in ["Marbles", "Marble"]:
                    # remove generator
                    assert type(self.grid.get(*self.marble.cur_pos)) in [Marble]
                    self.grid.set(*self.marble.cur_pos, None)

                    if self.marble.tee_uncovered:
                        self.grid.set(*self.marble.tee.cur_pos, None)

            elif self.problem in ["Apples", "Boxes"]:
                pass

            else:
                raise ValueError("Undefined problem {}".format(self.problem))

            # remove distractor
            if self.problem in ["Boxes", "Switches", "Generators", "Marbles", "Marble"]:
                assert type(self.grid.get(*self.distractor_current_pos)) in [LockableBox, Switch, AppleGenerator]
                self.grid.set(*self.distractor_current_pos, None)

        # apple
        self.left_apple = Apple()
        self.right_apple = Apple()

        # right apple
        self.right_box1 = LockableBox(
            self.right_box1_color,
            contains=self.right_apple,
            is_locked=False,
            block_set=self.right_boxes_block_set
        )
        self.right_boxes_block_set.append(self.right_box1)

        # right apple
        self.right_box2 = LockableBox(
            self.right_box2_color,
            contains=self.right_apple,
            is_locked=False,
            block_set=self.right_boxes_block_set
        )
        self.right_boxes_block_set.append(self.right_box2)

        # Box
        locked = self.problem == "Switches"

        self.box = LockableBox(
            self.box_color,
            # contains=self.left_apple,
            is_locked=locked,
            block_set=self.boxes_block_set
        )
        self.boxes_block_set.append(self.box)

        # Switch
        self.switch = Switch(
            color=self.switch_color,
            # lockable_object=self.box,
            locker_switch=True,
            no_turn_off=True,
            no_light=self.switch_no_light,
            block_set=self.switches_block_set,
        )

        self.switches_block_set.append(self.switch)

        # Generator
        self.generator = AppleGenerator(
            self.generator_color,
            block_set=self.generators_block_set,
            # on_push=lambda: self.grid.set(*self.left_apple_current_pos, self.left_apple),
            marble_activation=self.problem in ["Marble"],
        )
        self.generators_block_set.append(self.generator)

        self.left_generator_platform = GeneratorPlatform(self.left_generator_platform_color)

        self.marble = Marble(self.marble_color, env=self)

        # right side
        self.put_obj_np(self.right_box1, self.right_box1_current_pos)
        self.put_obj_np(self.right_box2, self.right_box2_current_pos)

        self.candidate_objects=[]
        # left side
        if self.problem == "Apples":
            self.put_obj_np(self.left_apple, self.left_apple_current_pos)
            self.candidate_objects.append(self.left_apple)

        elif self.problem in ["Boxes"]:
            self.put_obj_np(self.box, self.left_apple_current_pos)
            self.candidate_objects.append(self.box)

        elif self.problem in ["Switches"]:
            self.put_obj_np(self.box, self.left_apple_current_pos)
            self.put_obj_np(self.switch, self.switch_current_pos)
            self.candidate_objects.append(self.switch)

        elif self.problem in ["Generators", "Marble"]:
            self.put_obj_np(self.generator, self.generator_current_pos)
            self.put_obj_np(self.left_generator_platform, self.left_apple_current_pos)
            self.candidate_objects.append(self.generator)

            if self.problem in ["Marble"]:
                self.put_obj_np(self.marble, self.marble_current_pos)

        else:
            raise ValueError("Problem {} not defined. ".format(self.problem))

        # Distractors
        if self.problem == "Boxes":
            assert not locked

            self.other_box = LockableBox(
                self.left_color_2,
                is_locked=locked,
                block_set=self.boxes_block_set,
            )
            self.boxes_block_set.append(self.other_box)

            self.put_obj_np(self.other_box, self.distractor_current_pos)
            self.candidate_objects.append(self.other_box)

        elif self.problem == "Switches":
            self.other_switch = Switch(
                color=self.left_color_2,
                locker_switch=True,
                no_turn_off=True,
                no_light=self.switch_no_light,
                block_set=self.switches_block_set,
            )
            self.switches_block_set.append(self.other_switch)

            self.put_obj_np(self.other_switch, self.distractor_current_pos)
            self.candidate_objects.append(self.other_switch)

        elif self.problem in ["Generators", "Marble"]:
            self.other_generator = AppleGenerator(
                color=self.left_color_2,
                block_set=self.generators_block_set,
                marble_activation=self.problem in ["Marble"],
            )
            self.generators_block_set.append(self.other_generator)

            self.put_obj_np(self.other_generator, self.distractor_current_pos)
            self.candidate_objects.append(self.other_generator)

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

        self.agent_ate_an_apple = False
        self.chosen_right_box = None
        self.chosen_left_obj = None

        return obs

    def step(self, action):
        success = False
        p_action = action[0]
        utterance_action = action[1:]

        left_apple_had_been_eaten = self.left_apple.eaten
        right_apple_had_been_eaten = self.right_apple.eaten

        # primitive actions
        _, reward, done, info = super().step(p_action)

        if self.problem in ["Marbles", "Marble"]:
            # todo: create objects which can stepped automatically?
            self.marble.step()

        if not self.agent_ate_an_apple:
            if self.agent_side == "L":
                self.agent_ate_an_apple = self.left_apple.eaten and not left_apple_had_been_eaten
            else:
                self.agent_ate_an_apple = self.right_apple.eaten and not right_apple_had_been_eaten

        if self.right_box1.is_open:
            self.chosen_right_box = self.right_box1

        if self.right_box2.is_open:
            self.chosen_right_box = self.right_box2

        if self.chosen_right_box is not None:
            chosen_color = self.chosen_right_box.color
            self.chosen_left_obj = [o for o in self.candidate_objects if o.color == chosen_color][0]

            if type(self.chosen_left_obj) == LockableBox:
                self.chosen_left_obj.contains = self.left_apple

            elif type(self.chosen_left_obj) == Switch:
                self.chosen_left_obj.lockable_object = self.box
                self.box.contains = self.left_apple

            elif type(self.chosen_left_obj) == AppleGenerator:
                self.chosen_left_obj.on_push=lambda: self.grid.set(*self.left_apple_current_pos, self.left_apple)

            else:
                raise ValueError("Unknown target object.")

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

        if (self.role in ["A", "B"] or self.version == "Asocial") and self.agent_ate_an_apple:
                reward = self._reward()
                success = True
                done = True

        elif self.role == "Meta" and self.version == "Social" and self.agent_ate_an_apple and self.caretaker.ate_an_apple:

            if self.agent_side == "L":
                reward = self._reward() / 2
                success = True
                done = True

            else:
                # revert and rotate
                reward = self._reward() / 2
                self.agent_ate_an_apple = False
                self.caretaker.ate_an_apple = False
                self.agent_side = "L"
                self.put_objects_in_env(remove_objects=True)

                # teleport the agent and the NPC
                self.place_agent(size=self.left_half_size, top=self.left_half_top)

                self.grid.set(*self.caretaker.cur_pos, None)

                self.caretaker = Partner(self.npc_color, "Partner", self)
                self.place_obj(self.caretaker, size=self.right_half_size, top=self.right_half_top, reject_fn=ObjectsCollaborationEnv.is_in_marble_way)

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
    #                              **{'fontsize': 15, 'color': color, 'weight': "bold"})
    #
    #     self.window.show_img(obs)  # re-draw image to add changes to window
    #     return obs

register(
    id='SocialAI-ObjectsCollaboration-v0',
    entry_point='gym_minigrid.social_ai_envs:ObjectsCollaborationEnv'
)
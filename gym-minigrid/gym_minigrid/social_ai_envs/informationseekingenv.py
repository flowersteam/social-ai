import time
import random

import numpy as np
from gym_minigrid.social_ai_envs.socialaigrammar import SocialAIGrammar, SocialAIActions, SocialAIActionSpace
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import time
from collections import deque

def next_to(posa, posb):
    if type(posa) == tuple:
        posa = np.array(posa)

    if type(posb) == tuple:
        posb = np.array(posb)

    return abs(posa-posb).sum() == 1


class Caretaker(NPC):
    """
    A simple NPC that knows who is telling the truth
    """
    def __init__(self, color, name, env):
        super().__init__(color)
        self.name = name
        self.env = env
        self.npc_dir = 1  # NPC initially looks downward
        self.npc_type = 0  # this will be put into the encoding

        self.was_introduced_to = False
        self.decoy_color_given = False

        self.ate_an_apple = False
        self.demo_over = False
        self.demo_over_and_position_safe = False
        self.apple_unlocked_for_agent = False

        self.list_of_possible_utterances = [
            *self.list_of_possible_utterances,
            "Hot",
            "Warm",
            "Medium",
            "Cold",
            *COLOR_NAMES
        ]

        # target obj
        assert self.env.problem == self.env.parameters["Problem"] if self.env.parameters else "Apples"

        if self.env.problem in ["Apples"]:
            self.target_obj = self.env.apple
            self.distractor_obj = None

        elif self.env.problem == "Doors":
            self.target_obj = self.env.door
            self.distractor_obj = self.env.distractor_door

        elif self.env.problem == "Levers":
            self.target_obj = self.env.lever
            self.distractor_obj = self.env.distractor_lever

        elif self.env.problem == "Boxes":
            self.target_obj = self.env.box
            self.distractor_obj = self.env.distractor_box

        elif self.env.problem == "Switches":
            self.target_obj = self.env.switch
            self.distractor_obj = self.env.distractor_switch

        elif self.env.problem == "Generators":
            self.target_obj = self.env.generator
            self.distractor_obj = self.env.distractor_generator

        elif self.env.problem in ["Marble", "Marbles"]:
            self.target_obj = self.env.generator
            self.distractor_obj = self.env.distractor_generator

        if self.env.ja_recursive:
            if int(self.env.parameters["N"]) == 1:
                self.ja_decoy = self.env._rand_elem([self.target_obj])
            else:
                self.ja_decoy = self.env._rand_elem([self.target_obj, self.distractor_obj])

            # the other object is a decoy distractor
            self.ja_decoy_distractor = list({self.target_obj, self.distractor_obj} - {self.ja_decoy})[0]

            self.decoy_point_from_loc = self.find_point_from_loc(
                target_pos=self.ja_decoy.cur_pos,
                distractor_pos=self.ja_decoy_distractor.cur_pos if self.ja_decoy_distractor else None
            )

        self.point_from_loc = self.find_point_from_loc()

        assert self.env.grammar.contains_utterance(self.introduction_statement)

    def step(self, utterance):
        reply, info = super().step()

        if self.env.hidden_npc:
            return reply, info

        scaffolding = self.env.parameters.get("Scaffolding", "N") == "Y"
        language_color = False
        language_feedback = False
        pointing = False
        emulation = False

        if not scaffolding:
            cue_type = self.env.parameters["Cue_type"]

            if cue_type == "Language_Color":
                language_color = True
            elif cue_type == "Language_Feedback":
                language_feedback = True
            elif cue_type == "Pointing":
                pointing = True
            elif cue_type == "Emulation":
                emulation = True
            else:
                raise ValueError(f"Cue_type ({cue_type}) not defined.")
        else:
            # there are no cues if scaffolding is used (the peer gives the apples to the agent)
            assert "Cue_type" not in self.env.parameters

            # there is no additional test for joint attention (no cues are given so this wouldn't make sense)
            assert not self.env.ja_recursive

        reply, action = None, None
        if not self.was_introduced_to:
            # check introduction, updates was_introduced_to if needed
            reply, action = self.handle_introduction(utterance)

            assert action is None

            if self.env.ja_recursive:
                # look at the center of the room (this makes the cue giving in side and outisde JA different)
                action = self.look_at_action([self.env.current_width // 2, self.env.current_height // 2])
            else:
                # look at the agent
                action = self.look_at_action(self.env.agent_pos)

            if self.was_introduced_to:
                # was introduced just now
                if self.is_pointing():
                    action = self.stop_point

                if language_color:
                    # only say the color once
                    reply = self.target_obj.color

            elif self.env.ja_recursive:
                # was not introduced
                if language_feedback:
                    # random reply
                    reply = self.env._rand_elem([
                        "Hot",
                        "Warm",
                        "Medium",
                        "Cold"
                    ])

                if language_color and not self.decoy_color_given:
                    # color of a decoy (can be the correct one)
                    reply = self.ja_decoy.color
                    self.decoy_color_given=True

                if pointing:
                    # point to a decoy
                    action = self.goto_point_action(
                        point_from_loc=self.decoy_point_from_loc,
                        target_pos=self.ja_decoy.cur_pos,
                        distractor_pos=self.ja_decoy_distractor.cur_pos if self.ja_decoy_distractor else None
                    )

                    if self.is_pointing():
                        # if it's already pointing, turn to look at the center (to avoid looking at the wall)
                        action = self.look_at_action([self.env.current_width//2, self.env.current_height//2])


        else:

            if self.was_introduced_to and language_color:
                # language only once at introduction
                # reply = self.target_obj.color
                action = self.look_at_action(self.env.agent_pos)

            if self.was_introduced_to and language_feedback:
                # closeness string
                agent_distance_to_target = np.abs(self.target_obj.cur_pos - self.env.agent_pos).sum()
                if agent_distance_to_target <= 1:
                    reply = "Hot"
                elif agent_distance_to_target <= 2:
                    reply = "Warm"
                elif agent_distance_to_target <= 5:
                    reply = "Medium"
                elif agent_distance_to_target >= 5:
                    reply = "Cold"

                action = self.look_at_action(self.env.agent_pos)

            # pointing
            if self.was_introduced_to and pointing:
                if self.env.parameters["N"] == "1":
                    distractor_pos = None
                else:
                    distractor_pos = self.distractor_obj.cur_pos

                action = self.goto_point_action(
                    point_from_loc=self.point_from_loc,
                    target_pos=self.target_obj.cur_pos,
                    distractor_pos=distractor_pos,
                )

                if self.is_pointing():
                    action = self.look_at_action(self.env.agent_pos)

            # emulation or scaffolding
            emulation_demo = self.was_introduced_to and emulation and not self.demo_over
            scaffolding_help = self.was_introduced_to and scaffolding

            # do the demonstration / unlock the apple
            # in both of those two scenarios the NPC in essence solves the task
            # in demonstration - it eats the apple, and reverts the env at the end
            # in scaffolding - it doesn't eat the apple and looks at the agent
            if emulation_demo or scaffolding_help:

                if emulation_demo or (scaffolding_help and not self.apple_unlocked_for_agent):

                    if self.is_pointing():
                        # don't point during demonstration
                        action = self.stop_point

                    else:
                        # if apple unlocked go pick it up
                        if self.target_obj == self.env.switch and self.env.switch.is_on:
                            assert self.env.parameters["Problem"] == "Switches"
                            next_target_position = self.env.box.cur_pos

                        elif self.target_obj == self.env.generator and self.env.generator.is_pressed:
                            assert self.env.parameters["Problem"] in ["Generators", "Marbles", "Marble"]
                            next_target_position = self.env.generator_platform.cur_pos

                        elif self.target_obj == self.env.door and self.env.door.is_open:
                            next_target_position = self.env.apple.cur_pos

                        elif self.target_obj == self.env.lever and self.env.lever.is_on:
                            next_target_position = self.env.apple.cur_pos

                        else:
                            next_target_position = self.target_obj.cur_pos

                        if self.target_obj == self.env.generator and not self.env.generator.is_pressed:
                            if not self.env.generator.marble_activation:
                                # push generator
                                action = self.path_to_pos(next_target_position)
                            else:
                                # find angle
                                if self.env.marble.moving_dir is None:
                                    distance = (self.env.marble.cur_pos - self.env.generator.cur_pos)
                                    diff = np.sign(distance)

                                    if sum(abs(diff)) == 1:
                                        # if the agent pushed the ball during demo diff can be > 1, then it's unsolvable
                                        push_pos = self.env.marble.cur_pos+diff
                                        if all(self.cur_pos == push_pos):
                                            next_target_position = self.env.marble.cur_pos
                                        else:
                                            next_target_position = push_pos

                                        # go to loc in front of
                                        # push
                                        action = self.path_to_pos(next_target_position)

                        else:
                            # toggle all other objects
                            action = self.path_to_toggle_pos(next_target_position)

                        # for scaffolding check if trying to eat the apple
                        # if so, stop - apple is unlocked
                        if scaffolding_help:
                            if (
                                    self.env.get_cell(*self.front_pos) == self.env.apple and
                                    action == self.toggle_action
                            ):
                                # don't eat the apple
                                action = None
                                self.apple_unlocked_for_agent = True

                        # for emulation check if trying to toggle the eaten apple
                        # if so, stop and revert the env - demo is over
                        if emulation_demo:
                            if (
                                self.ate_an_apple and
                                self.env.get_cell(*self.front_pos) == self.env.apple and
                                action == self.toggle_action and
                                self.env.apple.eaten
                            ):
                                # trying to toggle an apple it ate
                                self.env.revert()
                                self.demo_over = True
                                action = None

                # if scaffolding apple unlocked, look at the agent
                if scaffolding_help and self.apple_unlocked_for_agent:
                    if all(self.cur_pos == self.initial_pos):
                        # if the apple is unlocked look at the agent
                        wanted_dir = self.compute_wanted_dir(self.env.agent_pos)
                        action = self.compute_turn_action(wanted_dir)
                    else:
                        # go to init pos, this removes problems in case the apple is unreachable now
                        action = self.path_to_pos(self.initial_pos)

            if self.was_introduced_to and emulation and self.demo_over and not self.demo_over_and_position_safe:
                if self.env.is_in_marble_way(self.cur_pos):
                    action = self.path_to_pos(self.find_point_from_loc())
                else:
                    self.demo_over_and_position_safe = True

            if self.demo_over_and_position_safe:
                assert emulation or scaffolding
                # look at the agent after demo is done
                action = self.look_at_action(self.env.agent_pos)

            if self.was_introduced_to and self.env.parameters["Scaffolding"] == "Y":
                if "Emulation" in self.env.parameters or "Pointing" in self.env.parameters or "Language_grounding" in self.env.parameters:
                    raise ValueError(
                        "Scaffolding cannot be used with information giving (Emulation, Pointing, Language_grounding)"
                    )

        eaten_before = self.env.apple.eaten

        if action is not None:
            action()

        # check if the NPC ate the apple
        eaten_after = self.env.apple.eaten
        self.ate_an_apple = not eaten_before and eaten_after

        info = self.create_info(
            action=action,
            utterance=reply,
            was_introduced_to=self.was_introduced_to,
        )

        assert (reply or "no_op") in self.list_of_possible_utterances

        return reply, info

    def create_info(self, action, utterance, was_introduced_to):
        info = {
            "prim_action": action.__name__ if action is not None else "no_op",
            "utterance": utterance or "no_op",
            "was_introduced_to": was_introduced_to
        }
        return info

    def is_point_from_loc(self, pos, target_pos=None, distractor_pos=None):

        if target_pos is None:
            target_pos = self.target_obj.cur_pos

        if distractor_pos is None:
            if self.distractor_obj is not None:
                distractor_pos = self.distractor_obj.cur_pos
            else:
                distractor_pos = [None, None]

        if self.env.is_in_marble_way(pos):
            return False

        if self.env.problem in ["Doors", "Levers"]:
            # must not be in front of a door
            if abs(self.env.door_current_pos - pos).sum() == 1:
                return False

            if self.env.problem in ["Doors"]:
                if abs(self.env.distractor_current_pos - pos).sum() == 1:
                    return False

        if any(pos == target_pos):
            same_ind = np.argmax(target_pos == pos)

            #  is there an occlusion in the way
            start = pos[1-same_ind]
            end = target_pos[1-same_ind]
            step = 1 if start <= end else -1
            for i in np.arange(start, end, step):
                p = pos.copy()
                p[1-same_ind] = i
                cell = self.env.grid.get(*p)

                if cell is not None:
                    if not cell.see_behind():
                        return False

            if pos[same_ind] != distractor_pos[same_ind]:
                return True

            if pos[same_ind] == distractor_pos[same_ind]:
                # if in between
                if distractor_pos[1-same_ind] < pos[1-same_ind] < target_pos[1-same_ind]:
                    return True

                if distractor_pos[1-same_ind] > pos[1-same_ind] > target_pos[1-same_ind]:
                    return True
        return False

    def find_point_from_loc(self, target_pos=None, distractor_pos=None):
        reject_fn = lambda env, p: not self.is_point_from_loc(p, target_pos=target_pos, distractor_pos=distractor_pos)

        point = self.env.find_loc(size=(self.env.wall_x, self.env.wall_y), reject_fn=reject_fn, reject_agent_pos=False)

        # assert all(point < np.array([self.env.wall_x, self.env.wall_y]))
        # assert all(point > np.array([0, 0]))

        return point

    def goto_point_action(self, point_from_loc, target_pos, distractor_pos):
        if self.is_point_from_loc(self.cur_pos, target_pos=target_pos, distractor_pos=distractor_pos):
            # point to a direction
            action = self.compute_wanted_point_action(target_pos)

        else:
            # do not point if not is_point_from_loc
            if self.is_pointing():
                # stop pointing
                action = self.stop_point

            else:
                # move
                action = self.path_to_pos(point_from_loc)

        return action


class InformationSeekingEnv(MultiModalMiniGridEnv):
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
        n_colors=None,
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

        if n_colors is None:
            self.n_colors = len(COLOR_NAMES)
        else:
            self.n_colors = n_colors

        self.grammar = SocialAIGrammar()

        self.init_done = False
        # parameters - to be set in reset
        self.parameters = None

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
        self.all_npc_utterance_actions = self.caretaker.list_of_possible_utterances
        self.prim_actions_dict = SocialAINPCActionsDict

    def revert(self):
        self.grid.set(*self.caretaker.cur_pos, None)
        self.place_npc()
        self.put_objects_in_env(remove_objects=True)

    def is_in_marble_way(self, pos):
        target_pos = self.generator_current_pos

        # generator distractor is in the same row / collumn as the marble and the generator
        # if self.distractor_current_pos is not None:
        #     distractor_pos = self.distractor_current_pos
        # else:
        #     distractor_pos = [None, None]

        if self.problem in ["Marbles", "Marble"]:
            # point can't be in the same row or column as both the marble and the generator
            # all three: marble, generator, loc are in the same row or column
            if any((pos == target_pos) * (pos == self.marble_current_pos)):
                # all three: marble, generator, loc are in the same row or column -> is in its way
                return True

            if int(self.parameters["N"]) > 1:
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

        # problem: Apples/Boxes/Switches/Generators/Marbles
        self.problem = self.parameters["Problem"] if self.parameters else "Apples"
        num_of_colors = self.parameters.get("Num_of_colors", None) if self.parameters else None
        if num_of_colors is None:
            num_of_colors = self.n_colors

        # additional test for recursivness of joint attention -> cues are given outside of JA
        self.ja_recursive = self.parameters.get("JA_recursive", False) if self.parameters else False

        self.add_obstacles()
        if self.obstacles != "No":
            warnings.warn("InformationSeeking should no be using obstacles.")

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.current_width, self.current_height)

        if self.problem in ["Doors", "Levers"]:
            # Add a second wall: this is needed so that an apple cannot be seen diagonally between the wall and the door
            self.grid.wall_rect(1, 1, self.wall_x-1, self.wall_y-1)

        # apple
        self.apple_pos = (self.current_width, self.current_height)

        # box
        locked = self.problem == "Switches"

        if num_of_colors is None:
            POSSIBLE_COLORS = COLOR_NAMES.copy()

        else:
            POSSIBLE_COLORS = COLOR_NAMES[:int(num_of_colors)].copy()

        self.box_color = self._rand_elem(POSSIBLE_COLORS)

        if self.problem in ["Doors", "Levers"]:
            # door

            # find the position on a wall
            self.apple_current_pos = self.find_loc(
                size=(self.current_width, self.current_height),
                reject_taken_pos=False,  # we will create a gap in the wall
                reject_agent_pos=True,
                reject_fn=lambda _, pos:
                not (pos[0] in [0, self.wall_x] or pos[1] in [0, self.wall_y]) or  # reject not on a wall
                tuple(pos) in [
                    (0, 0),
                    (0, 1),
                    (1, 0),

                    (0, self.wall_y),
                    (0, self.wall_y-1),
                    (1, self.wall_y),

                    (self.wall_x, self.wall_y),
                    (self.wall_x-1, self.wall_y),
                    (self.wall_x, self.wall_y-1),

                    (self.wall_x, 0),
                    (self.wall_x, 1),
                    (self.wall_x-1, 0),
                ]
            )
            self.grid.set(*self.apple_current_pos, None)  # hole in the wall

            # door is in front of the apple
            door_x = {
                0: 1,
                self.wall_x: self.wall_x - 1,
            }.get(self.apple_current_pos[0], self.apple_current_pos[0])
            door_y = {
                0: 1,
                self.wall_y: self.wall_y - 1,
            }.get(self.apple_current_pos[1], self.apple_current_pos[1])

            self.door_current_pos = np.array([door_x, door_y])
            self.grid.set(*self.door_current_pos, None)  # hole in the wall


            #  lever
            if self.problem in ["Levers"]:
                self.lever_current_pos = self.find_loc(
                    top=(2, 2),
                    size=(self.current_width-4, self.current_height-4),
                    reject_agent_pos=True,
                    reject_fn=lambda _, pos: next_to(pos, self.door_current_pos) # reject in front of the door
                )

        else:
            # find the position for the apple/box/generator_platform
            self.apple_current_pos = self.find_loc(size=self.apple_pos, reject_agent_pos=True)
            assert all(self.apple_current_pos < np.array([self.current_width-1, self.current_height-1]))

        # door
        self.door_color = self._rand_elem(POSSIBLE_COLORS)

        # lever
        self.lever_color = self._rand_elem(POSSIBLE_COLORS)

        # switch
        self.switch_pos = (self.current_width, self.current_height)
        self.switch_color = self._rand_elem(POSSIBLE_COLORS)
        self.switch_current_pos = self.find_loc(
            size=self.switch_pos,
            reject_agent_pos=True,
            reject_fn=lambda _, pos: tuple(pos) in map(tuple, [self.apple_current_pos]),
        )

        # generator
        self.generator_pos = (self.current_width, self.current_height)
        self.generator_color = self._rand_elem(POSSIBLE_COLORS)
        self.generator_current_pos = self.find_loc(
            size=self.generator_pos,
            reject_agent_pos=True,
            reject_fn=lambda _, pos: (
                tuple(pos) in map(tuple, [self.apple_current_pos])
                or
                (self.problem in ["Marble"] and tuple(pos) in [
                    # not in corners
                    (1, 1),
                    (self.current_width-2, 1),
                    (1, self.current_height-2),
                    (self.current_width-2, self.current_height-2),
                ])
                or
                # not in the same row collumn as the platform
                (self.problem in ["Marble"] and any(pos == self.apple_current_pos))
            ),
        )

        # generator platform
        self.generator_platform_color = self._rand_elem(POSSIBLE_COLORS)

        # marbles
        self.marble_pos = (self.current_width, self.current_height)
        self.marble_color = self._rand_elem(POSSIBLE_COLORS)
        self.marble_current_pos = self.find_loc(
            size=self.marble_pos,
            reject_agent_pos=True,
            reject_fn=lambda _, pos: self.problem in ["Marbles", "Marble"] and (
                tuple(pos) in map(tuple, [self.apple_current_pos, self.generator_current_pos])
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

        # distractor
        if self.problem == "Boxes":
            assert not locked
            POSSIBLE_COLORS.remove(self.box_color)

        elif self.problem == "Doors":
            POSSIBLE_COLORS.remove(self.door_color)

        elif self.problem == "Levers":
            POSSIBLE_COLORS.remove(self.lever_color)

        elif self.problem == "Switches":
            POSSIBLE_COLORS.remove(self.switch_color)

        elif self.problem in ["Generators", "Marble"]:
            POSSIBLE_COLORS.remove(self.generator_color)

        self.distractor_color = self._rand_elem(POSSIBLE_COLORS)
        self.distractor_pos = (self.current_width, self.current_height)

        # distractor reject function
        if self.problem in ["Apples", "Boxes"]:
            distractor_reject_fn = lambda _, pos: tuple(pos) in map(tuple, [self.apple_current_pos])

        elif self.problem in ["Switches"]:
            distractor_reject_fn = lambda _, pos: tuple(pos) in map(tuple, [self.apple_current_pos, self.switch_current_pos])

        elif self.problem in ["Generators"]:
            distractor_reject_fn = lambda _, pos: tuple(pos) in map(tuple, [self.apple_current_pos, self.generator_current_pos])

        elif self.problem in ["Marble"]:
            # problem is marbles
            if self.parameters["N"] == "1":
                distractor_reject_fn = lambda _, pos: tuple(pos) in map(tuple, [self.apple_current_pos, self.generator_current_pos, self.marble_current_pos])
            else:
                same_dim = (self.generator_current_pos == self.marble_current_pos).argmax()
                distactor_same_dim = 1-same_dim
                distractor_reject_fn = lambda _, pos: tuple(pos) in map(tuple, [
                    self.apple_current_pos,
                    self.generator_current_pos,
                    self.marble_current_pos
                ]) or pos[distactor_same_dim] != self.marble_current_pos[distactor_same_dim]

        elif self.problem in ["Doors"]:
            # reject not next to a wall
            distractor_reject_fn = lambda _, pos: (
                not (pos[0] in [1, self.wall_x-1] or pos[1] in [1, self.wall_y-1]) or  # reject not on a wall
                tuple(pos) in [
                    (1, 1),
                    (self.wall_x-1, self.wall_y - 1),
                    (1, self.wall_y-1),
                    (self.wall_x-1, 1),
                    tuple(self.door_current_pos)
                ]
            )

        elif self.problem in ["Levers"]:
            # not in front of the door
            distractor_reject_fn = lambda _, pos: next_to(pos, self.door_current_pos) or tuple(pos) in list(map(tuple, [self.door_current_pos, self.lever_current_pos]))

        else:
            raise ValueError("Problem {} indefined.".format(self.problem))

        if self.problem == "Doors":

            self.distractor_current_pos = self.find_loc(
                top=(1, 1),
                size=(self.current_width-2, self.current_height-2),
                reject_agent_pos=True,
                reject_fn=distractor_reject_fn,
                reject_taken_pos=False
            )

            if self.parameters["N"] != "1":
                self.grid.set(*self.distractor_current_pos, None)  # hole in the wall
        else:
            self.distractor_current_pos = self.find_loc(
                size=self.distractor_pos,
                reject_agent_pos=True,
                reject_fn=distractor_reject_fn
            )

        self.put_objects_in_env()


        # NPC
        put_peer = self.parameters["Peer"] if self.parameters else "N"
        assert put_peer in ["Y", "N"]

        color = self._rand_elem(COLOR_NAMES)
        self.caretaker = Caretaker(color, "Caretaker", self)

        if put_peer == "Y":
            self.place_npc()


        # Randomize the agent's start position and orientation
        self.place_agent(size=(self.current_width, self.current_height))

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

    def place_npc(self):
        if self.problem in ["Doors"]:
            self.place_obj(
                self.caretaker,
                size=(self.current_width, self.current_height),
                reject_fn=lambda _, pos: next_to(pos, self.door_current_pos) or next_to(pos, self.distractor_current_pos)
            )

        elif self.problem in ["Levers"]:
            self.place_obj(
                self.caretaker,
                size=(self.current_width, self.current_height),
                reject_fn=lambda _, pos: next_to(pos, self.door_current_pos)
            )

        else:
            self.place_obj(self.caretaker, size=(self.current_width, self.current_height), reject_fn=InformationSeekingEnv.is_in_marble_way)

        self.caretaker.initial_pos = self.caretaker.cur_pos

    def put_objects_in_env(self, remove_objects=False):

        assert self.apple_current_pos is not None
        assert self.switch_current_pos is not None

        self.doors_block_set = []
        self.levers_block_set = []
        self.switches_block_set = []
        self.boxes_block_set = []
        self.generators_block_set = []

        self.distractor_door = None
        self.distractor_lever = None
        self.distractor_box = None
        self.distractor_switch = None
        self.distractor_generator = None

        # problem: Apples/Boxes/Switches/Generators
        assert self.problem == self.parameters["Problem"] if self.parameters else "Apples"

        # move objects (used only in revert), not in gen_grid
        if remove_objects:
            # remove apple or box
            # assert type(self.grid.get(*self.apple_current_pos)) in [Apple, LockableBox]
            # self.grid.set(*self.apple_current_pos, None)

            # remove apple (after demo it must be an apple)
            assert type(self.grid.get(*self.apple_current_pos)) in [Apple]
            self.grid.set(*self.apple_current_pos, None)

            if self.problem in ["Doors"]:
                # assert type(self.grid.get(*self.door_current_pos)) in [Door]
                self.grid.set(*self.door.cur_pos, None)

            elif self.problem in ["Levers"]:
                # assert type(self.grid.get(*self.door_current_pos)) in [Door]
                self.grid.set(*self.remote_door.cur_pos, None)
                self.grid.set(*self.lever.cur_pos, None)

            elif self.problem in ["Switches"]:
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
            if self.problem in ["Boxes", "Switches", "Generators", "Marbles", "Marble", "Doors", "Levers"] and self.parameters["N"] != "1":
                assert type(self.grid.get(*self.distractor_current_pos)) in [LockableBox, Switch, AppleGenerator, Door, Lever]
                self.grid.set(*self.distractor_current_pos, None)

        # apple
        self.apple = Apple()

        # Box
        locked = self.problem == "Switches"

        self.box = LockableBox(
            self.box_color,
            contains=self.apple,
            is_locked=locked,
            block_set=self.boxes_block_set
        )
        self.boxes_block_set.append(self.box)

        # Doors
        self.door = Door(
            color=self.door_color,
            is_locked=False,
            block_set=self.doors_block_set,
        )
        self.doors_block_set.append(self.door)

        # Levers
        self.remote_door = RemoteDoor(
            color=self.door_color,
        )

        self.lever = Lever(
            color=self.lever_color,
            object=self.remote_door,
            active_steps=None,
            block_set=self.levers_block_set,
        )
        self.levers_block_set.append(self.lever)

        # Switch
        self.switch = Switch(
            color=self.switch_color,
            lockable_object=self.box,
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
            # on_push=lambda: self.put_obj_np(self.apple, self.apple_current_pos)
            on_push=lambda: self.grid.set(*self.apple_current_pos, self.apple),
            marble_activation=self.problem in ["Marbles", "Marble"],
        )
        self.generators_block_set.append(self.generator)

        self.generator_platform = GeneratorPlatform(self.generator_platform_color)

        self.marble = Marble(self.marble_color, env=self)

        if self.problem in ["Apples"]:
            self.put_obj_np(self.apple, self.apple_current_pos)

        elif self.problem in ["Doors"]:
            self.put_obj_np(self.apple, self.apple_current_pos)
            self.put_obj_np(self.door, self.door_current_pos)

        elif self.problem in ["Levers"]:
            self.put_obj_np(self.apple, self.apple_current_pos)
            self.put_obj_np(self.remote_door, self.door_current_pos)
            self.put_obj_np(self.lever, self.lever_current_pos)

        elif self.problem in ["Boxes"]:
            self.put_obj_np(self.box, self.apple_current_pos)

        elif self.problem in ["Switches"]:
            self.put_obj_np(self.box, self.apple_current_pos)
            self.put_obj_np(self.switch, self.switch_current_pos)

        elif self.problem in ["Generators", "Marbles", "Marble"]:
            self.put_obj_np(self.generator, self.generator_current_pos)
            self.put_obj_np(self.generator_platform, self.apple_current_pos)

            if self.problem in ["Marbles", "Marble"]:
                self.put_obj_np(self.marble, self.marble_current_pos)
        else:
            raise ValueError("Problem {} not defined. ".format(self.problem))

        # Distractors
        if self.problem not in ["Apples"]:

            N = int(self.parameters["N"])
            if N > 1:
                assert N == 2

                if self.problem == "Boxes":
                    assert not locked

                    self.distractor_box = LockableBox(
                        self.distractor_color,
                        is_locked=locked,
                        block_set=self.boxes_block_set,
                    )
                    self.boxes_block_set.append(self.distractor_box)

                    self.put_obj_np(self.distractor_box, self.distractor_current_pos)

                elif self.problem == "Doors":
                    self.distractor_door = Door(
                        color=self.distractor_color,
                        is_locked=False,
                        block_set=self.doors_block_set,
                    )
                    self.doors_block_set.append(self.distractor_door)

                    self.put_obj_np(self.distractor_door, self.distractor_current_pos)

                elif self.problem == "Levers":
                    self.distractor_lever = Lever(
                        color=self.distractor_color,
                        active_steps=None,
                        block_set=self.levers_block_set,
                    )
                    self.levers_block_set.append(self.distractor_lever)
                    self.put_obj_np(self.distractor_lever, self.distractor_current_pos)

                elif self.problem == "Switches":
                    self.distractor_switch = Switch(
                        color=self.distractor_color,
                        locker_switch=True,
                        no_turn_off=True,
                        no_light=self.switch_no_light,
                        block_set=self.switches_block_set,
                    )
                    self.switches_block_set.append(self.distractor_switch)

                    self.put_obj_np(self.distractor_switch, self.distractor_current_pos)

                elif self.problem in ["Generators", "Marbles", "Marble"]:
                    self.distractor_generator = AppleGenerator(
                        color=self.distractor_color,
                        block_set=self.generators_block_set,
                        marble_activation=self.problem in ["Marbles", "Marble"],
                    )
                    self.generators_block_set.append(self.distractor_generator)

                    self.put_obj_np(self.distractor_generator, self.distractor_current_pos)

                else:
                    raise ValueError("Undefined N for problem {}".format(self.problem))

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
        self.agent_opened_the_door = False
        self.agent_pulled_the_lever = False
        self.agent_turned_on_the_switch = False
        self.agent_pressed_the_generator = False
        self.agent_pushed_the_marble = False

        return obs

    def step(self, action):

        success = False
        p_action = action[0]
        utterance_action = action[1:]

        apple_had_been_eaten = self.apple.eaten
        box_had_been_opened = self.box.is_open
        door_had_been_opened = self.door.is_open
        lever_had_been_pulled = self.lever.is_on
        switch_had_been_turned_on = self.switch.is_on
        generator_had_been_pressed = self.generator.is_pressed
        marble_had_been_pushed = self.marble.was_pushed

        # primitive actions
        _, reward, done, info = super().step(p_action)

        if self.problem in ["Marbles", "Marble"]:
            # todo: create stepable objects which are stepped automatically?
            self.marble.step()

        # eaten just now by primitive actions of the agent
        if not self.agent_ate_the_apple:
            self.agent_ate_the_apple = self.apple.eaten and not apple_had_been_eaten

        if not self.agent_opened_the_box:
            self.agent_opened_the_box = self.box.is_open and not box_had_been_opened

        if not self.agent_opened_the_door:
            self.agent_opened_the_door = self.door.is_open and not door_had_been_opened

        if not self.agent_pulled_the_lever:
            self.agent_pulled_the_lever = self.lever.is_on and not lever_had_been_pulled

        if not self.agent_turned_on_the_switch:
            self.agent_turned_on_the_switch = self.switch.is_on and not switch_had_been_turned_on

        if not self.agent_pressed_the_generator:
            self.agent_pressed_the_generator = self.generator.is_pressed and not generator_had_been_pressed

        if not self.agent_pushed_the_marble:
            self.agent_pushed_the_marble = self.marble.was_pushed and not marble_had_been_pushed

        # utterances
        agent_spoke = not all(np.isnan(utterance_action))
        if agent_spoke:
            utterance = self.grammar.construct_utterance(utterance_action)

            if self.hear_yourself:
                self.utterance += "YOU: {} \n".format(utterance)
            self.full_conversation += "YOU: {} \n".format(utterance)
        else:
            utterance = None

        if self.parameters["Peer"] == "Y":
            reply, npc_info = self.caretaker.step(utterance)
        else:
            reply = None
            npc_info = self.caretaker.create_info(
                action=None,
                utterance=None,
                was_introduced_to=False
            )

        if reply:
            self.utterance += "{}: {} \n".format(self.caretaker.name, reply)
            self.full_conversation += "{}: {} \n".format(self.caretaker.name, reply)

        # aftermath
        if p_action == self.actions.done:
            done = True

        elif self.agent_ate_the_apple:
            # check that it is the agent who ate it
            assert self.actions(p_action) == self.actions.toggle
            assert self.get_cell(*self.front_pos) == self.apple

            if self.parameters.get("Cue_type", "nan") == "Emulation":

                # during emulation it can be the NPC who eats the apple, opens the box, and turns on the switch
                if self.parameters["Scaffolding"] and self.caretaker.apple_unlocked_for_agent:
                    # if the caretaker unlocked the apple the agent gets reward upon eating it
                    reward = self._reward()
                    success = True

                elif self.problem == "Apples":
                    reward = self._reward()
                    success = True

                elif self.problem == "Doors" and self.agent_opened_the_door:
                    reward = self._reward()
                    success = True

                elif self.problem == "Levers" and self.agent_pulled_the_lever:
                    reward = self._reward()
                    success = True

                elif self.problem == "Boxes" and self.agent_opened_the_box:
                    reward = self._reward()
                    success = True

                elif self.problem == "Switches" and self.agent_opened_the_box and self.agent_turned_on_the_switch:
                    reward = self._reward()
                    success = True

                elif self.problem == "Generators" and self.agent_pressed_the_generator:
                    reward = self._reward()
                    success = True

                elif self.problem in ["Marble"] and self.agent_pushed_the_marble:
                    reward = self._reward()
                    success = True

            else:
                reward = self._reward()
                success = True

            done = True

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

        info = {**info, **{"NPC_"+k: v for k, v in npc_info.items()}}

        info["NPC_observed"] = npc_observed
        info["success"] = success

        assert success == (reward > 0)

        return obs, reward, done, info

    def _reward(self):
        if self.diminished_reward:
            return super()._reward()
        else:
            return 1.0

    def render(self, *args, **kwargs):
        obs = super().render(*args, **kwargs)
        if args and args[0] == 'human':
            self.window.clear_text()  # erase previous text
            self.window.set_caption(self.full_conversation)

            # self.window.ax.set_title("correct color: {}".format(self.box.target_color), loc="left", fontsize=10)

            if self.outcome_info:
                color = None
                if "SUCCESS" in self.outcome_info:
                    color = "lime"
                elif "FAILURE" in self.outcome_info:
                    color = "red"
                self.window.add_text(*(0.01, 0.85, self.outcome_info),
                                     **{'fontsize': 15, 'color': color, 'weight': "bold"})

            self.window.show_img(obs)  # re-draw image to add changes to window
        return obs


register(
    id='SocialAI-InformationSeeking-v0',
    entry_point='gym_minigrid.social_ai_envs:InformationSeekingEnv'
)
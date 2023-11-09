import time

import numpy as np
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.social_ai_envs.socialaigrammar import SocialAIGrammar, SocialAIActions, SocialAIActionSpace
import time
from collections import deque


class AppleGuardingNPC(NPC):
    """
    A simple NPC that knows who is telling the truth
    """
    def __init__(self, color, name, env):
        super().__init__(color)
        self.name = name
        self.env = env
        self.npc_dir = 1  # NPC initially looks downward
        self.npc_dir = np.random.randint(0, 4)  # NPC initially looks downward
        self.npc_type = 1  # this will be put into the encoding

        self.was_introduced_to = False

        self.ate_an_apple = False
        self.demo_over = False
        self.demo_over_and_position_safe = False
        self.apple_unlocked_for_agent = False


        self.target_obj = self.env.apple

        self.waiting_counter = 0
        self.wait_steps = 4

        assert self.env.grammar.contains_utterance(self.introduction_statement)

    def draw_npc_face(self, c):
        assert self.npc_type == 1

        assert all(COLORS[self.color] == c)

        shapes = []
        shapes_colors = []

        # Draw eyes
        shapes.append(point_in_circle(cx=0.70, cy=0.50, r=0.10))
        shapes_colors.append(c)

        shapes.append(point_in_circle(cx=0.30, cy=0.50, r=0.10))
        shapes_colors.append(c)

        # Draw mouth
        shapes.append(point_in_rect(0.20, 0.80, 0.72, 0.81))
        shapes_colors.append(c)

        # Draw eyebrows
        shapes.append(point_in_triangle((0.15, 0.20),
                                            (0.85, 0.20),
                                            (0.50, 0.35)))
        shapes_colors.append(c)

        shapes.append(point_in_triangle((0.30, 0.20),
                                            (0.70, 0.20),
                                            (0.5, 0.35)))
        shapes_colors.append((0,0,0))

        return shapes, shapes_colors

    def can_see_pos(self, obj_pos):

        # is the npc seen by the agent
        npc_view_obj = self.relative_coords(*obj_pos)
        grid, vis_mask = self.gen_obs_grid()

        if npc_view_obj is not None:
            # in the agent's field of view
            ag_view_npc_x, ag_view_npc_y = npc_view_obj

            # is it occluded
            object_observed = vis_mask[ag_view_npc_x, ag_view_npc_y]
        else:
            object_observed = False
        
        return object_observed, grid, vis_mask

    def step(self, utterance):
        reply, info = super().step()

        if self.env.hidden_npc:
            return reply, info

        # reply, action = self.handle_introduction(utterance) # revert this?
        reply, action = None, None

        NPC_movement = self.env.parameters.get("NPC_movement", "Rotating")

        if self.waiting_counter >= self.wait_steps:
            self.waiting_counter = 0

            if NPC_movement == "Rotating":
                action = random.choice([self.rotate_left, self.rotate_right])

            elif NPC_movement == "Walking":
                action = random.choice([
                    random.choice([
                        self.rotate_left,  # 25 %
                        self.rotate_right  # 25 %
                    ]),
                    self.go_forward  # 50%
                ])
            else:
                raise DeprecationWarning(f"Undefined movement option {NPC_movement}")

        else:
            self.waiting_counter += 1

        if action is not None:
            action()

        info = {
            "prim_action": action.__name__ if action is not None else "no_op",
            "utterance": reply or "no_op",
            "was_introduced_to": self.was_introduced_to
        }

        assert (reply or "no_op") in self.list_of_possible_utterances

        return reply, info


class AppleStealingEnv(MultiModalMiniGridEnv):
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
        switch_no_light=False,
        reward_diminish_factor=0.1,
        see_through_walls=False,
        egocentric_observation=True,
        tagged_apple=False,
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
        self.all_npc_utterance_actions = AppleGuardingNPC.get_list_of_possible_utterances()
        self.prim_actions_dict = SocialAINPCActionsDict

        self.tagged_apple = tagged_apple

    def _gen_grid(self, width_, height_):
        # Create the grid
        self.grid = Grid(width_, height_, nb_obj_dims=self.encoding_size)

        # new
        self.current_width = self._rand_int(7, width_+1)
        self.current_height = self._rand_int(7, height_+1)
        # print("Room size: {}x{}".format(self.current_width, self.current_height))

        self.wall_x = self.current_width-1
        self.wall_y = self.current_height-1

        self.version = self.parameters["Version"] if self.parameters else "Asocial"

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, self.current_width, self.current_height)

        self.add_obstacles()

        # apple
        self.apple_pos = (self.current_width, self.current_height)

        # find the position for the apple/box/generator_platform

        self.apple_current_pos = self.find_loc(size=self.apple_pos, reject_agent_pos=True, reject_taken_pos=True)
        assert all(self.apple_current_pos < np.array([self.current_width-1, self.current_height-1]))

        self.apple = Apple()
        self.put_obj_np(self.apple, self.apple_current_pos)

        # NPC
        color = self._rand_elem(COLOR_NAMES)
        self.caretaker = AppleGuardingNPC(color, "Peer", self)

        if self.version == "Social":
            self.place_obj(self.caretaker, size=(self.current_width, self.current_height))

        # Randomize the agent's start position and orientation
        self.place_agent(size=(self.current_width, self.current_height))

        # Generate the mission string
        self.mission = 'undefined'

        # Dummy beginning string
        # self.beginning_string = "This is what you hear. \n"
        self.beginning_string = "Conversation: \n"
        self.utterance = self.beginning_string

        # utterance appended at the end of each step
        self.utterance_history = ""

        # used for rendering
        self.full_conversation = self.utterance
        self.outcome_info = None



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

        return obs

    def step(self, action):

        success = False

        p_action = action[0]
        utterance_action = action[1:]

        apple_had_been_eaten = self.apple.eaten
        if self.version == "Social":
            agent_seen_by_npc, _, _ = self.caretaker.can_see_pos(self.agent_pos)
        else:
            agent_seen_by_npc = False

        # primitive actions
        _, reward, done, info = super().step(p_action)

        if not self.agent_ate_the_apple:
            self.agent_ate_the_apple = self.apple.eaten and not apple_had_been_eaten

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
            assert self.get_cell(*self.front_pos) == self.apple

            if agent_seen_by_npc:
                reward = 0
                success = False

            else:
                reward = self._reward()
                success = True

            done = True

            # check that it is the agent who ate it
            assert self.actions(p_action) == self.actions.toggle
            assert self.get_cell(*self.front_pos) == self.apple

        # discount
        if self.step_penalty:
            reward = reward - 0.01

        # update obs with NPC movement
        obs = self.gen_obs(full_obs=self.full_obs)

        # fill observation with text
        self.append_existing_utterance_to_history()
        obs = self.add_utterance_to_observation(obs)
        self.reset_utterance()

        # for rendering
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
        assert success == (reward > 0)

        return obs, reward, done, info

    def _reward(self):
        if self.diminished_reward:
            return super()._reward()
        else:
            return 1.0

    def render(self, *args, **kwargs):
        obs = super().render(*args, show_dialogue=False, **kwargs)
        return obs


register(
    id='SocialAI-AppleStealingEnv-v0',
    entry_point='gym_minigrid.social_ai_envs:AppleStealingEnv'
)

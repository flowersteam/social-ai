import math
import random
import hashlib
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .rendering import *
from abc import ABC, abstractmethod
import warnings
import astar

import traceback
import warnings
from functools import wraps

SocialAINPCActionsDict = {
    "go_forward": 0,
    "rotate_left": 1,
    "rotate_right": 2,
    "toggle_action": 3,
    "point_stop_point": 4,
    "point_E": 5,
    "point_S": 6,
    "point_W": 7,
    "point_N": 8,
    "stop_point": 9,
    "no_op": 10
}

point_dir_encoding = {
    "point_E": 0,
    "point_S": 1,
    "point_W": 2,
    "point_N": 3,
}

def get_traceback():
    tb = traceback.extract_stack()
    return "".join(traceback.format_list(tb)[:-1])


# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'brown': np.array([82, 36, 19])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
    'brown' : 6,
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen'           : 0,
    'empty'            : 1,
    'wall'             : 2,
    'floor'            : 3,
    'door'             : 4,
    'key'              : 5,
    'ball'             : 6,
    'box'              : 7,
    'goal'             : 8,
    'lava'             : 9,
    'agent'            : 10,
    'npc'              : 11,
    'switch'           : 12,
    'lockablebox'      : 13,
    'apple'            : 14,
    'applegenerator'   : 15,
    'generatorplatform': 16,
    'marble'           : 17,
    'marbletee'        : 18,
    'fence'            : 19,
    'remotedoor'       : 20,
    'lever'            : 21,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    'open'  : 0,
    'closed': 1,
    'locked': 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = np.array((0, 0))

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_push(self):
        """Can the agent push the object?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self, nb_dims=3, absolute_coordinates=False):
        """Encode the a description of this object as a nb_dims-tuple of integers"""
        if absolute_coordinates:
            core = (OBJECT_TO_IDX[self.type], *self.cur_pos, COLOR_TO_IDX[self.color])
        else:
            core = (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color])

        return core + (0,) * (nb_dims - len(core))

    def cache(self, *args, **kwargs):
        """Used for cached rendering."""
        return self.encode(*args, **kwargs)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'marble':
            v = Marble(color)
        elif obj_type == 'apple':
            eaten = state == 1
            v = Apple(color, eaten=eaten)
        elif obj_type == 'apple_generator':
            is_pressed = state == 2
            v = AppleGenerator(color, is_pressed=is_pressed)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'lockablebox':
            is_locked = state == 2
            v = LockableBox(color, is_locked=is_locked)
        elif obj_type == 'door':
            # State, 0: open, 1: closed, 2: locked
            is_open = state == 0
            is_locked = state == 2
            v = Door(color, is_open, is_locked)
        elif obj_type == 'remotedoor':
            # State, 0: open, 1: closed
            is_open = state == 0
            v = RemoteDoor(color, is_open)
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'lava':
            v = Lava()
        elif obj_type == 'fence':
            v = Fence()
        elif obj_type == 'switch':
            v = Switch(color, is_on=state)
        elif obj_type == 'lever':
            v = Lever(color, is_on=state)
        elif obj_type == 'npc':
            warnings.warn("NPC's internal state  cannot be decoded. Only the icon is shown.")
            v = NPC(color)
            v.npc_type=0
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class BlockableWorldObj(WorldObj):

    def __init__(self, type, color, block_set):
        super(BlockableWorldObj, self).__init__(type, color)
        self.block_set = block_set
        self.blocked = False


    def can_push(self):
        return True

    def push(self, *args, **kwargs):
        return self.block_block_set()

    def toggle(self, *args, **kwargs):
        return self.block_block_set()

    def block_block_set(self):
        """A function that blocks the block set"""
        if not self.blocked:
            if self.block_set is not None:
                # cprint("BLOCKED!", "red")
                for e in self.block_set:
                    e.block()

            return True

        else:
            return False

    def block(self):
        self.blocked = True


class Goal(WorldObj):
    def __init__(self):
        super().__init__('goal', 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    def __init__(self):
        super().__init__('lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0,0,0))


class Fence(WorldObj):
    """
    Same as Lava but can't overlap.
    """
    def __init__(self):
        super().__init__('fence', 'grey')

    def can_overlap(self):
        return False

    def render(self, img):
        c = COLORS[self.color]

        # ugly fence
        fill_coords(img, point_in_rect(
            0.1, 0.9, 0.5, 0.9
            # (0.15, 0.9),
            # (0.10, 0.5),
            # (0.95, 0.9),
            # (0.90, 0.5),
            # (0.10, 0.9),
            # (0.10, 0.5),
            # (0.95, 0.9),
            # (0.95, 0.5),
        ), c)
        fill_coords(img, point_in_quadrangle(
            # (0.15, 0.9),
            # (0.10, 0.5),
            # (0.95, 0.9),
            # (0.90, 0.5),
            (0.10, 0.9),
            (0.10, 0.5),
            (0.95, 0.9),
            (0.95, 0.5),
        ), c)
        return

        # preety fence
        fill_coords(img, point_in_quadrangle(
            (0.15, 0.3125),
            (0.15, 0.4125),
            (0.85, 0.4875),
            (0.85, 0.5875),
        ), c)

        # h2
        fill_coords(img, point_in_quadrangle(
            (0.15, 0.6125),
            (0.15, 0.7125),
            (0.85, 0.7875),
            (0.85, 0.8875),
        ), c)

        # vm
        fill_coords(img, point_in_quadrangle(
            (0.45, 0.2875),
            (0.45, 0.8875),
            (0.55, 0.3125),
            (0.55, 0.9125),
        ), c)
        fill_coords(img, point_in_triangle(
            (0.45, 0.2875),
            (0.55, 0.3125),
            (0.5, 0.25),
        ), c)

        # vl
        fill_coords(img, point_in_quadrangle(
            (0.25, 0.2375),
            (0.25, 0.8375),
            (0.35, 0.2625),
            (0.35, 0.8625),
        ), c)
        # vl
        fill_coords(img, point_in_triangle(
            (0.25, 0.2375),
            (0.35, 0.2625),
            (0.3, 0.2),
        ), c)


        # vr
        fill_coords(img, point_in_quadrangle(
            (0.65, 0.3375),
            (0.65, 0.9375),
            (0.75, 0.3625),
            (0.75, 0.9625),
        ), c)
        fill_coords(img, point_in_triangle(
            (0.65, 0.3375),
            (0.75, 0.3625),
            (0.7, 0.3),
        ), c)


class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Lever(BlockableWorldObj):
    def __init__(self, color, object=None, is_on=False, block_set=None, active_steps=None):
        super().__init__('lever', color, block_set)
        self.is_on = is_on
        self.object = object

        self.active_steps = active_steps
        self.countdown = None  # countdown timer

        self.was_activated = False

        if self.block_set is not None:
            if self.is_on:
                raise ValueError("If using a block set, a Switch must be initialized as OFF")

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return False

    def see_behind(self):
        return True

    def step(self):
        if self.countdown is not None:
            self.countdown = self.countdown - 1

            if self.countdown <= 0:
                # if nothing is on the door, close the door and deactivate timer
                self.toggle()
                self.countdown = None

    def toggle(self, env=None, pos=None):

        if self.blocked:
            return False

        if self.was_activated and not self.is_on:
            # cannot be activated twice
            return False

        self.is_on = not self.is_on

        if self.is_on:
            if self.active_steps is not None:
                # activate countdown to shutdown
                self.countdown = self.active_steps
            self.was_activated = True

        if self.object is not None:
            # open object
            self.object.open_close()

        if self.is_on:
            self.block_block_set()

        return True

    def block(self):
        self.blocked = True

    def encode(self, nb_dims=3, absolute_coordinates=False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: off, 1: on
        state = 1 if self.is_on else 0

        count = self.countdown if self.countdown is not None else 255

        if absolute_coordinates:
            v = (OBJECT_TO_IDX[self.type], *self.cur_pos, COLOR_TO_IDX[self.color], state, count)
        else:
            v = (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state, count)

        v += (0,) * (nb_dims-len(v))

        return v

    def render(self, img):
        c = COLORS[self.color]
        black = (0, 0, 0)

        # off_angle = -math.pi/3
        off_angle = -math.pi/2
        on_angle = -math.pi/8


        rotating_lever_shapes = []
        rotating_lever_shapes.append((point_in_rect(0.5, 0.9, 0.77, 0.83), c))
        rotating_lever_shapes.append((point_in_circle(0.9, 0.8, 0.1), c))

        rotating_lever_shapes.append((point_in_circle(0.5, 0.8, 0.08), c))

        if self.is_on:
            if self.countdown is None:
                angle = on_angle
            else:
                angle = (self.countdown/self.active_steps) * (on_angle-off_angle) + off_angle

        else:
            angle = off_angle

        fill_coords(img, point_in_circle_clip(0.5, 0.8, 0.12, theta_end=-math.pi), c)
        # fill_coords(img, point_in_circle_clip(0.5, 0.8, 0.08, theta_end=-math.pi), black)

        rotating_lever_shapes = [(rotate_fn(v, cx=0.5, cy=0.8, theta=angle), col) for v, col in rotating_lever_shapes]

        for v, col in rotating_lever_shapes:
            fill_coords(img, v, col)

        fill_coords(img, point_in_rect(0.2, 0.8, 0.78, 0.82), c)
        fill_coords(img, point_in_circle(0.5, 0.8, 0.03), (0, 0, 0))


class RemoteDoor(BlockableWorldObj):
    """Door that are unlocked by a lever"""
    def __init__(self, color, is_open=False, block_set=None):
        super().__init__('remotedoor', color, block_set)
        self.is_open = is_open

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    # def toggle(self, env, pos=None):
    #     return False

    def open_close(self):
        # If the player has the right key to open the door

        self.is_open = not self.is_open
        return True

    def encode(self, nb_dims=3, absolute_coordinates=False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed
        state = 0 if self.is_open else 1

        if absolute_coordinates:
            v = (OBJECT_TO_IDX[self.type], *self.cur_pos, COLOR_TO_IDX[self.color], state)
        else:
            v = (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

        v += (0,) * (nb_dims-len(v))
        return v

    def block(self):
        self.blocked = True

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0,0,0))
        else:

            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0,0,0))

            #  wifi symbol
            fill_coords(img, point_in_circle_clip(cx=0.5, cy=0.8, r=0.5, theta_start=-np.pi/3, theta_end=-2*np.pi/3), c)
            fill_coords(img, point_in_circle_clip(cx=0.5, cy=0.8, r=0.45, theta_start=-np.pi/3, theta_end=-2*np.pi/3), (0,0,0))
            fill_coords(img, point_in_circle_clip(cx=0.5, cy=0.8, r=0.4, theta_start=-np.pi/3, theta_end=-2*np.pi/3), c)
            fill_coords(img, point_in_circle_clip(cx=0.5, cy=0.8, r=0.35, theta_start=-np.pi/3, theta_end=-2*np.pi/3), (0,0,0))
            fill_coords(img, point_in_circle_clip(cx=0.5, cy=0.8, r=0.3, theta_start=-np.pi/3, theta_end=-2*np.pi/3), c)
            fill_coords(img, point_in_circle_clip(cx=0.5, cy=0.8, r=0.25, theta_start=-np.pi/3, theta_end=-2*np.pi/3), (0,0,0))
            fill_coords(img, point_in_circle_clip(cx=0.5, cy=0.8, r=0.2, theta_start=-np.pi/3, theta_end=-2*np.pi/3), c)
            fill_coords(img, point_in_circle_clip(cx=0.5, cy=0.8, r=0.15, theta_start=-np.pi/3, theta_end=-2*np.pi/3), (0,0,0))
            fill_coords(img, point_in_circle_clip(cx=0.5, cy=0.8, r=0.1, theta_start=-np.pi/3, theta_end=-2*np.pi/3), c)

        return


class Door(BlockableWorldObj):
    def __init__(self, color, is_open=False, is_locked=False, block_set=None):
        super().__init__('door', color, block_set)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos=None):

        if self.blocked:
            return False

        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                ret = True
            ret = False

        else:
            self.is_open = not self.is_open
            ret = True

        self.block_block_set()

        return ret


    def encode(self, nb_dims=3, absolute_coordinates=False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        if absolute_coordinates:
            v = (OBJECT_TO_IDX[self.type], *self.cur_pos, COLOR_TO_IDX[self.color], state)
        else:
            v = (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

        v += (0,) * (nb_dims-len(v))
        return v

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0,0,0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0,0,0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Switch(BlockableWorldObj):
    def __init__(self, color, lockable_object=None, is_on=False, no_turn_off=True, no_light=True, locker_switch=False, block_set=None):
        super().__init__('switch', color, block_set)
        self.is_on = is_on
        self.lockable_object = lockable_object
        self.no_turn_off = no_turn_off
        self.no_light = no_light
        self.locker_switch = locker_switch

        if self.block_set is not None:

            if self.is_on:
                raise ValueError("If using a block set, a Switch must be initialized as OFF")

            if not self.no_turn_off:
                raise ValueError("If using a block set, a Switch must be initialized can't be turned off")


    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return False

    def see_behind(self):
        return True

    def toggle(self, env, pos=None):

        if self.blocked:
            return False

        if self.is_on:
            if self.no_turn_off:
                return False

        self.is_on = not self.is_on
        if self.lockable_object is not None:
            if self.locker_switch:
                # locker/unlocker switch
                self.lockable_object.is_locked = not self.lockable_object.is_locked
            else:
                # opener switch
                self.lockable_object.toggle(env, pos)


        if self.is_on:
            self.block_block_set()

        if self.no_turn_off:
            # assert that obj is toggled only once
            assert not hasattr(self, "was_toggled")
            self.was_toggled = True

        return True

    def block(self):
        self.blocked = True

    def encode(self, nb_dims=3, absolute_coordinates=False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: off, 1: on
        state = 1 if self.is_on else 0

        if self.no_light:
            state = 0

        if absolute_coordinates:
            v = (OBJECT_TO_IDX[self.type], *self.cur_pos, COLOR_TO_IDX[self.color], state)
        else:
            v = (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

        v += (0,) * (nb_dims-len(v))

        return v


    def render(self, img):
        c = COLORS[self.color]

        # Door frame and door
        if self.is_on and not self.no_light:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), 0.45 * np.array(c))

        else:

            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0,0,0))


class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0,0,0))


class MarbleTee(WorldObj):
    def __init__(self, color="red"):
        super(MarbleTee, self).__init__('marbletee', color)

    def can_pickup(self):
        return False

    def can_push(self):
        return False

    def render(self, img):
        c = COLORS[self.color]

        fill_coords(img, point_in_quadrangle(
            (0.2, 0.5),
            (0.8, 0.5),
            (0.4, 0.6),
            (0.6, 0.6),
        ), c)

        fill_coords(img, point_in_triangle(
            (0.4, 0.6),
            (0.6, 0.6),
            (0.5, 0.9),
        ), c)


class Marble(WorldObj):
    def __init__(self, color='blue', env=None):
        super(Marble, self).__init__('marble', color)
        self.is_tagged = False
        self.moving_dir = None
        self.env = env
        self.was_pushed = False
        self.tee = MarbleTee(color)
        self.tee_uncovered = False

    def can_pickup(self):
        return True

    def step(self):
        if self.moving_dir is not None:
            prev_pos = self.cur_pos
            self.go_forward()

            if any(prev_pos != self.cur_pos) and not self.tee_uncovered:
                assert self.was_pushed

                # if Marble was moved for the first time, uncover the Tee
                # self.env.grid.set(*prev_pos, self.tee)
                self.env.put_obj_np(self.tee, prev_pos)
                self.tee_uncovered = True

    @property
    def is_moving(self):
        return self.moving_dir is not None

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        if self.moving_dir is not None:
            return DIR_TO_VEC[self.moving_dir]
        else:
            return np.array((0, 0))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """
        return self.cur_pos + self.dir_vec

    def go_forward(self):
        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.env.grid.get(*fwd_pos)
        # Don't move if you are going to collide
        if fwd_pos.tolist() != self.env.agent_pos.tolist() and (fwd_cell is None or fwd_cell.can_overlap()):
            self.env.grid.set(*self.cur_pos, None)
            self.env.grid.set(*fwd_pos, self)
            self.cur_pos = fwd_pos
            return True

        # push object if pushable
        if fwd_pos.tolist() != self.env.agent_pos.tolist() and (fwd_cell is not None and fwd_cell.can_push()):
            fwd_cell.push(push_dir=self.moving_dir, pusher=self)
            self.moving_dir = None
            return True

        else:
            self.moving_dir = None
            return False

    def can_push(self):
        return True

    def push(self, push_dir, pusher=None):
        if type(push_dir) is not int:
            raise ValueError("Direction must be of type int and is of type {}".format(type(push_dir)))

        self.moving_dir = push_dir
        if self.moving_dir is not None:
            self.was_pushed = True

    def render(self, img):
        color = COLORS[self.color]
        if self.is_tagged:
            color = color / 2

        fill_coords(img, point_in_circle(0.5, 0.5, 0.20), color)
        fill_coords(img, point_in_circle(0.55, 0.45, 0.07), (0, 0, 0))

    def tag(self,):
        self.is_tagged = True

    def encode(self, nb_dims=3, absolute_coordinates=False):
        """Encode the a description of this object as a nb_dims-tuple of integers"""
        if absolute_coordinates:
            core = (OBJECT_TO_IDX[self.type], *self.cur_pos, COLOR_TO_IDX[self.color])
        else:
            core = (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color])

        return core + (1 if self.is_tagged else 0,) * (nb_dims - len(core))


class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)
        self.is_tagged = False

    def can_pickup(self):
        return True

    def render(self, img):
        color = COLORS[self.color]
        if self.is_tagged:
            color = color / 2
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), color)

    def tag(self,):
        self.is_tagged = True

    def encode(self, nb_dims=3, absolute_coordinates=False):
        """Encode the a description of this object as a nb_dims-tuple of integers"""
        if absolute_coordinates:
            core = (OBJECT_TO_IDX[self.type], *self.cur_pos, COLOR_TO_IDX[self.color])
        else:
            core = (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color])

        return core + (1 if self.is_tagged else 0,) * (nb_dims - len(core))


class Apple(WorldObj):
    def __init__(self, color='red', eaten=False):
        super(Apple, self).__init__('apple', color)
        self.is_tagged = False
        self.eaten = eaten
        assert self.color != "yellow"

    def revert(self, color='red', eaten=False):
        self.color = color
        self.is_tagged = False
        self.eaten = eaten
        assert self.color != "yellow"

    def can_pickup(self):
        return False

    def render(self, img):
        color = COLORS[self.color]

        if self.is_tagged:
            color = color / 2

        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), color)
        fill_coords(img, point_in_rect(0.1, 0.9, 0.1, 0.55), (0, 0, 0))
        fill_coords(img, point_in_circle(0.35, 0.5, 0.15), color)
        fill_coords(img, point_in_circle(0.65, 0.5, 0.15), color)

        fill_coords(img, point_in_rect(0.48, 0.52, 0.2, 0.45), COLORS["brown"])

        # quadrangle
        fill_coords(img, point_in_quadrangle(
            (0.52, 0.25),
            (0.65, 0.1),
            (0.75, 0.3),
            (0.90, 0.15),
        ), COLORS["green"])


        if self.eaten:
            assert self.color == "yellow"
            fill_coords(img, point_in_circle(0.74, 0.6, 0.23), (0,0,0))
            fill_coords(img, point_in_circle(0.26, 0.6, 0.23), (0,0,0))

    def toggle(self, env, pos):
        if not self.eaten:
            self.eaten = True

            assert self.color != "yellow"
            self.color = "yellow"

            return True

        else:
            assert self.color == "yellow"
            return False

    def tag(self,):
        self.is_tagged = True

    def encode(self, nb_dims=3, absolute_coordinates=False):
        """Encode the a description of this object as a nb_dims-tuple of integers"""

        # eaten <=> yellow
        assert self.eaten == (self.color == "yellow")
        if absolute_coordinates:
            core = (OBJECT_TO_IDX[self.type], *self.cur_pos, COLOR_TO_IDX[self.color])
        else:
            core = (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color])

        return core + (1 if self.is_tagged else 0,) * (nb_dims - len(core))


class GeneratorPlatform(WorldObj):
    def __init__(self, color="red"):
        super(GeneratorPlatform, self).__init__('generatorplatform', color)

    def can_pickup(self):
        return False

    def can_push(self):
        return False

    def render(self, img):
        c = COLORS[self.color]

        fill_coords(img, point_in_circle(0.5, 0.5, 0.2), c)
        fill_coords(img, point_in_circle(0.5, 0.5, 0.18), (0,0,0))

        fill_coords(img, point_in_circle(0.5, 0.5, 0.16), c)
        fill_coords(img, point_in_circle(0.5, 0.5, 0.14), (0,0,0))


class AppleGenerator(BlockableWorldObj):
    def __init__(self, color="red", is_pressed=False, block_set=None, on_push=None, marble_activation=False):
        super(AppleGenerator, self).__init__('applegenerator', color, block_set)
        self.is_pressed = is_pressed
        self.on_push = on_push
        self.marble_activation = marble_activation

    def can_pickup(self):
        return False

    def block(self):
        self.blocked = True

    def can_push(self):
        return True

    def push(self, push_dir=None, pusher=None):

        if self.marble_activation:
            # check that it is marble that pushed the generator
            if type(pusher) != Marble:
                return self.block_block_set()

        if not self.blocked:
            self.is_pressed = True

            if self.on_push:
                self.on_push()

            return self.block_block_set()

        else:
            return False

    def render(self, img):
        c = COLORS[self.color]

        if not self.marble_activation:
            # Outline
            fill_coords(img, point_in_rect(0.15, 0.85, 0.15, 0.85), c)
            # fill_coords(img, point_in_rect(0.17, 0.83, 0.17, 0.83), (0, 0, 0))
            fill_coords(img, point_in_rect(0.16, 0.84, 0.16, 0.84), (0, 0, 0))

            # Outline
            fill_coords(img, point_in_rect(0.22, 0.78, 0.22, 0.78), c)
            fill_coords(img, point_in_rect(0.24, 0.76, 0.24, 0.76), (0, 0, 0))
        else:
            # Outline
            fill_coords(img, point_in_circle(0.5, 0.5, 0.37), c)
            fill_coords(img, point_in_circle(0.5, 0.5, 0.35), (0, 0, 0))

            # Outline
            fill_coords(img, point_in_circle(0.5, 0.5, 0.32), c)
            fill_coords(img, point_in_circle(0.5, 0.5, 0.30), (0, 0, 0))

        # Apple inside
        fill_coords(img, point_in_circle(0.5, 0.5, 0.2), COLORS["red"])
        # fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.55), (0, 0, 0))
        fill_coords(img, point_in_rect(0.30, 0.65, 0.30, 0.55), (0, 0, 0))
        fill_coords(img, point_in_circle(0.42, 0.5, 0.12), COLORS["red"])
        fill_coords(img, point_in_circle(0.58, 0.5, 0.12), COLORS["red"])

        # peteljka
        fill_coords(img, point_in_rect(0.49, 0.50, 0.25, 0.45), COLORS["brown"])

        # leaf
        fill_coords(img, point_in_quadrangle(
            (0.52, 0.32),
            (0.60, 0.21),
            (0.70, 0.34),
            (0.80, 0.23),
        ), COLORS["green"])

    def encode(self, nb_dims=3, absolute_coordinates=False):
        """Encode the a description of this object as a 3-tuple of integers"""

        type = 2 if self.marble_activation else 1

        if absolute_coordinates:
            v = (OBJECT_TO_IDX[self.type], *self.cur_pos, COLOR_TO_IDX[self.color], type)
        else:
            v = (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], type)

        v += (0,) * (nb_dims - len(v))

        return v


class Box(WorldObj):
    def __init__(self, color="red", contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True


class LockableBox(BlockableWorldObj):
    def __init__(self, color="red", is_locked=False, contains=None, block_set=None):
        super(LockableBox, self).__init__('lockablebox', color, block_set)
        self.contains = contains
        self.is_locked = is_locked

        self.is_open = False

    def can_pickup(self):
        return True

    def encode(self, nb_dims=3, absolute_coordinates=False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        # 2 and 1 to be consistent with doors
        if self.is_locked:
            state = 2
        else:
            state = 1

        if absolute_coordinates:
            v = (OBJECT_TO_IDX[self.type], *self.cur_pos, COLOR_TO_IDX[self.color], state)
        else:
            v = (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

        v += (0,) * (nb_dims - len(v))

        return v

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)

        if self.is_locked:
            fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), 0.45 * np.array(c))
        else:
            fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        if self.blocked:
            return False

        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = True
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)

        self.block_block_set()

        # assert that obj is toggled only once
        assert not hasattr(self, "was_toggled")
        self.was_toggled = True

        return True

    def block(self):
        self.blocked = True


class NPC(ABC, WorldObj):
    def __init__(self, color, view_size=7):
        super().__init__('npc', color)
        self.point_dir = 255  # initially no point
        self.introduction_statement = "Help please "
        self.list_of_possible_utterances = NPC.get_list_of_possible_utterances()
        self.view_size = view_size
        self.carrying = False
        self.prim_actions_dict = SocialAINPCActionsDict

        self.reset_last_action()

    @staticmethod
    def get_list_of_possible_utterances():
        return ["no_op"]

    def _npc_action(func):
        """
        Decorator that logs the last action
        """
        @wraps(func)
        def func_wrapper(self, *args, **kwargs):

            if self.env.add_npc_last_prim_action:
                self.last_action = func.__name__

            return func(self, *args, **kwargs)

        return func_wrapper

    def reset_last_action(self):
        self.last_action = "no_op"

    def step(self):
        self.reset_last_action()

        if self.env.hidden_npc:
            info = {
                "prim_action": "no_op",
                "utterance": "no_op",
                "was_introduced_to": self.was_introduced_to
            }
            return None, info

        else:
            return None, None

    def handle_introduction(self, utterance):
        reply, action = None, None
        # introduction and language
        if self.env.parameters.get("Pragmatic_frame_complexity", "No") == "No":

            # only once
            if not self.was_introduced_to:
                self.was_introduced_to = True

        elif self.env.parameters["Pragmatic_frame_complexity"] == "Eye_contact":

            # only first time at eye contact
            if self.is_eye_contact() and not self.was_introduced_to:
                self.was_introduced_to = True

            # if not self.was_introduced_to:
                # rotate to see the agent
                # action = self.look_at_action(self.env.agent_pos)

        elif self.env.parameters["Pragmatic_frame_complexity"] == "Ask":

            # every time asked
            if utterance == self.introduction_statement:
                self.was_introduced_to = True

        elif self.env.parameters["Pragmatic_frame_complexity"] == "Ask_Eye_contact":

            # only first time at eye contact with the introduction statement
            if (self.is_eye_contact() and utterance == self.introduction_statement) and not self.was_introduced_to:
                self.was_introduced_to = True

            # if not self.was_introduced_to:
            #     # rotate to see the agent
            #     action = self.look_at_action(self.env.agent_pos)

        else:
            raise NotImplementedError()

        return reply, action

    def look_at_action(self, target_pos):
        # rotate to see the target_pos
        wanted_dir = self.compute_wanted_dir(target_pos)
        action = self.compute_turn_action(wanted_dir)
        return action

    @_npc_action
    def rotate_left(self):
        self.npc_dir -= 1
        if self.npc_dir < 0:
            self.npc_dir += 4
        return True

    @_npc_action
    def rotate_right(self):
        self.npc_dir = (self.npc_dir + 1) % 4
        return True

    def path_to_toggle_pos(self, goal_pos):
        """
        Return the next action from the path to toggling an object at goal_pos
        """
        if type(goal_pos) != np.ndarray or goal_pos.shape != (2,):
            raise ValueError(f"goal_pos must be a np.ndarray of shape (2,) and is {goal_pos}")

        assert type(self.front_pos) == np.ndarray and self.front_pos.shape == (2,)

        if all(self.front_pos == goal_pos):
            # in front of door
            return self.toggle_action

        else:
            return self.path_to_pos(goal_pos)

    def turn_to_see_agent(self):
        wanted_dir = self.compute_wanted_dir(self.env.agent_pos)
        action = self.compute_turn_action(wanted_dir)
        return action

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the npc's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy


    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        npc's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the npc's view size.
        """

        ax, ay = self.cur_pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx*lx + ry*ly)
        vy = -(dx*lx + dy*ly)

        return vx, vy

    def is_pointing(self):
        return self.point_dir != 255

    def path_to_pos(self, goal_pos):
        """
        Return the next action from the path to goal_pos
        """

        if type(goal_pos) != np.ndarray or goal_pos.shape != (2,):
            raise ValueError(f"goal_pos must be a np.ndarray of shape (2,) and is {goal_pos}")

        def neighbors(n):

            n_nd = np.array(n)

            adjacent_positions = [
                n_nd + np.array([ 0, 1]),
                n_nd + np.array([ 0,-1]),
                n_nd + np.array([ 1, 0]),
                n_nd + np.array([-1, 0]),
            ]
            adjacent_cells = map(lambda pos: self.env.grid.get(*pos), adjacent_positions)

            # keep the positions that don't have anything on or can_overlap
            neighbors = [
                tuple(pos) for pos, cell in
                zip(adjacent_positions, adjacent_cells) if (
                        all(pos == goal_pos)
                        or cell is None
                        or cell.can_overlap()
                ) and not all(pos == self.env.agent_pos)
            ]

            for n1 in neighbors:
                yield n1

        def distance(n1, n2):
            return 1

        def cost(n, goal):
            # manhattan
            return int(np.abs(np.array(n) - np.array(goal)).sum())

        # def is_goal_reached(n, goal):
        #     return all(n == goal)

        path = astar.find_path(
            # tuples because np.ndarray is not hashable
            tuple(self.cur_pos),
            tuple(goal_pos),
            neighbors_fnct=neighbors,
            heuristic_cost_estimate_fnct=cost,
            distance_between_fnct=distance,
            # is_goal_reached_fnct=is_goal_reached
        )

        if path is None:
            # no possible path
            return None

        path = list(path)
        assert all(path[0] == self.cur_pos)
        next_step = path[1]
        wanted_dir = self.compute_wanted_dir(next_step)

        if self.npc_dir == wanted_dir:
            return self.go_forward

        else:
            return self.compute_turn_action(wanted_dir)

    def gen_obs_grid(self):
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

        # Process ocluders and visibility
        # Note that this incurs some performance cost
        if not self.env.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(view_size // 2, view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the npc sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        npc_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*npc_pos, self.carrying)
        else:
            grid.set(*npc_pos, None)

        return grid, vis_mask

    def is_near_agent(self):
        ax, ay = self.env.agent_pos
        wx, wy = self.cur_pos
        if (ax == wx and abs(ay - wy) == 1) or (ay == wy and abs(ax - wx) == 1):
            return True
        return False

    def is_eye_contact(self):
        """
        Returns true if the agent and the NPC are looking at each other
        """
        if self.cur_pos[1] == self.env.agent_pos[1]:
            # same y
            if self.cur_pos[0] > self.env.agent_pos[0]:
                return self.npc_dir == 2 and self.env.agent_dir == 0
            else:
                return self.npc_dir == 0 and self.env.agent_dir == 2

        if self.cur_pos[0] == self.env.agent_pos[0]:
            # same x
            if self.cur_pos[1] > self.env.agent_pos[1]:
                return self.npc_dir == 3 and self.env.agent_dir == 1
            else:
                return self.npc_dir == 1 and self.env.agent_dir == 3

        return False

    def compute_wanted_dir(self, target_pos):
        """
        Computes the direction in which the NPC should look to see target pos
        """

        distance_vec = target_pos - self.cur_pos
        angle = np.degrees(np.arctan2(*distance_vec))
        if angle < 0:
            angle += 360

        if angle < 45:
            wanted_dir = 1  # S
        elif angle < 135:
            wanted_dir = 0  # E
        elif angle < 225:
            wanted_dir = 3  # N
        elif angle < 315:
            wanted_dir = 2  # W
        elif angle < 360:
            wanted_dir = 1  # S

        return wanted_dir

    def compute_wanted_point_dir(self, target_pos):
        point_dir = self.compute_wanted_dir(target_pos)

        return point_dir

    # dir = 0  # E
    # dir = 1  # S
    # dir = 2  # W
    # dir = 3  # N
    # dir = 255  # no point

    @_npc_action
    def stop_point(self):
        self.point_dir = 255
        return True

    @_npc_action
    def point_E(self):
        self.point_dir = point_dir_encoding["point_E"]
        return True

    @_npc_action
    def point_S(self):
        self.point_dir = point_dir_encoding["point_S"]
        return True

    @_npc_action
    def point_W(self):
        self.point_dir = point_dir_encoding["point_W"]
        return True

    @_npc_action
    def point_N(self):
        self.point_dir = point_dir_encoding["point_N"]
        return True

    def compute_wanted_point_action(self, target_pos):
        point_dir = self.compute_wanted_dir(target_pos)

        if point_dir == point_dir_encoding["point_E"]:
            return self.point_E
        elif point_dir == point_dir_encoding["point_S"]:
            return self.point_S
        elif point_dir == point_dir_encoding["point_W"]:
            return self.point_W
        elif point_dir == point_dir_encoding["point_N"]:
            return self.point_N
        else:
            raise ValueError("Unknown direction {}".format(point_dir))


    def compute_turn_action(self, wanted_dir):
        """
        Return the action turning for in the direction of wanted_dir
        """
        if self.npc_dir == wanted_dir:
            # return lambda *args: None
            return None
        if (wanted_dir - self.npc_dir) == 1 or (wanted_dir == 0 and self.npc_dir == 3):
            return self.rotate_right
        if (wanted_dir - self.npc_dir) == - 1 or (wanted_dir == 3 and self.npc_dir == 0):
            return self.rotate_left
        else:
            return self.env._rand_elem([self.rotate_left, self.rotate_right])

    @_npc_action
    def go_forward(self):
        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.env.grid.get(*fwd_pos)
        # Don't move if you are going to collide
        if fwd_pos.tolist() != self.env.agent_pos.tolist() and (fwd_cell is None or fwd_cell.can_overlap()):
            self.env.grid.set(*self.cur_pos, None)
            self.env.grid.set(*fwd_pos, self)
            self.cur_pos = fwd_pos
            return True

        # push object if pushable
        if fwd_pos.tolist() != self.env.agent_pos.tolist() and (fwd_cell is not None and fwd_cell.can_push()):
            fwd_cell.push(push_dir=self.npc_dir, pusher=self)

        else:
            return False

    @_npc_action
    def toggle_action(self):
        fwd_pos = self.front_pos
        fwd_cell = self.env.grid.get(*fwd_pos)
        if fwd_cell:
            return fwd_cell.toggle(self.env, fwd_pos)

        return False

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.npc_dir >= 0 and self.npc_dir < 4
        return DIR_TO_VEC[self.npc_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))


    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.cur_pos + self.dir_vec

    @property
    def back_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.cur_pos - self.dir_vec

    @property
    def right_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.cur_pos + self.right_vec

    @property
    def left_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.cur_pos - self.right_vec

    def draw_npc_face(self, c):
        assert self.npc_type == 0

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

        # Draw bottom hat
        shapes.append(point_in_triangle((0.15, 0.28),
                                            (0.85, 0.28),
                                            (0.50, 0.05)))
        shapes_colors.append(c)
        # Draw top hat
        shapes.append(point_in_rect(0.30, 0.70, 0.05, 0.28))
        shapes_colors.append(c)
        return shapes, shapes_colors

    def render(self, img):


        c = COLORS[self.color]

        npc_shapes = []
        npc_shapes_colors = []


        npc_face_shapes, npc_face_shapes_colors = self.draw_npc_face(c=c)

        npc_shapes.extend(npc_face_shapes)
        npc_shapes_colors.extend(npc_face_shapes_colors)

        if hasattr(self, "npc_dir"):
            # Pre-rotation to ensure npc_dir = 1 means NPC looks downwards
            npc_shapes = [rotate_fn(v, cx=0.5, cy=0.5, theta=-1*(math.pi / 2)) for v in npc_shapes]
            # Rotate npc based on its direction
            npc_shapes = [rotate_fn(v, cx=0.5, cy=0.5, theta=(math.pi/2) * self.npc_dir) for v in npc_shapes]

        if hasattr(self, "point_dir"):
            if self.is_pointing():
                # default points east
                finger = point_in_triangle((0.85, 0.1),
                                           (0.85, 0.3),
                                           (0.99, 0.2))
                finger = rotate_fn(finger, cx=0.5, cy=0.5, theta=(math.pi/2) * self.point_dir)

                npc_shapes.append(finger)
                npc_shapes_colors.append(c)

        if self.last_action == self.toggle_action.__name__:
            # T symbol
            t_symbol = [point_in_rect(0.8, 0.84, 0.02, 0.18), point_in_rect(0.8, 0.95, 0.08, 0.12)]
            t_symbol = [rotate_fn(v, cx=0.5, cy=0.5, theta=(math.pi/2) * self.npc_dir) for v in t_symbol]
            npc_shapes.extend(t_symbol)
            npc_shapes_colors.extend([c, c])

        elif self.last_action == self.go_forward.__name__:
            # symbol for Forward (ommited for speed)
            pass

        if self.env.hidden_npc:
            # crossed eye symbol
            dx, dy = 0.15, -0.2

            # draw eye
            npc_shapes.append(point_in_circle(cx=0.70+dx, cy=0.48+dy, r=0.11))
            npc_shapes_colors.append((128,128,128))

            npc_shapes.append(point_in_circle(cx=0.30+dx, cy=0.52+dy, r=0.11))
            npc_shapes_colors.append((128,128,128))

            npc_shapes.append(point_in_circle(0.5+dx, 0.5+dy, 0.25))
            npc_shapes_colors.append((128, 128, 128))

            npc_shapes.append(point_in_circle(0.5+dx, 0.5+dy, 0.20))
            npc_shapes_colors.append((0, 0, 0))

            npc_shapes.append(point_in_circle(0.5+dx, 0.5+dy, 0.1))
            npc_shapes_colors.append((128, 128, 128))

            # cross it
            npc_shapes.append(point_in_line(0.2+dx, 0.7+dy, 0.8+dx, 0.3+dy, 0.04))
            npc_shapes_colors.append((128, 128, 128))


        # Draw shapes
        for v, c in zip(npc_shapes, npc_shapes_colors):
            fill_coords(img, v, c)

    def cache(self, *args, **kwargs):
        """Used for cached rendering."""
        # adding npc_dir and point_dir because, when egocentric coordinates are used,
        # they can result in the same encoding but require new rendering
        return self.encode(*args, **kwargs) + (self.npc_dir, self.point_dir,)

    def can_overlap(self):
        # If the NPC is hidden, agent can overlap on it
        return self.env.hidden_npc

    def encode(self, nb_dims=3, absolute_coordinates=False):
        if not hasattr(self, "npc_type"):
            raise ValueError("An NPC class must implement the npc_type (int)")

        if not hasattr(self, "env"):
            raise ValueError("An NPC class must have the env")

        assert nb_dims == 6+2*bool(absolute_coordinates)

        if self.env.hidden_npc:
            return (1,) + (0,) * (nb_dims-1)

        assert self.env.egocentric_observation == (not absolute_coordinates)

        if absolute_coordinates:
            v = (OBJECT_TO_IDX[self.type], *self.cur_pos, COLOR_TO_IDX[self.color], self.npc_type)
        else:
            v = (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.npc_type)

        if self.env.add_npc_direction:
            assert hasattr(self, "npc_dir"), "4D but there is no npc dir in NPC state"
            assert self.npc_dir >= 0

            if self.env.egocentric_observation:
                assert self.env.agent_dir >= 0

                # 0 - eye contact; 2 - gaze in same direction; 1 - to left; 3 - to right
                npc_dir_enc = (self.npc_dir - self.env.agent_dir + 2) % 4

                v += (npc_dir_enc,)
            else:
                v += (self.npc_dir,)

        if self.env.add_npc_point_direction:
            assert hasattr(self, "point_dir"), "5D but there is no npc point dir in NPC state"

            if self.point_dir == 255:
                # no pointing
                v += (self.point_dir,)

            elif 0 <= self.point_dir <= 3:
                # pointing

                if self.env.egocentric_observation:
                    assert self.env.agent_dir >= 0

                    #  0  - pointing at agent; 2 - point in direction of agent gaze; 1 - to left; 3 - to right
                    point_enc = (self.point_dir - self.env.agent_dir + 2) % 4
                    v += (point_enc,)

                else:
                    v += (self.point_dir,)

            else:
                raise ValueError(f"Undefined point direction {self.point_dir}")

        if self.env.add_npc_last_prim_action:
            assert hasattr(self, "last_action"), "6D but there is no last action in NPC state"

            if self.last_action in ["point_E", "point_S", "point_W", "point_N"] and self.env.egocentric_observation:

                # get the direction of the last point
                last_action_point_dir = point_dir_encoding[self.last_action]

                # convert to relative dir
                #  0  - pointing at agent; 2 - point in direction of agent gaze; 1 - to left; 3 - to right
                last_action_relative_point_dir = (last_action_point_dir - self.env.agent_dir + 2) % 4

                # the point_X action ids are in range [point_E, ... , point_N]
                # id of point_E is the starting one, we use the same range [E, S, W ,N ] -> [at, left, same, right]
                last_action_id = self.prim_actions_dict["point_E"] + last_action_relative_point_dir

            else:
                last_action_id = self.prim_actions_dict[self.last_action]

            v += (last_action_id,)

        assert self.point_dir >= 0
        assert len(v) == nb_dims

        return v


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height, nb_obj_dims):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height
        self.nb_obj_dims = nb_obj_dims

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1  = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            o = obj_type()
            o.cur_pos = np.array((x+i, y))
            self.set(x + i, y, o)

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            o = obj_type()
            o.cur_pos = np.array((x, y+j))
            self.set(x, y + j, o)

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width, self.nb_obj_dims)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height, self.nb_obj_dims)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        obj,
        agent_dir=None,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3,
        nb_obj_dims=3,
        mask_unobserved=False
    ):
        """
        Render a tile and cache the result
        """
        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size, mask_unobserved)
        # key = obj.encode(nb_dims=nb_obj_dims) + key if obj else key
        key = obj.cache(nb_dims=nb_obj_dims) + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)  # 3D for rendering

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)
        elif mask_unobserved:
            # mask unobserved and not highlighted -> unobserved by the agent
            img *= 0

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size,
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None,
        mask_unobserved=False,
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                    nb_obj_dims=self.nb_obj_dims,
                    mask_unobserved=mask_unobserved
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask=None, absolute_coordinates=False):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, self.nb_obj_dims), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1:] = 0

                    else:
                        array[i, j, :] = v.encode(nb_dims=self.nb_obj_dims, absolute_coordinates=absolute_coordinates)

        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels in [5, 4, 3]

        vis_mask = np.ones(shape=(width, height), dtype=np.bool)

        grid = Grid(width, height, nb_obj_dims=channels)
        for i in range(width):
            for j in range(height):
                if len(array[i, j]) == 3:
                    type_idx, color_idx, state = array[i, j]
                else:
                    type_idx, color_idx, state, orient = array[i, j]

                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

        return grid, vis_mask

    def process_vis(grid, agent_pos):
        # mask = np.zeros(shape=(grid.width, grid.height), dtype=np.bool)
        #
        # mask[agent_pos[0], agent_pos[1]] = True
        #
        # for j in reversed(range(0, grid.height)):
        #     for i in range(0, grid.width-1):
        #         if not mask[i, j]:
        #             continue
        #
        #         cell = grid.get(i, j)
        #         if cell and not cell.see_behind():
        #             continue
        #
        #         mask[i+1, j] = True
        #         if j > 0:
        #             mask[i+1, j-1] = True
        #             mask[i, j-1] = True
        #
        #     for i in reversed(range(1, grid.width)):
        #         if not mask[i, j]:
        #             continue
        #
        #         cell = grid.get(i, j)
        #         if cell and not cell.see_behind():
        #             continue
        #
        #         mask[i-1, j] = True
        #         if j > 0:
        #             mask[i-1, j-1] = True
        #             mask[i, j-1] = True

        mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)
        # handle frontal occlusions

        # 45 deg
        for j in reversed(range(0, agent_pos[1]+1)):
            dy = abs(agent_pos[1] - j)

            # in front of the agent
            i = agent_pos[0]
            cell = grid.get(i, j)
            if (cell and not cell.see_behind()) or mask[i, j] == False:

                if j < agent_pos[1] and j > 0:
                    # 45 deg
                    mask[i-1,j-1] = False
                    mask[i,j-1] = False
                    mask[i+1,j-1] = False

            # agent -> to the left
            for i in reversed(range(1, agent_pos[0])):
                dx = abs(agent_pos[0] - i)
                cell = grid.get(i, j)

                if (cell and not cell.see_behind()) or mask[i,j] == False:
                    # angle
                    if dx >= dy:
                        mask[i - 1, j] = False

                    if j > 0:
                        mask[i - 1, j - 1] = False
                        if dy >= dx:
                            mask[i, j - 1] = False

            # agent -> to the right
            for i in range(agent_pos[0]+1, grid.width-1):
                dx = abs(agent_pos[0] - i)
                cell = grid.get(i, j)

                if (cell and not cell.see_behind()) or mask[i,j] == False:
                    # angle
                    if dx >= dy:
                        mask[i + 1, j] = False

                    if j > 0:
                        mask[i + 1, j - 1] = False
                        if dy >= dx:
                            mask[i, j - 1] = False

            # for i in range(0, grid.width):
            #     cell = grid.get(i, j)
            #     if (cell and not cell.see_behind()) or mask[i,j] == False:
            #         mask[i, j-1] = False

        # grid
        # for j in reversed(range(0, agent_pos[1]+1)):
        #
        #     i = agent_pos[0]
        #     cell = grid.get(i, j)
        #     if (cell and not cell.see_behind()) or mask[i, j] == False:
        #         if j < agent_pos[1]:
        #             # grid
        #             mask[i,j-1] = False
        #
        #     for i in reversed(range(1, agent_pos[0])):
        #         # agent -> to the left
        #         cell = grid.get(i, j)
        #         if (cell and not cell.see_behind()) or mask[i,j] == False:
        #             # grid
        #             mask[i-1, j] = False
        #             if j < agent_pos[1] and j > 0:
        #                 mask[i, j-1] = False
        #
        #     for i in range(agent_pos[0]+1, grid.width-1):
        #         # agent -> to the right
        #         cell = grid.get(i, j)
        #         if (cell and not cell.see_behind()) or mask[i,j] == False:
        #             # grid
        #             mask[i+1, j] = False
        #             if j < agent_pos[1] and j > 0:
        #                 mask[i, j-1] = False

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5

        # Done completing task
        done = 6

    def __init__(
        self,
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        see_through_walls=False,
        full_obs=False,
        seed=None,
        agent_view_size=7,
        actions=None,
        action_space=None,
        add_npc_direction=False,
        add_npc_point_direction=False,
        add_npc_last_prim_action=False,
        reward_diminish_factor=0.9,
        egocentric_observation=True,
    ):

        # sanity check params for SocialAI experiments
        if "SocialAI" in type(self).__name__:
            assert egocentric_observation
            assert grid_size == 10
            assert not see_through_walls
            assert max_steps == 80
            assert agent_view_size == 7
            assert not full_obs
            assert add_npc_direction and add_npc_point_direction and add_npc_last_prim_action

        self.egocentric_observation = egocentric_observation

        if hasattr(self, "lever_active_steps"):
            assert self.lever_active_steps == 10

        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        if actions:
            self.actions = actions
        else:
            self.actions = MiniGridEnv.Actions

        # Actions are discrete integer values
        if action_space:
            self.action_space = action_space
        else:
            self.action_space = spaces.MultiDiscrete([len(self.actions)])

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.agent_view_size = agent_view_size

        # Number of object dimensions (i.e. number of channels in symbolic image)
        self.add_npc_direction = add_npc_direction
        self.add_npc_point_direction = add_npc_point_direction
        self.add_npc_last_prim_action = add_npc_last_prim_action
        self.nb_obj_dims = 3 + 2*bool(not self.egocentric_observation) + int(self.add_npc_direction) + int(self.add_npc_point_direction) + int(self.add_npc_last_prim_action)

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, self.nb_obj_dims),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls
        self.full_obs = full_obs

        self.reward_diminish_factor = reward_diminish_factor

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs(full_obs=self.full_obs)
        return obs

    def reset_with_info(self, *args, **kwargs):
        obs = self.reset(*args, **kwargs)
        info = self.generate_info(done=False, reward=0)
        return obs, info

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode('utf8'))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def is_near(self, pos1, pos2):
        ax, ay = pos1
        wx, wy = pos2
        if (ax == wx and abs(ay - wy) == 1) or (ay == wy and abs(ax - wx) == 1):
            return True
        return False

    def get_cell(self, x, y):
        return self.grid.get(x, y)

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall'          : 'W',
            'floor'         : 'F',
            'door'          : 'D',
            'key'           : 'K',
            'ball'          : 'A',
            'box'           : 'B',
            'goal'          : 'G',
            'lava'          : 'V',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1 - self.reward_diminish_factor * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """
        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def find_loc(self,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf,
        reject_agent_pos=True,
        reject_taken_pos=True
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')
            if num_tries % 10000 == 0 and num_tries > 0:
                warnings.warn("num_tries = {}. This is probably an infinite loop. {}".format(num_tries, get_traceback()))
                # warnings.warn("num_tries = {}. This is probably an infinite loop.".format(num_tries))
                exit()
                break

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if reject_taken_pos:
                if self.grid.get(*pos) != None:
                    continue

            # Don't place the object where the agent is
            if reject_agent_pos and np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        return pos

    def place_obj(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        # if top is None:
        #     top = (0, 0)
        # else:
        #     top = (max(top[0], 0), max(top[1], 0))
        #
        # if size is None:
        #     size = (self.grid.width, self.grid.height)
        #
        # num_tries = 0
        #
        # while True:
        #     # This is to handle with rare cases where rejection sampling
        #     # gets stuck in an infinite loop
        #     if num_tries > max_tries:
        #         raise RecursionError('rejection sampling failed in place_obj')
        #
        #     num_tries += 1
        #
        #     pos = np.array((
        #         self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
        #         self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
        #     ))
        #
        #     # Don't place the object on top of another object
        #     if self.grid.get(*pos) != None:
        #         continue
        #
        #     # Don't place the object where the agent is
        #     if np.array_equal(pos, self.agent_pos):
        #         continue
        #
        #     # Check if there is a filtering criterion
        #     if reject_fn and reject_fn(self, pos):
        #         continue
        #
        #     break
        #
        # self.grid.set(*pos, obj)
        #
        # if obj is not None:
        #     obj.init_pos = pos
        #     obj.cur_pos = pos
        #
        # return pos

        pos = self.find_loc(
            top=top,
            size=size,
            reject_fn=reject_fn,
            max_tries=max_tries
        )

        if obj is None:
            self.grid.set(*pos, obj)
        else:
            self.put_obj_np(obj, pos)

        return pos

    def put_obj_np(self, obj, pos):
        """
        Put an object at a specific position in the grid
        """

        assert isinstance(pos, np.ndarray)

        i, j = pos

        cell = self.grid.get(i, j)
        if cell is not None:
            raise ValueError("trying to put {} on {}".format(obj, cell))

        self.grid.set(i, j, obj)
        obj.init_pos = np.array((i, j))
        obj.cur_pos = np.array((i, j))

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """
        warnings.warn(
            "This function is kept for backwards compatiblity with minigrid. It is recommended to use put_object_np()."
        )
        raise DeprecationWarning("Deprecated use put_obj_np. (or remove this Warning)")

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
        self,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = None
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        return pos

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """
        assert self.agent_dir >= 0 and self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.dir_vec

    @property
    def back_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos - self.dir_vec

    @property
    def right_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.right_vec

    @property
    def left_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos - self.right_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.agent_pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.agent_view_size
        hs = self.agent_view_size // 2
        tx = ax + (dx * (sz-1)) - (rx * hs)
        ty = ay + (dy * (sz-1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx*lx + ry*ly)
        vy = -(dx*lx + dy*ly)

        return vx, vy

    def get_view_exts(self, dir=None, view_size=None, pos=None):
        """
        Get the extents of the square set of tiles visible to the agent (or to an npc if specified
        Note: the bottom extent indices are not included in the set
        """

        # by default compute view exts for agent
        if (dir is None) and (view_size is None) and (pos is None):
            dir = self.agent_dir
            view_size = self.agent_view_size
            pos = self.agent_pos

        # Facing right
        if dir == 0:
            topX = pos[0]
            topY = pos[1] - view_size // 2
        # Facing down
        elif dir == 1:
            topX = pos[0] - view_size // 2
            topY = pos[1]
        # Facing left
        elif dir == 2:
            topX = pos[0] - view_size + 1
            topY = pos[1] - view_size // 2
        # Facing up
        elif dir == 3:
            topX = pos[0] - view_size // 2
            topY = pos[1] - view_size + 1
        else:
            assert False, "invalid agent direction: {}".format(dir)

        botX = topX + view_size
        botY = topY + view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.agent_view_size or vy >= self.agent_view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def agent_sees(self, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = self.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates
        assert not self.full_obs, "agent sees function not implemented with full_obs"
        obs = self.gen_obs()
        obs_grid, _ = Grid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell != None and fwd_cell.can_push():
                fwd_cell.push(push_dir=self.agent_dir, pusher="agent")

            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif hasattr(self.actions, "pickup") and action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif hasattr(self.actions, "drop") and action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        elif action in map(int, self.actions):
            # action that was added in an inheriting class (ex. talk action)
            pass

        elif np.isnan(action):
            # action skip
            pass

        else:
            assert False, f"unknown action {action}"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs(full_obs=self.full_obs)

        info = self.generate_info(done, reward)

        return obs, reward, done, info

    def generate_info(self, done, reward):

        success = done and reward > 0

        info = {"success": success}

        gen_extra_info_dict = self.gen_extra_info()  # add stuff needed for textual observations here

        assert not any(item in info for item in gen_extra_info_dict), "Duplicate keys found with gen_extra_info"

        info = {
            **info,
            **gen_extra_info_dict,
        }
        return info

    def gen_extra_info(self):
        grid, vis_mask = self.gen_obs_grid()
        carrying = self.carrying
        agent_pos_vx, agent_pos_vy = self.get_view_coords(self.agent_pos[0], self.agent_pos[1])
        npc_actions_dict = SocialAINPCActionsDict

        extra_info = {
            "image": grid.encode(vis_mask),
            "vis_mask": vis_mask,
            "carrying": carrying,
            "agent_pos_vx": agent_pos_vx,
            "agent_pos_vy": agent_pos_vy,
            "npc_actions_dict": npc_actions_dict
        }
        return extra_info

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def add_agent_to_grid(self, image):
        """
        Add agent to symbolic pixel image, used only for full observation
        """
        ax, ay = self.agent_pos
        image[ax,ay] = [9,9,9,self.agent_dir]  # could be made cleaner by creating an Agent_id (here we use Lava_id)
        return image

    def gen_obs(self, full_obs=False):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        Fully observable view can be returned when full_obs is set to True
        """
        if full_obs:
            image = self.add_agent_to_grid(self.grid.encode())

        else:
            grid, vis_mask = self.gen_obs_grid()

            # Encode the partially observable view into a numpy array
            image = grid.encode(vis_mask, absolute_coordinates=not self.egocentric_observation)

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'direction': self.agent_dir,
            'mission': self.mission
        }

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask
        )

        return img

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS, mask_unobserved=False):
        """
        Render the whole-grid human view
        """
        if mode == 'human' and close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=np.bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=highlight_mask if highlight else None,
            mask_unobserved=mask_unobserved
        )
        if mode == 'human':
            # self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def get_mission(self):
        return self.mission

    def close(self):
        if self.window:
            self.window.close()
        return

    def gen_text_obs(self):
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        # (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)
        # State, 0: open, 1: closed, 2: locked
        IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))
        IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

        list_textual_descriptions = []

        if self.carrying is not None:
            list_textual_descriptions.append("You carry a {} {}".format(self.carrying.color, self.carrying.type))

        agent_pos_vx, agent_pos_vy = self.get_view_coords(self.agent_pos[0], self.agent_pos[1])

        view_field_dictionary = dict()

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j][0] != 0 and image[i][j][0] != 1 and image[i][j][0] != 2:
                    if i not in view_field_dictionary.keys():
                        view_field_dictionary[i] = dict()
                        view_field_dictionary[i][j] = image[i][j]
                    else:
                        view_field_dictionary[i][j] = image[i][j]

        # Find the wall if any
        #  We describe a wall only if there is no objects between the agent and the wall in straight line

        # Find wall in front
        add_wall_descr = False
        if add_wall_descr:
            j = agent_pos_vy - 1
            object_seen = False
            while j >= 0 and not object_seen:
                if image[agent_pos_vx][j][0] != 0 and image[agent_pos_vx][j][0] != 1:
                    if image[agent_pos_vx][j][0] == 2:
                        list_textual_descriptions.append(
                            f"A wall is {agent_pos_vy - j} steps in front of you. \n")  # forward
                        object_seen = True
                    else:
                        object_seen = True
                j -= 1
            # Find wall left
            i = agent_pos_vx - 1
            object_seen = False
            while i >= 0 and not object_seen:
                if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
                    if image[i][agent_pos_vy][0] == 2:
                        list_textual_descriptions.append(
                            f"A wall is {agent_pos_vx - i} steps to the left. \n")  # left
                        object_seen = True
                    else:
                        object_seen = True
                i -= 1
            # Find wall right
            i = agent_pos_vx + 1
            object_seen = False
            while i < image.shape[0] and not object_seen:
                if image[i][agent_pos_vy][0] != 0 and image[i][agent_pos_vy][0] != 1:
                    if image[i][agent_pos_vy][0] == 2:
                        list_textual_descriptions.append(
                            f"A wall is {i - agent_pos_vx} steps to the right. \n")  # right
                        object_seen = True
                    else:
                        object_seen = True
                i += 1

        # list_textual_descriptions.append("You see the following objects: ")
        # returns the position of seen objects relative to you
        for i in view_field_dictionary.keys():
            for j in view_field_dictionary[i].keys():
                if i != agent_pos_vx or j != agent_pos_vy:
                    object = view_field_dictionary[i][j]

                    front_dist = agent_pos_vy - j
                    left_right_dist = i-agent_pos_vx

                    loc_descr = ""
                    if front_dist == 1 and left_right_dist == 0:
                        loc_descr += "Right in front of you "

                    elif left_right_dist == 1 and front_dist == 0:
                        loc_descr += "Just to the right of you"

                    elif left_right_dist == -1 and front_dist == 0:
                        loc_descr += "Just to the left of you"

                    else:
                        front_str = str(front_dist)+" steps in front of you " if front_dist > 0 else ""

                        loc_descr += front_str

                        suff = "s" if abs(left_right_dist) > 0 else ""
                        and_ = "and" if loc_descr != "" else ""

                        if left_right_dist < 0:
                            left_right_str = f"{and_} {-left_right_dist} step{suff} to the left"
                            loc_descr += left_right_str

                        elif left_right_dist > 0:
                            left_right_str = f"{and_} {left_right_dist} step{suff} to the right"
                            loc_descr += left_right_str

                        else:
                            left_right_str = ""
                            loc_descr += left_right_str

                    loc_descr += f" there is a "
                    
                    obj_type = IDX_TO_OBJECT[object[0]]
                    if obj_type == "npc":
                        IDX_TO_STATE = {0: 'friendly', 1: 'antagonistic'}

                        description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} peer. "

                        # gaze
                        gaze_dir = {
                            0: "towards you",
                            1: "to the left of you",
                            2: "in the same direction as you",
                            3: "to the right of you",
                        }
                        description += f"It is looking {gaze_dir[object[3]]}. "

                        # point
                        point_dir = {
                            0: "towards you",
                            1: "to the left of you",
                            2: "in the same direction as you",
                            3: "to the right of you",
                        }

                        if object[4] != 255:
                            description += f"It is pointing {point_dir[object[4]]}. "

                        # last action
                        last_action = {v: k for k, v in SocialAINPCActionsDict.items()}[object[5]]


                        last_action = {
                            "go_forward": "foward",
                            "rotate_left": "turn left",
                            "rotate_right": "turn right",
                            "toggle_action": "toggle",
                            "point_stop_point": "stop pointing",
                            "point_E": "",
                            "point_S": "",
                            "point_W": "",
                            "point_N": "",
                            "stop_point": "stop pointing",
                            "no_op": ""
                        }[last_action]

                        if last_action not in ["no_op", ""]:
                            description += f"It's last action is {last_action}. "

                    elif obj_type in ["switch", "apple", "generatorplatform", "marble", "marbletee", "fence"]:

                        if obj_type == "switch":
                            # assumes that Switch.no_light == True
                            assert object[-1] == 0

                        description = f"{IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                        assert object[2:].mean() == 0

                    elif obj_type == "lockablebox":
                        IDX_TO_STATE = {0: 'open', 1: 'closed', 2: 'locked'}
                        description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                        assert object[3:].mean() == 0

                    elif obj_type == "applegenerator":
                        IDX_TO_STATE = {1: 'square', 2: 'round'}
                        description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                        assert object[3:].mean() == 0

                    elif obj_type == "remotedoor":
                        IDX_TO_STATE = {0: 'open', 1: 'closed'}
                        description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                        assert object[3:].mean() == 0

                    elif obj_type == "door":
                        IDX_TO_STATE = {0: 'open', 1: 'closed', 2: 'locked'}
                        description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} "
                        assert object[3:].mean() == 0

                    elif obj_type == "lever":
                        IDX_TO_STATE = {1: 'activated', 0: 'unactivated'}
                        if object[3] == 255:
                            countdown_txt = ""
                        else:
                            countdown_txt = f"with {object[3]} timesteps left. "

                        description = f"{IDX_TO_STATE[object[2]]} {IDX_TO_COLOR[object[1]]} {IDX_TO_OBJECT[object[0]]} {countdown_txt}"

                        assert object[4:].mean() == 0
                    else:
                        raise ValueError(f"Undefined object type {obj_type}")

                    full_destr = loc_descr + description + "\n"

                    list_textual_descriptions.append(full_destr)

        if len(list_textual_descriptions) == 0:
            list_textual_descriptions.append("\n")

        return {'descriptions': list_textual_descriptions}

class MultiModalMiniGridEnv(MiniGridEnv):

    grammar = None

    def reset(self, *args, **kwargs):
        obs = super().reset()
        self.append_existing_utterance_to_history()
        obs = self.add_utterance_to_observation(obs)
        self.reset_utterance()
        return obs

    def append_existing_utterance_to_history(self):
        if self.utterance != self.empty_symbol:
            if self.utterance.startswith(self.empty_symbol):
                self.utterance_history += self.utterance[len(self.empty_symbol):]
            else:
                assert self.utterance == self.beginning_string
                self.utterance_history += self.utterance

    def add_utterance_to_observation(self, obs):
        obs["utterance"] = self.utterance
        obs["utterance_history"] = self.utterance_history
        obs["mission"] = "Hidden"
        return obs

    def reset_utterance(self):
        # set utterance to empty indicator
        self.utterance = self.empty_symbol

    def render(self, *args, show_dialogue=True, **kwargs):

        obs = super().render(*args, **kwargs)

        if args and args[0] == 'human':
            # draw text to the side of the image
            self.window.clear_text()  # erase previous text
            if show_dialogue:
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

    def add_obstacles(self):
        self.obstacles = self.parameters.get("Obstacles", "No") if self.parameters else "No"

        if self.obstacles != "No":
            n_stumps_range = {
                "A_bit": (1, 2),
                "Medium": (3, 4),
                "A_lot": (5, 6),
            }[self.obstacles]

            n_stumps = random.randint(*n_stumps_range)

            for _ in range(n_stumps):
                self.wall_start_x = self._rand_int(1, self.current_width - 2)
                self.wall_start_y = self._rand_int(1, self.current_height - 2)
                if random.choice([True, False]):
                    self.grid.horz_wall(
                        x=self.wall_start_x,
                        y=self.wall_start_y,
                        length=1
                    )
                else:
                    self.grid.horz_wall(
                        x=self.wall_start_x,
                        y=self.wall_start_y,
                        length=1
                    )
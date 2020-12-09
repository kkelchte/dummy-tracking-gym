import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding, EzPickle
from collections import namedtuple
from math import sqrt

import pyglet
from skimage import draw
import matplotlib.pyplot as plt

pyglet.options["debug_gl"] = False
from pyglet import gl

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 500
WINDOW_H = 500
HUNT = True
RUN = True
PLAYFIELD = 500
SQUARE_SIZE = 100
SPEED = 5  # number of units moved each step
MIN_START_DISTANCE = 60  # two squares should be at least this far apart
FINAL_DISTANCE = 100  # the tracker has caught the target if the distance is less than FINAL_DISTANCE
FINAL_IOU = 0.9  # the tracker has caught the target if the IoU is more than FINAL_IOU
MAX_DURATION = 300  # max number of time steps, if max is reached fleeing agent wins
MAX_DIST = sqrt((WINDOW_H-SQUARE_SIZE)**2+(WINDOW_W-SQUARE_SIZE)**2)


def distance(position_a, position_b):
    return np.sqrt((position_a[0] - position_b[0])**2 +
                   (position_a[1] - position_b[1])**2)


def intersection_over_union(position_a, position_b):
    square = namedtuple('square', 'xmin ymin xmax ymax')

    square_a = square(position_a[0] - SQUARE_SIZE // 2, position_a[1] - SQUARE_SIZE // 2,
                      position_a[0] + SQUARE_SIZE // 2, position_a[1] + SQUARE_SIZE // 2)
    square_b = square(position_b[0] - SQUARE_SIZE // 2, position_b[1] - SQUARE_SIZE // 2,
                      position_b[0] + SQUARE_SIZE // 2, position_b[1] + SQUARE_SIZE // 2)

    dx = min(square_a.xmax, square_b.xmax) - max(square_a.xmin, square_b.xmin)
    dy = min(square_a.ymax, square_b.ymax) - max(square_a.ymin, square_b.ymin)
    if (dx >= 0) and (dy >= 0):
        intersection = dx * dy
    else:
        intersection = 0

    union = 2*SQUARE_SIZE**2 - intersection

    return intersection / union


class DummyTrackingEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
    }

    def __init__(self, random_location: bool = True):
        EzPickle.__init__(self)
        self.random_location = random_location
        self.time_steps = 0
        self.viewer = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = True
        center = PLAYFIELD//2
        self.square_zero_position = (center - SQUARE_SIZE // 2, center + SQUARE_SIZE // 2)

        self.square_one_position = self.square_zero_position[:]
        while distance(self.square_zero_position, self.square_one_position) < MIN_START_DISTANCE:
            self.square_one_position = (np.random.randint(0 + SQUARE_SIZE//2, PLAYFIELD - SQUARE_SIZE//2),
                                         np.random.randint(0 + SQUARE_SIZE//2, PLAYFIELD - SQUARE_SIZE//2))

        self.action_space = spaces.Box(np.array([-1, -1, -1, -1]), np.array([+1, +1, +1, +1]), dtype=np.float32)
        self.observation_space = spaces.Box(0, PLAYFIELD, shape=(4, 1), dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        def apply_action(p, a):
            return (np.clip(p[0] + SPEED * a[0], 0 + SQUARE_SIZE//2, PLAYFIELD - SQUARE_SIZE//2).astype(np.int),
                    np.clip(p[1] + SPEED * a[1], 0 + SQUARE_SIZE//2, PLAYFIELD - SQUARE_SIZE//2).astype(np.int))

        self.square_zero_position = apply_action(self.square_zero_position, action[:2])
        self.square_one_position = apply_action(self.square_one_position, action[2:])
        self.time_steps += 1
        iou = intersection_over_union(self.square_one_position, self.square_zero_position)
        dist = distance(self.square_one_position, self.square_zero_position)
        r = 9*iou + (MAX_DIST-dist)/MAX_DIST
        if self.time_steps > MAX_DURATION != -1:
            d = 1
        else:
            d = 0
        frame = self.get_tiny_frame()
        return np.asarray([*self.square_zero_position, *self.square_one_position], dtype=np.float) / PLAYFIELD, r, \
               d, {'frame': frame}

    def get_tiny_frame(self):
        """Create tiny frame 5x smaller to provide back at each step for visualisation"""
        scale = 5
        frame = np.zeros((WINDOW_H//scale, WINDOW_W//scale)).astype(np.uint8)

        frame[
            self.square_zero_position[0] // scale - SQUARE_SIZE // (2 * scale):
            self.square_zero_position[0] // scale + SQUARE_SIZE // (2 * scale),
            self.square_zero_position[1] // scale - SQUARE_SIZE // (2 * scale):
            self.square_zero_position[1] // scale + SQUARE_SIZE // (2 * scale)
        ] = 255
        rr, cc = draw.circle_perimeter(self.square_zero_position[0] // scale,
                                       self.square_zero_position[1] // scale,
                                       radius=SQUARE_SIZE // (4 * scale),
                                       shape=frame.shape)
        frame[rr, cc] = 0
        frame[
            self.square_one_position[0] // scale - SQUARE_SIZE // (2 * scale):
            self.square_one_position[0] // scale + SQUARE_SIZE // (2 * scale),
            self.square_one_position[1] // scale - SQUARE_SIZE // (2 * scale):
            self.square_one_position[1] // scale + SQUARE_SIZE // (2 * scale)
        ] = 255
        rr, cc = draw.line(self.square_one_position[0] // scale,
                           self.square_one_position[1] // scale - SQUARE_SIZE // (4 * scale),
                           self.square_one_position[0] // scale,
                           self.square_one_position[1] // scale + SQUARE_SIZE // (4 * scale))
        frame[rr, cc] = 0

        # rotate frame to align with pyglet view
        return np.rot90(frame)

    def reset(self):
        self.time_steps = 0
        center = PLAYFIELD//2
        if self.random_location:
            def sample_random_location():
                return (np.random.randint(0 + SQUARE_SIZE // 2, PLAYFIELD - SQUARE_SIZE // 2),
                        np.random.randint(0 + SQUARE_SIZE // 2, PLAYFIELD - SQUARE_SIZE // 2))
            self.square_zero_position = sample_random_location()
            self.square_one_position = sample_random_location()
            while distance(self.square_zero_position, self.square_one_position) < MIN_START_DISTANCE:
                self.square_zero_position = sample_random_location()
                self.square_one_position = sample_random_location()
        else:
            self.square_zero_position = (center + SQUARE_SIZE,
                                         center + SQUARE_SIZE)

            self.square_one_position = (center - SQUARE_SIZE,
                                        center - SQUARE_SIZE)

        return np.asarray([*self.square_zero_position, *self.square_one_position], dtype=np.float) / PLAYFIELD

    def render_playfield(self):
        colors = [0.4, 0.8, 0.4, 1.0] * 4
        polygons = [+PLAYFIELD,
                    +PLAYFIELD,
                    0,
                    +PLAYFIELD,
                    -PLAYFIELD,
                    0,
                    -PLAYFIELD,
                    -PLAYFIELD,
                    0,
                    -PLAYFIELD,
                    +PLAYFIELD,
                    0]
        vl = pyglet.graphics.vertex_list(
            len(polygons) // 3, ("v3f", polygons), ("c4f", colors)  # gl.GL_QUADS,
        )
        vl.draw(gl.GL_QUADS)

    def render_agents(self):
        def draw_square(position, color):
            x, y = position
            colors = color * 4
            polygons = [x + SQUARE_SIZE//2,
                        y + SQUARE_SIZE//2,
                        0,
                        x + SQUARE_SIZE//2,
                        y - SQUARE_SIZE//2,
                        0,
                        x - SQUARE_SIZE/2,
                        y - SQUARE_SIZE/2,
                        0,
                        x - SQUARE_SIZE/2,
                        y + SQUARE_SIZE/2,
                        0]
            vl = pyglet.graphics.vertex_list(
                len(polygons) // 3, ("v3f", polygons), ("c4f", colors)  # gl.GL_QUADS,
            )
            vl.draw(gl.GL_QUADS)
        draw_square(self.square_zero_position, [0.6, 0.1, 0.1, 1.0])
        draw_square(self.square_one_position, [0.1, 0.1, 0.6, 1.0])

    def render(self, mode='human'):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == "state_pixels":
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_playfield()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_agents()

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(format='RGBA', pitch=4 * 96), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]
        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def get_slow_hunt(state: np.ndarray) -> np.ndarray:
    agent_blue = state[:2]
    agent_red = state[2:]
    difference = agent_blue - agent_red
    difference = np.sign(difference)
    return 0.5 * difference


def get_slow_run(state: np.ndarray) -> np.ndarray:
    agent_blue = state[:2]
    agent_red = state[2:]
    difference = (agent_blue - agent_red)
    for diff in difference:
        if diff == 0:
            difference += (np.random.rand(2) - 0.5)/10
    difference = np.sign(difference)
    return 0.6 * difference


if __name__ == "__main__":
    from pyglet.window import key

    MAX_DURATION = 300
    a = np.array([0.0, 0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = -1.0
        if k == key.RIGHT:
            a[0] = +1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[1] = -1.0
        if k == key.F:
            a[2] = +1.0
        if k == key.S:
            a[2] = -1.0
        if k == key.E:
            a[3] = +1.0
        if k == key.D:
            a[3] = -1.0

    def key_release(k, mod):
        if k == key.LEFT:
            a[0] = 0
        if k == key.RIGHT:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[1] = 0
        if k == key.F:
            a[2] = 0
        if k == key.S:
            a[2] = 0
        if k == key.E:
            a[3] = 0
        if k == key.D:
            a[3] = 0

    env = DummyTrackingEnv()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    isopen = True
    while isopen:
        state = env.reset()
        total_reward = 0.0
        restart = False
        while True:
            if HUNT:
                a[2:] = get_slow_hunt(state)
            if RUN:
                a[:2] = get_slow_run(state)

            state, reward, done, info = env.step(a)
            total_reward += reward
            print(f'state: zero: {state[:2]}, one: {state[2:]} \t reward: {total_reward} \t '
                  f'done: {done} \t info: {info["frame"].shape}')

            #plt.imshow(info['frame'])
            # plt.show()
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()

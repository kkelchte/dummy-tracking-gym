import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding, EzPickle

import pyglet

pyglet.options["debug_gl"] = False
from pyglet import gl

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 500
WINDOW_H = 500
HUNT = False
PLAYFIELD = 500
SQUARE_SIZE = 100
SPEED = 5  # number of units moved each step
MIN_START_DISTANCE = 60  # two squares should be at least this far apart
FINAL_DISTANCE = 20  # the tracker has caught the target if the distance is less than FINAL_DISTANCE
TRACK_REWARD = 100  # reward when tracker wins
MAX_DURATION = 1000  # max number of time steps, if max is reached fleeing agent wins


def distance(position_a, position_b):
    return np.sqrt((position_a[0] - position_b[0])**2 +
                   (position_a[1] - position_b[1])**2)


class DummyTrackingEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
    }

    def __init__(self):
        EzPickle.__init__(self)
        self.time_steps = 0
        self.viewer = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = True
        center = PLAYFIELD//2
        self.red_square_position = (center - SQUARE_SIZE // 2, center + SQUARE_SIZE // 2)

        self.blue_square_position = self.red_square_position[:]
        while distance(self.red_square_position, self.blue_square_position) < MIN_START_DISTANCE:
            self.blue_square_position = (np.random.randint(0 + SQUARE_SIZE//2, PLAYFIELD - SQUARE_SIZE//2),
                                         np.random.randint(0 + SQUARE_SIZE//2, PLAYFIELD - SQUARE_SIZE//2))

        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]), np.array([+1, +1, +1, +1]), dtype=np.float32
        )
        self.observation_space = spaces.Box(0, PLAYFIELD, shape=(4, 1), dtype=np.float32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        def apply_action(p, a):
            return (np.clip(p[0] + SPEED * a[0], 0 + SQUARE_SIZE//2, PLAYFIELD - SQUARE_SIZE//2).astype(np.int),
                    np.clip(p[1] + SPEED * a[1], 0 + SQUARE_SIZE//2, PLAYFIELD - SQUARE_SIZE//2).astype(np.int))

        self.blue_square_position = apply_action(self.blue_square_position, action[:2])
        self.red_square_position = apply_action(self.red_square_position, action[2:])
        self.time_steps += 1
        dis = distance(self.blue_square_position, self.red_square_position)
        if dis < FINAL_DISTANCE:
            d = 1
        elif self.time_steps > MAX_DURATION != -1:
            d = 1
        else:
            d = 0
        frame = self.get_tiny_frame()
        return np.asarray([*self.blue_square_position, *self.red_square_position]), -dis, d, {'frame': frame}

    def get_tiny_frame(self):
        """Create tiny frame 5x smaller to provide back at each step for visualisation"""
        frame = np.zeros((WINDOW_H//5, WINDOW_W//5)).astype(np.uint8)

        frame[self.blue_square_position[0]//5-SQUARE_SIZE//10:self.blue_square_position[0]//5+SQUARE_SIZE//10,
              self.blue_square_position[1]//5-SQUARE_SIZE//10:self.blue_square_position[1]//5+SQUARE_SIZE//10] = 100
        frame[self.red_square_position[0]//5-SQUARE_SIZE//10:self.red_square_position[0]//5+SQUARE_SIZE//10,
              self.red_square_position[1]//5-SQUARE_SIZE//10:self.red_square_position[1]//5+SQUARE_SIZE//10] = 255
        # rotate frame to align with pyglet view
        return np.rot90(frame)

    def reset(self):
        self.time_steps = 0
        center = PLAYFIELD//2
        self.red_square_position = (center - SQUARE_SIZE // 2, center + SQUARE_SIZE // 2)
        #self.blue_square_position = self.red_square_position[:]
        #while distance(self.red_square_position, self.blue_square_position) < MIN_START_DISTANCE:
        #    self.blue_square_position = (np.random.randint(0 + SQUARE_SIZE // 2, PLAYFIELD - SQUARE_SIZE // 2),
        #                                 np.random.randint(0 + SQUARE_SIZE // 2, PLAYFIELD - SQUARE_SIZE // 2))
        self.blue_square_position = (SQUARE_SIZE//2, SQUARE_SIZE//2)

        return np.asarray([*self.blue_square_position, *self.red_square_position])

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
        draw_square(self.blue_square_position, [0.1, 0.1, 0.6, 1.0])
        draw_square(self.red_square_position, [0.6, 0.1, 0.1, 1.0])

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
    return 0.3 * difference


if __name__ == "__main__":
    from pyglet.window import key

    MAX_DURATION = -1
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

            state, reward, done, info = env.step(a)
            print(f'state: blue: {state[:2]}, red: {state[2:]} \t reward: {reward} \t '
                  f'done: {done} \t info: {info["frame"].shape}')
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()

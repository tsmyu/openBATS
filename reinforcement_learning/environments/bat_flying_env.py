import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import path

# from .lidar_bat import *
from .lidar_bat import *


FPS = 60


class BatFlyingEnv(gym.Env):
    """
    Description:
        Bats emit a pulse and receive the echo to calculate the distance and
        the direction of a object. So, they can fly without bumping some
        obstacle, and forage in the dark.

        In this environment, an agent can get the distance and the direction of
        the nearest obstacle when emits a pulse.

    Observation:
        Type: Box(2)
        Num  Observation     Min      Max
        0    echo distance  0        Inf
        1    echo direction -180 deg 180 deg

    Actions:
        Type: Box(3)
        Num   Action
        1     Flying direction
        2     Pulse direction
        3     Emit Pulse

    Reward:
        Reword is 1 for every step take, including the termination step

    Starting State:
        position
        direction
        speed

    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(
            self,
            world_width=6.0,
            world_height=4.5,
            discrete_length=0.01,
            dt=0.005,
            bat=None,
            walls=None,
            goal_area=None,
            max_accel=None,
            max_accel_angle=None,
            max_pulse_angle=None):
        # world settings
        self.world_width = world_width
        self.world_height = world_height
        self.discrete_length = discrete_length
        # self.dt = 0.01  # [s]
        self.dt = 0.005  # [s]
        # time from emitting pulse [s]
        self.spend_time_from_pulse = 0.1
        self.min_IPI = 0.1

        self.lower_bound_freq_emit_pulse = 0.3

        self.flying_angle_reward = -10.0
        self.pulse_reward = -0.0001
        self.pulse_angle_reward = -0.0001
        self.bump_reward = -100
        self.low_speed_reward = -100
        self.fliyng_reward = 1
        self.previous_bat_angle = 0

        # walls settings
        margin = 0.01
        p0 = Point(margin, margin)
        p1 = Point(margin, world_height - margin)
        p2 = Point(world_width - margin, world_height - margin)
        p3 = Point(world_width - margin, margin)
        w0 = Segment(p0, p1)
        w1 = Segment(p1, p2)
        w2 = Segment(p2, p3)
        w3 = Segment(p3, p0)
        walls = [w0, w1, w2, w3]
        self.walls = [] if walls is None else walls

        # setting chains
        l = 0.05
        chains_point = ((1.0, 1.8), (3.0, 2.8),
                        (3.0, 1.8), (3.0, 0.8), (5.0, 1.8))
        for c in chains_point:
            c1 = Segment(Point(c[0], c[1]), Point(c[0], c[1]+l))
            self.walls.append(c1)
            c2 = Segment(Point(c[0], c[1]), Point(c[0]+l, c[1]))
            self.walls.append(c2)
            c3 = Segment(Point(c[0]+l, c[1]), Point(c[0]+l, c[1]+l))
            self.walls.append(c3)
            c4 = Segment(Point(c[0], c[1]+l), Point(c[0]+l, c[1]+l))
            self.walls.append(c4)

        # self.wallの最初の4データは周囲の壁のため使用しない
        self.wall_points = []
        for wall in self.walls[4:]:
            self.wall_points.append(
                [int(wall.p0.x / self.discrete_length), int(wall.p0.y / self.discrete_length)])
            self.wall_points.append(
                [int(wall.p1.x / self.discrete_length), int(wall.p1.y / self.discrete_length)])

        # self.goal_area = () if goal_area is None else goal_area
        self.max_flying_angle = math.pi / 18  # [rad]
        # self.max_pulse_angle = math.pi / 4  # [rad]
        self.straight_angle = math.pi * 0  # [rad]

        # env settings
        self.action_low = np.array([-1.0, -1.0, 0])
        self.action_high = np.ones(3)
        self.action_space = spaces.Box(
            self.action_low,
            self.action_high,
            dtype=np.float32)

        # bat settings
        self.seed()
        if bat is None:
            self._reset_bat()
        else:
            self.bat = bat

        # observation
        # TODO : 変更する 12245 shold be change
        # high = np.ones(self.bat.n_memory * 2)
        # self.observation_space = spaces.Box(
        #     -high,
        #     high,
        #     dtype=np.float32)
        # high = np.ones((2, 81, 181), dtype=int)
        # low = np.zeros((2, 81, 181), dtype=int)
        high = np.ones((181), dtype=int)
        low = np.zeros((181), dtype=int)
        self.observation_space = spaces.Box(np.array(
            [low]*self.bat.n_memory), np.array([high]*self.bat.n_memory), dtype=int)

        self.viewer = None
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def point_rotaion(self, position, angle):

        return [position[0]*math.cos(angle) - position[1]*math.sin(angle), position[0]*math.sin(angle) + position[1]*math.cos(angle)]

    def get_square_points(self, bat_position, bat_angle):
        bat_angle = -bat_angle
        bat_wing_span = self.bat.wing_span
        bat_total_length = self.bat.total_length
        # 原点中心での座標を求めた後、コウモリの位置をたす
        p0_origin = [bat_wing_span/2, -bat_total_length/2]
        p1_origin = [bat_wing_span/2, bat_total_length/2]
        p2_origin = [-bat_wing_span/2, bat_total_length/2]
        p3_origin = [-bat_wing_span/2, -bat_total_length/2]

        p0_rotaion = self.point_rotaion(p0_origin, bat_angle)
        p1_rotaion = self.point_rotaion(p1_origin, bat_angle)
        p2_rotaion = self.point_rotaion(p2_origin, bat_angle)
        p3_rotaion = self.point_rotaion(p3_origin, bat_angle)

        bat_x_pos = bat_position.x
        bat_y_pos = bat_position.y
        p0 = [p0_rotaion[0] + bat_x_pos, p0_rotaion[1] + bat_y_pos]
        p1 = [p1_rotaion[0] + bat_x_pos, p1_rotaion[1] + bat_y_pos]
        p2 = [p2_rotaion[0] + bat_x_pos, p2_rotaion[1] + bat_y_pos]
        p3 = [p3_rotaion[0] + bat_x_pos, p3_rotaion[1] + bat_y_pos]

        return [p0, p1, p2, p3]

    def get_distancce(self, bat_position_p0, bat_position_p1):
        distance = np.sqrt(
            (bat_position_p0[0] - bat_position_p1[0])**2 + (bat_position_p0[1] - bat_position_p1[1])**2)

        return distance

    def get_bat_area(self, bat_position_p0, bat_position_p1):
        bat_area = []
        if self.get_distancce(bat_position_p1[0], bat_position_p0[0]) > self.get_distancce(bat_position_p1[0], bat_position_p0[1]):
            return [bat_position_p0[1], bat_position_p1[1], bat_position_p1[2], bat_position_p0[2]]
        elif self.get_distancce(bat_position_p1[0], bat_position_p0[0]) < self.get_distancce(bat_position_p1[0], bat_position_p0[1]):
            return [bat_position_p1[0], bat_position_p0[0], bat_position_p1[3], bat_position_p0[3]]
        else:
            raise ValueError

    def get_flag_in_bat(self, bat_area):
        flag_in_bat = False
        bat_points_list = []
        for i in range(len(bat_area)):
            if i == len(bat_area)-1:
                bat_point_s = bat_area[i]
                bat_point_e = bat_area[0]
            else:
                bat_point_s = bat_area[i]
                bat_point_e = bat_area[i+1]
            bat_points_list.append(
                [bat_point_e[0] / self.discrete_length, bat_point_s[0] / self.discrete_length, bat_point_e[1] / self.discrete_length, bat_point_s[1] / self.discrete_length])
        for wall_p in self.wall_points:
            if flag_in_bat:
                return True
            for bat_point in bat_points_list:
                wall_vec = [wall_p[0] - bat_point[1], wall_p[1] - bat_point[3]]
                bat_vec = [bat_point[0] - bat_point[1],
                           bat_point[2] - bat_point[3]]
                corss_product = np.cross(bat_vec, wall_vec)
                if corss_product < 0:
                    flag_in_bat = True
                elif corss_product >= 0:
                    flag_in_bat = False
                    break

        return False

    def step(self, action):
        '''
        stepが刻まれると飛行・パルス放射・旋回角度による報酬が与えられる
        '''
        action = np.clip(action, self.action_low, self.action_high)
        step_reward = self.fliyng_reward
        done = False
        flying_angle, pulse_angle, pulse_proba = action

        bat_p0 = Point(*self.bat.bat_vec)
        self.bat.move(self.straight_angle)

        # freq emit pulse [0.3, 0.8]
        self.spend_time_from_pulse += self.dt
        if self.spend_time_from_pulse >= self.min_IPI:
            if pulse_proba >= 0.5:
                print("pulse_emit")
                step_reward += self.flying_angle_reward * np.abs(flying_angle)
                self.bat.move(flying_angle * self.max_flying_angle)
                self.bat.emit_pulse(
                    pulse_angle, self.walls)
                self.bat.emit = True
                self.last_pulse_angle = pulse_angle
                step_reward += self.pulse_reward
                step_reward += self.pulse_angle_reward * np.abs(pulse_angle)
                self.spend_time_from_pulse = 0.0
                self._update_observation()
            else:
                self.bat.emit = False
        else:
            self.bat.emit = False

        bat_p1 = Point(*self.bat.bat_vec)
        bat_position_p0 = self.get_square_points(
            bat_p0, self.previous_bat_angle * self.max_flying_angle)
        bat_position_p1 = self.get_square_points(
            bat_p1, flying_angle * self.max_flying_angle)
        bat_area = self.get_bat_area(bat_position_p0, bat_position_p1)
        bat_seg_list = []
        for i in range(len(bat_area)):
            if i == len(bat_area)-1:
                bat_seg_list.append(Segment(
                    Point(bat_area[i][0], bat_area[i][1]), Point(bat_area[0][0], bat_area[0][1])))
            else:
                bat_seg_list.append(Segment(Point(bat_area[i][0], bat_area[i][1]), Point(
                    bat_area[i+1][0], bat_area[i+1][1])))

        for bat_seg in bat_seg_list:
            for w in self.walls:
                c_p = cal_cross_point(bat_seg, w)
                if is_point_in_segment(c_p, w) and is_point_in_segment(c_p, bat_seg):
                    wall_vec = w.p0.unpack() - w.p1.unpack()
                    self.bat.bump(bat_p0.unpack(), wall_vec)
                    step_reward += self.bump_reward
                    done = True
        if not done:
            flag_bat_in = self.get_flag_in_bat(bat_area)
            if flag_bat_in:
                step_reward += self.bump_reward
                done = True

        if np.linalg.norm(self.bat.v_vec) < 1:
            step_reward += self.low_speed_reward
            # done = True

        self.t += self.dt
        if 5 < self.t:
            done = True

        if self.bat.emit:
            self.bat.pulse_count += 1
        if done:
            print("pulse count:", self.bat.pulse_count)

        print(f"state:\n{self.bat.state}")

        #　一個前のbat_angleを更新する
        self.previous_bat_angle = flying_angle

        self.bat.emit = False

        return self.state, step_reward, done, {}

    def reset(self):
        self._reset_bat()
        self._reset_walls()
        self.t = 0.0
        self.spend_time_from_pulse = 0.1
        self.close()
        self._update_observation()
        return self.state

    def _reset_bat(self):
        low = np.array([-math.pi/12, 0.1, 0.1])
        high = np.array([math.pi/12, 4.4, 4.4])
        init_bat_params = self.np_random.uniform(low=low, high=high)
        init_speed = 5
        self.bat = LidarBat(*init_bat_params, init_speed, self.dt)
        self._update_observation()

    def _reset_walls(self):
        pass
        # self.walls = self.walls[:4]
        # p = np.linspace(1.5, 3.5, 3)
        # # p = np.array([1.5, 3])
        # # xs, ys = np.meshgrid(p, p)
        # # xs = xs.ravel() + self.np_random.uniform(-0.3, 0.3, 9)
        # # ys = ys.ravel() + self.np_random.uniform(-0.3, 0.3, 9)
        # # angles = self.np_random.uniform(-math.pi, math.pi, 9)
        # # l = 0.3  # wall length
        # # for x, y, a in zip(xs, ys, angles):
        # #     c, s = (l / 2) * cos_sin(a)
        # #     p0 = Point(x + c, y + s)
        # #     p1 = Point(x - c, y - s)
        # #     self.walls.append(Segment(p0, p1))
        # l = 0.66  # wall length
        # acrilyc_panel0 = Segment(Point(1, 1.5), Point(1, 1.5-l))
        # self.walls.append(acrilyc_panel0)
        # acrilyc_panel1 = Segment(Point(2, 0), Point(2, l))
        # self.walls.append(acrilyc_panel1)
        # acrilyc_panel2 = Segment(Point(3, 1.5), Point(3, 1.5-l))
        # self.walls.append(acrilyc_panel2)

    def _update_observation(self):
        obs = np.copy(self.bat.state)
        obs = np.clip(obs, -1, 1)
        self.state = np.ravel(obs).astype(np.float32)

    def render(self, mode='human', screen_width=1000):
        # whether draw pulse and echo source
        draw_pulse_direction = True
        draw_echo_source = False

        # settings screen
        aspect_ratio = self.world_height / self.world_width
        screen_height = int(aspect_ratio * screen_width)
        scale = screen_width / self.world_width

        # initilize screen
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #r = (self.bat.size * scale) / 2
            # wing = 4 * math.pi / 5  # angle [rad]
            #nose_x, nose_y = r, 0
            #r_x, r_y = r * math.cos(-wing), r * math.sin(-wing)
            #l_x, l_y = r * math.cos(+wing), r * math.sin(+wing)
            # bat_geom = rendering.FilledPolygon([
            #(nose_x, nose_y),
            #(r_x, r_y),
            # (l_x, l_y)])
            # bat_geom.set_color(0, 0, 0)
            fname = path.join(path.dirname(__file__), "bat.PNG")
            bat_geom = rendering.Image(
                fname, self.bat.total_length*scale, self.bat.wing_span*scale)
            self.battrans = rendering.Transform()
            bat_geom.add_attr(self.battrans)
            self.viewer.add_geom(bat_geom)
            self._bat_geom = bat_geom

            for w in self.walls:
                x0, y0, x1, y1 = w.unpack() * scale
                line = rendering.Line((x0, y0), (x1, y1))
                line.linewidth = rendering.LineWidth(10)
                line.set_color(0.5, 0.5, 0.5)
                self.viewer.add_geom(line)

        bat_geom = self._bat_geom
        self.battrans.set_translation(*self.bat.bat_vec * scale)
        self.battrans.set_rotation(self.bat.angle)

        if self.bat.emit == True:
            if draw_pulse_direction == True:
                # pulse_length = 0.5
                # pulse_vec = pulse_length * cos_sin(self.last_pulse_angle)
                for e in self.bat.eyes:
                    pulse_vec = e.vec
                    pulse_vec = rotate_vector(
                        pulse_vec, self.bat.angle) + self.bat.bat_vec
                    x0, y0 = self.bat.bat_vec * scale
                    x1, y1 = pulse_vec * scale
                    line = self.viewer.draw_line([x0, y0], [x1, y1])
                    self.viewer.add_geom(line)

            if draw_echo_source == True:
                radius = 4  # pixel
                echo_source_vec = self.bat.state[0] * self.bat.lidar_length
                echo_source_vec = rotate_vector(
                    echo_source_vec, self.bat.angle) + self.bat.bat_vec
                x, y = echo_source_vec * scale
                echo_source = rendering.make_circle(radius)
                echo_source.set_color(0.9, 0.65, 0.4)
                echotrans = rendering.Transform()
                echo_source.add_attr(echotrans)
                echotrans.set_translation(x, y)
                self.viewer.add_geom(echo_source)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

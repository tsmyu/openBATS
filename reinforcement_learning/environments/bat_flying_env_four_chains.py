import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

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
        'video.frames_per_second' : FPS
    }

    def __init__(
            self,
            world_width=6.0,
            world_height=3.6,
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
        #time from emitting pulse [s]
        self.spend_time_from_pulse = 0.1
        self.min_IPI = 0.1

        self.lower_bound_freq_emit_pulse = 0.3

        self.flying_angle_reward = -10
        self.pulse_reward = -0.001
        self.pulse_angle_reward = -0.001
        self.bump_reward = -100
        self.low_speed_reward = -100
        self.fliyng_reward = 1

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

        # self.goal_area = () if goal_area is None else goal_area
        self.max_flying_angle = math.pi / 6 # [rad]
        self.max_pulse_angle = math.pi / 4 # [rad]

        ## env settings
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
        high = np.ones(self.bat.n_memory * 2)
        self.observation_space = spaces.Box(
            -high,
            high,
            dtype=np.float32)
        
        self.viewer = None
        self.state = None
        self.seed()

        # counter intialize
        self.count = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
   
    def step(self, action):
        action = np.clip(action, self.action_low, self.action_high)
        step_reward = self.fliyng_reward
        done = False
        flying_angle, pulse_angle, pulse_proba = action
        step_reward += self.flying_angle_reward * np.abs(flying_angle)
        bat_p0 = Point(*self.bat.bat_vec)
        self.bat.move(flying_angle * self.max_flying_angle)
        bat_p1 = Point(*self.bat.bat_vec)
        bat_seg = Segment(bat_p0, bat_p1)
        for w in self.walls:
            c_p = cal_cross_point(bat_seg, w)
            if is_point_in_segment(c_p, w) and is_point_in_segment(c_p, bat_seg):
                wall_vec = w.p0.unpack() - w.p1.unpack()
                self.bat.bump(bat_p0.unpack(), wall_vec)
                step_reward += self.bump_reward
                done = True

        #self.bat.emit = False
        # freq emit pulse [0.3, 0.8]
        self.spend_time_from_pulse += self.dt
        if self.spend_time_from_pulse >= self.min_IPI:
            if np.random.rand() < pulse_proba/2 + self.lower_bound_freq_emit_pulse:
                self.bat.emit_pulse(pulse_angle * self.max_pulse_angle, self.walls)
                self.bat.emit = True
                self.last_pulse_angle = pulse_angle
                step_reward += self.pulse_reward
                step_reward += self.pulse_angle_reward * np.abs(pulse_angle)
                self.spend_time_from_pulse = 0.0
            else:
                self.bat.emit = False
        else:
            self.bat.emit = False            

        if np.linalg.norm(self.bat.v_vec) < 1:
            step_reward += self.low_speed_reward
            # done = True

        self.t += self.dt
        if 10 < self.t:
            done = True
        self._update_observation()

        if self.bat.emit:
            self.count+=1
        if done:
            print("pulse count:", self.count)

        return self.state, step_reward, done, {}

    def reset(self):
        self._reset_bat()
        self._reset_walls()
        self.t = 0.0
        self.spend_time_from_pulse = 0.1
        self.close()
        self._update_observation()
        # counter intialize
       
        return self.state

    def _reset_bat(self):
        low = np.array([-math.pi/12, 0.1, 0.1])
        high = np.array([math.pi/12, 5.9, 3.5])
        init_bat_params = self.np_random.uniform(low=low, high=high)
        init_speed = 5
        self.bat = LidarBat(*init_bat_params, init_speed, self.dt)
        self._update_observation()
    

    def _reset_walls(self):
        self.walls = self.walls[:4]
        #p = np.linspace(1.5, 3.5, 3)
        #p = np.array([1.5, 3])
         #xs, ys = np.meshgrid(p, p)
         #xs = xs.ravel() + self.np_random.uniform(-0.3, 0.3, 9)
         #ys = ys.ravel() + self.np_random.uniform(-0.3, 0.3, 9)
         #angles = self.np_random.uniform(-math.pi, math.pi, 9)
         #l = 0.3  # wall length
        #for x, y, a in zip(xs, ys, angles):
         #   c, s = (l / 2) * cos_sin(a)
          #  p0 = Point(x + c, y + s)
           # p1 = Point(x - c, y - s)
           # self.walls.append(Segment(p0, p1))
        l = 0.05 # wall length
        
        


        '''
        chains_point=((0.25,0.75),(1.3,1.2),(1.9,0.7),(2.4,1.35),(3.3,0.45),(4.9,0.5),(5.45,0.95),
                        (0.2,2.2),(0.8,2.2),(3.6,2.2),(4.8,1.8),(5.6,1.8),(2.55,3.3),(5.6,1.95),
                            (0.6,3.6),(1.3,3.1),(2.4,4.15),(3.25,3.75),(4.0,3.95),(4.95,3.0),(5.3,3.65))
        '''
        chains_point=((1.0,1.8),(3.0,2.8),(3.0,1.8),(3.0,0.8),(5.0,1.8))

        

        for c in chains_point:
            c1 = Segment(Point(c[0], c[1]),Point(c[0], c[1]+l))
            self.walls.append(c1)
            c2 = Segment(Point(c[0], c[1]),Point(c[0]+l,c[1]))
            self.walls.append(c2)
            c3 = Segment(Point(c[0]+l, c[1]),Point(c[0]+l, c[1]+l))
            self.walls.append(c3)
            c4 = Segment(Point(c[0], c[1]+l),Point(c[0]+l, c[1]+l))
            self.walls.append(c4)
       
    
    def _update_observation(self):
        obs = np.copy(self.bat.state)
        obs = np.clip(obs, -1, 1)
        self.state = np.ravel(obs).astype(np.float32)

    def render(self, mode='human', screen_width=1000):
        # whether draw pulse and echo source
        draw_pulse_direction = True
        draw_echo_source = True

        # settings screen
        aspect_ratio = self.world_height / self.world_width
        screen_height = int(aspect_ratio * screen_width)
        scale = screen_width / self.world_width

       
        # initilize screen
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            r = (self.bat.size * scale) / 2
            wing =  4*math.pi /5# angle [rad]
            nose_x, nose_y = r, 0
            r_x, r_y = r * math.cos(-wing), r * math.sin(-wing)
            l_x, l_y = r * math.cos(+wing), r * math.sin(+wing)
            bat_geom = rendering.FilledPolygon([
                (nose_x, nose_y),
                (r_x, r_y),
                (l_x, l_y)])
            bat_geom.set_color(0, 0, 0)
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
                
                pulse_length = 0.5
                pulse_vec = pulse_length * cos_sin(self.last_pulse_angle)
                pulse_vec = rotate_vector(
                    pulse_vec, self.bat.angle) + self.bat.bat_vec
                x0, y0 =self.bat.bat_vec*scale
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




        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


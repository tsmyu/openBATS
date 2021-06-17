import os
import math
import subprocess
import numpy as np
import environments.scat as scat
from .ears import Ears


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def unpack(self):
        return np.array([self.x, self.y])


class Segment(object):
    def __init__(self, p0: Point, p1: Point):
        self.p0 = p0
        self.p1 = p1

    def unpack(self):
        return np.array([self.p0.x, self.p0.y, self.p1.x, self.p1.y])


def cos_sin(theta) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)])


def cal_cross_point(s0: Segment, s1: Segment) -> Point:
    x0, y0, x1, y1 = s0.p0.x, s0.p0.y, s0.p1.x, s0.p1.y
    x2, y2, x3, y3 = s1.p0.x, s1.p0.y, s1.p1.x, s1.p1.y
    den = (x3 - x2) * (y1 - y0) - (x1 - x0) * (y3 - y2)
    if den == 0:
        return Point(np.inf, np.inf)

    d1 = (y2 * x3 - x2 * y3)
    d2 = (y0 * x1 - x0 * y1)

    x = (d1 * (x1 - x0) - d2 * (x3 - x2)) / den
    y = (d1 * (y1 - y0) - d2 * (y3 - y2)) / den
    return Point(x, y)


def rotate_vector(v, angle):
    return np.array(
        [[np.cos(angle), -np.sin(angle)],
         [np.sin(angle), np.cos(angle)]]) @ v


def is_point_in_segment(p: Point, s: Segment) -> bool:
    e = 1e-8  # e is small number, for excuse
    x_ok = (min(s.p0.x, s.p1.x) - e <=
            p.x) and (p.x <= max(s.p0.x, s.p1.x) + e)
    y_ok = (min(s.p0.y, s.p1.y) - e <=
            p.y) and (p.y <= max(s.p0.y, s.p1.y) + e)
    return x_ok and y_ok


def convert2vec(v):
    if type(v) == Point:
        v = v.unpack()
    if type(v) == Segment:
        v = v.p1.unpack() - v.p0.unpack()
    return v


def cos_similarity(v0, v1):
    v0 = convert2vec(v0)
    v1 = convert2vec(v1)
    return np.inner(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))


def rotation_direction(v0, v1):
    v0 = convert2vec(v0)
    v1 = convert2vec(v1)
    outer = v0[0] * v1[1] - v1[0] * v0[1]
    return 1 if outer >= 0 else -1


class LidarBat(object):
    def __init__(self, init_angle, init_x, init_y, init_speed, dt):
        self.Ears = Ears()
        self.angle = init_angle
        self.angle_r_ear = -60
        self.angle_l_ear = 60
        self.x = init_x  # [m]
        self.y = init_y  # [m]
        self.bat_vec = np.array([self.x, self.y])
        self.v_x, self.v_y = init_speed * cos_sin(init_angle)  # [m/s]
        self.v_vec = np.array([self.v_x, self.v_y])
        self.dt = dt  # [s]

        self.body_weight = 23e-3  # [kg]
        self.size = 7e-2  # [m]

        self.emit = False

        self.lidar_length = 10
        self.lidar_left_angle = (math.pi / 6) / 2
        self.lidar_right_angle = -(math.pi / 6) / 2
        self.lidar_range = np.array([
            self.lidar_left_angle, self.lidar_right_angle])  # [rad]

        self.pulse = self.create_pulse()

        self.n_memory = 5  # number of states
        # TODO 12245 shold be change
        low = np.zeros((2, 81, 12245), dtype=int)
        self.state = np.array([low for i in range(self.n_memory)])

    def create_pulse(self):
        f1 = 68e3         # start frequency
        f2 = 50e3         # end frequency
        fs = int(1/self.Ears.dt_fdtd)        # sampling freq
        d = 2e-3  # duration d*fsで測定したデータの個数
        time = np.linspace(0, 1, fs, endpoint=False)  # array を生成　0~1までfs*10個の点
        time_sig = time[:int(d*fs)]
        # swav = np.sin((2*np.pi*f1*(f1/f2)**(1/d)**time_sig)/np.log((f1/f2)**(1/d)))#exponential型
        FM2 = np.sin(2*np.pi*(f1*time_sig+(f2-f1)*time_sig**2/2/d))  # linear型
        FM1 = np.sin(2*np.pi*(f1*time_sig+(f2/2-f1/2)
                              * time_sig**2/2/d))/100  # linear型
        FM3 = np.sin(2*np.pi*(f1*time_sig+(3*f2/2-3*f1/2)
                              * time_sig**2/2/d))/100  # linear型
        CF2 = np.sin(2*np.pi*(f1*time_sig))
        CF1 = np.sin(2*np.pi*(f1/2*time_sig))/100
        CF3 = np.sin(2*np.pi*(3*f1/2*time_sig))/100
        # FM or CF
        CF = CF1 + CF2 + CF3
        FM = FM1 + FM2 + FM3
        waves = np.concatenate([CF, FM])
        # waves = FM
        # waves = np.fliplr([waves])[0]

        return waves

    def conv_pulse(self, echoes):

        corr_left = np.correlate(echoes[0], self.pulse, "full")[
            len(self.pulse):]
        corr_right = np.correlate(echoes[1], self.pulse, "full")[
            len(self.pulse):]

        return corr_left, corr_right
    
    def convert_to_fdtdmap(self):
        dl = self.Ears.dl
        bat_vec_fdtd = []
        for bat_vec in self.bat_vec:
            bat_vec_fdtd.append(int(bat_vec/dl))

        return bat_vec_fdtd

    def emit_pulse(self, pulse_angle, obstacle_segments):
        """
        get obsevation (echoes) by using FDTD
        """
        # check if echoes' data are in database
        bat_vec_fdtd = self.convert_to_fdtdmap()
        if self.Ears.check_data_in_database(bat_vec_fdtd):
            pass
        else:
            print(
                f'{bat_vec_fdtd[0]}_{bat_vec_fdtd[1]}.bin is not exist in data base.')
            print("FDTD.exe for sound pressure start")
            subprocess.run(
                f"./environments/Bat2d1.1AI2/WE-FDTD_T.exe {bat_vec_fdtd[0]} {bat_vec_fdtd[1]} 0", shell=True)
            print("FDTD.exe for particle velocity x start")
            subprocess.run(
                f"./environments/Bat2d1.1AI2/WE-FDTD_T.exe {bat_vec_fdtd[0]} {bat_vec_fdtd[1]} 1", shell=True)
            print("FDTD.exe for particle velocity y start")
            subprocess.run(
                f"./environments/Bat2d1.1AI2/WE-FDTD_T.exe {bat_vec_fdtd[0]} {bat_vec_fdtd[1]} 2", shell=True)
        # get echoes impulse response
        echoes = self.Ears.get_echoes(
            bat_vec_fdtd, pulse_angle, self.angle, self.angle_r_ear, self.angle_l_ear)
        # get echoes by conv pulse
        left_echo, right_echo = self.conv_pulse(echoes)

        # get peak times by running cochlear_block of SCAT model
        emit_spike_list, echo_right_spike_list, echo_left_spike_list = scat.run(
            self.pulse, left_echo, right_echo, self.dt_fdtd)
        observation = [echo_right_spike_list, echo_left_spike_list]
        self._update_state(observation)
        return observation

    def _lidar_segments(self, lidar_vec):
        lidar_vec = rotate_vector(lidar_vec, self.angle)
        v_left = rotate_vector(lidar_vec, self.lidar_left_angle)
        v_right = rotate_vector(lidar_vec, self.lidar_right_angle)
        v_left, v_right = v_left + self.bat_vec, v_right + self.bat_vec
        bat_p = Point(*self.bat_vec)
        left_p = Point(*v_left)
        right_p = Point(*v_right)
        return Segment(bat_p, left_p), Segment(bat_p, right_p)

    def _update_state(self, new_observation):
        self.state[1:] = self.state[:-1]
        self.state[0] = new_observation

    def move(self, angle):
        self.v_vec = rotate_vector(self.v_vec, angle)
        self.bat_vec += self.v_vec * self.dt
        self._cal_angle()

    def bump(self, bat_vec, surface_vec, e=1):
        '''
        simulate partially inelastic collisions.
        e: coefficient of restitution
        '''
        T = surface_vec / np.linalg.norm(surface_vec)  # Tangent vector
        N = np.array([-T[1], T[0]])  # Normal vector
        v_T = np.inner(self.v_vec, T) * T
        v_N = np.inner(self.v_vec, N) * N
        self.v_vec = -e * v_N + v_T
        self.bat_vec = bat_vec + self.v_vec * self.dt
        self._cal_angle()

    def _cal_angle(self):
        self.angle = math.atan2(*self.v_vec[::-1])

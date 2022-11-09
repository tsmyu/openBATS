from distutils.log import error
import math
import numpy as np


class Point(object):
    '''
    objectのxy座標を定義
    '''
    def __init__(self, x, y):
        '''
        __init__メソッドはclassを使うには必要なもの
        '''
        self.x = x
        self.y = y

    def unpack(self):
        '''
        xy座標を返す
        '''
        return np.array([self.x, self.y])

class Segment(object):
    '''
    座標の名前
    '''
    def __init__(self, p0: Point, p1: Point):
        self.p0 = p0
        self.p1 = p1

    def unpack(self):
        return np.array([self.p0.x, self.p0.y, self.p1.x, self.p1.y])


def cos_sin(theta) -> np.ndarray:
    '''
    角度から座標を定義
    '''
    return np.array([math.cos(theta), math.sin(theta)])

def cal_cross_point(s0: Segment, s1: Segment) -> Point:
    '''
    2つのセグメントの重なっている点を計算...
    '''
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

def get_flag_dump(bat_position_arr, field_arr) -> bool:

    field_all = bat_position_arr + field_arr
    print(np.max(field_all))
    if np.max(field_all) == 5:
        return True
    else:
        return False

def rotate_vector(v, angle):
    '''
    vを回転
    '''
    return np.array(
        [[np.cos(angle), -np.sin(angle)], 
        [np.sin(angle), np.cos(angle)]]) @ v

def is_point_in_segment(p: Point, s: Segment) -> bool:
    '''
    p.xが適切な値か
    '''
    e = 1e-8  # e is small number, for excuse 
    x_ok = (min(s.p0.x, s.p1.x) - e <= p.x) and (p.x <= max(s.p0.x, s.p1.x) + e)
    y_ok = (min(s.p0.y, s.p1.y) - e <= p.y) and (p.y <= max(s.p0.y, s.p1.y) + e)
    return x_ok and y_ok

def convert2vec(v):
    '''
    v 
    '''
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

class EYE(object):
    def __init__(self, NUM_EYES, i):
        # 進行方向に対して左右30°，181本（NUM_EYES）
        self.OffSetAngle = -math.pi / 6 + i * (math.pi / 6) / (NUM_EYES-1)
        self.SightDistance = 0
        self.obj           = -1
        self.FOV           = 10 # TODO 
        self.vec = self.FOV * cos_sin(self.OffSetAngle)
class LidarBat(object):
    def __init__(self, init_angle, init_x, init_y, init_speed, dt):
        self.angle = init_angle
        self.x = init_x  # [m]
        self.y = init_y  # [m]
        self.bat_vec = np.array([self.x, self.y])
        self.v_x, self.v_y = init_speed * cos_sin(init_angle)  # [m/s]
        self.v_vec = np.array([self.v_x, self.v_y])
        self.dt = dt  # [s]

        self.body_weight = 23e-3 # [kg]
        self.total_length = 15e-2  # [m]
        self.wing_span = 40e-2  # [m]

        NUM_EYES = 181
        self.eyes = [EYE(NUM_EYES, i) for i in range(0, NUM_EYES)]

        self.n_memory = 5  # number of states
        self.state = np.array([[0]*NUM_EYES for i in range(self.n_memory)], dtype=np.float32)
        self.emit = False
        self.pulse_count = 0

        # self.lidar_length = 10
        # self.lidar_left_angle = (math.pi / 6) / 2
        # self.lidar_right_angle = -(math.pi / 6) / 2
        # self.lidar_range = np.array([
        #     self.lidar_left_angle, self.lidar_right_angle])  # [rad]
    
    def detect_points(self, lidar_seg, obstacle_segments):
        detected_points = []
        for s in obstacle_segments:
            c_p  = cal_cross_point(lidar_seg, s)
            is_cross  = (is_point_in_segment(c_p, s) and 
                              is_point_in_segment(c_p, lidar_seg))
            if is_cross:
                detected_points.append(c_p)
        
        return detected_points
    
    def calc_min_detect_length(self, detected_points, e):
        min_length = 1e5
        if len(detected_points) == 0:
            detected_length = e.FOV * 2
        else:
            for p in detected_points:
                detected_vec = p.unpack() - self.bat_vec
                detected_length = np.linalg.norm(detected_vec)
                if min_length > detected_length:
                    min_length = detected_length
        
        return round(min_length / e.FOV, 2)


    def emit_pulse(self, lidar_angle, obstacle_segments):
        obs_length_list = []
        for e in self.eyes:
            lidar_vec = cos_sin(lidar_angle+e.OffSetAngle) * e.FOV
            lidar_seg = self._lidar_segments(lidar_vec)
            detected_points = self.detect_points(lidar_seg, obstacle_segments)
            min_length = self.calc_min_detect_length(detected_points, e)
            obs_length_list.append(min_length)

        observation = obs_length_list
        self._update_state(observation)

        # return observation

    def _lidar_segments(self, lidar_vec):
        lidar_vec = rotate_vector(lidar_vec, self.angle)
        bat_p = Point(*self.bat_vec)
        eye_p = Point(*lidar_vec)

        return Segment(bat_p, eye_p)

    def _update_state(self, new_observation):
        self.state[1:] = self.state[:-1]
        self.state[0] = new_observation
        print(f"new observation: {new_observation}")
        print(f"self.state: {self.state[0]}")

    def move(self, angle):
        self.v_vec = rotate_vector(self.v_vec, angle)
        self.bat_vec += self.v_vec * self.dt
        self._cal_angle()
    
    def bump(self, bat_vec, surface_vec, e=1):
        '''
        simulate partially inelastic collisions.
        e: coefficient of restitution
        '''
        T = surface_vec / np.linalg.norm(surface_vec) # Tangent vector
        N = np.array([-T[1], T[0]])  # Normal vector
        v_T = np.inner(self.v_vec, T) * T
        v_N = np.inner(self.v_vec, N) * N
        self.v_vec = -e * v_N + v_T
        self.bat_vec = bat_vec + self.v_vec * self.dt
        self._cal_angle()

    def _cal_angle(self):
        self.angle = math.atan2(*self.v_vec[::-1])
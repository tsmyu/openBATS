
import os
import json
import math
import numpy as np

components_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
setting_file_path = components_dir + "settings.json"
with open(setting_file_path, "r") as setting_file_obj:
    config_param = json.load(setting_file_obj)
    dx = config_param["resolution"]["distance"]
    dt = config_param["resolution"]["time"]
    sig_freq = config_param["signal"]["frequency"]
    sig_wave_number = config_param["signal"]["wave_number"]
    sig_duration = round(sig_wave_number/2/sig_freq/ dt)
    e_r_distance = int(config_param["ear_point_from_signal"] / 2 / dx)
    sound_speed_list = config_param["soundspeed"]
    emit_dis = sound_speed_list["air"]/sig_freq/2/dx


class Bat:
    def __init__(self):
        self.echo_right_ear = []
        self.echo_left_ear = []
        # cardioid
        self.Vxb_e = 0.0
        self.Vxf_e = 0.0
        self.Vyb_e = 0.0
        self.Vyf_e = 0.0

    def set_position(self, pulse_info):
        self.emit_position_x = pulse_info[0]
        self.emit_position_y = pulse_info[1]
        self.emit_angle = pulse_info[2]
        self.emit_right, self.emit_left = self.__calc_emit_point()
        self.ear_points = self.__calc_receive_point()
        self.echo_right_ear = []
        self.echo_left_ear = []
        # cardioid
        self.Vxb_e = 0.0
        self.Vxf_e = 0.0
        self.Vyb_e = 0.0
        self.Vyf_e = 0.0

        return self.ear_points

    def __calc_emit_point(self):
        right_point = (round(self.emit_position_x - (emit_dis* math.sin(math.radians(self.emit_angle)))),
                       round(self.emit_position_y - (emit_dis * math.cos(math.radians(self.emit_angle)))))
        left_point = (round(self.emit_position_x + (emit_dis * math.sin(math.radians(self.emit_angle)))),
                      round(self.emit_position_y + (emit_dis*math.cos(math.radians(self.emit_angle)))))

        return (right_point, left_point)

    def __calc_receive_point(self):
        right_point = (round(self.emit_position_x - (e_r_distance * math.sin(math.radians(self.emit_angle)))),
                       round(self.emit_position_y - (e_r_distance*math.cos(math.radians(self.emit_angle)))))
        left_point = (round(self.emit_position_x + (e_r_distance * math.sin(math.radians(self.emit_angle)))),
                      round(self.emit_position_y + (e_r_distance*math.cos(math.radians(self.emit_angle)))))

        return (right_point, left_point)

    def emit_pulse(self, i, P2, density):
        if i < sig_duration:
            sig = self.__create_pulse(i)
        else:
            sig = 0.0
        P2[self.emit_right] += sig
        # P2[self.emit_left] += sig
        if i <= 1 and i < sig_duration:
            sig_p = self.__create_pulse(i-1)
            P2 = self.__calc_directivity(sig_p, P2, density)

        return P2

    def __calc_directivity(self, sig_p, p2, density):
        x_t = self.emit_position_x
        y_t = self.emit_position_y
        # cardioid
        p2[self.emit_right[0]+1, self.emit_right[1]] += sig_p
        p2[self.emit_right] -= sig_p

        return p2

    def __create_pulse(self, i):
        # sig_wave = sig_amp * math.sin(2 * math.pi * sig_freq * i * dt)
        # sig = sig_wave * signal.gaussian(sig_duration, std=sigma)[i]
        sig_wave = math.sin(2*math.pi*sig_freq*(i-sig_duration)*dt)/2*math.pi*sig_freq*(i-sig_duration)*dt

        return sig_wave

    def get_echo(self, p1, p2, density):

        (x_r, y_r) = self.ear_points[0]
        (x_l, y_l) = self.ear_points[1]
        # tmp for test
        # (x_r, y_r) = (300, 300)
        # (x_l, y_l) = (400, 300)

        w_x = math.cos(math.radians(self.emit_angle))
        w_y = math.sin(math.radians(self.emit_angle))

        # cardioid
        # right ear
        self.Vxb_e = self.Vxb_e - dt / (density[x_r, y_r] * dx) * \
            (p2[x_r, y_r] - p2[x_r - 1, y_r])
        self.Vxf_e = self.Vxf_e - dt / (density[x_r, y_r] * dx) * \
            (p2[x_r + 1, y_r] - p2[x_r, y_r])
        self.Vyb_e = self.Vyb_e - dt / (density[x_r, y_r] * dx) * \
            (p2[x_r, y_r] - p2[x_r, y_r - 1])

        self.Vyf_e = self.Vyf_e - dt / (density[x_r, y_r] * dx) * \
            (p2[x_r, y_r + 1] - p2[x_r, y_r])
        Ix = p1[x_r, y_r] * (self.Vxf_e + self.Vxb_e)
        Iy = p1[x_r, y_r] * (self.Vyf_e + self.Vyb_e)
        c = (-w_x * Ix - w_y * Iy) / (Ix ** 2 + Iy ** 2) ** 0.5
        r = (1 + c) / 2
        p_right = p1[x_r, y_r] * r
        if np.isnan(p_right):
            p_right = 0
        self.echo_right_ear.append(p_right)
        # self.echo_right_ear.append(p1[x_r, y_r])

        # left ear
        self.Vxb_e = self.Vxb_e - dt / (density[x_l, y_l] * dx) * \
            (p2[x_l, y_l] - p2[x_l - 1, y_l])
        self.Vxf_e = self.Vxf_e - dt / (density[x_l, y_l] * dx) * \
            (p2[x_l + 1, y_l] - p2[x_l, y_l])
        self.Vyb_e = self.Vyb_e - dt / (density[x_l, y_l] * dx) * \
            (p2[x_l, y_l] - p2[x_l, y_l - 1])
        self.Vyf_e = self.Vyf_e - dt / (density[x_l, y_l] * dx) * \
            (p2[x_l, y_l + 1] - p2[x_l, y_l])

        Ix = p1[x_l, y_l] * (self.Vxf_e + self.Vxb_e)
        Iy = p1[x_l, y_l] * (self.Vyf_e + self.Vyb_e)
        c = (-w_x * Ix - w_y * Iy) / (Ix ** 2 + Iy ** 2) ** 0.5
        r = (1 + c) / 2
        p_left = p1[x_l, y_l] * r
        if np.isnan(p_left):
            p_left = 0
        self.echo_left_ear.append(p_left)


import os
import glob
from scipy import signal
import math
import numpy as np
import struct


class Ears:
    def __init__(self):
        self.database_path = os.path.dirname(os.path.abspath(__file__)) + '/Bat2d1.1AI2/'
        cfl = 0.98
        c0 = 340.0
        self.dl = 0.0005
        dt = cfl * dl / c0
        self.dt_fdtd = dt

    def check_data_in_database(self, position):
        # print(self.database_path + f'wave0_x{position[0]}_y{position[1]}.bin')
        if os.path.isfile(self.database_path + f'wave0_x{position[0]}_y{position[1]}.bin'):
            return True
        else:
            return False

    def get_echoes(self, position, pulse_angle, head_angle, rear_angle, lear_angle):
        data_files = self.__get_file_from_database(position)
        bp_data, bxv_data, byv_data = self.__read_echo_data(data_files)
        Nt, p_arr, xv_arr, yv_arr = self.__bin_to_array(bp_data, bxv_data, byv_data)
        echoes = self.__add_directivity(
            p_arr, xv_arr, yv_arr, pulse_angle, head_angle, rear_angle, lear_angle)

        return echoes
    
    def __bin_to_array(self, bp_data, bxv_data, byv_data):
        Nt = struct.unpack_from('i', bp_data)[0]
        p_data_list = []
        xv_data_list  = []
        yv_data_list = []
        # TODO 111 * Nt での読み込み
        for i in range(Nt*111):
            p_data = struct.unpack_from('f', bp_data, 4*(i+1))[0]
            p_data_list.append(p_data)
            xv_data = struct.unpack_from('f', bxv_data, 4*(i+1))[0]
            xv_data_list.append(xv_data)
            yv_data = struct.unpack_from('f', byv_data, 4*(i+1))[0]
            yv_data_list.append(yv_data)
        
        p_arr = np.array(p_data_list).reshape(Nt, 111).T
        xv_arr = np.array(xv_data_list).reshape(Nt, 111).T
        yv_arr = np.array(yv_data_list).reshape(Nt, 111).T
            

        return Nt, p_arr, xv_arr, yv_arr

    def __get_file_from_database(self, position):

        return glob.glob(f'{self.database_path}/wave*_x{position[0]}_y{position[1]}.bin')

    def __read_echo_data(self, data_files):
        for d_file in data_files:
            file_name = os.path.splitext(os.path.basename(d_file))[
                0].split('_')[0]
            with open(d_file, mode='rb') as f:
                if file_name == 'wave0':
                    p_data = f.read()
                elif file_name == 'waveX':
                    xv_data = f.read()
                elif file_name == 'waveY':
                    yv_data = f.read()

        return p_data, xv_data, yv_data
    
    def butter_pass(self, wave, fs, f_pass, f_filter, gpass, gstop):
        fn = fs / 2
        wp = f_pass / fn
        ws = f_filter / fn
        N, Wn = signal.buttord(wp, ws, gpass, gstop)
        b, a = signal.butter(N, Wn, "high")
        y = signal.filtfilt(b, a, wave)
        
        return y

    def high_pass_filter(self, wave):

        f_pass = 300
        f_filter = 1
        gpass = 3
        gstop = 40

        wave_after_filter = self.butter_pass(wave,
                                            1/self.dt_fdtd,
                                            f_pass, f_filter,
                                            gpass, gstop)

        return wave_after_filter

    def __calc_directivity(self, p, head_angle, rear_angle, lear_angle):

        head_angle = np.rad2deg(head_angle)

        Nangle_L = round(head_angle / 10) + 9
        if Nangle_L > 36:
            Nangle_L = Nangle_L - 18
        elif Nangle_L < 0:
            Nangle_L = Nangle_L + 18

        Nangle_R = round(head_angle / 10) + 27
        if Nangle_R > 36:
            Nangle_R = Nangle_R - 18
        elif Nangle_R < 0:
            Nangle_R = Nangle_R + 18
        
        Nangle_L = int(Nangle_L)
        Nangle_R = int(Nangle_R)
        print(f'NangleL:{Nangle_L}')
        print(f'NangleR:{Nangle_R}')

        # matlab(p[Nangle_L*3+1, :]) --> python(p[Nangle_L*3, :] )
        left_echo = (p[Nangle_L*3, :] + p[Nangle_L*3+1, :] *
                     math.cos(lear_angle) + p[Nangle_L*3+2] * math.sin(lear_angle)) / 2

        right_echo = (p[Nangle_R*3, :] + p[Nangle_R*3+1, :] *
                      math.cos(rear_angle) + p[Nangle_R*3+2, :] * math.sin(rear_angle)) / 2

        return left_echo, right_echo

    def __add_directivity(self, p_data, xv_data, yv_data,
                          pulse_angle, head_angle, rear_angle, lear_angle):

        p0_list = []
        px_list = []
        py_list = []
        
        for i in range(len(p_data)):
            p0 = self.high_pass_filter(p_data[i])
            p0_list.append(p0)
            px = self.high_pass_filter(xv_data[i])
            px_list.append(px)
            py = self.high_pass_filter(yv_data[i])
            py_list.append(py)
        p0 = np.vstack(p0_list)
        px = np.vstack(px_list)
        py = np.vstack(py_list)

        ps = (p0 + px * math.cos(pulse_angle) +
              py * math.sin(pulse_angle))/2

        left_echo, right_echo = self.__calc_directivity(
            ps, head_angle, rear_angle, lear_angle)

        left_echo = self.high_pass_filter(left_echo)
        right_echo = self.high_pass_filter(right_echo)

        return [left_echo, right_echo]

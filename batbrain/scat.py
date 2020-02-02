import os
import sys
import glob
import numpy as np
from scipy import signal
from scipy.signal import argrelmax
import itertools
import math
import re
import csv
import matplotlib.pyplot as plt
import collections

# simulation parameter
dx = 2.5e-4
dt = 5.0e-7
fs = 1/dt
calc_time = 4.0
# distance between nose and ears [m]
distance_ears = 0.002
velocity_air = 331.5
limit_time = distance_ears / velocity_air


class DataField:
    def __init__(self, input_data, emit_csv_list):
        # threash
        self.threash = 0.05
        self.base_name = input_data
        file_name = os.path.basename(input_data)
        # self.pixel_area = (
        #     int(file_name.split("_")[-3]), int(file_name.split("_")[-2]))
        self.emit_angle = int(
            re.split("[g]", file_name.split("_")[-1].split(".")[0])[-1])
        self.emit_point = (int(os.path.basename(input_data).split("_")[-3]), int(
            os.path.basename(input_data).split("_")[-2]))
        for emit_csv in emit_csv_list:
            emit_name = os.path.basename(emit_csv)
            _emit_angle = int(re.split("[g]", emit_name.split("_")[-1].split(".")[0])[-1])
            if self.emit_angle == _emit_angle:
                emit_data_list = np.genfromtxt(emit_csv, usecols=(
                    0, 1, 2, 3), skip_header=1, skip_footer=1, delimiter=",")
                break

        echo_data_list = np.genfromtxt(input_data, usecols=(
            0, 1, 2, 3), skip_header=1, skip_footer=1, delimiter=",")

        self.time_line = echo_data_list[:, 0]
        self.emit_wave = echo_data_list[:, 1]
        right_wave = echo_data_list[:, 2] - \
            emit_data_list[:, 2][:len(echo_data_list[:, 2])]
        left_wave = echo_data_list[:, 3] - \
            emit_data_list[:, 3][:len(echo_data_list[:, 3])]
        self.echo_without_emit_wave = [right_wave, left_wave]
        self.data_len = 0

    def calc_cochlear_block(self):
        emit_spike_list = []
        echo_right_spike_list = []
        echo_left_spike_list = []
        # IIR filter
        f_emit_list, f_echo_right_list, f_echo_left_list = self.__iir_filtering()

        emit_length = len(f_emit_list[0])
        echo_right_length = len(f_echo_right_list[0])
        echo_left_length = len(f_echo_left_list[0])
        for idx, f_emit in enumerate(f_emit_list):
            # half wave rectification
            fh_emit = self.__half_wave_rectification(f_emit)
            # low pass filter
            fhl_emit = self.__low_pass_filter(fh_emit)
            # detect spike
            emit_point = self.__detect_peak(fhl_emit)
            emit_spike = self.__make_spike(emit_point, emit_length, 1e-6)
            emit_spike_list.append(emit_spike)
            if idx >= 30:
                fig = plt.figure()
                ax1 = fig.add_subplot(511)
                ax1.plot(self.emit_wave)
                ax2 = fig.add_subplot(512)
                ax2.plot(f_emit)
                ax3 = fig.add_subplot(513)
                ax3.plot(fh_emit)
                ax4 = fig.add_subplot(514)
                ax4.plot(fhl_emit)
                ax5 = fig.add_subplot(515)
                ax5.plot(emit_spike)
                plt.show()
        # right echo
        for f_echo_right in f_echo_right_list:
            # half wave rectification
            fh_echo_right = self.__half_wave_rectification(f_echo_right)
            # low pass filter
            fhl_echo_right = self.__low_pass_filter(fh_echo_right)
            # detect spike
            echo_right_point = self.__detect_peak(fhl_echo_right)
            echo_right_spike = self.__make_spike(
                echo_right_point, echo_right_length, 1e-6)
            echo_right_spike_list.append(echo_right_spike)
        # left echo
        for f_echo_left in f_echo_left_list:
            # half wave rectification
            fh_echo_left = self.__half_wave_rectification(f_echo_left)
            # low pass filter
            fhl_echo_left = self.__low_pass_filter(fh_echo_left)
            # detect peak and make spike
            echo_left_point = self.__detect_peak(fhl_echo_left)
            echo_left_spike = self.__make_spike(
                echo_left_point, echo_left_length, 1e-6)
            echo_left_spike_list.append(echo_left_spike)

        return emit_spike_list, echo_right_spike_list, echo_left_spike_list

    def calc_temporal_block(self, emit_spike_list, echo_right_spike_list, echo_left_spike_list):
        dtx = self.__calc_dtx(emit_spike_list)
        print(dtx)
        input()
    
    def __calc_dtx(self, emit_spike_list):
        first_spike = []
        dtx = 0
        tmp_dtx = 0
        for idx, emit_spike in enumerate(emit_spike_list):
            print(f"{idx+20}kHz:{collections.Counter(emit_spike)}")
            for idx, spike in enumerate(emit_spike):
                if spike == 1.0:
                    first_spike.append(idx)
                    break
        print(first_spike)
        print(len(first_spike))
        for i in range(len(first_spike)):
            if i == len(first_spike):
                pass
            elif i == 0:
                dtx = first_spike[i] - first_spike[i + 1]
            else:
                tmp_dtx = first_spike[i] - first_spike[i + 1]
                if dtx != tmp_dtx:
                    print(f"dtx is not match between different frequency.dtx1:{dtx}, dtx2:{tmp_dtx}")
                    input()
        
        return dtx

                

    def __iir_filtering(self):
        """
        calc iir filter
        """
        max_freq = 100e3
        min_freq = 20e3
        bin_num = 81
        bandwidth = 4e3
        f_emit_list = []
        f_echo_right_list = []
        f_echo_left_list = []
        for i in range(bin_num):
            hz = i*1e3
            sos = signal.iirfilter(N=10,
                                   Wn=[min_freq + hz - bandwidth / 2,
                                       min_freq + hz + bandwidth / 2],
                                   btype="bandpass",
                                   analog=False,
                                   ftype="butter",
                                   output="sos",
                                   fs=fs)
            f_emit_wave = signal.sosfiltfilt(sos, self.emit_wave, padlen=0)
            f_echo_wave_right = signal.sosfiltfilt(
                sos, self.echo_without_emit_wave[0])
            f_echo_wave_left = signal.sosfiltfilt(
                sos, self.echo_without_emit_wave[1])
            f_emit_list.append(f_emit_wave)
            f_echo_right_list.append(f_echo_wave_right)
            f_echo_left_list.append(f_echo_wave_left)

        return f_emit_list, f_echo_right_list, f_echo_left_list

    def __half_wave_rectification(self, wave):

        return np.where(wave <= 0, 0, wave)

    def __low_pass_filter(self, wave):
        sos = signal.iirfilter(N=1,
                               Wn=3000,
                               btype="lowpass",
                               analog=False,
                               ftype="butter",
                               output="sos",
                               fs=fs)

        l_wave = signal.sosfiltfilt(sos, wave, padlen=0)

        return l_wave

    def __detect_peak(self, wave):
        peak_arr = argrelmax(wave, order=1)[0]
        if peak_arr == []:
            peak_list = []
        else:
            peak_power_arr = wave[peak_arr]
            max_peak_power = max(peak_power_arr)
            peak_list = []
            for i in range(len(peak_arr)):
                if peak_power_arr[i] <= 0.001:
                    peak_list.append(0)
                elif peak_power_arr[i] < max_peak_power / 2:
                    peak_list.append(0)
                else:
                    peak_list.append(peak_arr[i])

            # peak_arr = [0 if (peak_power_arr[i] < max_peak_power / 2)
            #             else peak_arr[i] for i in range(len(peak_arr))]
            for i, peak in enumerate(peak_list):
                if peak == 0:
                    pass
                elif i == len(peak_list):
                    pass
                else:
                    diff = peak_power_arr[i + 1] - peak_power_arr[i]
                    if diff < 0:
                        peak_list[i:] = [0] * len(peak_list[i:])
                        break
            peak_list = [peak for peak in peak_list if peak != 0]
        return peak_list

    def __make_spike(self, peak_arr, arr_length, spike_length):
        spike_lengh_points = round(spike_length * fs)
        spike_arr = np.zeros(arr_length)
        rise_line = [i / spike_lengh_points /
                     2 for i in range(int(spike_lengh_points / 2))]
        down_line = [1 - i / spike_lengh_points /
                     2 for i in range(int(spike_lengh_points / 2))]
        spike_area = rise_line + down_line
        for peak in peak_arr:
            peak_start = int(peak - spike_lengh_points / 2)
            peak_fin = int(peak + spike_lengh_points / 2)
            spike_arr[peak_start:peak_fin] = spike_area

        return spike_arr

    def __get_envelope(self, corr_wave):
        envelope_emit = abs(signal.hilbert(corr_wave[0]))
        envelope_right = abs(signal.hilbert(corr_wave[1]))
        envelope_left = abs(signal.hilbert(corr_wave[2]))

        return [envelope_emit, envelope_right, envelope_left]

    def __normalization(self, wave_list, base="emit"):
        """
        normalize data
        """
        norm_list = []
        if base == "emit":
            for wave in wave_list:
                wave = np.array(wave)
                wave = wave - wave.mean()
                wave_max = max(np.array(self.emit_wave))
                wave_n = wave / wave_max
                norm_list.append(wave_n)
        elif base == "echo":
            for wave in wave_list:
                wave = np.array(wave)
                wave = wave - wave.mean()
                wave_max = max(wave)
                wave_n = wave / wave_max
                norm_list.append(wave_n)

        return norm_list

    def get_info(self):
        right_echo_point, left_echo_point = self.__get_echo_point()
        self.right_echo_time, self.left_echo_time = self.__get_echo_time(
            right_echo_point, left_echo_point)
        self.right_echo_power, self.left_echo_power = self.__get_echo_power(
            right_echo_point, left_echo_point)
        self.distance_list, self.angle_list = self.__get_distance_angle()
        self.echo_points = self.__get_echo_points()

    def __get_echo_point(self):
        right_echo_point_raw = np.where(self.right_corr >= self.threash)[0]
        left_echo_point_raw = np.where(self.left_corr >= self.threash)[0]
        right_echo_list = []
        left_echo_list = []
        pre_r_point = 0
        pre_point = 0
        for r_point in right_echo_point_raw:
            if r_point - pre_r_point != 1:
                if pre_point == 0:
                    pre_point = r_point
                elif pre_point != 0:
                    echo_point = round((pre_point+r_point)/2)
                    right_echo_list.append(int(echo_point))
                    pre_point = 0
            pre_r_point = r_point
        pre_l_point = 0
        pre_point = 0
        for l_point in left_echo_point_raw:
            if l_point - pre_l_point != 1:
                if pre_point == 0:
                    pre_point = l_point
                elif pre_point != 0:
                    echo_point = round((pre_point+l_point)/2)
                    left_echo_list.append(int(echo_point))
                    pre_point = 0
            pre_l_point = l_point

        return right_echo_list, left_echo_list

    def __get_echo_time(self, right_points, left_points):

        return self.time_line[right_points], self.time_line[left_points]

    def __get_echo_power(self, right_points, left_points):

        return self.right_corr[right_points], self.left_corr[left_points]

    def __get_distance_angle(self):
        if min(len(self.right_echo_time), len(self.left_echo_time)) == 0:
            distance_list = [-100]
            angle_list = [-100]
        else:
            distance_list, angle_list = self.__calc_distance_angle()

        return distance_list, angle_list

    def __calc_distance_angle(self):
        distance_list = []
        angle_list = []
        for right_tim in self.right_echo_time:
            tmp_distance_list = []
            tmp_angle_list = []
            for left_tim in self.left_echo_time:
                if abs(right_tim - left_tim) < limit_time:
                    distance = (right_tim + left_tim) * velocity_air / 4
                    if -1 < (right_tim-left_tim)*velocity_air/distance_ears < 1:
                        angle = math.asin((right_tim-left_tim) *
                                          velocity_air / distance_ears)
                        tmp_angle_list.append(math.degrees(angle))
                        tmp_distance_list.append(distance)
            distance_list.append(tmp_distance_list)
            angle_list.append(tmp_angle_list)

        distance_l = list(itertools.chain(*distance_list))
        angle_l = list(itertools.chain(*angle_list))
        distance_list = sorted(set(distance_l), key=distance_l.index)
        angle_list = sorted(set(angle_l), key=angle_l.index)
        return distance_list, angle_list

    def __get_echo_points(self):
        pixel_points = []
        for (dis, ang) in zip(self.distance_list, self.angle_list):
            f_angle = math.radians(self.emit_angle - 90 + ang)
            pixel_points.append((self.emit_point[0]+round(dis*math.cos(f_angle)/dx),
                                 self.emit_point[1]+round(dis*math.sin(f_angle)/dx)))
        return pixel_points

    def save(self):
        print(self.right_echo_time)
        with open("./echo_point_{}/echo_point_{}.csv".format(os.path.basename(os.path.dirname(self.base_name)),
                                                             os.path.splitext(self.base_name)[0].split("/")[-1]), "a") as f:
            writer = csv.writer(f, lineterminator='\n')

            writer.writerow(["right_echo_point"])
            writer.writerow(self.right_echo_time)
            writer.writerow(["left_echo_point"])
            writer.writerow(self.left_echo_time)
            writer.writerow(["distance"])
            writer.writerow(self.distance_list)
            writer.writerow(["angle"])
            writer.writerow(self.angle_list)
            writer.writerow(["emit_points"])
            writer.writerow([self.emit_point])
            writer.writerow(["echo_points"])
            writer.writerow(self.echo_points)

        tim = self.time_line[:self.data_len][:, np.newaxis]
        echo_right = self.echo_without_emit_wave[0][:self.data_len][:, np.newaxis]
        echo_left = self.echo_without_emit_wave[1][:self.data_len][:, np.newaxis]
        corr_right = self.right_corr[:, np.newaxis]
        corr_left = self.left_corr[:, np.newaxis]
        data_for_csv = np.c_[tim, echo_right, echo_left, corr_right, corr_left]
        with open("./corr_{}/corr_{}.csv".format(os.path.basename(os.path.dirname(self.base_name)), os.path.splitext(self.base_name)[0].split("/")[-1]), "a") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["tim", "echo_right", "echo_left",
                             "corr_right", "corr_left"])
            writer.writerows(data_for_csv)

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


def calc_cochlear_block(pulse, right_echo, left_echo, fs):
    emit_spike_list = []
    echo_right_spike_list = []
    echo_left_spike_list = []
    # IIR filter
    f_emit_list, f_echo_right_list, f_echo_left_list = __iir_filtering(
        pulse, right_echo, left_echo, fs)

    emit_length = len(f_emit_list[0])
    echo_right_length = len(f_echo_right_list[0])
    echo_left_length = len(f_echo_left_list[0])
    for idx, f_emit in enumerate(f_emit_list):
        # half wave rectification
        fh_emit = __half_wave_rectification(f_emit)
        # low pass filter
        fhl_emit = __low_pass_filter(fh_emit, fs)
        # detect spike
        emit_point = __detect_peak(fhl_emit)
        emit_spike = __make_spike(emit_point, emit_length, 1e-6, fs)
        emit_spike_list.append(emit_spike)
        # for figure
        # if idx >= 30:
        #     fig = plt.figure()
        #     ax1 = fig.add_subplot(511)
        #     ax1.plot(pulse)
        #     ax2 = fig.add_subplot(512)
        #     ax2.plot(f_emit)
        #     ax3 = fig.add_subplot(513)
        #     ax3.plot(fh_emit)
        #     ax4 = fig.add_subplot(514)
        #     ax4.plot(fhl_emit)
        #     ax5 = fig.add_subplot(515)
        #     ax5.plot(emit_spike)
        #     plt.show()
    # right echo
    for f_echo_right in f_echo_right_list:
        # half wave rectification
        fh_echo_right = __half_wave_rectification(f_echo_right)
        # low pass filter
        fhl_echo_right = __low_pass_filter(fh_echo_right, fs)
        # detect spike
        echo_right_point = __detect_peak(fhl_echo_right)
        echo_right_spike = __make_spike(
            echo_right_point, echo_right_length, 1e-6, fs)
        echo_right_spike_list.append(echo_right_spike)
    # left echo
    for f_echo_left in f_echo_left_list:
        # half wave rectification
        fh_echo_left = __half_wave_rectification(f_echo_left)
        # low pass filter
        fhl_echo_left = __low_pass_filter(fh_echo_left, fs)
        # detect peak and make spike
        echo_left_point = __detect_peak(fhl_echo_left)
        echo_left_spike = __make_spike(
            echo_left_point, echo_left_length, 1e-6, fs)
        echo_left_spike_list.append(echo_left_spike)

    return np.vstack(emit_spike_list), np.vstack(echo_right_spike_list), np.vstack(echo_left_spike_list)


def calc_temporal_block(emit_spike_list, echo_right_spike_list, echo_left_spike_list):
    dtx = __calc_dtx(emit_spike_list)
    print(dtx)
    return dtx


def run(pulse, right_echo, left_echo, dt):
    """
    scat model 
    """
    fs = 1 / dt
    # cochlear_block
    emit_spike_list, echo_right_spike_list, echo_left_spike_list = calc_cochlear_block(
        pulse, right_echo, left_echo, fs)
    # temporal_block
    # dtx = calc_temporal_block(
    #     emit_spike_list, echo_right_spike_list, echo_left_spike_list)

    return emit_spike_list, echo_right_spike_list, echo_left_spike_list


def __make_spike(peak_arr, arr_length, spike_length, fs):

    spike_arr = np.zeros(arr_length)
    # spike_lengh_points = round(spike_length * fs)
    # rise_line = [i / spike_lengh_points /
    #              2 for i in range(int(spike_lengh_points / 2))]
    # down_line = [1 - i / spike_lengh_points /
    #              2 for i in range(int(spike_lengh_points / 2))]
    # spike_area = rise_line + down_line
    for peak in peak_arr:
        # peak_start = int(peak - spike_lengh_points / 2)
        # peak_fin = int(peak + spike_lengh_points / 2)
        # spike_arr[peak_start:peak_fin] = spike_area
        spike_arr[peak] = 1

    return spike_arr


def __detect_peak(wave):
    peak_arr = argrelmax(wave, order=1)[0]
    if peak_arr == []:
        peak_list = []
    else:
        peak_power_arr = wave[peak_arr]
        max_peak_power = max(peak_power_arr)
        peak_list = []
        # TODO fix threshold
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


def __half_wave_rectification(wave):

    return np.where(wave <= 0, 0, wave)


def __low_pass_filter(wave, fs):
    sos = signal.iirfilter(N=1,
                           Wn=3000,
                           btype="lowpass",
                           analog=False,
                           ftype="butter",
                           output="sos",
                           fs=fs)

    l_wave = signal.sosfiltfilt(sos, wave, padlen=0)

    return l_wave


def __iir_filtering(pulse, right_echo, left_echo, fs):
    """
    calc iir filter
    """
    max_freq = 100e3
    min_freq = 20e3
    min_T = 1 / max_freq
    max_T = 1 / min_freq
    bin_num = 81
    dT = (max_T - min_T) / bin_num
    bandwidth = 4e3
    f_emit_list = []
    f_echo_right_list = []
    f_echo_left_list = []
    for i in range(bin_num):
        if i == 0:
            hz = 0
        else:
            hz = 1 / i*dT
        sos = signal.iirfilter(N=10,
                               Wn=[min_freq + hz - bandwidth / 2,
                                   min_freq + hz + bandwidth / 2],
                               btype="bandpass",
                               analog=False,
                               ftype="butter",
                               output="sos",
                               fs=fs)
        f_emit_wave = signal.sosfiltfilt(sos, pulse, padlen=0)
        f_echo_wave_right = signal.sosfiltfilt(
            sos, right_echo)
        f_echo_wave_left = signal.sosfiltfilt(
            sos, left_echo)
        f_emit_list.append(f_emit_wave)
        f_echo_right_list.append(f_echo_wave_right)
        f_echo_left_list.append(f_echo_wave_left)

    return f_emit_list, f_echo_right_list, f_echo_left_list


def __calc_dtx(emit_spike_list):
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
                print(
                    f"dtx is not match between different frequency.dtx1:{dtx}, dtx2:{tmp_dtx}")
                input()

    return dtx

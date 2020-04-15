import os
import json
import time
import numpy as np
import math
from scipy import signal
from numba import jit, jitclass, float32

if not __debug__:
    debug_flag = True
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
else:
    debug_flag = False

components_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
setting_file_path = components_dir + "settings.json"
with open(setting_file_path, "r") as setting_file_obj:
    config_param = json.load(setting_file_obj)
    nx = config_param["field"]["xaxis"]
    ny = config_param["field"]["yaxis"]
    dx = config_param["resolution"]["distance"]
    dt = config_param["resolution"]["time"]
    nmax = config_param["cycle_number"]
    savestep = config_param["save_step"]
    sound_speed = config_param["soundspeed"]
    sound_speed_air = sound_speed["air"]
    sig_freq = config_param["signal"]["frequency"]
    sig_amp = config_param["signal"]["amplitude"]
    sig_duration = config_param["signal"]["duration"]
    sigma = config_param["signal"]["sigma"]
    signal_point = eval(config_param["signal"]["point"])
    receive_point = int(config_param["ear_point_from_signal"] / 1000 / dx)
    gpu_flag = config_param["GPU"]

if gpu_flag:
    import chainer
    import cupy as cp

class CalcField:
    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2
        self.ims = []
        self.tim = 0

        self.recived_wave1 = []
        self.recived_wave2 = []

    def update(self, i):
        time_start = time.perf_counter()
        if i < sig_duration:
            sig = self.CreatePulse(i)
            self.P2[signal_point[0], signal_point[1]
            ] = self.P1[signal_point[0], signal_point[1]] + sig
        self.P1[1: nx - 1, 1: ny - 1] = (
            2 * self.P2[1: nx - 1, 1: ny - 1]
            - self.P1[1: nx - 1, 1: ny - 1]
            + (sound_speed_air * dt / dx) ** 2
            * ((self.P2[2:nx, 1: ny - 1] + self.P2[: nx - 2, 1: ny - 1] + self.P2[1: nx - 1, 2:ny] + self.P2[1: nx - 1, : ny - 2]))
            - 4 * (sound_speed_air * dt / dx) ** 2 * self.P2[1: nx - 1, 1: ny - 1]
        )
        time_end = time.perf_counter()
        self.tim += time_end - time_start
        self.get_wave(self.P1)
        if debug_flag:
            im = self.MakeFigure(self.P1)
            self.ims.append([im])
        self.P1, self.P2 = self.P2, self.P1
    
    def CreatePulse(self, i):
        sig_wave = sig_amp * math.sin(2 * math.pi * sig_freq * i * dt)
        sig = sig_wave * signal.gaussian(sig_duration, std=sigma)[i]
        return sig
    
    def MakeFigure(self, P):
        if gpu_flag:
            P_to_img = cp.abs(P) / cp.max(abs(P))
            im = plt.imshow(chainer.cuda.to_cpu(P_to_img), cmap="jet")
        else:
            P_to_img = np.round(np.abs(P) / np.max(abs(P)), 2)
            im = plt.imshow(P_to_img, cmap="jet")

        return im
    
    def get_wave(self, p):
        self.recived_wave1.append(p[300, 500])
        self.recived_wave2.append(p[300,1500])



def Calc(P1, P2):
    calc_field = CalcField(P1, P2)
    [calc_field.update(i) for i in range(nmax)]
    plt.plot(calc_field.recived_wave1)
    plt.plot(calc_field.recived_wave2)
    plt.show()

    return calc_field.ims, calc_field.tim


def main():
    if gpu_flag:
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        P1 = cp.zeros((nx + 1, ny + 1), dtype=cp.float32)
        P2 = cp.zeros((nx + 1, ny + 1), dtype=cp.float32)
        if debug_flag:
            fig = plt.figure()
        image_list, tim = Calc(P1, P2)
    else:
        P1 = np.zeros((nx + 1, ny + 1), dtype=np.float32)
        P2 = np.zeros((nx + 1, ny + 1), dtype=np.float32)
        if debug_flag:
            fig = plt.figure()
        image_list, tim = Calc(P1, P2)
    
    if debug_flag:
        ani = animation.ArtistAnimation(
            fig, image_list, interval=100, blit=True)
        plt.show()
    
    print(f"finish. calc time{tim}s")


if __name__ == "__main__":
    main()

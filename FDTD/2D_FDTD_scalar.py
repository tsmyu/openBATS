import os
import json
import time
import numpy as np
import math
from scipy import signal
import Mur1

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
    abcs = config_param["ABC"]
    abc_name = [key for key, value in abcs.items() if value][0]
    sound_speed = config_param["soundspeed"]
    sound_speed_air = sound_speed["air"]
    sig_freq = config_param["signal"]["frequency"]
    sig_amp = config_param["signal"]["amplitude"]
    sig_duration = config_param["signal"]["duration"]
    sigma = config_param["signal"]["sigma"]
    signal_point = eval(config_param["signal"]["point"])
    receive_point = int(config_param["ear_point_from_signal"] / 1000 / dx)
    gpu_flag = config_param["GPU"]

# eval(f"import {abc_name} as abc")
abc_field = Mur1.Mur1(0, nx, 0, ny, sound_speed_air, dt, dx)

if gpu_flag:
    import chainer
    import cupy as cp


def MakeFigure(P):
    if gpu_flag:
        P_to_img = cp.abs(P) / cp.max(abs(P))
        im = plt.imshow(chainer.cuda.to_cpu(P_to_img), cmap="jet")
    else:
        P_to_img = np.round(np.abs(P) / np.max(abs(P)), 2)
        im = plt.imshow(P_to_img, cmap="jet")

    return im


def CreatePulse(i):
    sig_wave = sig_amp * math.sin(2 * math.pi * sig_freq * i * dt)
    sig = sig_wave * signal.gaussian(sig_duration, std=sigma)[i]
    return sig


def Calc(P1, P2):
    ims = []
    tim = 0
    for i in range(nmax):
        print("step:{}".format(i))
        time_start = time.perf_counter()
        if i < sig_duration:
            sig = CreatePulse(i)
            P2[signal_point[0], signal_point[1]] = P1[signal_point[0], signal_point[1]] + sig
        P1[1 : nx - 1, 1 : ny - 1] = (
            2 * P2[1 : nx - 1, 1 : ny - 1]
            - P1[1 : nx - 1, 1 : ny - 1]
            + (sound_speed_air * dt / dx) ** 2
            * ((P2[2:nx, 1 : ny - 1] + P2[: nx - 2, 1 : ny - 1] + P2[1 : nx - 1, 2:ny] + P2[1 : nx - 1, : ny - 2]))
            - 4 * (sound_speed_air * dt / dx) ** 2 * P2[1 : nx - 1, 1 : ny - 1]
        )
        P1 = abc_field.calc(P2, P1)
        time_end = time.perf_counter()
        tim += time_end - time_start
        if debug_flag:
            im = MakeFigure(P1)
            ims.append([im])
        P1, P2 = P2, P1

    print(f"calc time:{np.round(tim,2)}[s]")
    return ims


def main():
    if gpu_flag:
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        P1 = cp.zeros((nx + 1, ny + 1), dtype=cp.float32)
        P2 = cp.zeros((nx + 1, ny + 1), dtype=cp.float32)
        if debug_flag:
            fig = plt.figure()
        image_list = Calc(P1, P2)
    else:
        P1 = np.zeros((nx + 1, ny + 1), dtype=np.float32)
        P2 = np.zeros((nx + 1, ny + 1), dtype=np.float32)
        if debug_flag:
            fig = plt.figure()
        image_list = Calc(P1, P2)
    if debug_flag:
        ani = animation.ArtistAnimation(fig, image_list, interval=100, blit=True)
        plt.show()


if __name__ == "__main__":
    main()

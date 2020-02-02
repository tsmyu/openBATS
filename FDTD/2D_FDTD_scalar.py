import os
import sys
import json
import time
import field
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt

if not __debug__:
    debug_flag = True
    import matplotlib.animation as animation
else:
    debug_flag = False

components_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
setting_file_path = components_dir + "settings.json"
with open(setting_file_path, "r") as setting_file_obj:
    config_param = json.load(setting_file_obj)
    # nx = config_param["field"]["xaxis"]
    # ny = config_param["field"]["yaxis"]
    dx = config_param["resolution"]["distance"]
    dt = config_param["resolution"]["time"]
    nmax = config_param["cycle_number"]
    savestep = config_param["save_step"]
    abcs = config_param["ABC"]
    abc_name = [key for key, value in abcs.items() if value][0]
    sound_speed = config_param["soundspeed"]
    sound_speed_air = sound_speed["air"]
    sound_speed_acrylic = sound_speed["acrylic"]
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

recived_wave1 = []
recived_wave2 = []


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
    # sig = sig_wave * signal.gaussian(sig_duration, std=sigma)[i]
    
    return sig_wave


def Calc(field_data, P1, P2):
    ims = []
    tim = 0
    width = field_data.width
    height = field_data.height
    for i in range(nmax):
        print("step:{}".format(i))
        time_start = time.perf_counter()
        if i < sig_duration:
            sig = CreatePulse(i)
            P2[signal_point[0], signal_point[1]
               ] = P2[signal_point[0], signal_point[1]] + sig
        P1[1: width - 1, 1: height - 1] = (
            2 * P2[1: width - 1, 1: height - 1]
            - P1[1: width - 1, 1: height - 1]
            + (sound_speed_air * dt / dx) ** 2
            * (
                (
                    P2[2:width, 1: height - 1]
                    + P2[: width - 2, 1: height - 1]
                    + P2[1: width - 1, 2:height]
                    + P2[1: width - 1, : height - 2]
                )
            )
            - 4 * (sound_speed_air * dt / dx) ** 2 *
            P2[1: width - 1, 1: height - 1]
        )
        P1 = field_data.update(P2, P1)
        time_end = time.perf_counter()
        tim += time_end - time_start
        get_wave(P1)
        if debug_flag:
            im = MakeFigure(P1)
            ims.append([im])
        P1, P2 = P2, P1
    # plt.plot(recived_wave1)
    recived_wave_arr_1 = abs(np.array(recived_wave1))
    recived_wave_arr_2 = abs(np.array(recived_wave2))
    # plt.plot(recived_wave_arr_1)
    plt.plot(recived_wave_arr_2)
    plt.savefig("recived_wave_raw.png")
    plt.close()
    recived_wave1_max = max(recived_wave_arr_1)
    recived_wave2_max = max(recived_wave_arr_2)
    recived_wave1_log = 20 * np.log(recived_wave_arr_1 / recived_wave1_max)
    recived_wave2_log = 20 * np.log(recived_wave_arr_2 / recived_wave2_max)
    # recived_wave1_log = [20 * np.log(20e-6/20e-6) if recived_wave <= 0 else 20 * np.log(recived_wave / 20e-6) for recived_wave in recived_wave1]
    # recived_wave2_log=[20 * np.log(20e-6/20e-6) if recived_wave <= 0 else 20 * np.log(recived_wave / 20e-6) for recived_wave in recived_wave2]
    # plt.plot(recived_wave1_log)
    plt.plot(recived_wave2_log)
    plt.ylim(-200, 0)
    plt.xlabel("time [2e-6 s]")
    plt.ylabel("P [dB]")
    plt.savefig("recived_wave_num.png")

    print(f"calc time:{np.round(tim,2)}[s]")
    return ims


def get_wave(p):
    recived_wave1.append(p[200, 300])
    recived_wave2.append(p[400, 300])


def main(field_image):
    print("field setting....")
    field_data = field.Field(field_image, abc_name, sound_speed_air, dt, dx)
    print("done")
    width = field_data.width
    height = field_data.height

    if abc_name == "Mur1":
        import mur1

        abc_field = mur1.Mur1(0, width, 0, height, sound_speed_air, dt, dx)

    if gpu_flag:
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        P1 = cp.zeros((width, height), dtype=cp.float32)
        P2 = cp.zeros((width, height), dtype=cp.float32)
        if debug_flag:
            fig = plt.figure()
        image_list = Calc(field_data, P1, P2)
    else:
        P1 = np.zeros((width, height), dtype=np.float32)
        P2 = np.zeros((width, height), dtype=np.float32)
        if debug_flag:
            fig = plt.figure()
        image_list = Calc(field_data, P1, P2)
    if debug_flag:
        ani = animation.ArtistAnimation(
            fig, image_list, interval=100, blit=True)
        plt.show()


if __name__ == "__main__":
    argvs = sys.argv
    if len(argvs) < 2:
        print(f"Usage: python {argvs[0]} [field image]")
    field_image = argvs[1]
    main(field_image)

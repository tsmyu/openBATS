
import os
import json
import numpy as np
import math
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import chainer
import time
import copy
import csv

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
    wall_list = eval(config_param["wall"]["point"])
    wall_length_x = config_param["wall"]["length_x"]
    wall_length_y = config_param["wall"]["length_y"]
    density = config_param["density"]
    kappa = config_param["kappa"]
    sig_freq = config_param["signal"]["frequency"]
    sig_amp = config_param["signal"]["amplitude"]
    sig_duration = config_param["signal"]["duration"]
    sigma = config_param["signal"]["sigma"]
    signal_point = eval(config_param["signal"]["point"])
    receive_point = int(config_param["ear_point_from_signal"]/1000 / dx)
    gpu_flag = config_param["GPU"]

if gpu_flag:
    import cupy as cp

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(1, 1, 1)
ims = []

def CreatePulse(i):
    # sig = sig_amp * (1.0-math.cos(2.0 * math.pi * sig_freq * i * dt))
    sig_wave = sig_amp * math.sin(2 * math.pi * sig_freq * i * dt)
    sig = sig_wave * signal.gaussian(sig_duration, std = sigma)[i]

    return sig


def MakeField(kappa_f, density_f):
    for wall in wall_list:
        kappa_f[wall[0]:wall[0]+wall_length_x, wall[1]:wall[1]+wall_length_y] = kappa["acrylic"]
        density_f[wall[0]:wall[0]+wall_length_x, wall[1]:wall[1]+wall_length_y] = density["acrylic"]

    return kappa_f, density_f

def MakeFigure(P):
    start = time.time()
    if gpu_flag:
        P_to_img = cp.abs(P)/cp.max(abs(P))
        p = ax1.imshow(chainer.cuda.to_cpu(P_to_img), cmap="jet")
    else:
        P_to_img = np.abs(P)/np.max(abs(P))
        p = ax1.imshow((P_to_img), cmap="jet")
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return [p]

def Calc(Vx, Vy, P, kappa_f, density_f):
    kappa_field, density_field = MakeField(kappa_f, density_f)
    for i in range(nmax):
        print("step:{}".format(i))
        if i < sig_duration:
            sig = CreatePulse(i)
            P[signal_point[0], signal_point[1]] = P[signal_point[0], signal_point[1]] + sig
            Vx[signal_point[0], signal_point[1]] = Vx[signal_point[0], signal_point[1]] + sig
            # Vy[sig_x, sig_y] = Vy[sig_x, sig_y] + sig/2
        Vx[1:nx+1, :ny] = Vx[1:nx+1, :ny] - dt / (density_field[:nx, :] * dx) * (P[1:nx+1, :ny] - P[:nx, :ny])
        Vy[:nx, 1:ny+1] = Vy[:nx, 1:ny+1] - dt / (density_field[:, :ny] * dx) * (P[:nx, 1:ny+1] - P[:nx, :ny])
        P[:nx, :ny] = P[:nx, :ny] - (kappa_field * dt / dx) * ((Vx[1:nx+1, :ny] - Vx[:nx, :ny]) + (Vy[:nx, 1:ny+1] - Vy[:nx, :ny]))
        with open("ear_point_wave.csv", "a") as f:
            writer = csv.writer(f, lineterminator="\n")
            if i == 0:
                writer.writerow(["left", "right"])
            writer.writerow([P[signal_point[0], signal_point[1] - receive_point], P[signal_point[0], signal_point[1] + receive_point]])
        for wall in wall_list:
            P[wall[0]:wall[0]+wall_length_x, wall[1]:wall[1]+wall_length_y] = 0.0
        # P[sig_x:, :] = 0.0
        if i % savestep == 0 and i != 0:

            # P_to_img = cp.round(cp.abs(P)/cp.max(abs(P)), 2)
            im = MakeFigure(P)
            ims.append(im)

    return ims

if __name__=="__main__":
    if gpu_flag:
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        Vx = cp.zeros((nx+1, ny+1))
        Vy = cp.zeros((nx+1, ny+1))
        P = cp.zeros((nx+1, ny+1))
        kappa_field = cp.ones((nx, ny)) * kappa["air"]
        density_field = cp.ones((nx, ny)) * density["air"]
    else:
        Vx = np.zeros((nx+1, ny+1))
        Vy = np.zeros((nx+1, ny+1))
        P = np.zeros((nx+1, ny+1))
        kappa_field = np.ones((nx, ny)) * kappa["air"]
        density_field = np.ones((nx, ny)) * density["air"]
    image_list = Calc(Vx, Vy, P, kappa_field, density_field)
    ani = animation.ArtistAnimation(fig, image_list, interval=100, blit=False)
    # ani.save("fdtd_test1.mp4", writer="ffmpeg")
    plt.show()

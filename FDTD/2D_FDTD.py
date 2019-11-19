
import numpy as np
import math
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation

nx = 1000
ny = 800
dx = 10.0e-4
dt = 20.0e-7
nmax = 700
savestep = 10

x_obs = 700
y_obs = 390
R_max = 20.0
density = {"air": 1.293,
           "acrylic": 1180}
kappa = {"air": 142.0e3,
         "acrylic": 322.0e4}

# signal
# frequency [Hz]
sig_freq = 1.0e4
# amplitude [Pa]
sig_amp = 5
# duration [ms]
sig_duration = 100
# if gaussian sigma
sigma = 7
# signal point
sig_x = 800
sig_y = 400


def CreatePulse(i):
    # sig = sig_amp * (1.0-math.cos(2.0 * math.pi * sig_freq * i * dt))
    sig_wave = sig_amp * math.sin(2*math.pi*sig_freq*i*dt)
    sig = sig_wave * signal.gaussian(sig_duration, std=sigma)[i]
    return sig


def MakeField(kappa_f, density_f, obs_flag=True):
    if obs_flag:
        kappa_f[x_obs:x_obs+10, y_obs:y_obs+20] = kappa["acrylic"]
        density_f[x_obs:x_obs + 10, y_obs:y_obs + 20] = density["acrylic"]

    return kappa_f, density_f


def Calc(Vx, Vy, P, kappa_f, density_f):
    ims = []
    kappa_field, density_field = MakeField(kappa_f, density_f, False)
    for i in range(nmax):
        if i < sig_duration:
            sig = CreatePulse(i)
            P[sig_x, sig_y] = P[sig_x, sig_y] + sig
            # Vx[sig_x, sig_y] = Vx[sig_x, sig_y] + sig
            # Vy[sig_x, sig_y] = Vy[sig_x, sig_y] + sig
        print("step:{}".format(i))
        Vx[1:nx+1, :ny] = Vx[1:nx+1, :ny] - dt / \
            (density_field[:nx, :] * dx) * (P[1:nx+1, :ny] - P[:nx, :ny])
        Vy[:nx, 1:ny+1] = Vy[:nx, 1:ny+1] - dt / \
            (density_field[:, :ny] * dx) * (P[:nx, 1:ny+1] - P[:nx, :ny])
        P[:nx, :ny] = P[:nx, :ny] - (kappa_field*dt/dx) * (
            (Vx[1:nx+1, :ny] - Vx[:nx, :ny]) + (Vy[:nx, 1:ny+1] - Vy[:nx, :ny]))
        # P[x_obs:x_obs+10, y_obs:y_obs+20] = 0.0
        # P[sig_x:, :] = 0.0
        P_to_img = np.round(np.abs(P)/np.max(abs(P)), 2)
        im = plt.imshow(P_to_img, cmap="jet")
        ims.append([im])

    return ims


if __name__ == "__main__":
    fig = plt.figure()
    Vx = np.zeros((nx+1, ny+1))
    Vy = np.zeros((nx+1, ny+1))
    P = np.zeros((nx+1, ny+1))
    kappa_field = np.ones((nx, ny)) * kappa["air"]
    density_field = np.ones((nx, ny)) * density["air"]
    image_list = Calc(Vx, Vy, P, kappa_field, density_field)
    ani = animation.ArtistAnimation(fig, image_list, interval=100, blit=True)
    plt.show()

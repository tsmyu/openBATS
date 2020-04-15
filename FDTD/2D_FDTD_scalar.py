import matplotlib.pyplot as plt
import os
import sys
import json
import time
import csv
import numpy as np
import math
from scipy import signal
import field
import bat
import matplotlib
import copy
matplotlib.use("Tkagg")

if not __debug__:
    debug_flag = True
    import matplotlib.animation as animation
else:
    debug_flag = False

components_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
setting_file_path = components_dir + "settings.json"
with open(setting_file_path, "r") as setting_file_obj:
    config_param = json.load(setting_file_obj)
    dx = config_param["resolution"]["distance"]
    dt = config_param["resolution"]["time"]
    nmax = config_param["cycle_number"]
    savestep = config_param["save_step"]
    abcs = config_param["ABC"]
    abc_name = [key for key, value in abcs.items() if value][0]
    sound_speed_list = config_param["soundspeed"]
    density_list = config_param["density"]
    sig_freq = config_param["signal"]["frequency"]
    receive_dis = int(config_param["ear_point_from_signal"] / 2 / dx)
    gpu_flag = config_param["GPU"]

if gpu_flag:
    import chainer
    import cupy as cp

n = 17
receive_points = [[] for i in range(n)]
N_list = [[] for i in range(n - 1)]
legends = [i for i in range(n - 1)]
lamda = sound_speed_list["air"] / sig_freq
for i in range(n):
    if i // 4 == 0:
        j = i
        N_list[i].append(2 * ((200 ** 2 + 150 ** 2) ** 0.5+((1100 + 100 * j - 1000) ** 2 + (250 - 100)
                                                            ** 2) ** 0.5 - ((1100 + 100 * j - 800) ** 2 + (100 - 100) ** 2) ** 0.5)*dx / lamda)
    elif i // 4 == 1:
        j = i - 4
        N_list[i].append(2 * ((200 ** 2 + 150 ** 2) ** 0.5+((1100 + 100 * j - 1000) ** 2 + (250 - 200)
                                                            ** 2) ** 0.5 - ((1100 + 100 * j - 800) ** 2 + (200 - 100) ** 2) ** 0.5)*dx / lamda)
    elif i // 4 == 2:
        j = i - 8
        N_list[i].append(2 * ((200 ** 2 + 150 ** 2) ** 0.5+((1100 + 100 * j - 1000) ** 2 + (250 - 300)
                                                            ** 2) ** 0.5 - ((1100 + 100 * j - 800) ** 2 + (300 - 100) ** 2) ** 0.5)*dx / lamda)
    elif i // 4 == 3:
        j = i - 12
        N_list[i].append(2 * ((200 ** 2 + 150 ** 2) ** 0.5+((1100 + 100 * j - 1000) ** 2 + (250 - 400)
                                                            ** 2) ** 0.5 - ((1100 + 100 * j - 800) ** 2 + (400 - 100) ** 2) ** 0.5) * dx / lamda)

recived_wave1 = []
ims_list = []


def MakeFigure(P, pulse_info, receive_points, field_data):
    if gpu_flag:
        P_to_img = cp.abs(P) / cp.max(abs(P))
        plt.imshow(chainer.cuda.to_cpu(P_to_img), cmap="jet")
        plt.scatter(pulse_info[0], pulse_info[1], c="w")
        plt.scatter(receive_points[0][0], receive_points[0][1], c="w")
        img = plt.scatter(receive_points[1], receive_points[1][1], c="w")
    else:
        _P = copy.copy(P)
        _P[field_data.wall_area] = 0
        P_to_img = np.abs(_P) / np.max(abs(_P))
        # P_to_img = np.round(np.abs(P) / 30, 2)
        P_to_img[field_data.wall_area] = 1
        img = plt.imshow(P_to_img, cmap="jet")
        # plt.scatter(pulse_info[0], pulse_info[1], c="w", s=5)
        # plt.scatter(receive_points[0][0], receive_points[0][1], c="w", s=5)
        # img = plt.scatter(receive_points[1][0],
        #                   receive_points[1][1], c="w", s=5)

    return img


def Calc(field_data, P1, P2, pulse_info_list):
    """
    main calc
    P : sound pressure
    n : time step
    i : x-axis
    j : y-axis
    d : density
    v : sound speed
    dx : spatial resolution
    dt : time resolution

    P[n+1](i,j) = 2P[n](i,j) - P[n-1](i,j)+d(i,j)*(v(i,j)**2*dt**2/dx**2)
    """
    tim = 0
    width = field_data.width
    height = field_data.height
    if gpu_flag:
        density = cp.round(field_data.density_arr, 2)
        alpha = cp.round((sound_speed_list["air"] * dt / dx) ** 2, 2)
    else:
        density = np.round(field_data.density_arr, 2)
        alpha = np.round((sound_speed_list["air"] * dt / dx) ** 2, 2)
    velocity = field_data.velocity_arr
    # alpha = np.round((velocity[1:width - 1, 1:height - 1] * dt / dx)** 2, 2)

    BAT = bat.Bat()
    # test
    directivity_list = []
    for pulse_info in pulse_info_list:
        print(f"-----calc with emit pulse{pulse_info}-----")
        receive_points = BAT.set_position(pulse_info)
        time_start = time.perf_counter()
        ims = []
        tim_list = []
        emit_list = []
        for i in range(nmax):
            print("step:{}".format(i))
            P2 = BAT.emit_pulse(i, P2, density)
            # if i < sig_duration:
            #     P2[pulse_info[0], pulse_info[1]
            #        ] = P2[pulse_info[0], pulse_info[1]] + sig

            P1[1:width - 1, 1:height - 1] = (
                2*P2[1:width - 1, 1:height - 1]
                - P1[1:width - 1, 1:height - 1]
                + alpha * density[1:width - 1, 1:height - 1]
                * (P2[2:width, 1: height - 1]/density[2:width, 1: height - 1]
                   + P2[: width - 2, 1: height - 1] /
                   density[: width - 2, 1: height - 1]
                   + P2[1: width - 1, 2:height]/density[1: width - 1, 2:height]
                   + P2[1: width - 1, : height - 2] /
                   density[1: width - 1, : height - 2]
                   )
                - 4 * alpha * P2[1: width - 1, 1: height - 1]
            )
            P1 = field_data.update(P2, P1)
            BAT.get_echo(P1, P2, density)
            tim_list.append(i * dt)
            emit_list.append(P2[pulse_info[0], pulse_info[1]])
            if debug_flag and i % savestep == 0:
                im = MakeFigure(P1, pulse_info, receive_points, field_data)
                ims.append([im])
            P1, P2 = P2, P1
        with open(f"{pulse_info}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "emit", "right_echo", "left_echo"])
            tim_list = np.array(tim_list)[:, np.newaxis]
            emit_list = np.array(emit_list)[:, np.newaxis]
            echo_r = np.array(BAT.echo_right_ear)[:, np.newaxis]
            echo_l = np.array(BAT.echo_left_ear)[:, np.newaxis]
            target_data = np.hstack([tim_list, emit_list, echo_r, echo_l])
            writer.writerows(target_data)
        time_end = time.perf_counter()
        tim = time_end - time_start
        print(f"calc time:{np.round(tim,2)}[s]")
        # plt.plot(BAT.echo_right_ear, label=f"{pulse_info[2]}")

        # directivity test
        # directivity_list.append([pulse_info[2], np.nanmax(BAT.echo_right_ear)])
        # plt.plot(BAT.echo_left_ear, label="p1")
        # plt.plot(receive_test)
        ims_list.append(ims)
    # plt.plot(receive_left)
    # plt.show()
    # emit_power = max(abs(np.array(receive_points[-1])))
    # for idx, receive_point in enumerate(receive_points[:-1]):
    #     recived_wave_arr = abs(np.array(receive_point))
    #     recived_wave_max = max(recived_wave_arr)
    #     recived_wave_log = 20 * np.log(recived_wave_arr / emit_power)
    #     recived_max_log = 20 * np.log(recived_wave_max / emit_power)
    #     plt.scatter(N_list[idx][0], recived_max_log, label=f"{idx}")

    # # plt.ylim(-200, 0)
    # # plt.subplots_adjust(right=0.1)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # plt.xlabel("N")
    # plt.ylabel("P [dB]")
    # plt.savefig("recived_wave_diffraction_N.png",bbox_inches='tight', pad_inches=0)
    # plt.plot(BAT.echo_left_ear, label="no_directivity")

    # # test
    # ax1 = plt.subplot(111, projection="polar")
    # ax1.set_theta_direction(-1)
    # ax1.set_rlabel_position(0)
    # ax1.set_xticks(np.pi / 180. * np.linspace(0, 360, 12, endpoint=False))
    # ax1.spines['polar'].set_color('darkgray')
    # # plt.ylim(-60.1, 5)
    # for directivity_data in directivity_list:
    #     ax1.scatter(directivity_data[0] * np.pi / 180, directivity_data[1])
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 12
    # plt.legend()
    # plt.show()
    # plt.savefig("echo.png")
    return ims_list


def read_pulse_info(data):
    pulse_info_list = []
    with open("{}".format(data)) as f:
        txt_line = f.readlines()
    for txt in txt_line:
        if txt.split(",")[0] == "index":
            flight_path_x = txt.split(",")[1:]
        elif txt.split(",")[0] == "columns":
            flight_path_y = txt.split(",")[1:]
        elif txt.split(",")[0] == "pulse_direction":
            pulse_dir = txt.split(",")[1:]
        else:
            pass
    for (x, y, direction) in zip(flight_path_x, flight_path_y, pulse_dir):
        if x != "\n" and y != "\n" and direction != "\n" and x != "" and y != "" and direction != "":
            x = x.strip('\n')
            y = y.strip("\n")
            direction = 90 - int(direction.strip("\n"))
            # if direction > 180:
            #     direction = direction - 360
            pulse_info_list.append((int(x), int(y), int(direction)))

    return pulse_info_list


def main(field_image, txt_data):
    print("field setting....")
    if gpu_flag:
        field_data = field.Field_GPU(field_image, abc_name,
                                     sound_speed_list, density_list, dt, dx)
    else:
        field_data = field.Field(field_image, abc_name,
                                 sound_speed_list, density_list, dt, dx)
    width = field_data.width
    height = field_data.height

    if abc_name == "Mur1":
        import ABC
        abc_field = ABC.Mur1(0, width, 0, height,
                             sound_speed_list["air"], dt, dx)
    print("done")
    print("pulse information reading....")
    pulse_info_list = read_pulse_info(txt_data)
    print("done")

    if gpu_flag:
        print("calc with GPU start....")
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        P1 = cp.zeros((width, height), dtype=cp.float32)
        P2 = cp.zeros((width, height), dtype=cp.float32)
        if debug_flag:
            fig = plt.figure()
        image_list = Calc(field_data, P1, P2, pulse_info_list)
    else:
        print("calc without GPU start....")
        P1 = np.zeros((width, height), dtype=np.float32)
        P2 = np.zeros((width, height), dtype=np.float32)
        if debug_flag:
            fig = plt.figure()
        image_list = Calc(field_data, P1, P2, pulse_info_list)
    if debug_flag:
        print("make animation....")
        ani = animation.ArtistAnimation(
            fig, image_list[0], interval=100, blit=True)
        ani.save(f'ani.gif', writer="imagemagick")
        # for idx, img in enumerate(image_list):
        #     ani = animation.ArtistAnimation(
        #         fig, img, interval=100, blit=True)
        #     ani.save(f'anim_{idx}.gif', writer="imagemagick")
        # plt.show()


if __name__ == "__main__":
    argvs = sys.argv
    if len(argvs) < 2:
        print(
            f"Usage: python {argvs[0]} [field image] [pulse information text]")
    field_image = argvs[1]
    txt_data = argvs[2]
    main(field_image, txt_data)

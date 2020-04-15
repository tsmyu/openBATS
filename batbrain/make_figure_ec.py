import os
import sys
from natsort import natsorted
import csv
import glob
import cv2
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math


def __make_figure(name, w, x, y, pulse_dir, echo_point_list, power_max):

    # 1 pixel = 0.25 mm
    #base_img = np.ones((600, 1800, 3), dtype=np.uint8)*255
    #cv2.line(base_img, (400, 0), (400, 264), (0, 0, 0), thickness=4)
    #cv2.line(base_img, (800, 600), (800, 336), (0, 0, 0), thickness=4)
    #cv2.line(base_img, (1200, 0), (1200, 264), (0, 0, 0), thickness=4)
    pulse_x = [500 * math.sin(math.radians(float(pulse)))
               for pulse in pulse_dir if (pulse != "\n" and pulse != "")]
    pulse_y = [500 * math.cos(math.radians(float(pulse)))
               for pulse in pulse_dir if (pulse != "\n" and pulse != "")]
    pulse_x = x + np.array(pulse_x)
    pulse_y = 3000 - np.array(y) + np.array(pulse_y)
    xs = np.linspace(x[0], x[-1], len(x))
    ys = 3000 - np.polyval(w, xs)
    fig = plt.figure(figsize=(12, 5))
    plt.xlim(0, 10000)
    plt.ylim(0, 3000)
    plt.xticks(np.arange(0, 10001, 2000))
    plt.yticks(np.arange(0, 3001, 1000))
    plt.xlabel("[m]")
    plt.ylabel("[m]")
    ax = fig.add_subplot(111)
    y = 3000 - np.array(y)
    # ax.plot(xs, ys, 'r-', lw=4, label='flight path')
    ax.plot(xs, ys, 'r-', lw=4)
    ax.scatter(x, y, s=50)
    for i, (xx, yy, px, py) in enumerate(zip(x, y, pulse_x, pulse_y)):
        c_length = 255 / len(echo_point_list)

        if i <= len(echo_point_list):
            if len(echo_point_list) == 0:
                c_r = 204 / 255
                c_g = 204 / 255
                c_b = 204 / 255
            else:
                # c_r = (255 - i * c_length) / 255
                # c_g = (255 - i * c_length) / 255
                # c_b = (255 - i * c_length) / 255
                c_r = i / len(echo_point_list)
                c_g = 1 - abs(1 - i*2 / len(echo_point_list))
                c_b = 1 - i / len(echo_point_list)
            # ax.plot((xx, int(px)), (yy, int(py)), 'b-', lw=2, label='pulse direction')
            ax.annotate('', xy=(int(px), int(py)), xytext=(xx, yy), arrowprops=dict(shrink=0, width=1, headwidth=8,
                                                                                    headlength=10, connectionstyle='arc3',
                                                                                    facecolor=(c_r, c_g, c_b), edgecolor=(c_r, c_g, c_b)))
        else:
            ax.annotate('', xy=(int(px), int(py)), xytext=(xx, yy), arrowprops=dict(shrink=0, width=1, headwidth=8,
                                                                                    headlength=10, connectionstyle='arc3',
                                                                                    facecolor='gray', edgecolor='gray'))
        for echo_point in echo_point_list:
            if len(echo_point_list) == 0:
                pass
            elif echo_point[0] == i:
                for i in range(len(echo_point[1])):
                    print("power:", echo_point[2])
                    ax.scatter(eval(echo_point[1][i])[0], 3000-eval(echo_point[1][i])[1],
                               s=300*echo_point[2][i]/power_max,
                               c=np.array((c_r, c_g, c_b)),
                               marker="s",
                               alpha=echo_point[2][i] / power_max,
                               edgecolors="k")
    ax.plot((3000, 3000), (3000, 1680), 'k-', lw=4, linestyle="dashed")
    ax.plot((5000, 5000), (0, 1320), 'k-', lw=4, linestyle="dashed")
    ax.plot((7000, 7000), (3000, 1680), 'k-', lw=4, linestyle="dashed")
    ax.set_xticklabels(["0",
                        "1.0",
                        "2.0",
                        "3.0",
                        "4.0",
                        "5.0"])
    ax.set_yticklabels(["0",
                        "0.5",
                        "1.0",
                        "1.5"])
    ax.legend()
    plt.plot(xs, ys)
    path = os.getcwd()
    if not os.path.exists(path+"/png_tmp"):
        os.makedirs(path+"/png_tmp")
    plt.savefig("{}/png_tmp/{}.png".format(path, name))
    # for i in range(len(xs)):
    #    cv2.circle(base_img, (int(ys[i]), int(xs[i])), 2, (255, 255, 0), -1)

    # return base_img


def __calc_poly(flight_path_y_list, flight_path_x_list):
    x = [int(float(i)) for i in flight_path_y_list if (i != "\n" and i != "")]
    y = [int(float(i)) for i in flight_path_x_list if (i != "\n" and i != "")]
    w = np.polyfit(x, y, 10)

    return w, x, y


def calc_echo_point(target_csv):
    print("target_csv:", os.path.basename(target_csv))
    emit_data_line = 1000
    echo_data_line_p = 1000
    echo_power_line = 1000
    echo_points_time_r_l_line = 1000
    echo_points_power_r_l_line = 1000
    with open(target_csv, 'r') as f:
        data_map = csv.reader(f)
        for idx, data in enumerate(data_map):
            if len(data) > 0:
                if data[0] == "emit_points":
                    emit_data_line = idx + 1
                if data[0] == "echo_points":
                    echo_data_line_p = idx + 1
                if data[0] == "echo_points_power":
                    echo_power_line = idx + 1
                if data[0] == "echo_points_time_r_l":
                    echo_points_time_r_l_line = idx + 1
                if data[0] == "echo_points_power_r_l":
                    echo_points_power_r_l_line = idx + 1
                if idx == emit_data_line:
                    emit_data = data
                if idx == echo_data_line_p:
                    echo_data = data
                if idx == echo_power_line:
                    echo_power = data
                if idx == echo_points_time_r_l_line:
                    echo_r_l_tim = data
                if idx == echo_points_power_r_l_line:
                    echo_r_l_power = data
            else:
                emit_data = ['(-100, -100)']
                echo_data = ['(-100, -100)']
                echo_power = ['0']
                echo_r_l_tim = []
                echo_r_l_power = []
    return emit_data, echo_data, echo_power, echo_r_l_tim, echo_r_l_power


def calc_path(csv_data):
    bname = os.path.splitext(os.path.basename(csv_data))[0]
    pulse_dir = bname.split("g")[-1]
    flight_path_x = bname.split("_")[-3]
    flight_path_y = bname.split("_")[-2]

    return flight_path_x, flight_path_y, pulse_dir


def __make_tim_power_diff(echo_r_l_tim, echo_r_l_power):
    print(echo_r_l_tim)
    if not echo_r_l_tim == []:
        r_l_tim = eval(echo_r_l_tim[0])
        r_l_power = eval(echo_r_l_power[0])
        # left - right (time), right - left (power)
        first_diff_tim = float(r_l_tim[1]) - float(r_l_tim[0])
        first_diff_power = float(r_l_power[0]) - float(r_l_power[1])
    else:
        first_diff_tim = np.nan
        first_diff_power = np.nan

    return first_diff_tim, first_diff_power


def __norm(flight_path_y_list, diff_tim_list, diff_power_list):
    # str -> int
    flight_y = [-1*int(i) for i in flight_path_y_list]
    flight_path_y = np.array(flight_y)

    # y
    flight_path_y_0 = flight_path_y- flight_path_y[0]
    flight_path_y_max = max(abs(flight_path_y_0))
    flight_path_y_n = flight_path_y_0 / flight_path_y_max
    if diff_tim_list != []:
        diff_tim = np.array(diff_tim_list)
        diff_power = np.array(diff_power_list)
        # time
        diff_tim_n = diff_tim - np.nanmean(diff_tim)
        # diff_tim_max = max(abs(diff_tim_0))
        # diff_tim_n = diff_tim_0 / diff_tim_max
        # power
        diff_power_n = diff_power - np.nanmean(diff_power)
        # diff_power_max = max(abs(diff_power_0))
        # diff_power_n = diff_power / diff_power_max
    else:
        diff_tim_n = None
        diff_power_n = None


    return flight_path_y_n, diff_tim_n, diff_power_n


def main(csv_list):
    echo_data_p_list = []
    power_max = 0
    flight_path_y_list = []
    flight_path_x_list = []
    pulse_dir_list = []
    diff_tim_list = []
    diff_power_list = []
    for idx, csv_data in enumerate(csv_list):
        print("csv target:", csv_data)
        _, echo_data, echo_power, echo_r_l_tim, echo_r_l_power = calc_echo_point(
            csv_data)
        echo_power_data = [float(power) for power in echo_power]
        echo_data_p_list.append((idx, echo_data, echo_power_data))
        # emit power 2360
        if power_max < max(echo_power_data):
            power_max = max(echo_power_data)
        flight_path_x, flight_path_y, pulse_dir = calc_path(csv_data)
        flight_path_y_list.append(flight_path_y)
        flight_path_x_list.append(flight_path_x)
        pulse_dir_list.append(pulse_dir)
        first_diff_tim, first_diff_power = __make_tim_power_diff(
            echo_r_l_tim, echo_r_l_power)
        diff_tim_list.append(first_diff_tim)
        diff_power_list.append(first_diff_power)
    y_n, diff_tim_n, diff_power_n = __norm(flight_path_y_list, diff_tim_list, diff_power_list)
    x_axis = np.linspace(0, len(csv_list) - 1, len(csv_list))
    fig, ax1 = plt.subplots()
    ax1.plot(x_axis, y_n, label="flight path", c="r")
    ax1.plot(x_axis, diff_power_n, label="power diff", c="b")
    plt.legend(loc='upper left', borderaxespad=0, fontsize=10)
    ax1.set_ylim([-4, 4])
    ax2 = ax1.twinx()
    ax2.plot(x_axis, diff_tim_n, label="time diff", c="g")
    plt.legend(loc='upper right', borderaxespad=0, fontsize=10)
    ax2.set_ylim([-0.00006, 0.00006])
    path = os.getcwd()
    name_ = csv_list[0].split("/")[-2].split("_")[-1]
    plt.savefig(f"{path}/{name_}_diff.png")

    w, x, y = __calc_poly(flight_path_x_list, flight_path_y_list)
    __make_figure(csv_list[0].split("/")[-2], w, x, y,
                  pulse_dir_list, echo_data_p_list, power_max)


if __name__ == "__main__":
    argvs = sys.argv

    if len(argvs) < 2:
        print(
            "Usage: python {} [target pulse position txt] [folder of echo csv]".format(argvs[0]))
        exit()
    csv_list = natsorted(glob.glob("{}/*.csv".format(argvs[1])))

    main(csv_list)

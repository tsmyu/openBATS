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


def __make_figure(txt_name, w, x, y, pulse_dir, echo_point_list, power_max):

    # 1 pixel = 0.25 mm
    #base_img = np.ones((600, 1800, 3), dtype=np.uint8)*255
    #cv2.line(base_img, (400, 0), (400, 264), (0, 0, 0), thickness=4)
    #cv2.line(base_img, (800, 600), (800, 336), (0, 0, 0), thickness=4)
    #cv2.line(base_img, (1200, 0), (1200, 264), (0, 0, 0), thickness=4)
    pulse_x = [100 * math.sin(math.radians(float(pulse)))
               for pulse in pulse_dir if (pulse != "\n" and pulse != "")]
    pulse_y = [100 * math.cos(math.radians(float(pulse)))
               for pulse in pulse_dir if (pulse != "\n" and pulse != "")]
    pulse_x = x + np.array(pulse_x)
    pulse_y = 600 - np.array(y) + np.array(pulse_y)
    xs = np.linspace(x[0], x[-1], len(x))
    ys = 600 - np.polyval(w, xs)
    fig = plt.figure(figsize=(12, 5))
    plt.xlim(0, 2000)
    plt.ylim(0, 600)
    plt.xticks(np.arange(0, 2001, 400))
    plt.yticks(np.arange(0, 601, 200))
    plt.xlabel("[m]")
    plt.ylabel("[m]")
    ax = fig.add_subplot(111)
    y = 600 - np.array(y)
    # ax.plot(xs, ys, 'r-', lw=4, label='flight path')
    ax.plot(xs, ys, 'r-', lw=4)
    ax.scatter(x, y, s=50)
    for i, (xx, yy, px, py) in enumerate(zip(x, y, pulse_x, pulse_y)):

        if i <= len(echo_point_list):
            if len(echo_point_list) == 0:
                c_r = 204 / 255
                c_g = 204 / 255
                c_b = 204 / 255
            else:
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
                    print("power:",echo_point[2])
                    ax.scatter(eval(echo_point[1][i])[0], 600-eval(echo_point[1][i])[1],
                               s=300*echo_point[2][i]/power_max,
                               c=np.array((c_r, c_g, c_b)),
                               marker="s",
                               alpha=echo_point[2][i] / power_max)
    ax.plot((600, 600), (600, 336), 'k-', lw=4, linestyle="dashed")
    ax.plot((1000, 1000), (0, 264), 'k-', lw=4, linestyle="dashed")
    ax.plot((1400, 1400), (600, 336), 'k-', lw=4, linestyle="dashed")
    ax.set_xticklabels(["0",
                        "1",
                        "2",
                        "3",
                        "4.0",
                        "5.0"])
    ax.set_yticklabels(["0",
                        "0.5",
                        "1.0",
                        "1.5"])
    ax.legend()
    plt.plot(xs, ys)
    path = "/".join(txt_name.split("/")[:7])
    if not os.path.exists(path+"/png_tmp"):
        os.makedirs(path+"/png_tmp")
    plt.savefig("{}/png_tmp/{}.png".format(path,
                                           txt_name.split("/")[-1].split(".")[0]))
    # for i in range(len(xs)):
    #    cv2.circle(base_img, (int(ys[i]), int(xs[i])), 2, (255, 255, 0), -1)

    # return base_img


def __calc_poly(flight_path):
    x = [int(float(i)) for i in flight_path[0] if (i != "\n" and i != "")]
    y = [int(float(i)) for i in flight_path[1] if (i != "\n" and i != "")]
    w = np.polyfit(x, y, 10)

    return w, x, y


def calc_echo_point(target_csv):
    print("target_csv:", os.path.basename(target_csv))
    emit_data_line = 1000
    echo_data_line_p = 1000
    echo_power_line = 1000
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
                if idx == emit_data_line:
                    emit_data = data
                if idx == echo_data_line_p:
                    echo_data = data
                if idx == echo_power_line:
                    echo_power = data
            else:
                emit_data = ['(-100, -100)']
                echo_data = ['(-100, -100)']
                echo_power = ['0']
    return emit_data, echo_data, echo_power


def calc_path(data):
    with open("{}".format(data)) as f:
        txt_line = f.readlines()
    for txt in txt_line:
        if txt.split(",")[0] == "index":
            flight_path_x = txt.split(",")[1:]
        elif txt.split(",")[0] == "columns":
            flight_path_z = txt.split(",")[1:]
        elif txt.split(",")[0] == "pulse_direction":
            pulse_dir = txt.split(",")[1:]
        else:
            pass

    return [flight_path_z, flight_path_x], pulse_dir


def main(txt_data, csv_list):
    echo_data_p_list = []
    print("txt target:", txt_data)
    flight_path, pulse_dir = calc_path(txt_data)
    w, x, y = __calc_poly(flight_path)
    power_max = 0
    for idx, csv_data in enumerate(csv_list):
        print("csv target:", csv_data)
        _, echo_data_p, echo_power = calc_echo_point(csv_data)
        echo_power_data = [int(power) for power in echo_power]
        echo_data_p_list.append((idx, echo_data_p, echo_power_data))
        # emit power 2360
        if power_max < max(echo_power_data):
            power_max = max(echo_power_data)
    print("max power:", power_max)
    __make_figure(txt_data, w, x, y, pulse_dir, echo_data_p_list, power_max)


if __name__ == "__main__":
    argvs = sys.argv

    if len(argvs) < 2:
        print(
            "Usage: python {} [target pulse position txt] [folder of echo csv]".format(argvs[0]))
        exit()
    txt_data = argvs[1]
    csv_list = natsorted(glob.glob("{}/*.csv".format(argvs[2])))

    main(txt_data, csv_list)

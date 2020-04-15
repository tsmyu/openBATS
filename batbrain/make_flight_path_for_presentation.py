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


def __make_figure(fig, txt_data, pulse_dir, w, x, y):

    # 1 pixel = 0.25 mm
    #base_img = np.ones((600, 1800, 3), dtype=np.uint8)*255
    #cv2.line(base_img, (400, 0), (400, 264), (0, 0, 0), thickness=4)
    #cv2.line(base_img, (800, 600), (800, 336), (0, 0, 0), thickness=4)
    #cv2.line(base_img, (1200, 0), (1200, 264), (0, 0, 0), thickness=4)
    pulse_x = [100 * math.sin(math.radians(float(pulse))) for pulse in pulse_dir if (pulse != "\n" and pulse != "")]
    pulse_y = [100 * math.cos(math.radians(float(pulse))) for pulse in pulse_dir if (pulse != "\n" and pulse != "")]
    pulse_x = x + np.array(pulse_x)
    pulse_y = 600 - np.array(y) + np.array(pulse_y)
    xs = np.linspace(x[0], x[-1], len(x))
    ys = 600 - np.polyval(w, xs)
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
    # for i, (xx, yy, px, py) in enumerate(zip(x, y, pulse_x, pulse_y)):
    #     ax.annotate('', xy=(int(px), int(py)), xytext=(xx, yy),arrowprops=dict(shrink=0, width=1, headwidth=8, 
    #                         headlength=10, connectionstyle='arc3',
    #                         facecolor='gray', edgecolor='gray'))
    for i, (xx, yy, px, py) in enumerate(zip(x, y, pulse_x, pulse_y)):
        ax.annotate('', xy=(int(px), int(py)), xytext=(xx, yy),arrowprops=dict(shrink=0, width=1, headwidth=8, 
                            headlength=10, connectionstyle='arc3',
                            facecolor='gray', edgecolor='gray'))

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
    #for i in range(len(xs)):
    #    cv2.circle(base_img, (int(ys[i]), int(xs[i])), 2, (255, 255, 0), -1)

    #return base_img

def __calc_poly(flight_path):
    print(flight_path)
    x = [int(float(i)) for i in flight_path[0] if (i != "\n" and i != "")]
    y = [int(float(i)) for i in flight_path[1] if (i != "\n" and i != "")]
    w = np.polyfit(x, y, 12)
    
    return w, x, y

def calc_echo_point(target_csv):
    print("target_csv:", os.path.basename(target_csv))
    emit_data_line = 1000
    echo_data_line_p = 1000
    with open(target_csv, 'r') as f:
        data_map = csv.reader(f)
        for idx, data in enumerate(data_map):
            if len(data) > 0:
                if data[0] == "emit_points":
                    emit_data_line = idx + 1
                if data[0] == "echo_points":
                    echo_data_line_p = idx + 1
                if idx == emit_data_line:
                    emit_data = data
                if idx == echo_data_line_p:
                    echo_data = data
            else:
                emit_data = ['(-100, -100)']
                echo_data = ['(-100, -100)']

    return emit_data, echo_data

def calc_path(data):
    with open("{}".format(data)) as f:
        txt_line=f.readlines()
    for txt in txt_line:
        if txt.split(",")[0] == "index":
            flight_path_x=txt.split(",")[1:]
        elif txt.split(",")[0] == "columns":
            flight_path_z=txt.split(",")[1:]
        elif txt.split(",")[0] == "pulse_direction":
            pulse_dir=txt.split(",")[1:]
        else:
            pass

    return [flight_path_z, flight_path_x], pulse_dir

def main(txt_list):
    print("txt target1:", txt_data_1)
    print("txt target2:", txt_data_2)
    fig = plt.figure(figsize=(12, 5))
    for txt_data in txt_list:
        flight_path, pulse_dir = calc_path(txt_data)
        w, x, y = __calc_poly(flight_path)
        __make_figure(fig, txt_data, pulse_dir, w, x, y)
    path = "/".join(txt_data_1.split("/")[:7])
    if not os.path.exists(path+"/flight_png"):
        os.makedirs(path+"/flight_png")
    plt.savefig("{}/flight_png/{}.png".format(path, txt_data_1.split("/")[-1].split(".")[0]))


if __name__ == "__main__":
    argvs=sys.argv

    if len(argvs) < 2:
        print(
            "Usage: python {} [txt1] [txt2]".format(argvs[0]))
        exit()
    txt_data_1 = argvs[1]
    txt_data_2 = argvs[2]
    txt_list = [txt_data_1, txt_data_2]

    main(txt_list)

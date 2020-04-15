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


def remake(data):
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
    print(flight_path_x)
    input()
    index_list = []
    column_list = []
    for (x, z) in zip(flight_path_x, flight_path_z):
        index_list.append(int(x) * 5)
        column_list.append(int(z) * 5)
        
        


def main(txt_data):
    echo_data_p_list = []
    print("txt target:", txt_data)
    remake(txt_data)


if __name__ == "__main__":
    argvs = sys.argv

    if len(argvs) < 2:
        print(
            "Usage: python {} [target pulse position txt]".format(argvs[0]))
        exit()
    txt_data = argvs[1]

    main(txt_data)

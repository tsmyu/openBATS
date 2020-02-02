

import os
import sys
import glob
import cv2
from natsort import natsorted


def create_movie(bat_data, png_list):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(f'{os.path.basename(bat_data)}.mp4', fourcc, 1.0, (640, 480))

    for png in png_list:
        img = cv2.imread(png)
        img = cv2.resize(img, (640, 480))
        video.write(img)

    video.release()


def main(bat_list):
    for bat_data in bat_list:
        png_list = natsorted(glob.glob(f"{bat_data}/*.png"))
        create_movie(bat_data, png_list)


if __name__ == "__main__":
    argvs = sys.argv

    if len(argvs) < 2:
        print(
            "Usage: python {} [folder of png]".format(argvs[0]))
        exit()
    bat_list = glob.glob(f"{argvs[1]}/*")
    print(bat_list)
    main(bat_list)

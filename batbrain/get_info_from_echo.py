import os
import sys
import glob
import numpy as np
from scipy import signal
import itertools
import math
import re
import csv

# simulation parameter
dx = 2.5e-4
dt = 5.0e-7
calc_time = 4.0
# distance between nose and ears [m]
distance_ears = 0.002
velocity_air = 331.5
limit_time = distance_ears / velocity_air


class DataField:
    def __init__(self, input_data, emit_csv):
        # threash
        self.threash = 0.05
        self.base_name = input_data
        file_name = os.path.basename(input_data)
        self.pixel_area = (
            int(file_name.split("_")[0]), int(file_name.split("_")[1]))
        self.emit_angle = int(
            re.split("[g]", file_name.split("_")[-1].split(".")[0])[-1])
        self.emit_point = (int(os.path.basename(input_data).split("_")[6]), int(
            os.path.basename(input_data).split("_")[7]))
        emit_data_list = np.genfromtxt(emit_csv, usecols=(
            0, 1, 2, 3), skip_header=1, skip_footer=1, delimiter=",")

        echo_data_list = np.genfromtxt(input_data, usecols=(
            0, 1, 2, 3), skip_header=1, skip_footer=1, delimiter=",")

        self.time_line = echo_data_list[:, 0]
        self.emit_wave = echo_data_list[:, 1]
        right_wave = echo_data_list[:, 2] - \
            emit_data_list[:, 2][:len(echo_data_list[:, 2])]
        left_wave = echo_data_list[:, 3] - \
            emit_data_list[:, 3][:len(echo_data_list[:, 3])]
        self.echo_without_emit_wave = [right_wave, left_wave]

    def preprocessing(self):
        # correlation
        corr_wave = self.__get_correlation()
        # envelope
        corr_wave_envelop = self.__get_envelope(corr_wave)
        # notmalize
        corr_wave_envelop_norm = self.__normalization(
            corr_wave_envelop, "echo")
        # set length of data
        self.data_len = min(len(self.echo_without_emit_wave[0]), len(
            corr_wave[1]), len(self.time_line))

        self.right_corr = corr_wave_envelop_norm[1][:self.data_len]
        self.left_corr = corr_wave_envelop_norm[2][:self.data_len]

    def __get_correlation(self):
        """
        calc correlation value
        """
        corr_emit = np.correlate(self.emit_wave, self.emit_wave, "full")[
            len(self.emit_wave):]
        corr_right = np.correlate(self.echo_without_emit_wave[0], self.emit_wave, "full")[
            len(self.emit_wave):]
        corr_left = np.correlate(self.echo_without_emit_wave[1], self.emit_wave, "full")[
            len(self.emit_wave):]

        return [corr_emit, corr_right, corr_left]

    def __get_envelope(self, corr_wave):
        envelope_emit = abs(signal.hilbert(corr_wave[0]))
        envelope_right = abs(signal.hilbert(corr_wave[1]))
        envelope_left = abs(signal.hilbert(corr_wave[2]))

        return [envelope_emit, envelope_right, envelope_left]

    def __normalization(self, wave_list, base="emit"):
        """
        normalize data
        """
        norm_list = []
        if base == "emit":
            for wave in wave_list:
                wave = np.array(wave)
                wave = wave - wave.mean()
                wave_max = max(np.array(self.emit_wave))
                wave_n = wave / wave_max
                norm_list.append(wave_n)
        elif base == "echo":
            for wave in wave_list:
                wave = np.array(wave)
                wave = wave - wave.mean()
                wave_max = max(wave)
                wave_n = wave / wave_max
                norm_list.append(wave_n)

        return norm_list

    def get_info(self):
        right_echo_point, left_echo_point = self.__get_echo_point()
        self.right_echo_time, self.left_echo_time = self.__get_echo_time(
            right_echo_point, left_echo_point)
        self.right_echo_power, self.left_echo_power = self.__get_echo_power(
            right_echo_point, left_echo_point)
        self.distance_list, self.angle_list = self.__get_distance_angle()
        self.echo_points = self.__get_echo_points()

    def __get_echo_point(self):
        right_echo_point_raw = np.where(self.right_corr >= self.threash)[0]
        left_echo_point_raw = np.where(self.left_corr >= self.threash)[0]
        right_echo_list = []
        left_echo_list = []
        pre_r_point = 0
        pre_point = 0
        for r_point in right_echo_point_raw:
            if r_point - pre_r_point != 1:
                if pre_point == 0:
                    pre_point = r_point
                elif pre_point != 0:
                    echo_point = round((pre_point+r_point)/2)
                    right_echo_list.append(echo_point)
                    pre_point = 0
            pre_r_point = r_point
        pre_l_point = 0
        pre_point = 0
        for l_point in left_echo_point_raw:
            if l_point - pre_l_point != 1:
                if pre_point == 0:
                    pre_point = l_point
                elif pre_point != 0:
                    echo_point = round((pre_point+l_point)/2)
                    left_echo_list.append(echo_point)
                    pre_point = 0
            pre_l_point = l_point

        return right_echo_list, left_echo_list

    def __get_echo_time(self, right_points, left_points):

        return self.time_line[right_points], self.time_line[left_points]

    def __get_echo_power(self, right_points, left_points):

        return self.right_corr[right_points], self.left_corr[left_points]

    def __get_distance_angle(self):
        if min(len(self.right_echo_time), len(self.left_echo_time)) == 0:
            distance_list = [-100]
            angle_list = [-100]
        else:
            distance_list, angle_list = self.__calc_distance_angle()

        return distance_list, angle_list

    def __calc_distance_angle(self):
        distance_list = []
        angle_list = []
        for right_tim in self.right_echo_time:
            tmp_distance_list = []
            tmp_angle_list = []
            for left_tim in self.left_echo_time:
                if abs(right_tim - left_tim) < limit_time:
                    distance = (right_tim + left_tim) * velocity_air / 4
                    if -1 < (right_tim-left_tim)*velocity_air/distance_ears < 1:
                        angle = math.asin((right_tim-left_tim) *
                                          velocity_air / distance_ears)
                        tmp_angle_list.append(math.degrees(angle))
                        tmp_distance_list.append(distance)
            distance_list.append(tmp_distance_list)
            angle_list.append(tmp_angle_list)

        distance_l = list(itertools.chain(*distance_list))
        angle_l = list(itertools.chain(*angle_list))
        distance_list = sorted(set(distance_l), key=distance_l.index)
        angle_list = sorted(set(angle_l), key=angle_l.index)
        return distance_list, angle_list

    def __get_echo_points(self):
        pixel_points = []
        for (dis, ang) in zip(self.distance_list, self.angle_list):
            f_angle = math.radians(self.emit_angle - 90 + ang)
            pixel_points.append((self.emit_point[0]+round(dis*math.cos(f_angle)/dx),
                                 self.emit_point[1]+round(dis*math.sin(f_angle)/dx)))
        return pixel_points

    def save(self):
        print(self.right_echo_time)
        with open("./echo_point_{}/echo_point_{}.csv".format(os.path.basename(os.path.dirname(self.base_name)),
                                                             os.path.splitext(self.base_name)[0].split("/")[-1]), "a") as f:
            writer = csv.writer(f, lineterminator='\n')

            writer.writerow(["right_echo_point"])
            writer.writerow(self.right_echo_time)
            writer.writerow(["left_echo_point"])
            writer.writerow(self.left_echo_time)
            writer.writerow(["distance"])
            writer.writerow(self.distance_list)
            writer.writerow(["angle"])
            writer.writerow(self.angle_list)
            writer.writerow(["emit_points"])
            writer.writerow([self.emit_point])
            writer.writerow(["echo_points"])
            writer.writerow(self.echo_points)

        tim = self.time_line[:self.data_len][:, np.newaxis]
        echo_right = self.echo_without_emit_wave[0][:self.data_len][:, np.newaxis]
        echo_left = self.echo_without_emit_wave[1][:self.data_len][:, np.newaxis]
        corr_right = self.right_corr[:, np.newaxis]
        corr_left = self.left_corr[:, np.newaxis]
        data_for_csv = np.c_[tim, echo_right, echo_left, corr_right, corr_left]
        with open("./corr_{}/corr_{}.csv".format(os.path.basename(os.path.dirname(self.base_name)), os.path.splitext(self.base_name)[0].split("/")[-1]), "a") as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["tim", "echo_right", "echo_left",
                             "corr_right", "corr_left"])
            writer.writerows(data_for_csv)


def main(csv_list, emit_csv):
    """
    main for detect point from echo
    """
    for idx, csv in enumerate(csv_list):
        data_field = DataField(csv, emit_csv)
        print("analyze target:{}".format(csv))
        data_field.preprocessing()
        data_field.get_info()
        data_field.save()
        print("-------finish:{}th/{}-----------".format(idx+1, len(csv_list)))


if __name__ == "__main__":
    argvs = sys.argv
    if len(argvs) < 2:
        print(
            "Usage: python {} [folder of csv] [emit_pulse csv]".format(argvs[0]))
        exit()
    csv_list = sorted(glob.glob("{}/*.csv".format(argvs[1])))
    if not os.path.exists("./corr_{}".format(os.path.basename(argvs[1]))):
        os.makedirs("./corr_{}".format(os.path.basename(argvs[1])))
    if not os.path.exists("./echo_point_{}".format(os.path.basename(argvs[1]))):
        os.makedirs("./echo_point_{}".format(os.path.basename(argvs[1])))
    print("csv_list:{}".format(csv_list))
    emit_csv = argvs[2]
    main(csv_list, emit_csv)

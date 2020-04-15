import os
import json
import numpy as np
import copy
from PIL import Image
import matplotlib.pyplot as plt
import ABC

components_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
setting_file_path = components_dir + "settings.json"
with open(setting_file_path, "r") as setting_file_obj:
    config_param = json.load(setting_file_obj)
    gpu_flag = config_param["GPU"]

if gpu_flag:
    import chainer
    import cupy as cp


class Field:
    def __init__(self, f_image, abc_name, sound_speed, density, dt, dx):
        img = Image.open(f_image)
        self.width, self.height = img.size

        if abc_name == "Mur1":
            self.abc_field = ABC.Mur1(
                0, self.width, 0, self.height, sound_speed["air"], dt, dx)
        elif abc_name == "Mur2":
            u_list = np.zeros([self.width-2, self.height-2])
            self.abc_field = ABC.Mur2(
                0, self.width, 0, self.height, sound_speed["air"], dt, dx, density, u_list)

        self.ref_points = []
        field_arr = np.array(img, dtype=np.int32).T
        # (self.wall_area,
        # self.ref_points_w,
        # self.ref_points_h) = self.__read_field(field_arr)
        (self.wall_area,
         self.ref_points_w_p,
         self.ref_points_w_m,
         self.ref_points_h_p,
         self.ref_points_h_m) = self.__read_field(field_arr)
        self.velocity_arr, self.density_arr = self.__make_velocity_density_field(
            self.wall_area, sound_speed, density)

        R = 1
        self.coef_obs = ((1+R)*(sound_speed["air"]*dt/dx) - (1-R)) / \
            ((1 + R) * (sound_speed["air"] * dt / dx) + (1 - R))

    def __read_field(self, field_arr):
        # 障害物のエリア
        wall_points = np.where(field_arr == 255)
        width_diff = field_arr[:-1, :] - field_arr[1:, :]
        height_diff = field_arr[:, :-1] - field_arr[:, 1:]
        # -255の箇所は-1したポイントが反射地点
        ref_points_w_p = list(zip(*np.where(width_diff == 255)))
        ref_points_w_m_tmp = list(zip(*np.where(width_diff == -255)))
        ref_points_w_m = [(ref_point[0]+1, ref_point[1])
                          for ref_point in ref_points_w_m_tmp]
        ref_points_h_p = list(zip(*np.where(height_diff == 255)))
        ref_points_h_m_tmp = list(zip(*np.where(height_diff == -255)))
        ref_points_h_m = [(ref_point[0], ref_point[1]-1)
                          for ref_point in ref_points_h_m_tmp]
        ref_points_w = set(ref_points_w_p + ref_points_w_m)
        ref_points_h = set(ref_points_h_p + ref_points_h_m)

        # return wall_points, ref_points_w_p, ref_points_h

        return wall_points, ref_points_w_p, ref_points_w_m, ref_points_h_p, ref_points_h_m

        # ref_points_w_p = np.where(width_diff == 255)
        # ref_points_w_m = np.where(width_diff == -255)
        # if not len(ref_points_w_m_tmp[0]) == 0:
        #     ref_points_w_m = copy.copy(ref_points_w_m_tmp)
        #     ref_points_w_m[0] = ref_points_w_m_tmp[0] - 1
        # ref_points_h_p = np.where(height_diff == 255)
        # ref_points_h_m = np.where(height_diff == -255)
        # if not len(ref_points_h_m_tmp[1]) == 0:
        #     ref_points_h_m = copy.copy(ref_points_h_m_tmp)
        #     ref_points_h_m[1] = ref_points_h_m_tmp[1] - 1

        # if not len(ref_points_w_p) == 0:
        #     print(ref_points_w_p)
        #     ref_points_w_p = list(zip(*ref_points_w_p))
        # if not len(ref_points_w_m ) == 0:
        #     ref_points_w_m = list(zip(*ref_points_w_m))
        # if not len(ref_points_h_p) == 0:
        #     ref_points_h_p = list(zip(*ref_points_h_p))
        # if not len(ref_points_h_m) == 0:
        #     ref_points_h_m = list(zip(*ref_points_h_m))
        # ref_points_w = set(ref_points_w_p + ref_points_w_m)
        # ref_points_h = set(ref_points_h_p + ref_points_h_m)

        # return wall_points, ref_points_w, ref_points_h

    def __make_velocity_density_field(self, wall_area, sound_speed, density):
        density_arr = np.full((self.width, self.height),
                              density["air"], dtype=np.float32)
        density_arr[wall_area] = density["acrylic"]
        velocity_arr = np.full((self.width,self.height),
                               sound_speed["air"], dtype=np.float32)
        velocity_arr[wall_area] = sound_speed["acrylic"]

        return velocity_arr, density_arr

    def update(self, Pn, Pn1):
        # x方向の反射
        for (x, y) in self.ref_points_w_m:
            Pn1[x+1, y] = Pn[x, y] + self.coef_obs * \
                (Pn1[x, y] - Pn[x + 1, y])
        for (x, y) in self.ref_points_w_p:
            Pn1[x-1, y] = Pn[x, y] + self.coef_obs * \
                (Pn1[x, y] - Pn[x-1, y])
        # y方向の反射
        for (x, y) in self.ref_points_h_m:
            Pn1[x, y+1] = Pn[x, y] + self.coef_obs * \
                (Pn1[x, y] - Pn[x, y + 1])
        for (x, y) in self.ref_points_h_p:
            Pn1[x, y-1] = Pn[x, y] + self.coef_obs * \
                (Pn1[x, y] - Pn[x, y-1])
        # 境界条件（現在mur1）
        Pn1 = self.abc_field.calc(Pn, Pn1)
        # 障害物内音圧0
        # Pn1[self.wall_area] = 0

        return Pn1


class Field_GPU:
    def __init__(self, f_image, abc_name, sound_speed, density, dt, dx):
        img = Image.open(f_image)
        self.width, self.height = img.size

        if abc_name == "Mur1":
            import mur1

            self.abc_field = mur1.Mur1(
                0, self.width, 0, self.height, sound_speed["air"], dt, dx)

        self.ref_points = []
        field_arr = cp.array(img, dtype=cp.float32).T
        (self.wall_area,
         self.ref_points_w_p,
         self.ref_points_w_m,
         self.ref_points_h_p,
         self.ref_points_h_m) = self.__read_field(field_arr)
        self.velocity_arr, self.density_arr = self.__make_velocity_density_field(
            self.wall_area, sound_speed, density)

        R = 1
        self.coef_obs = ((1+R)*(sound_speed["air"]*dt/dx) - (1-R)) / \
            ((1 + R) * (sound_speed["air"] * dt / dx) + (1 - R))

    def __read_field(self, field_arr):
        # 障害物のエリア
        wall_points = cp.where(field_arr == 255)
        width_diff = field_arr[:-1, :] - field_arr[1:, :]
        height_diff = field_arr[:, :-1] - field_arr[:, 1:]
        # -255の箇所は-1したポイントが反射地点
        ref_points_w_p = list(zip(*cp.where(width_diff == 255)))
        ref_points_w_m_tmp = list(zip(*cp.where(width_diff == -255)))
        ref_points_w_m = [(ref_point[0]+1, ref_point[1])
                          for ref_point in ref_points_w_m_tmp]
        ref_points_h_p = list(zip(*cp.where(height_diff == 255)))
        ref_points_h_m_tmp = list(zip(*cp.where(height_diff == -255)))
        ref_points_h_m = [(ref_point[0], ref_point[1]-1)
                          for ref_point in ref_points_h_m_tmp]
        ref_points_w = set(ref_points_w_p + ref_points_w_m)
        ref_points_h = set(ref_points_h_p + ref_points_h_m)

        # return wall_points, ref_points_w_p, ref_points_h

        return wall_points, ref_points_w_p, ref_points_w_m, ref_points_h_p, ref_points_h_m

    def __make_velocity_density_field(self, wall_area, sound_speed, density):
        density_arr = cp.full((self.width, self.height),
                              density["air"], dtype=cp.float32)
        density_arr[wall_area] = density["acrylic"]
        velocity_arr = cp.full((self.width, self.height),
                               sound_speed["air"], dtype=cp.float32)
        velocity_arr[wall_area] = sound_speed["acrylic"]

        return velocity_arr, density_arr

    def update(self, Pn, Pn1):
        # x方向の反射
        for (x, y) in self.ref_points_w_m:
            Pn1[x+1, y] = Pn[x, y] + self.coef_obs * \
                (Pn1[x, y] - Pn[x + 1, y])
        for (x, y) in self.ref_points_w_p:
            Pn1[x-1, y] = Pn[x, y] + self.coef_obs * \
                (Pn1[x, y] - Pn[x-1, y])
        # y方向の反射
        for (x, y) in self.ref_points_h_m:
            Pn1[x, y+1] = Pn[x, y] + self.coef_obs * \
                (Pn1[x, y] - Pn[x, y + 1])
        for (x, y) in self.ref_points_h_p:
            Pn1[x, y-1] = Pn[x, y] + self.coef_obs * \
                (Pn1[x, y] - Pn[x, y-1])
        # 境界条件（現在mur1）
        Pn1 = self.abc_field.calc(Pn, Pn1)
        # 障害物内音圧0
        # Pn1[self.wall_area] = 0

        return Pn1

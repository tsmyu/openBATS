import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class Field:
    def __init__(self, f_image, abc_name, sound_speed_air, dt, dx):
        img = Image.open(f_image)
        self.width, self.height = img.size

        if abc_name == "Mur1":
            import mur1

            self.abc_field = mur1.Mur1(0, self.width, 0, self.height, sound_speed_air, dt, dx)

        self.ref_points = []
        field_arr = np.array(img).T
        (self.wall_points,
         self.ref_points_w,
         self.ref_points_h) = self.__read_field(field_arr)

        self.coef = 1

    def __read_field(self, field_arr):
        # 障害物のエリア
        wall_points = []
        # 障害物の反射面
        ref_points = []

        wall_points = np.where(field_arr == 255)
        width_diff = field_arr[:-1, :] - field_arr[1:, :]
        height_diff = field_arr[:, :-1] - field_arr[:, 1:]
        # -255の箇所は-1したポイントが反射地点
        ref_points_w_p = np.where(width_diff == 255)
        ref_points_w_m = np.where(width_diff == -255)
        if not len(ref_points_w_m[0]) == 0:
            print(ref_points_w_m)
            ref_points_w_m[0] = ref_points_w_m[0] - 1
        ref_points_h_p = np.where(height_diff == 255)
        ref_points_h_m = np.where(height_diff == -255)
        if not len(ref_points_h_m[0]) == 0:
            ref_points_h_m[1] = ref_points_h_m[1] - 1

        ref_points_w_p = list(zip(*ref_points_w_p))
        ref_points_w_m = list(zip(*ref_points_w_m))
        ref_points_h_p = list(zip(*ref_points_h_p))
        ref_points_h_m = list(zip(*ref_points_h_m))
        ref_points_w = set(ref_points_w_p + ref_points_w_m)
        ref_points_h = set(ref_points_h_p + ref_points_h_m)

        return wall_points, ref_points_w, ref_points_h

    def update(self, Pn, Pn1):
        for (x, y) in self.ref_points_w:
            Pn1[x, y] = Pn[x - 1, y] + self.coef * (Pn1[x - 1, y] - Pn[x, y])
        for (x, y) in self.ref_points_h:
            Pn1[x, y] = Pn[x, y - 1] + self.coef * (Pn1[x, y - 1] - Pn[x, y])
            # Pn1[x, y] = Pn[x + 1, y] + self.coef * (Pn1[x + 1, y] - Pn[x, y])
            # Pn1[x, y] = Pn[x, y + 1] + self.coef * (Pn1[x, y + 1] - Pn[x, y])

        Pn1 = self.abc_field.calc(Pn, Pn1)

        Pn1[self.wall_points] = 0

        return Pn1

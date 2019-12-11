import numpy as np
from PIL import Image


class Field:
    def __init__(self, filed_image,
                 abc_name,
                 sound_speed_air,
                 dt,
                 dx):
        img = Image.open(f_image)
        self.width, self.height = img.size

        if abc_name == "Mur1":
            import mur1
            self.abc_field = mur1.Mur1(0,
                                       self.width,
                                       0,
                                       self.height,
                                       sound_speed_air,
                                       dt,
                                       dx)
        
        field_arr = np.array(img)
        


    def update(self, Pn, Pn1):
        Pn1[self.x_min, :] = Pn[self.x_min + 1, :] + self.coef * \
            (Pn1[self.x_min + 1, :] - Pn[self.x_min, :])
        Pn1[:, self.y_min] = Pn[:, self.y_min + 1] + self.coef * \
            (Pn1[:, self.y_min + 1] - Pn[:, self.y_min])
        Pn1[self.x_max, :] = Pn[self.x_max - 1, :] + self.coef * \
            (Pn1[self.x_max - 1, :] - Pn[self.x_max, :])
        Pn1[:, self.y_max] = Pn[:, self.y_max - 1] + self.coef * \
            (Pn1[:, self.y_max - 1] - Pn[:, self.y_max])
        
        Pn1 = self.abc_field.calc(Pn, Pn1)

        return Pn1

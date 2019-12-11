class Mur1:
    def __init__(self, x_min, x_max, y_min, y_max, v, dt, dx):
        self.x_min = x_min
        self.x_max = x_max - 1
        self.y_min = y_min
        self.y_max = y_max - 1
        self.v = v
        self.dt = dt
        self.dx = dx
        self.coef = (self.v * self.dt - self.dx) / (self.v * self.dt + self.dx)

    def calc(self, Pn, Pn1):
        Pn1[self.x_min, :] = Pn[self.x_min + 1, :] + self.coef * \
            (Pn1[self.x_min + 1, :] - Pn[self.x_min, :])
        Pn1[:, self.y_min] = Pn[:, self.y_min + 1] + self.coef * \
            (Pn1[:, self.y_min + 1] - Pn[:, self.y_min])
        Pn1[self.x_max, :] = Pn[self.x_max - 1, :] + self.coef * \
            (Pn1[self.x_max - 1, :] - Pn[self.x_max, :])
        Pn1[:, self.y_max] = Pn[:, self.y_max - 1] + self.coef * \
            (Pn1[:, self.y_max - 1] - Pn[:, self.y_max])

        return Pn1

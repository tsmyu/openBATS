class Mur1:
    def __init__(self, x_min, x_max, y_min, y_max, v, dt, dx):
        self.x_min = x_min
        self.x_max = x_max - 1
        self.y_min = y_min
        self.y_max = y_max - 1
        self.dt = dt
        self.dx = dx
        # self.coef = (v * self.dt - self.dx) / (v * self.dt + self.dx)
        R = 0
        self.coef = ((1+R)*(v*dt/dx) - (1-R)) / \
            ((1+R)*(v*dt/dx) + (1-R))

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


class Mur2:
    def __init__(self, x_min, x_max, y_min, y_max, v, dt, dx, density, u_list):
        self.x_min = x_min
        self.x_max = x_max - 1
        self.y_min = y_min
        self.y_max = y_max - 1
        self.dt = dt
        self.dx = dx
        self.density = density["air"]
        self.u_list_x = u_list[0:2, :]
        self.u_list_y = u_list[:, 0:2]
        # self.coef = (v * self.dt - self.dx) / (v * self.dt + self.dx)
        R = 0
        self.coef1 = ((1+R)*(v*dt/dx) - (1-R)) / \
            ((1 + R) * (v * dt / dx) + (1 - R))
        self.coef2 = self.density * v / 2 * (v * dt + dx)

    def calc(self, Pn, Pn1):
        self.u_list_x[0, :] = self.u_list_x[0, :] - \
            (self.dt/self.density*self.dx) * \
            (Pn[self.x_min, :] - Pn[self.x_max + 1, :])
        self.u_list_x[1, :] = self.u_list_x[1, :] - \
            (self.dt / self.density * self.dx) * \
            (Pn[self.x_max, :] - Pn[self.x_max - 1, :])
        self.u_list_x[:, 0] = self.u_list_x[:, 0] - \
            (self.dt/self.density*self.dx) * \
            (Pn[:, self.y_min] - Pn[:, self.y_min+1])
        self.u_list_x[:, 1] = self.u_list_x[:, 1] - \
            (self.dt/self.density*self.dx) * \
            (Pn[:, self.y_max] - Pn[:, self.y_max-1])

        # Pn1[self.x_min, :] = Pn[self.x_min + 1, :] + self.coef1 * \
        #     (Pn1[self.x_min + 1, :] - Pn[self.x_min, :]) - self.coef2*()
        # Pn1[:, self.y_min] = Pn[:, self.y_min + 1] + self.coef * \
        #     (Pn1[:, self.y_min + 1] - Pn[:, self.y_min])
        # Pn1[self.x_max, :] = Pn[self.x_max - 1, :] + self.coef * \
        #     (Pn1[self.x_max - 1, :] - Pn[self.x_max, :])
        # Pn1[:, self.y_max] = Pn[:, self.y_max - 1] + self.coef * \
        #     (Pn1[:, self.y_max - 1] - Pn[:, self.y_max])

        return Pn1

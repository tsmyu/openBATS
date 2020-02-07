import sympy as sp
from sympy import sqrt, symbols
import math
import matplotlib.pyplot as plt

dx = 2.5e-4
dt = 1e-7
distance_ears = 0.002
velocity_air = 331.5
print(distance_ears / velocity_air)
right_tim = 0.0000010000
left_tim = 0.0000005000
# right_tim = 0.0010520066
# left_tim = 0.0010515066
x = symbols("x", positive=True)
y = symbols("y", positive=True)
a1 = symbols("a1", positive=True)
a2 = symbols("a2", positive=True)
sp.var('x, y, a1, a2')
eq1 = (x-(distance_ears / 4)) ** 2 / a1 ** 2 + \
    y ** 2 / (a1 ** 2 - (distance_ears / 4) ** 2) - 1
eq2 = (x+(distance_ears / 4)) ** 2 / a2 ** 2 + \
    y ** 2 / (a2 ** 2 - (distance_ears / 4) ** 2) - 1
ans = sp.solve([eq1, eq2], [x, y])
for j in range(1, 11):
    l = 0.0001 * 10 * j

    for i in range(int(6e-6/dt)):
        right_tim = 2 * l / velocity_air + i * dt
        left_tim = 2 * l / velocity_air

        a_1 = right_tim * velocity_air / 2
        a_2 = left_tim * velocity_air / 2
        x_ = 0
        y_ = 0
        for i in range(4):
            x0 = ans[i][0].subs([(a1, a_1), (a2, a_2)])
            y0 = ans[i][1].subs([(a1, a_1), (a2, a_2)])
            if x0.is_real and y0.is_real:
                if y0 >= 0:
                    x_ = x0
                    y_ = y0
        if x_ == 0 and y_ == 0:
            print("done")
            break
        theta = math.atan(x_ / y_)
        print(
            f"rgiht_tim:{right_tim}, left_tim:{left_tim }, diff:{right_tim-left_tim}")
        print(f"degree_eclipse:{math.degrees(theta)}")

        angle = math.asin((right_tim-left_tim) *
                          velocity_air / distance_ears)
        if right_tim - left_tim > 0:
            angle = - angle
        print(f"degree_apro:{math.degrees(angle)}")
        plt.ylim(0,0.011)
        plt.scatter(math.degrees(theta), l, c="r", marker= "x")
        plt.scatter(math.degrees(angle), l, c="g", marker="x")
    
plt.show()

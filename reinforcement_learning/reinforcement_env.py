# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 13:30:56 2017

@author: p000526832
"""
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import *


Poly = 16
obj_num = 20
obstacle_agent_num = 3
NUM_EYES = 16


class Walls(object):
    def __init__(self, x0, y0, x1, y1):
        self.xList = [x0, x1]
        self.yList = [y0, y1]
        self.P_color = QColor(50, 50, 50)

    def addPoint(self, x, y):
        self.xList.append(x)
        self.yList.append(y)

    def Draw_Course(self, dc, i):
        dc.setPen(self.P_color)

        for j in range((i-1)*2*Poly+(i-1), i*2*Poly+(i-1)):
            dc.drawLine(self.xList[j], self.yList[j],
                        self.xList[j+1], self.yList[j+1])

    def Draw(self, dc):
        for i in range(0, len(self.xList)-1):
            dc.drawLine(self.xList[i], self.yList[i],
                        self.xList[i+1], self.yList[i+1])

    def IntersectLine(self, p0, v0, i):
        dp = [p0[0] - self.xList[i], p0[1] - self.yList[i]]
        v1 = [self.xList[i+1] - self.xList[i], self.yList[i+1] - self.yList[i]]

        denom = float(v1[1]*v0[0] - v1[0]*v0[1])
        if denom == 0.0:
            return [False, 1.0]

        ua = (v1[0] * dp[1] - v1[1] * dp[0])/denom
        ub = (v0[0]*dp[1] - v0[1] * dp[0])/denom

        if 0 < ua and ua < 1.0 and 0 < ub and ub < 1.0:
            return [True, ua]

        return [False, 1.0]

    def IntersectLines(self, p0, v0):

        tmpt = 1.0
        tmpf = False
        for i in range(0, len(self.xList)-1):
            f, t = self.IntersectLine(p0, v0, i)
            if f:
                tmpt = min(tmpt, t)
                tmpf = True

        return [tmpf, tmpt]

    def adLine(self, p0, i):
        dp = [p0[0] - self.xList[i], p0[1] - self.yList[i]]

        v = [self.xList[i+1] - self.xList[i], self.yList[i+1] - self.yList[i]]
        vl = (v[0]**2+v[1]**2)

        if(vl == 0.0):
            p1l = (dp[0]**2+dp[1]**2)**0.5
        else:
            t = max(0.0, min(1.0, (dp[0]*v[0] + dp[1]*v[1])/vl))

            p1 = [self.xList[i] + t * v[0] - p0[0],
                  self.yList[i] + t * v[1] - p0[1]]
            p1l = (p1[0]**2+p1[1]**2)**0.5

        return p1l

    def adLines(self, p0, d):

        for i in range(1, obj_num + 1):
            for j in range((i-1)*2*Poly+(i-1), i*2*Poly+(i-1)):
                if self.adLine(p0, j) <= d:
                    return True
#        for i in range(0, len(self.xList)-1):
#            if self.adLine( p0, i) <= d:
#                return True
        return False

    def adLines_wall(self, p0, d):
        for i in range(0, len(self.xList) - 1):
            if self.adLine(p0, i) <= d:
                return True

        return False

    def adLines_obstacle(self, p0, d):

        for i in range(1, obstacle_agent_num + 1):
            for j in range((i-1)*2*Poly+(i-1), i*2*Poly+(i-1)):
                if self.adLine(p0, j) <= d:
                    return True

        return False

    def crossLine(self, x, y, p0, j):
        xpb = (p0[0]-x)
        y21 = (self.yList[j+1] - self.yList[j])
        x21 = (self.xList[j+1] - self.xList[j])
        ypb = (p0[1]-y)

        denominator = xpb*y21 - x21*ypb
        if denominator == 0:
            r = 2.0
            s = 2

        else:

            r = (self.xList[j]*y21 + y*x21 - x*y21 -
                 self.yList[j]*x21) / denominator
            s = (x*ypb - self.xList[j]*ypb + self.yList[j]
                 * xpb - y*xpb) / (-1 * denominator)

            r = np.round(r, 1)

        return r, s

    def crossLines(self, x, y, p0):
        R = []
        for i in range(1, obj_num + 1):
            for j in range((i-1)*2*Poly+(i-1), i*2*Poly+(i-1)):
                r, s = self.crossLine(x, y, p0, j)
                r = np.round(r, 1)
                if 0 <= r <= 1 and 0 <= s <= 1:
                    R.append(r)

        if len(R) >= 1:
            return True, min(R)
        else:

            return False, 2.0

    def crossLines_wall(self, x, y, p0):
        R = []
        for i in range(0, len(self.xList) - 1):
            r, s = self.crossLine(x, y, p0, i)
            r = np.round(r, 1)
            if 0 <= r <= 1 and 0 <= s <= 1:
                R.append(r)

        if len(R) >= 1:
            return True, min(R)
        else:

            return False, 2.0

    def crossLines_obstacle(self, x, y, p0):
        R = []
        for i in range(1, obstacle_agent_num + 1):
            for j in range((i-1)*2*Poly+(i-1), i*2*Poly+(i-1)):
                r, s = self.crossLine(x, y, p0, j)
                r = np.round(r, 1)
                if 0 <= r <= 1 and 0 <= s <= 1:
                    R.append(r)

        if len(R) >= 1:
            return True, min(R)
        else:

            return False, 2.0

    def reset(self):
        self.xList = []
        self.yList = []


class Ball(object):
    def __init__(self, x, y, color, property=0):
        self.pos_x = x
        self.pos_y = y
        self.rad = 10

        self.property = property

        self.B_color = color
        self.P_color = QColor(50, 50, 50)

    def Draw(self, dc):
        dc.setPen(self.P_color)
        dc.setBrush(self.B_color)
        dc.drawEllipse(QPoint(self.pos_x, self.pos_y), self.rad, self.rad)

    def SetPos(self, x, y):
        self.pos_x = x
        self.pos_y = y

    def IntersectBall(self, p0, v0):
        # StackOverflow:Circle line-segment collision detection algorithm?
        # http://goo.gl/dk0yO1

        o = [-self.pos_x + p0[0], -self.pos_y + p0[1]]

        a = v0[0] ** 2 + v0[1] ** 2
        b = 2 * (o[0]*v0[0]+o[1]*v0[1])
        c = o[0] ** 2 + o[1] ** 2 - self.rad ** 2

        discriminant = float(b * b - 4 * a * c)

        if discriminant < 0:
            return [False, 1.0]

        discriminant = discriminant ** 0.5

        t1 = (- b - discriminant)/(2*a)
        t2 = (- b + discriminant)/(2*a)

        if t1 >= 0 and t1 <= 1.0:
            return [True, t1]

        if t2 >= 0 and t2 <= 1.0:
            return [True, t2]

        return [False, 1.0]


class Sens(object):
    def __init__(self, i):
        #        self.OffSetAngle   = - math.pi/3 + i * math.pi*2/3/NUM_EYES
        self.OffSetAngle = 2*math.pi*i/NUM_EYES
#        self.SightDistance = 0
        self.OverHang = 100.0
        self.obj = -1


class Agent(Ball):
    def __init__(self, canvasSize, x, y, epsilon=0.99, model=None):
        super(Agent, self).__init__(
            x, y, QColor(1, 1, 190)
        )
        self.dir_Angle = 0.0  # -math.pi/2.0
        self.speed = 10.0

        self.pos_x_max, self.pos_y_max = canvasSize
        self.pos_y_max = 460

        self.EYEs = [Sens(i) for i in range(0, NUM_EYES)]

    def Sens(self, Course, BBox, obstacle=None, i=0):
        self.EYE = self.EYEs[i]
        p = [self.pos_x + self.EYE.OverHang*math.cos(self.dir_Angle + self.EYE.OffSetAngle),
             self.pos_y - self.EYE.OverHang*math.sin(self.dir_Angle + self.EYE.OffSetAngle)]

        # Line Width = 6.0
        C, rC = Course.crossLines(self.pos_x, self.pos_y, p)
        B, rB = BBox.crossLines_wall(self.pos_x, self.pos_y, p)

        if C:
            self.EYE.obj = 1
            self.r = rC
        elif B:
            self.EYE.obj = 1
            self.r = rB

        else:
            self.EYE.obj = -1
            self.r = rC

        if obstacle is not None:

            O, rO = obstacle.crossLines_obstacle(self.pos_x, self.pos_y, p)
            if C:
                self.EYE.obj = 1
                self.r = rC
            elif B:
                self.EYE.obj = 1
                self.r = rB
            elif O:
                self.EYE.obj = 1
                self.r = rO
            else:
                self.EYE.obj = -1
                self.r = rC

    def Draw(self, dc):
        dc.setPen(self.P_color)
        for EYE in self.EYEs:

            dc.drawLine(self.pos_x, self.pos_y,
                        self.pos_x + EYE.OverHang *
                        math.cos(self.dir_Angle + EYE.OffSetAngle),
                        self.pos_y - EYE.OverHang*math.sin(self.dir_Angle + EYE.OffSetAngle))
        super(Agent, self).Draw(dc)

    def Move(self, WallsList):
        HitBoundary = False

        dp = [self.speed * math.cos(self.dir_Angle),
              -self.speed * math.sin(self.dir_Angle)]

        for w in WallsList:
            if w.IntersectLines([self.pos_x, self.pos_y], dp)[0]:
                dp = [0.0, 0.0]
                HitBoundary = True

        self.pos_x += dp[0]
        self.pos_y += dp[1]

        if not(self.pos_x > 0 and self.pos_x < self.pos_x_max
                and self.pos_y > 0 and self.pos_y < self.pos_y_max):
            HitBoundary = True

        return HitBoundary

    def Move_obstacle(self, WallsList):
        r = np.random.rand(1)

        action = [r, 1-r]
        self.speed = (action[0] + action[1])/2.0 * 10.0
        self.dir_Angle += math.atan((action[0] - action[1])
                                    * self.speed / 2.0 / 5.0)
        dp = [self.speed * math.cos(self.dir_Angle),
              -self.speed * math.sin(self.dir_Angle)]
        self.pos_x += dp[0]
        self.pos_y += dp[1]
        self.pos_x = max(0, min(self.pos_x, self.pos_x_max))
        self.pos_y = max(0, min(self.pos_y, self.pos_y_max))

    def HitBall(self, b):
        if ((b.pos_x - self.pos_x)**2+(b.pos_y - self.pos_y)**2)**0.5 < (self.rad + b.rad):
            return True
        return False


logger = logging.getLogger(__name__)


class Reinforcement_Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, obs_agent):
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        Rad = 20.0
        self.obs_agent = obs_agent

        self.Course = Walls(Rad*math.cos(np.pi * -16 / Poly) + 100, Rad*math.sin(np.pi * -16 / Poly) + 100,
                            Rad*math.cos(np.pi * -16 / Poly) + 100, Rad*math.sin(np.pi * -16 / Poly) + 100)

        for i in range(1, obj_num + 1):
            if i <= 4:
                for j in range(-Poly, Poly+1):
                    self.Course.addPoint(Rad*math.cos(np.pi * j / Poly) + 100,
                                         Rad*math.sin(np.pi * j / Poly) + i*100)
            elif 4 < i <= 8:
                for j in range(-Poly, Poly+1):
                    self.Course.addPoint(Rad*math.cos(np.pi * j / Poly) + 200,
                                         Rad*math.sin(np.pi * j / Poly) + (i-4)*100)
            elif 8 < i <= 12:
                for j in range(-Poly, Poly+1):
                    self.Course.addPoint(Rad*math.cos(np.pi * j / Poly) + 300,
                                         Rad*math.sin(np.pi * j / Poly) + (i-8)*100)
            elif 12 < i <= 16:
                for j in range(-Poly, Poly+1):
                    self.Course.addPoint(Rad*math.cos(np.pi * j / Poly) + 400,
                                         Rad*math.sin(np.pi * j / Poly) + (i-12)*100)
            elif 16 < i <= 20:
                for j in range(-Poly, Poly+1):
                    self.Course.addPoint(Rad*math.cos(np.pi * j / Poly) + 500,
                                         Rad*math.sin(np.pi * j / Poly) + (i-16)*100)
            else:
                pass
        self.Course.xList = self.Course.xList[2:]
        self.Course.yList = self.Course.yList[2:]

        # Agent
        self.A = Agent((620, 450), 320, 240)
        # Mono Sensor moving obstacle
        self.obstacle = []
        if self.obs_agent:
            self.B = Agent((600, 400), 50, 240)
            self.B.B_color = QColor(0, 0, 0)
            self.C = Agent((600, 400), 40, 240)
            self.C.B_color = QColor(0, 0, 0)
            self.D = Agent((600, 400), 30, 240)
            self.D.B_color = QColor(0, 0, 0)
            self.obstacle_agent = [self.B, self.C, self.D]
            self.obstacle = Walls(320, 240, 320, 240)
            self.obstacle.xList = self.obstacle.xList[2:]
            self.obstacle.yList = self.obstacle.yList[2:]

        # Outr Boundary Box
        self.BBox = Walls(630, 460, 10, 460)
        self.BBox.addPoint(10, 10)
        self.BBox.addPoint(630, 10)
        self.BBox.addPoint(630, 460)

        self.action_space = spaces.Box(
            np.array([-1., -1.]), np.array([+1., +1.]))
        self.observation_space = spaces.Discrete(1)
        self._seed()
        self.reset()
        self.viewer = None
        self.steps_beyond_done = None
        self._configure()
        self.step_num = 0

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        # moving obstacle
        if self.obs_agent:
            for OAgent in self.obstacle_agent:
                OAgent.Sens(self.Course, self.BBox)
                OAgent.Move_obstacle([self.BBox])
                for j in range(-Poly, Poly+1):
                    self.obstacle.addPoint(OAgent.rad*math.cos(np.pi * j / Poly) + OAgent.pos_x,
                                           OAgent.rad*math.sin(np.pi * j / Poly) + OAgent.pos_y)

        # Action Step
        self.A.speed = (action[0]+action[1])/2.0 * 10.0
        self.A.dir_Angle += math.atan((action[0] - action[1])
                                      * self.A.speed / 2.0 / 5.0)
        self.A.dir_Angle = (self.A.dir_Angle + np.pi) % (2 * np.pi) - np.pi
        done = self.A.Move([self.BBox])
        self.states = []
        self.distance = []
        for i in range(0, NUM_EYES):

            #            self.A.Sens(self.Course,self.BBox, self.obstacle, i)
            self.A.Sens(self.Course, self.BBox, None, i)

            if self.A.EYE.obj == 1:
                self.state = 1
            else:
                self.state = 0
            self.states.append(self.state)
            self.distance.append(self.A.r)

        # Reward
        proximity_reward = 0.0


#        elif self.Course.adLines([self.A.pos_x, self.A.pos_y], 10.0):
#            proximity_reward -= 5.0
        # 壁にぶつかったら罰則
#        if self.BBox.adLines_wall([self.A.pos_x, self.A.pos_y], 3.0):
#            proximity_reward -= 10.0

        if done:
            proximity_reward -= 10.0
        else:
            self.step_num = 0.1

        if self.Course.adLines([self.A.pos_x, self.A.pos_y], 5.0):
            proximity_reward -= 10.0
            done = True

#        if self.obstacle.adLines_obstacle([self.A.pos_x, self.A.pos_y], 5.0):
#            proximity_reward -= 10.0
#            done = True
        if self.obs_agent:
            self.obstacle.reset()

        reward = proximity_reward + self.step_num

        return np.array(self.distance), reward, done, {'AgentPos': (self.A.pos_x, self.A.pos_y), 'AgentDir': self.A.dir_Angle}

    def reset(self):
        self.state = (1,)
        self.A.pos_x = 320.0
        self.A.pos_y = 240.0
        self.A.dir_Angle = 0.0
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        return


class APPWINDOW(QWidget):
    def __init__(self, obs_agent, WorkerThread, parent=None, id=-1, title=None):
        super().__init__()

        self.obs_agent = obs_agent
        self.resize(640, 480)
        self.setWindowTitle(title)
        self.setStyleSheet("background-color : white;")

        self.World = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.OnTimer)
        self.timer.start(20)

        self.WThread = WorkerThread
        self.WThread.start()
        print("thread start")

    def SetWorld(self, World):
        self.World = World
        print("set world")

    def paintEvent(self, QPaintEvent):
        if self.World is not None:
            # Graphics Update
            qp = QPainter(self)

            qp.setPen(QColor(Qt.white))
            qp.setBrush(QColor(Qt.white))
            qp.drawRect(0, 0, 640, 480)

            for ag in [self.World.A]:
                ag.Draw(qp)

            if self.obs_agent:
                self.World.B.Draw(qp)
                self.World.C.Draw(qp)
                self.World.D.Draw(qp)
            self.World.BBox.Draw(qp)
            for i in range(1, obj_num + 1):
                self.World.Course.Draw_Course(qp, i)

    def OnTimer(self):
        self.update()

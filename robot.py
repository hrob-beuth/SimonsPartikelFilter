import numpy as np
from mapping import MAP_SIZE, bresenham
from numpy import pi


class Robot:
    # class for robot and particles

    def __init__(self, x=None, y=None, phi=None, w=0.001):
        self.x = x or MAP_SIZE // 2
        self.y = y or MAP_SIZE // 2
        self.phi = phi or np.random.uniform(0, 2 * pi)
        self.w = w
        self.oldRight = 0  # for odometry
        self.oldLeft = 0

    def setPosition(self, newX, newY, newPhi):
        self.x = max(0, min(MAP_SIZE, newX))
        self.y = max(0, min(MAP_SIZE, newY))
        self.phi = newPhi % (2 * pi)

    def getPosition(self):
        return self.x, self.y, self.phi

    def move(self, right, left):
        dright = right - self.oldRight  # angular velocity
        dleft = left - self.oldLeft
        # update postion of robot
        forward = -2.5 * (dleft + dright)  # wheel raduis = 5cm
        turn = 0.625 * (dright - dleft)  # axle length = 8cm
        self.setPosition(newX=self.x + forward * np.cos(self.phi),
                         newY=self.y + forward * np.sin(self.phi),
                         newPhi=self.phi + turn)
        # update old values
        self.oldLeft = left
        self.oldRight = right

    def measurementProbalilty(self, m, distance):
        # get position of self (sx, sy) and measurement (mx, my) in map
        sx = int(self.x + 0.5)
        sy = int(self.y + 0.5)
        mx = int(self.x + distance * np.cos(self.phi) + .5)
        my = int(self.y + distance * np.sin(self.phi) + .5)
        # check if robot not in valid position of map
        if (sx, sy) not in m.valid:
            return 0.0001  # very small prob
        # check if measurement is inside of map bounds
        if not ((0 <= mx < MAP_SIZE) and (0 <= my < MAP_SIZE)):
            return 0.0001  # very small prob
        # receive measurement trajectory
        path = bresenham(sx, sy, mx, my)
        # check if measurement passes through illegal wall
        for x, y in path[:-5]:
            if m.map[y,x] < 0.9:
                return 0.0001  # very small prob
        # if robot belief of wall corresponds to OBST in map, prob is higher
        return 1-m.map[sy,sx]

    def addNoise(self, position_noise, orientation_noise):
        self.setPosition(newX=self.x + position_noise * np.random.randn(),
                         newY=self.y + position_noise * np.random.randn(),
                         newPhi=self.phi + orientation_noise * np.random.randn())

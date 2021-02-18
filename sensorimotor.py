import numpy as np

class MB_Sensors:
    def __init__(self):
        self.accel = (0,0,0)
        self.gyro =(0,0,0)
        self.distance = 0
        self.key = 0
        self.pos_left = 0
        self.pos_right = 0
        self.image = np.zeros((80,60,3)) # image data

    def setValues(self, data):
        self.accel = tuple(data[0:3])
        self.gyro = tuple(data[3:6])
        self.distance = data[6]
        self.key = data[7]
        self.pos_left = data[8]
        self.pos_right = data[9]
        self.image = np.array(data[10:]).reshape((80,60,3))

class MB_Motor:
    def __init__(self):
        self.motor = [0,0]

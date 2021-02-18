import numpy as np

class Controller:

    def __init__(self, p, i, d):
        self.p = p
        self.i = i
        self.d = d
        self.errSum = 0.  # for integrating
        self.oldErr = 0.  # for differantiating


    def clamp(self, input, maxVal):
        return np.clip(input, -maxVal, +maxVal)

    def actuate(self, err):
        self.errSum = self.clamp(self.errSum + err, 10)
        errDiff = err - self.oldErr
        self.oldErr = err

        return self.clamp( self.p * err + 
                           self.i * self.errSum + 
                           self.d * errDiff, 4.0)

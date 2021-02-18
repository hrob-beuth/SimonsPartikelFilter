import numpy as np
import matplotlib.pyplot as plt

MAP_SIZE = 256
OBST = 0.001
NOOBST = 0.999
NOTVALID = 0.9995
HOLEWIDTH = 15  # pixels


class Map:

    def __init__(self):
        self.map = NOTVALID * np.ones((MAP_SIZE, MAP_SIZE))
        self.valid = []
        self.fig, self.ax = plt.subplots()

    def showMap(self, particleFilter=None, best=None):
        self.ax.imshow(self.map, cmap="gray")
        if particleFilter:
            xCoords = [p.x for p in particleFilter.particles]
            yCoords = [p.y for p in particleFilter.particles]
            sizes = [500 * p.w for p in particleFilter.particles]
            plt.scatter(xCoords, yCoords, s=sizes)
        if best:
            plt.scatter(best.x, best.y, s=1, c="r")
        plt.show(block=False)

    def updateMap(self, particleFilter=None, best=None):
        self.ax.clear()
        plt.imshow(self.map, cmap="gray")
        if particleFilter:
            xCoords = [p.x for p in particleFilter.particles]
            yCoords = [p.y for p in particleFilter.particles]
            sizes = [500 * p.w for p in particleFilter.particles]
            self.ax.scatter(xCoords, yCoords, s=sizes)
        if best:
            plt.scatter(best.x, best.y, s=500, c="r")
        plt.draw()
        plt.pause(0.0001)

def createMap():
    m = Map()
    m.showMap()
    m.map[4, 4:152] = OBST
    m.map[52, 4:102] = OBST
    m.map[152, 152:254] = OBST
    m.map[204, 102:254] = OBST
    m.map[4:52, 4] = OBST
    m.map[4:152, 152] = OBST
    m.map[52:204, 102] = OBST
    m.map[152:205, 254] = OBST
    m.map[5:52, 5:152] = NOOBST
    m.map[51:204, 103:152] = NOOBST
    m.map[153:204, 151:254] = NOOBST
    blurrMap(m)
    setValid(m)
    return m


def setPixel(m, i, j, val):
    for x in range(i - 1, i + 2):
        for y in range(j - 1, j + 2):
            if m.map[x, y] == NOOBST or m.map[x, y] == NOTVALID:
                m.map[x, y] = val
    return m


def blurrPixel(m, val1, val2):
    for x in range(1, MAP_SIZE - 1):
        for y in range(1, MAP_SIZE - 1):
            if m.map[x, y] == val1:
                setPixel(m, x, y, val2)
    return m


def blurrMap(m):
    val1 = OBST
    for i in range(HOLEWIDTH):
        val2 = OBST + (i + 1) / HOLEWIDTH * (NOOBST - OBST)
        blurrPixel(m, val1, val2)
        val1 = val2


def setValid(m):
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            if m.map[x, y] == NOOBST:
                m.valid.append((y, x))  # matrix index, to coords
    return m


def bresenham(x0, y0, x1, y1):
    # initialization
    dx = abs(int(x1 - x0))
    dy = - abs(int(y1 - y0))
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    path = []

    while True:
        path.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > dy:
            err += dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return path

if __name__ == "__main__":
    m = createMap()
    plt.show()
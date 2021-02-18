import numpy as np
import mapping as mapping
from robot import Robot


def initParticles(N, m):
    particles = []
    for _ in range(N):
        x, y = m.valid[np.random.randint(len(m.valid))]
        particles.append(Robot(x, y, w=1. / N))
    return particles


class ParticleFilter:

    def __init__(self, N, m):
        self.N = N
        self.particles = initParticles(N, m)
        self.map = m
        self.best = self.getBestParticle()

    def moveParticles(self, forward, turn):
        for p in self.particles:
            p.move(forward, turn)

    def normWeights(self):
        sum = np.sum(np.sqrt(p.w) for p in self.particles)  # p-norm p=2
        # sum = np.sqrt(np.sum([p.w ** 2 for p in self.particles]))
        for p in self.particles:
            p.w /= sum

    def evalParticles(self, measurement):
        for p in self.particles:
            p.w = p.measurementProbalilty(self.map, measurement)

    def getBestParticle(self):
        return max(self.particles, key=lambda x: x.w)

    def getEfficiency(self):
        self.normWeights()
        return 1/sum(p.w**2 for p in self.particles)

    def stateEstimation(self, forward, turn, measurement):
        self.moveParticles(forward, turn)
        self.addNoise()
        self.evalParticles(measurement)
        self.best = self.getBestParticle()
        eff = self.getEfficiency()
        if eff < 0.5*self.N:
            self.resample()
        return self.best

    def addNoise(self):
        for p in self.particles:
            p.addNoise(1, 0.08)

    def resample(self):
        sm = sum(p.w for p in self.particles)
        start = np.random.uniform(0, sm/self.N)
        newParticles = []
        i = 0
        s = self.particles[0].w
        for k in range(self.N):
            p = start + k*sm/self.N
            while s < p:
                i += 1
                s += self.particles[i].w
            x, y, phi = self.particles[i].getPosition()
            newParticles.append(Robot(x, y, phi, w = 1/self.N))
        self.particles = newParticles

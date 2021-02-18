from particle_filter import ParticleFilter
import mapping as mp
import numpy as np

m = mp.createMap()
p = ParticleFilter(100, m)
rob = p.best
m.showMap(p)

for i in range(1000):
    fakedist = 10
    print(i, fakedist, end=" ")
    rob = p.stateEstimation(0.1, -0.1, fakedist)
    m.updateMap(p, best=rob)

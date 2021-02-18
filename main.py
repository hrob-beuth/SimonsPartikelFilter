from particle_filter import ParticleFilter
import mapping as mp
from tcpip import TCPIP
from sensorimotor import MB_Sensors, MB_Motor
from controller import Controller
from numpy import pi 
import numpy as np
import matplotlib.pyplot as plt
from keycodes import KEY_A, KEY_D, KEY_S, KEY_W
from cnnModel.data import symbolNames
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['Tf_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

# Load CNN
cnn = tf.saved_model.load("/Users/vincirist/Documents/HackerMaterial/pythonWS/hrobml/SimonPySlam/cnnModel/model/symbolFinder")


m = mp.createMap()
p = ParticleFilter(100, m)
tcpip = TCPIP()
mbSen = MB_Sensors()
mbMot = MB_Motor()
rob = p.best
m.showMap(p)

phi = 0.0
phiReq = 0.0
psi = 0.0
psiReq = 0.0

phiController = Controller(p=180, i=150, d=80)
psiController = Controller(p=1, i=.01, d=0)

def updatePhi(phi):
    deltaPhi = 0.01 * mbSen.gyro[0] # angluar velocity pitch
    accel2 = -mbSen.accel[2] / 9.81 # accaleration 
    return (0.01-1) * (deltaPhi-phi) - (0.01) * accel2

def updatePsi(psi):
    return 2*pi/10.121*(mbSen.pos_right-mbSen.pos_left)

loopCnt = 0
while True:
    try:
        tcpip.receive(mbSen) # receive senor data
    except:
        m.showMap(p)    
    tcpip.send(mbMot) # send motor data

    
    phiReq = 0.0
    if KEY_A == mbSen.key:
        psiReq -= 0.01 # turn left
    elif KEY_D == mbSen.key:
        psiReq += 0.01 # turn right
    elif KEY_W == mbSen.key:
        phiReq = 0.01 # bend forward
    elif KEY_S == mbSen.key:
        phiReq = -0.01 # bend forward

    phi = updatePhi(phi)
    psi = updatePsi(psi)

    uPhi = phiController.actuate(err= phiReq-phi)
    uPsi = psiController.actuate(err= psiReq-psi)
    mbMot.motor = [uPhi - uPsi, uPhi + uPsi]

    rob = p.stateEstimation(mbSen.pos_right, mbSen.pos_left, mbSen.distance / 10)

    # run CNN
    im = np.array(mbSen.image).reshape((1,60,80,3))
    prediction = cnn(im)
    symbol = symbolNames[np.argmax(prediction)]
    maximum = np.max(prediction)
    print(symbol, maximum)

    # if loopCnt%100 == 0:
    #     m.updateMap(p, rob)
    # loopCnt += 1

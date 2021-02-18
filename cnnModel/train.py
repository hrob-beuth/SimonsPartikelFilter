from data import createTrainingSet, symbolNames
import tensorflow as tf
from MobileNet.mobileNetHenni import MobileNetV2_K210 as MobNet
import numpy as np
from pathlib import Path
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

dataPath = Path("/Users/vincirist/Documents/HackerMaterial/pythonWS/hrobml/SimonPySlam/cnnModel/MobileNet/data")
datafile = dataPath / "1000.npz"

if datafile.exists():
    print("Loading Training Data ...")
    data = np.load(datafile)
    xTrain, yTrain, xTest, yTest = data['a'], data['b'], data['c'], data['d']
else:
    print("Creating and Saving Training Data ... ")
    xTrain, yTrain, xTest, yTest = createTrainingSet(1000)
    np.savez_compressed(datafile, a=xTrain , b=yTrain, c=xTest, d=yTest)

model = MobNet(input_shape=(60,80,3), alpha=1, include_top=True, weights=None, classes=len(symbolNames))
print(model.summary())

print("Training Model ... ")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(xTrain, yTrain, epochs=1, validation_data=(xTest, yTest))

print("Save Model ... ")
model.save("./MobileNet/models", overwrite=True, include_optimizer=True)

print("Convert ")
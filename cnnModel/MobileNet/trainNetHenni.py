import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

from data import createTrainingSet, symbolNames 
import tensorflow as tf
from mobilenet_v2_k210 import MobileNetV2_K210
import model 
import numpy as np
if os.path.exists("./MobileNet/data/prepped/1000.npz"):
    print("Load existing data")
    data = np.load("./MobileNet/data/prepped/1000.npz")
    xTrn, yTrn, xTst, yTst = data['a'], data['b'], data['c'], data['d']

else:
    print("Create trainign data")
    xTrn, yTrn, xTst, yTst = createTrainingSet(1000)
    print("Done")
    np.saved_compressed("./MobileNet/data/prepped/1000.npz", a=xTrn, b=yTrn, c=xTst, d=yTst)


model = MobileNetV2_K210(input_shape=(60,80,3), alpha=1, include_top=True, weights=None, classes=len(symbolNames))
print(model.summary())
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(xTrn, yTrn, epochs=1, validation_data=(xTst,yTst))
print("save model")
model.save("./MobileNet/models/", overwrite=True, include_optimizer=True)
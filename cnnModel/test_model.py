import tensorflow as tf
from tensorflow.keras import Model as KModel
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

class Model(KModel):

    def __init__(self, dimension=(60,80,3), targets=9):
        super(Model, self).__init__()
        self.conv1 =  Conv2D(filters=16, kernel_size=3, strides=(2,2), input_shape=dimension, activation="relu")
        self.pool1 = MaxPool2D(pool_size=(2,2))
        self.conv2 =  Conv2D(filters=32, kernel_size=3, activation="relu")
        self.pool2 = MaxPool2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation="relu")
        self.dense2 = Dense(targets)
        scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.compile(optimizer="adam", loss=scce, metrics=["accuracy"])

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

    def train(self, xTrain, yTrain, xTest, yTest, epochs=40):
        # callCP = ModelCheckpoint(
        #     filepath='/Users/vincirist/Documents/HackerMaterial/pythonWS/hrobml/SimonPySlam/cnnModel/models/chpt{epoch:02d}.keras',
        #     monitor='val_loss',
        #     verbose=0,
        #     save_weights_only=True)
        callTB = TensorBoard(
            log_dir='/Users/vincirist/Documents/HackerMaterial/pythonWS/hrobml/SimonPySlam/cnnModel/tfLogs/',
            histogram_freq=0,
            write_graph=True,
            update_freq=100)
        callbacks = [callTB]
        self.fit(xTrain, yTrain, batch_size=64, epochs=epochs, validation_data=(xTest, yTest), callbacks=callbacks)

    def save(self):
        tf.saved_model.save(self, "/Users/vincirist/Documents/HackerMaterial/pythonWS/hrobml/SimonPySlam/cnnModel/model/symbolFinder")

        
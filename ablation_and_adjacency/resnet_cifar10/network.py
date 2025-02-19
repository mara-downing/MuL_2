#replace return statement with code to execute input on network and return classification number
import numpy as np
import tensorflow as tf

#For performance, load model outside of run_network function
model = tf.keras.models.load_model("resnet.h5")

def run_network(inputfeatures):
    newinputfeatures = np.array(inputfeatures[:])
    data = newinputfeatures.reshape(32,32,3)
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(32):
        for j in range(32):
            data[i][j][0] = (data[i][j][0] - mean[0])/std[0]
            data[i][j][1] = (data[i][j][1] - mean[1])/std[1]
            data[i][j][2] = (data[i][j][2] - mean[2])/std[2]
    classification = model.predict(np.array([data]),verbose=0).argmax()
    return classification

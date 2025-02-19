#replace return statement with code to execute input on network and return classification number
import numpy as np
import tensorflow as tf

#For performance, load model outside of run_network function
model = tf.keras.models.load_model("mnist_complicated.h5")

def run_network(inputfeatures):
    inputfeatures = np.array(inputfeatures)
    data = inputfeatures.reshape(28,28,1)
    classification = model.predict(np.array([data])).argmax()
    return classification

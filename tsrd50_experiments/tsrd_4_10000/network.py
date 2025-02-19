#replace return statement with code to execute input on network and return classification number
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("../../networks_and_inputs/tsrd50_4_10000/tsrd.keras")

def run_network(inputfeatures):
    inputfeatures = np.array(inputfeatures)
    tempinput = inputfeatures[np.newaxis]
    return model.predict(tempinput).argmax()

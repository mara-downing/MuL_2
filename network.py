#replace return statement with code to execute input on network and return classification number
import numpy as np
import tensorflow as tf

#For performance, load model outside of run_network function
model = tf.keras.models.load_model("REPLACE_WITH_FILENAME")

def run_network(inputfeatures):
    return 0
import sys
import re

import tensorflow as tf
import numpy as np
import random
import os
import math
from PIL import Image

inputlist = []
outputlist = []

datafile = open("tsrd_train_50.csv", "r")
data = datafile.readlines()

x_train = []
y_train = []
x_test = []
y_test = []

for line in data:
  if(line[-1] == "\n"):
    line = line[:-1]
  if(line == ""):
    break
  linelist = line.split(",")
  for i in range(len(linelist)):
      linelist[i] = int(linelist[i])
  inputlist += [linelist[0:-1]]
  outputlist += [linelist[-1]]
  # x_train += [linelist[0:-1]]
  # y_train += [linelist[-1]]

datafile.close()

# datafile = open("tsrd_test_small.csv", "r")
# data = datafile.readlines()

# for line in data:
#   if(line[-1] == "\n"):
#     line = line[:-1]
#   if(line == ""):
#     break
#   linelist = line.split(",")
#   for i in range(len(linelist)):
#       linelist[i] = int(linelist[i])
#   x_test += [linelist[0:-1]]
#   y_test += [linelist[-1]]

test_index = set(random.sample(range(0, len(inputlist)), (int)(len(inputlist) * 0.10)))
for i in range(len(inputlist)):
  if i in test_index:
    x_test.append(inputlist[i])
    y_test.append(outputlist[i])
  else:
    x_train.append(inputlist[i])
    y_train.append(outputlist[i])

for i in range(len(x_train)):
  for j in range(len(x_train[i])):
    x_train[i][j] = (float(x_train[i][j]))/255.0

for i in range(len(x_test)):
  for j in range(len(x_test[i])):
    x_test[i][j] = (float(x_test[i][j]))/255.0

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# x_train = tf.keras.utils.normalize(x_train, axis=-1)
# x_test = tf.keras.utils.normalize(x_test, axis=-1)

linear_model = tf.keras.Sequential([
    # tf.keras.layers.Flatten(input_shape=(9, 1)),
    tf.keras.layers.Dense(5000, activation=tf.nn.relu),
    tf.keras.layers.Dense(5000, activation=tf.nn.relu),
    tf.keras.layers.Dense(58)#, activation=tf.nn.softmax)
])
linear_model.compile(
    optimizer='sgd',#tf.optimizers.Adam(learning_rate=0.1),
    loss=loss_fn,
    metrics=['accuracy'])

history = linear_model.fit(
    x_train,
    y_train,
    epochs=50,    # Calculate validation results on 20% of the training data.
    validation_split = 0.2)
linear_model.evaluate(
    x_test,
    y_test,
    verbose=2)



networkdirname = "../networks_and_inputs/tsrd50_2_5000/"

#TODO: find how to create an h5 file from this network, put it into this folder and name mnist.keras
linear_model.save(networkdirname + "tsrd.keras")

# new_model = tf.keras.models.load_model(networkdirname + "mnist.keras")

# Show the model architecture
# new_model.summary()


#get a set of all correctly classified inputs (with output) for this network:
outfile = open(networkdirname + "correctinputs.csv", "w")
for i in range(len(x_test)):
    tempinput = x_test[i][np.newaxis]
    singleres = linear_model.predict(tempinput).argmax()
    if(singleres == y_test[i]):
        #code to put input into line in output file
        for elem in x_test[i]:
            outfile.write(str(round(elem * 255)) + ",")
        outfile.write(str(y_test[i]) + "\n")
    # else:
    #   print(str(i) + "," + str(singleres) + "," + str(y_test[i]))
    #   im = Image.new(mode="RGB", size=(150, 150))
    #   pixel = []
    #   count = 0
    #   for elem in x_test[i]:
    #         pixel += [int(round(elem * 255))]
    #         if(len(pixel) == 3):
    #           im.putpixel((count % 150,int(count/150)),(pixel[0],pixel[1],pixel[2],255))
    #           pixel = []
    #           count += 1
    #   im.save("tsrd_incorrect/" + str(i) + ".png")
for i in range(len(x_train)):
    tempinput = x_train[i][np.newaxis]
    singleres = linear_model.predict(tempinput).argmax()
    if(singleres == y_train[i]):
        #code to put input into line in output file
        for elem in x_train[i]:
            outfile.write(str(round(elem * 255)) + ",")
        outfile.write(str(y_train[i]) + "\n")

outfile.close()

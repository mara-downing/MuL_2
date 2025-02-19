import sys
from PIL import Image
import os
import re
import math

infile = open("input5.txt", "r")
data = infile.readlines()
infile.close()

data = data[2:]

for line in data:
    if "," in line:
        linelist = line.split(",")
        linelist = linelist[:-1]
        linelist = [math.floor(float(x)) for x in linelist]
        im = Image.new(mode="RGB", size=(28, 28))
        for i in range(len(linelist)):
            im.putpixel((math.floor(i%28),math.floor(i/28)), (linelist[i], linelist[i], linelist[i], 255))
        im.save("test5.png")
        exit()

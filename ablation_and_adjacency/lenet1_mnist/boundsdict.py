#replace the dictionary definition with one containing upper and lower bounds for each bounded input feature.

boundsdict = {}

#boundsdict[0] = (0,1)
for i in range(784):
    boundsdict[i] = (0,1)
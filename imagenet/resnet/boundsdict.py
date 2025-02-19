#replace the dictionary definition with one containing upper and lower bounds for each bounded input feature.

boundsdict = {}

# boundsdict[0] = (0,1)
for i in range(3*224*224):
    boundsdict[i] = (0,1)

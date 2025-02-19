import sys
import os
import re
import time
import random
import math
import numpy as np

from boundsdict import *
from denormalize import *
from network import *
from parseinputs import *

rejectedsamples = 0
acceptedsamples = 0

# def polar_to_rectangular(center, r, coords):
#     #center is the input that serves as the origin for the polar coordinates
#     #should return a set of rectangular coordinates
#     rectcoords = []
#     oneminuspastsines = 1
#     pastsines = 1
#     for i in range(len(coords)):
#         rectcoords += [center[i] + (pastsines * r * cos(coords[i]))]
#         oneminuspastsines = pastsines
#         pastsines *= sin(coords[i])
#     rectcoords += [center[-1] + (oneminuspastsines * r * sin(coords[-1]))]
#     return rectcoords

def createAdjacent(inputs, outputs, r):
    #inputs is nested list: each element is list of normalized input features
    #fills adjacency dictionary and returns it: each key is an element in inputs, each value is list of input keys within 2r distance
    #also prints to close_different.txt output file if adjacent inputs are classified differently (uses outputs list)
    adjdict = {}
    for i in range(len(inputs)):
        for j in range(i+1, len(inputs)):
            inner = 0
            for k in range(len(inputs[i])):
                inner += (inputs[i][k] - inputs[j][k])**2
            dist = math.sqrt(inner)
            if(dist < 2*r):
                if i in adjdict.keys():
                    adjdict[i] += [j]
                else:
                    adjdict[i] = [j]
                if j in adjdict.keys():
                    adjdict[j] += [i]
                else:
                    adjdict[j] = [i]
                #check outputs and print to close_different.txt if different classes
                if(outputs[i] != outputs[j]):
                    outfile = open(outputdirname + "/close_different.txt", "a")
                    outfile.write(str(i) + "," + str(j) + "\n")
                    outfile.close()
    return adjdict

def take_sample(center, r, indbounds, adj, inputs, pbs):
    global rejectedsamples
    global acceptedsamples
    #first generate a sample in polar coordinates using the bounds
    sample = generate_sample(center, r, indbounds)
    # print("Sample generated")
    #then check if outside bounds (look into global vars), reject if outside
    if(pbs == "u"):
        for i in range(len(sample)):
            if i in boundsdict.keys():
                if(sample[i] < boundsdict[i][0] or sample[i] > boundsdict[i][1]):
                    rejectedsamples += 1
                    return (center, -1)
    elif(pbs == "a"):
        for i in range(len(sample)):
            if i in boundsdict.keys():
                if(sample[i] < boundsdict[i][0]):
                    sample[i] = boundsdict[i][0]
                elif(sample[i] > boundsdict[i][1]):
                    sample[i] = boundsdict[i][1]
    #implied else: ignore
    #then, if any elements in adj (list pulled from adjdict), check distance to those
    for elem in adj:
        inner = 0
        for k in range(len(sample)):
            inner += (inputs[elem][k] - sample[k])**2
        dist = math.sqrt(inner)
        if(dist < r):
            rejectedsamples += 1
            return (center, -1)
    #if not too close to another, call run_network on input
    classification = run_network(sample)
    acceptedsamples += 1
    #return tuple: sample in rectangular and classification
    # print("Sample accepted")
    return (sample, classification)

def generate_sample(center, r, bounds):
    #should generate random values for each angle within the given bounds
    #Create and fill sample_rect with sample chosen from bounding rectangle
    #From http://corysimon.github.io/articles/uniformdistn-on-sphere/
    sample = [0] * len(center)
    distfromorigin = 0
    for i in range(len(sample)):
        distfromorigin += (sample[i] - 0)**2
    distfromorigin = math.sqrt(distfromorigin)
    while(distfromorigin < 0.0001):
        for i in range(len(sample)):
            sample[i] = np.random.normal(0,1)
        for i in range(len(sample)):
            distfromorigin += (sample[i] - 0)**2
        distfromorigin = math.sqrt(distfromorigin)
    u = np.random.uniform(0,1)
    for i in range(len(sample)):
        sample[i] = (sample[i]/distfromorigin)*r*(u**(1/len(center)))
        sample[i] = center[i] + sample[i]
    return sample

def individual_bounds(center, bounds):
    indbounds = []
    halves = 0
    for i in range(len(center)):
        if i in bounds.keys():
            if(bounds[i][0] == center[i]):
                indbounds += [1]
                halves += 1
            elif(bounds[i][1] == center[i]):
                indbounds += [-1]
                halves += 1
            else:
                indbounds += [0]
        else:
            indbounds += [0]
    return (halves, indbounds)



#initialize variables from cmd line input
inputsfile = sys.argv[1]
radius = float(sys.argv[2])
timelimit = float(sys.argv[3])
partialboundsstrat = sys.argv[4][0]#options are (u)se, (i)gnore, or (a)djust
weightingstrat = sys.argv[5][0]#options are (b)alanced or (u)niform
#use: guarantees every sample is exactly radius from its input and a valid sample given the network bounds. Uniformity is preserved, but may result in a high proportion of rejected samples
#ignore: Uniformity around hypersphere and radius from input is preserved but does not guarantee all generated inputs are within the network bounds. Guarantees no rejected samples
#adjust: Uniformity is not preserved, nor is radius from input. Generated samples are guaranteed to be along edge of allowed region but may be moved inwards by partial bounds. Guarantees no rejected inputs and all valid samples.
if(partialboundsstrat != "u" and partialboundsstrat != "a" and partialboundsstrat != "i"):
    assert(False)
if(weightingstrat != "b" and weightingstrat != "u"):
    assert(False)

#check if folder "output[specs]" exists, otherwise create
outputdirname = "output_b_r" + str(radius) + "_t" + str(timelimit) + "_" + partialboundsstrat + weightingstrat
if not os.path.isdir(outputdirname):
    os.mkdir(outputdirname)


starttime = time.time()
#read in inputs to lists:
(inputlist, outputlist) = parse_inputs(inputsfile)
# print(inputlist)
# print(outputlist)
adjacencies = createAdjacent(inputlist, outputlist, radius)

#create a list of dictionaries, each containing the bounds for each input:
boundslistlist = []
weightlist = []
for inp in inputlist:
    (numhalvestaken, boundslist) = individual_bounds(inp, boundsdict)
    boundslistlist += [boundslist]
    weightlist += [numhalvestaken]

# print(boundslistlist)
# print(weightlist)

maxnumhalves = 0
for elem in weightlist:
    if(elem > maxnumhalves):
        maxnumhalves = elem

if(weightingstrat == "u"):
    for i in range(len(weightlist)):
        weightlist[i] = 2**(maxnumhalves - weightlist[i])
else:
    for i in range(len(weightlist)):
        weightlist[i] = 1

# print(weightlist)

assert(len(boundslistlist) == len(inputlist))

# correctlist = [0]*len(inputlist)
# incorrectlist = [0]*len(inputlist)
#replacing with confusion matrix: first index is what it should've been, second is what it is, value is count
numoutputs = max(outputlist) + 1
cmcount = []
for i in range(numoutputs):
    cmcount += [[0]*numoutputs]

for i in range(len(inputlist)):
    outfile = open(outputdirname + "/input" + str(i) + ".txt", "w")
    outfile.write("Correct class: " + str(outputlist[i]) + "\n")
    outfile.close()

while(time.time() - starttime < timelimit):
    #decide which input to sample around using weightlist
    index = random.choices(range(len(inputlist)), weights=weightlist, k=1)[0]
    #call take_sample with input and relevant other info
    if index in adjacencies.keys():
        (sample, result) = take_sample(inputlist[index], radius, boundslistlist[index], adjacencies[index], inputlist, partialboundsstrat)
    else:
        (sample, result) = take_sample(inputlist[index], radius, boundslistlist[index], [], inputlist, partialboundsstrat)
    #with result, if result equals expected add tally to correctlist
    #otherwise add tally to incorrectlist and append sample and output to relevant file
    if(result != -1):
        if(result == outputlist[index]):
            cmcount[result][result] += 1
        else:
            cmcount[outputlist[index]][result] += 1
            outfile = open(outputdirname + "/input" + str(index) + ".txt", "a")
            denormsample = denormalize(sample)
            for elem in denormsample:
                outfile.write(str(elem) + ",")
            outfile.write(str(result) + "\n")
            outfile.close()

outfile = open(outputdirname + "/summary.txt", "w")
totalcorrect = 0
totalincorrect = 0
for i in range(len(cmcount)):
    for j in range(len(cmcount[i])):
        if(i == j):
            totalcorrect += cmcount[i][j]
        else:
            totalincorrect += cmcount[i][j]
assert(acceptedsamples == totalcorrect + totalincorrect)
print("Rejected samples: " + str(rejectedsamples))
print("Accepted samples: " + str(acceptedsamples))
outfile.write(str(100 * totalcorrect / (totalcorrect + totalincorrect)) + "% correctly classified of " + str((totalcorrect + totalincorrect)) + " samples (" + str(rejectedsamples) + " samples rejected)\n")
for i in range(len(cmcount)):
    for j in range(len(cmcount[i])-1):
        outfile.write(str(cmcount[i][j])+",")
    outfile.write(str(cmcount[i][-1])+"\n")
outfile.close()

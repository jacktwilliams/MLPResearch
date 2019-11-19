#!/usr/bin/python3
import numpy as np

file = open("Dataset.arff")
lines = file.read().split("\n")
lines = lines[36:] #throw away cruft
resCol = 30
print(lines[0])

counters = np.zeros(30, dtype=np.int32)
print(counters)
numLegit = 0

for line in lines:
    vals = line.split(",")
    if vals[resCol] == '1':  #if result is one
        numLegit += 1
        for i in range(0,30):  #iterate over every column, add one if that column is one
            if vals[i] == '1':
                counters[i] += 1

print(counters)
counters = list(map(lambda count: (count/numLegit), counters))
#counters = list(map(lambda count: abs(.5 - (count/numLegit)), counters))

for i in range(len(counters)):
    print(str(i) + ": " + str(counters[i]))

counters.sort()
print(counters)
print(numLegit)

# RightClick:20, SSLfinal_State:7, Iframe:22, Google_Index:27, on_mouseover:19, Redirect:18

selections = [20,7,22,27,19,18,resCol]
cleaned = []
for line in lines:
    vals = line.split(",")
    newline = ""
    for selection in selections:
        newline += (str(vals[selection]) + ",")
    newline = newline[:len(newline) - 1] # remove last comma
    cleaned.append(newline)    

#!/usr/bin/python3
import numpy as np
import mlp
import datetime

file = open("Dataset.arff")
lines = file.read().split("\n")
lines = lines[36:] #throw away cruft
resCol = 30

#feature selection based on results from featureSelection.py
selections = [20,7,22,27,19,18,resCol]
cleaned = []
for line in lines:
    vals = line.split(",")
    newline = []
    for selection in selections:
        newline.append(int(vals[selection]))
    cleaned.append(newline)    

cleaned = np.array(cleaned)
data = cleaned[:, :len(selections) - 1]
targets = cleaned[:,len(selections)-1]

#change -1 for "phishy" over to 0
newTargets = []
for target in targets:
    if target == -1:
        newTargets.append(0)
    else:
        newTargets.append(target)
targets = np.array(newTargets).reshape((len(targets),1))

def train_and_eval(num_iterations, learning_rate, num_neurons, momentum):
    split = int(len(data) * .9) #no ten-fold. Use 75% for training and rest for validation
    train = data[:split,:]
    train_tgt = targets[:split]
    validation = data[split:,:]
    val_tgt = targets[split:]
    p = mlp.mlp(train, train_tgt, num_neurons, momentum=momentum, outtype='logistic')
    p.mlptrain(train, train_tgt, learning_rate, num_iterations)
    conf_mat = p.confmat(validation, val_tgt)
    acc = get_accuracy(conf_mat)
    return acc

def get_accuracy(conf_mat):
    total = conf_mat.sum()
    correct = (conf_mat * np.eye(2)).sum()
    return correct / total

#print(train_and_eval(10000, .001, 10, .9))

#lets see what accuracy we can get manually
print(datetime.datetime.now())
hidden_to_itrs = {}
best_hidden = 0
best_acc = 0

for num_hidden in range(1,30):
    best_itrs = 0
    itrs_acc = 0
    for itrs in [1,100,1000]:
        acc = train_and_eval(itrs, .001, num_hidden, .9)
        print(str(num_hidden) + " : " + str(itrs) + " : " + str(acc))
        if acc > itrs_acc:
            best_itrs = itrs
            itrs_acc = acc
            hidden_to_itrs[num_hidden] = itrs

    if itrs_acc > best_acc:
        best_acc = itrs_acc
        best_hidden = num_hidden
        print(str(num_hidden) + " better than previous with accuracy of " + str(itrs_acc))

print(datetime.datetime.now())

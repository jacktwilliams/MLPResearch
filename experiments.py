#!/usr/bin/python3
import numpy as np
import mlp
import datetime
import BatAlgorithm

file = open("Dataset.arff")
lines = file.read().split("\n")
lines = lines[36:] #throw away cruft
resCol = 30

def pick_selections_and_clean(selections):
    cleaned = []
    for line in lines:
        vals = line.split(",")
        newline = []
        for selection in selections:
            newline.append(int(vals[selection]))
        cleaned.append(newline)
        
    cleaned = np.array(cleaned)
    data = cleaned[:, :len(selections) - 1]
    targets = cleaned[:, len(selections) - 1]
    #clean targets
    #change -1 for "phishy" over to 0
    newTargets = []
    for target in targets:
        if target == -1:
            newTargets.append(0)
        else:
            newTargets.append(target)
    targets = np.array(newTargets).reshape((len(targets),1))
    return data, targets


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

def train_eval_ten_fold(num_iterations, learning_rate, num_neurons, momentum):
    total = 0
    leave = 0
    for i in range(10):
        start = int(len(data) * leave)
        end = int(len(data) * (leave + .1))
        train = np.concatenate((data[:start,:], data[end:,:]))
        train_tgt = np.concatenate((targets[:start,:], targets[end:,:]))
        validation = data[start:end,:]
        val_tgt = targets[start:end,:]
        p = mlp.mlp(train, train_tgt, num_neurons, momentum=momentum, outtype='logistic')
        p.mlptrain(train, train_tgt, learning_rate, num_iterations)
        conf_mat = p.confmat(validation, val_tgt)
        acc = get_accuracy(conf_mat)
        total += acc
    return total / 10
        

def get_accuracy(conf_mat):
    total = conf_mat.sum()
    correct = (conf_mat * np.eye(2)).sum()
    return correct / total

def manual_parameter_discovery():
    print(datetime.datetime.now())
    hidden_to_itrs = {}
    best_hidden = 0
    best_acc = 0

    for num_hidden in range(1,10):
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

def train_eval_for_bat(D, sol):
    acc = train_eval_ten_fold(map_num_itrs(sol[0]), map_learning_rate(sol[1]),
                                      map_num_hidden(sol[2]), map_momentum(sol[3]))
    print("Accuracy: " + str(acc) + "\n")
    batResults.append((sol, acc))
    return 1 - acc

def map_num_itrs(val):
    x = int(val * 100 + 100)
    print("itrs " + str(x))
    return x

def map_learning_rate(val):
    x = val / 10
    print("learning " + str(x))
    return x

def map_num_hidden(val):
    x = int((val * 100) / 2 + 10)
    print("hidden " + str(x))
    return x

def map_momentum(val):
    x = val * 1.5 #range 0 - 1.5
    print ("momentum " + str(x))
    return x

def getBestBatSolution():
    bestAcc = 0
    bestSol = None
    for res in batResults:
        if res[1] > bestAcc:
            bestAcc = res[1]
            bestSol = res
    return bestSol

#feature selection based on results from featureSelection.py
selections = [20,7,22,27,19,18,resCol]
data,targets = pick_selections_and_clean(selections)

numItrs = 1

print("Starting Bat Experiment")
print(datetime.datetime.now())
batF = open("batResults.txt", "w")
for i in range(numItrs):
    bats = BatAlgorithm.BatAlgorithm(4, 1, 1, .5, .5, 0, 2, 0, 1, train_eval_for_bat)
    batResults = []
    bats.move_bat()
    sol = getBestBatSolution()
    batF.write(str(sol[1]))
batF.close()
batResults = None #gargage collect this long list
print("***Done with bats")
print(datetime.datetime.now())

print("Starting Manual runs")
print(datetime.datetime.now())
regF = open("manualResults.txt", "w")
for i in range(numItrs):
    regF.write(str(train_eval_ten_fold(100, .001, 1, .9)))
regF.close()
print("***Done with manual runs")
print(datetime.datetime.now())

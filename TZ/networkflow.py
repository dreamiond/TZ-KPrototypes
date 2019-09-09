#!/usr/bin/env python

import numpy as np
from kmodes.kprototypes_TZ import KPrototypes
import datetime

# fileNames = ['广东.csv','河北.csv','浙江.csv','陕西.csv','贵州.csv','辽宁.csv','湖北.csv','江苏.csv','北京.csv','上海.csv']
fileNames = ['kprototypes_nor_5000.csv']
resSyms = np.genfromtxt('kprototypes_nor_5000_res.csv',dtype=int,delimiter=',')[:,0]
resClas = np.genfromtxt('kprototypes_nor_5000_res.csv',dtype=str,delimiter=',')[:,1]
resMatrix = {}
for i in range(len(resSyms)):
    if resClas[i] != '':
        if resClas[i] == 'normal':
            resMatrix[resSyms[i]] = 0
        else:
            resMatrix[resSyms[i]] = 1

startTime = datetime.datetime.now()
preMatrix = {}
zeroIds = []
for idx in range(len(fileNames)):
    fileName = fileNames[idx]
    syms = np.genfromtxt(fileName, dtype=str, delimiter=',')[:, 0]
    X = np.genfromtxt(fileName, dtype=object, delimiter=',')[:, 1:]
    X[:, 0] = X[:, 0].astype(float)

    columnNum = len(X[0])
    kproto = KPrototypes(n_clusters=5, init='Huang', verbose=0,n_init=1)
    clusters = kproto.fit_predict(X, categorical=[columnNum-3, columnNum-2, columnNum-1])

    # Print cluster centroids of the trained model.
    print(kproto.cluster_centroids_)
    # Print training statistics
    print(kproto.cost_)
    print(kproto.n_iter_)

    for s, c in zip(syms, clusters):
        s = int(float(s))
        preMatrix[s] = c
        # if c == 0:
        #     zeroIds.append(s)
endTime = datetime.datetime.now()
print("聚类用时："+str(endTime-startTime))

clustNum = [0,0,0,0,0]
for i in preMatrix.keys():
    clustNum[preMatrix[i]] += 1
normalClust = np.argmax(clustNum)

for i in preMatrix.keys():
    if preMatrix[i] == normalClust:
        preMatrix[i] = 0
    else:
        preMatrix[i] = 1

tp = 0
fp = 0
fn = 0
tn = 0
for i in preMatrix.keys():
    if preMatrix[i] == 1 and resMatrix[i] == 1: tp += 1
    elif preMatrix[i] == 1 and resMatrix[i] == 0: fp += 1
    elif preMatrix[i] == 0 and resMatrix[i] == 1: fn += 1
    else: tn += 1

print('TP: '+str(tp))
print('FP: '+str(fp))
print('FN: '+str(fn))
print('TN: '+str(tn))
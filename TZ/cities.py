#!/usr/bin/env python

import numpy as np
from kmodes.kprototypes_TZ import KPrototypes
import matplotlib.pyplot as plt

fileNames = ['广东.csv','河北.csv','浙江.csv','陕西.csv','贵州.csv','辽宁.csv','湖北.csv','江苏.csv','北京.csv','上海.csv']
ks = [4,4,4,3,3,4,5,3,4,3]
# fileNames = ['广东.csv']

for idx in range(len(fileNames)):
    fileName = fileNames[idx]
    syms = np.genfromtxt(fileName, dtype=str, delimiter=',')[:, 0]
    X = np.genfromtxt(fileName, dtype=object, delimiter=',')[:, 1:]
    X[:, 0] = X[:, 0].astype(float)

    columnNum = len(X[0])
    # ks = []
    # costs = []
    kproto = KPrototypes(n_clusters=ks[idx], init='Huang', verbose=0, n_init=10)
    clusters = kproto.fit_predict(X, categorical=[columnNum - 2, columnNum - 1])

    # Print cluster centroids of the trained model.
    # print(kproto.cluster_centroids_)
    # Print training statistics
    # print("{} k取{} 时的损失为: {}".format(fileName, k, kproto.cost_ ))
    # print(kproto.n_iter_)

    # for s, c in zip(syms, clusters):
    #     print("Symbol: {}, cluster:{}".format(s, c))


    clust2name = {}
    for s,c in zip(syms,clusters):
        clust2name.setdefault(c,[]).append(s)

    for k in clust2name.keys():
        print("{}地区第{}类:".format(fileName,k))
        for n in clust2name[k]:
            print(n)
"""
对各省数据进行归一化处理
"""

import csv
import numpy
import sys

def kprototypesNor():
    fromPath = 'kprototypes.csv'
    toPath = 'kprototypes_nor.csv'

    X = numpy.genfromtxt(fromPath,dtype=object,delimiter=',')
    X[:,:-3] = X[:,:-3].astype(float)
    X[:,-3:] = X[:,-3:].astype(str)
    rows = len(X)
    columns = len(X[0])

    minNumInEachLine = numpy.zeros(columns, dtype=float)
    maxNumInEachLine = numpy.zeros(columns, dtype=float)
    for i in range(1, columns-3):
        maxNumInEachLine[i] = -sys.maxsize
        minNumInEachLine[i] = sys.maxsize
    for i in range(rows):
        for j in range(1, columns-3):
            maxNumInEachLine[j] = max(maxNumInEachLine[j],X[i][j])
            minNumInEachLine[j] = min(minNumInEachLine[j],X[i][j])

    for i in range(rows):
        for j in range(1, columns-3):
            if maxNumInEachLine[j] != minNumInEachLine[j]:
                X[i][j] = (X[i][j]-minNumInEachLine[j])/(maxNumInEachLine[j]-minNumInEachLine[j])

    with open(toPath, 'w', newline='') as f:
        f_csv = csv.writer(f)
        for row in X:
            f_csv.writerow(row)


def kmeansNor():
    fromPath = 'KDDTrain_PCA.csv'
    toPath = 'KDDTrain_nor.csv'

    X = numpy.genfromtxt(fromPath,dtype=float,delimiter=',')
    rows = len(X)
    columns = len(X[0])

    minNumInEachLine = numpy.zeros(columns,dtype=float)
    maxNumInEachLine = numpy.zeros(columns,dtype=float)
    for i in range(columns):
        maxNumInEachLine[i] = -sys.maxsize
        minNumInEachLine[i] = sys.maxsize
    for i in range(rows):
        for j in range(columns):
            maxNumInEachLine[j] = max(maxNumInEachLine[j],X[i][j])
            minNumInEachLine[j] = min(minNumInEachLine[j],X[i][j])

    for i in range(rows):
        for j in range(columns):
            if maxNumInEachLine[j] != minNumInEachLine[j]:
                X[i][j] = (X[i][j]-minNumInEachLine[j])/(maxNumInEachLine[j]-minNumInEachLine[j])

    with open(toPath, 'w', newline='') as f:
        f_csv = csv.writer(f)
        for row in X:
            f_csv.writerow(row)


def main():
    kprototypesNor()


if __name__ == '__main__':
    main()
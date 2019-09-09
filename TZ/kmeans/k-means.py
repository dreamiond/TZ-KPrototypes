import numpy as np
import matplotlib.pyplot as plt

# 加载数据
def loadDataSet(fileName):
    data = np.genfromtxt(fileName,dtype=float,delimiter=',')
    return data

def loadResData(fileName):
    res = np.genfromtxt(fileName,dtype=int,delimiter=',')
    return res

# 欧氏距离计算
def distEclud(vecA,vecB):
    return np.linalg.norm(vecA-vecB)  # 计算欧氏距离

# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m)) #
        centroids[i,:] = dataSet[index,:]
    return centroids

# k均值聚类
def KMeans(dataSet,k):

    m = np.shape(dataSet)[0]  #行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.zeros((m,2),dtype=float)
    clusterChange = True

    # 第1步 初始化centroids
    centroids = randCent(dataSet,k)
    while clusterChange:
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            # 遍历所有的质心
            #第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j,:],dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex,minDist**2
        #第 4 步：更新质心
        numOfPointsInClusters = np.zeros(k, dtype=float)
        totalVecOfThisCluster = np.zeros((k, len(dataSet[0])), dtype=float)
        for j in range(k):
            for idx in range(len(dataSet)):
                numOfPointsInClusters[int(clusterAssment[idx][0])] += 1
                totalVecOfThisCluster[int(clusterAssment[idx][0])] += dataSet[idx]
        for j in range(k):
            for idx in range(len(dataSet[0])):
                totalVecOfThisCluster[j][idx] /= numOfPointsInClusters[j]
            centroids[j] = totalVecOfThisCluster[j]

    print("Congratulations,cluster complete!")
    return centroids,clusterAssment

def showCluster(dataSet, clusterAssment):
    m,n = dataSet.shape
    for i in range(m):
        markIndex = int(clusterAssment[i,0])
        print('id: {}, symbol: {}'.format(i,markIndex))

def showConfusionMatrix(dataSet, res, clusterAssment, k):

    fact = {} # normal对应0, abnormal对应1
    for i in range(len(res)):
        if res[i][0] == 1:# abnormal
            fact[i] = 1
        else:
            fact[i] = 0

    numOfPointsInEachCluster = np.zeros(k,dtype=int)
    for ca in clusterAssment:
        numOfPointsInEachCluster[int(ca[0])] += 1
    normalClust = np.argmax(numOfPointsInEachCluster)

    predict = {}
    for i in range(len(clusterAssment)):
        if int(clusterAssment[i][0]) == normalClust:
            predict[i] = 0
        else:
            predict[i] = 1

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(len(fact)):
        if fact[i] == 1 and predict[i] == 1: TP += 1
        elif fact[i] == 1 and predict[i] == 0: FP += 1
        elif fact[i] == 0 and predict[i] == 1: FN += 1
        else: TN += 1
    print('TP: {}'.format(TP))
    print('FP: {}'.format(FP))
    print('FN: {}'.format(FN))
    print('TN: {}'.format(TN))

def main():
    dataSet = loadDataSet("KDDTrain_nor_20000.csv")
    res = loadResData('KDDTrain_nor_20000_res.csv')
    k = 5
    centroids, clusterAssment = KMeans(dataSet, k)
    # showCluster(dataSet,clusterAssment)
    showConfusionMatrix(dataSet,res,clusterAssment,k)


if __name__ == '__main__':
    main()
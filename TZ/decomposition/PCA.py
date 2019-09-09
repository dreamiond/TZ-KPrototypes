from sklearn.decomposition import PCA
import numpy
import csv

fromPath = 'KDDTrain.csv'
toPath = 'KDDTrain_PCA.csv'

def getArrFromFile(path = fromPath):
    """
    读取原始csv文件，返回numpy数组
    :param
    path: 原始csv文件路径 string类型 default:fromPath
    :return:
    X: 由原始文件生成的数组 二维numpy数组类型
    """
    X = numpy.genfromtxt(path,dtype=float,delimiter=',')[1:,:-2]
    return X

def writeFile(newX,path = toPath):
    """
    将降维后的数组写入csv文件
    :param newX: 降维后的数组 二维numpy类型
    :param path: 写入csv文件的路径 string类型 default:toPath
    :return:
    """
    with open(path, 'w', newline='') as f:
        f_csv = csv.writer(f)
        for row in newX:
            f_csv.writerow(row)

def decom(X,indices):
    """
    使用PCA方法降维的主函数
    :param X: 要进行降维的原始数组 二维numpy数组类型
    :param indices:数组中需要降维的列数 二维数组类型
            形式如[
                    [第1组需要降维的第1列列号，第一组需要降维的最后一列列号],
                    [第2组需要降维的第1列列号，第一组需要降维的最后一列列号],
                    …………
                    [最后一组需要降维的第1列列号，第一组需要降维的最后一列列号]
                ]
    :return:降维后的数组 二维numpy数组类型
    """
    numOfColumnsDontNeedToChange = len(X[0])
    for ind in indices:
        numOfColumnsDontNeedToChange -= (ind[-1]-ind[0]+1)

    newX = X[:,0:numOfColumnsDontNeedToChange]
    decompositionedArray = numpy.zeros((len(X), len(indices)),dtype=float)
    pca = PCA(n_components=1)
    for i in range(len(indices)):
        arr = X[:,indices[i][0]:indices[i][-1]+1]
        newArr = pca.fit_transform(arr)
        for j in range(len(newArr)):
            decompositionedArray[j][i] = newArr[j][0]

    return numpy.concatenate((newX,decompositionedArray), axis=1)

def main():
    X = getArrFromFile(fromPath)
    indices = [[38,40],[41,106],[107,117]]
    newX = decom(X,indices)
    writeFile(newX)

if __name__ == '__main__':
    main()
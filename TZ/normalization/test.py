from kmodes.util.dissim import num_TZ_dissim,cat_TZ_dissim
from sklearn.decomposition import PCA
import numpy
centroid = [
    [1,2,3],
    [5,6,6]
]
Xnum = [
    [54,2,44],
    [89,6,4],
    [1.5,0,-5],
    [5346,874,212]
]
centroid = numpy.array(centroid)
Xnum = numpy.array(Xnum)

x = numpy.array([[1,2,3],[2,3,3],[12938,9999,666],[54,11,21354]])
pca = PCA(n_components=1)
newx = pca.fit_transform(x)
print(newx)
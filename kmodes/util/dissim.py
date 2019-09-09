"""
Dissimilarity measures for clustering
"""

import numpy as np


def matching_dissim(a, b, **_):
    """Simple matching dissimilarity function"""
    return np.sum(a != b, axis=1)


def euclidean_dissim(a, b, **_):
    """Euclidean distance dissimilarity function"""
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("Missing values detected in numerical columns.")
    return np.sum((a - b) ** 2, axis=1)

# -----------------------------------------创新点2-----------------------------------------------
def cat_TZ_dissim(centroids, Xcat):
    """
    tz文字类距离算法
    Parameters
    -----------
    centroids:所有聚类中心 list类型
        形如[
            [1,2,3,4,6,7,8,9],
            [9,8,7,6,5,4,3,2,1]
            …………………………
        ]
    Xcat:所有要计算距离的数据 list类型
        形如[
            [1,2,3,4,6,7,8,9],
            [9,8,7,6,5,4,3,2,1]
            …………………………
        ]
    Return
    ------------
    聚类中心与其它点的距离 list类型
    [
        [第1个点与第一个中心的距离，第1个点与第二个中心的距离，第1个点与第三个中心的距离……],
        [第2个点与第一个中心的距离，第2个点与第二个中心的距离，第2个点与第三个中心的距离……],
        ………………
    ]
    """
    # pearson = np.corrcoef(Xcat)
    # maxR = 0
    # minR = 1
    # for row in pearson:
    #     for n in row:
    #         if n != 1:
    #             maxR = max(maxR,abs(n))
    #             minR = min(minR,abs(n))
    #
    # res = np.zeros((len(Xcat),len(centroids)),dtype=float)
    # for i in range(0,len(Xcat)):
    #     point = Xcat[i]
    #     arr = [point]
    #     for row in centroids:
    #         arr.append(row)
    #     arr = np.array(arr)
    #     cor = np.corrcoef(arr)[0][1:]
    #     for j in range(0, len(cor)):
    #         res[i][j] = (1 - (abs(cor[j]) - minR) / (maxR - minR + 0.001) + 0.001) * np.sum(point != centroids[j], axis=1)
    # return res


def num_TZ_dissim(centroids, Xnum, **_):
    """
    tz数值类距离算法
    Parameters
    -----------
    centroids:所有聚类中心 list类型
        形如[
            [1,2,3,4,6,7,8,9],
            [9,8,7,6,5,4,3,2,1]
            …………………………
        ]
    Xnum:所有要计算距离的数据 list类型
        形如[
            [1,2,3,4,6,7,8,9],
            [9,8,7,6,5,4,3,2,1]
            …………………………
        ]
    Return
    ------------
    聚类中心与其它点的距离 list类型
    [
        [第1个点与第一个中心的距离，第1个点与第二个中心的距离，第1个点与第三个中心的距离……],
        [第2个点与第一个中心的距离，第2个点与第二个中心的距离，第2个点与第三个中心的距离……],
        ………………
    ]
    """
    pearson = np.corrcoef(Xnum)
    maxR = 0
    minR = 1
    for row in pearson:
        for n in row:
            if n != 1:
                maxR = max(maxR,abs(n))
                minR = min(minR,abs(n))
    # for centroid in centroids:
    #     arr = [centroid]
    #     for row in Xnum:
    #         arr.append(row)
    #     arr = np.array(arr)
    #     cor = np.corrcoef(arr)[0][1:]  # 加权系数
    #     for i in range(0, len(cor)):
    #         cor[i] = 1 - (abs(cor[i]) - minR) / (maxR - minR + 0.001) + 0.001
    #     res = np.zeros(len(Xnum), dtype=float)
    #     for i in range(0, len(Xnum)):
    #         res[i] = np.linalg.norm(centroid - Xnum[i]) * cor[i]
    res = np.zeros((len(Xnum),len(centroids)),dtype=float)
    for i in range(0,len(Xnum)):
        point = Xnum[i]
        arr = [point]
        for row in centroids:
            arr.append(row)
        arr = np.array(arr)
        cor = np.corrcoef(arr)[0][1:]
        for j in range(0, len(cor)):
            res[i][j] = (1 - (abs(cor[j]) - minR) / (maxR - minR + 0.001) + 0.001) * np.linalg.norm(centroids[j]-point)
    return res
# ------------------------------------------------------------------------------------------------

def ng_dissim(a, b, X=None, membship=None):
    """Ng et al.'s dissimilarity measure, as presented in
    Michael K. Ng, Mark Junjie Li, Joshua Zhexue Huang, and Zengyou He, "On the
    Impact of Dissimilarity Measure in k-Modes Clustering Algorithm", IEEE
    Transactions on Pattern Analysis and Machine Intelligence, Vol. 29, No. 3,
    January, 2007

    This function can potentially speed up training convergence.

    Note that membship must be a rectangular array such that the
    len(membship) = len(a) and len(membship[i]) = X.shape[1]

    In case of missing membship, this function reverts back to
    matching dissimilarity (e.g., when predicting).
    """
    # Without membership, revert to matching dissimilarity
    if membship is None:
        return matching_dissim(a, b)

    def calc_cjr(b, X, memj, idr):
        """Num objects w/ category value x_{i,r} for rth attr in jth cluster"""
        xcids = np.where(memj == 1)
        return float((np.take(X, xcids, axis=0)[0][:, idr] == b[idr]).sum(0))

    def calc_dissim(b, X, memj, idr):
        # Size of jth cluster
        cj = float(np.sum(memj))
        return (1.0 - (calc_cjr(b, X, memj, idr) / cj)) if cj != 0.0 else 0.0

    if len(membship) != a.shape[0] and len(membship[0]) != X.shape[1]:
        raise ValueError("'membship' must be a rectangular array where "
                         "the number of rows in 'membship' equals the "
                         "number of rows in 'a' and the number of "
                         "columns in 'membship' equals the number of rows in 'X'.")

    return np.array([np.array([calc_dissim(b, X, membship[idj], idr)
                               if b[idr] == t else 1.0
                               for idr, t in enumerate(val_a)]).sum(0)
                     for idj, val_a in enumerate(a)])

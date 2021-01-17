from sklearn.cluster import KMeans
import numpy as np
from Dataloading.Datasets.InferenceDataset import InferenceDataset


def cluster_routine(net, overview, tilesize, num_clusters, batchsize):
    ids = InferenceDataset(overview, tilesize, tilesize, batchsize)
    result = ids.infer_flattened(net, True)
    result_features = np.array(list(result.values()))
    result_inds = list(result.keys())

    kmeans, ccff = kmeans_cluster(num_clusters, result_features)

    labeledim = np.ones((ids.tilecounty * tilesize, ids.tilecountx * tilesize))
    labeledim *= -1
    for i in range(len(ccff)):
        indices = result_inds[i]
        if ccff[i] == True: labeledim[indices[0] * tilesize : (indices[0] + 1) * tilesize, indices[1] * tilesize : (indices[1] + 1) * tilesize] = kmeans.labels_[i]


    return labeledim


def kmeans_cluster(num_clusters, result_features):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(result_features)
    traf = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(result_features)
    dists = traf[range(result_features.shape[0]), kmeans.labels_]
    ccff = kmeans_cluster_criterion(dists, kmeans)

    return kmeans, ccff


def kmeans_cluster_criterion(dists, kmeans):
    ccff = np.zeros((dists.shape)).astype(np.bool)
    for l in np.unique(kmeans.labels_):
        ccff[kmeans.labels_ == l] = dists[kmeans.labels_ == l] - np.mean(dists[kmeans.labels_ == l]) < 0
    return ccff
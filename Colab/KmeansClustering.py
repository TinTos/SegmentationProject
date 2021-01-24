from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np
from Dataloading.Datasets.InferenceDataset import InferenceDataset


def cluster_routine(net, overview, tilesize, num_clusters, batchsize, labelsureonly=True, thresh = 0.5, algo = 'kmeans'):
    ids = InferenceDataset(overview, tilesize, tilesize, batchsize)
    result = ids.infer_flattened(net, True)
    result_features = np.array(list(result.values()))
    result_inds = list(result.keys())

    labels, ccff = kmeans_cluster(num_clusters, result_features) if algo == 'kmeans' else gmm_cluster(num_clusters, result_features, thresh)

    labeledim = np.ones((ids.tilecounty * tilesize, ids.tilecountx * tilesize))
    labeledim *= -1
    for i in range(len(ccff)):
        indices = result_inds[i]
        if ccff[i] == True or not labelsureonly: labeledim[indices[0] * tilesize : (indices[0] + 1) * tilesize, indices[1] * tilesize : (indices[1] + 1) * tilesize] = labels[i]


    return labeledim


def kmeans_cluster(num_clusters, result_features):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(result_features)
    traf = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(result_features)
    dists = traf[range(result_features.shape[0]), kmeans.labels_]
    ccff = kmeans_cluster_criterion(dists, kmeans)

    return kmeans.labels_, ccff


def kmeans_cluster_criterion(dists, kmeans):
    ccff = np.zeros((dists.shape)).astype(np.bool)
    for l in np.unique(kmeans.labels_):
        ccff[kmeans.labels_ == l] = dists[kmeans.labels_ == l] - np.mean(dists[kmeans.labels_ == l]) < 0
    return ccff


def gmm_cluster(num_clusters, result_features, thresh):
    gmm = GaussianMixture(n_components=num_clusters, random_state=0).fit(result_features)
    labels = gmm.fit_predict(result_features)
    probs = gmm.predict_proba(result_features)

    ccff = gmm_cluster_criterion(probs, thresh)

    return labels, ccff


def gmm_cluster_criterion(probs, thresh):
    ccff = np.zeros((probs.shape[0])).astype(np.bool)
    for i in range(probs.shape[0]):
        ccff[i] = max(probs[i]) > thresh
    return ccff
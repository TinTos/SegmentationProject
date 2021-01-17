import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np

from Dataloading.Datasets.InferenceDataset import InferenceDataset
from TorchLearning.PretrainedModel import preprocess
from sklearn.cluster import KMeans

def training_routine(net, dataloader):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(1):  # loop over the dataset multiple times


        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics

            if i % 1 == 0:    # print every 2000 mini-batches
                print('[%d, %8d] loss: %.3f' %
                      (epoch + 1, i + 1, loss))


    print('Finished Training')


def inference_routine(net, dataloader, overview, tilesize, labelundecisive = False, decisionthresh = 0.5):

    with torch.no_grad():
        if (len(overview.shape) == 2):
            overview = overview.reshape((1, overview.shape[0], overview.shape[1]))

        isRGB = overview.shape[0] == 3

        tilecounty = int(overview.shape[1] // tilesize)
        tilecountx = int(overview.shape[2] // tilesize)
        result = np.zeros((tilecounty*tilesize, tilecountx*tilesize))

        net.eval()

        for i, data in enumerate(dataloader):
            inputs, inds = data

            inputs = preprocess(inputs, isRGB)

            probs, outputs = torch.max(torch.sigmoid(net(inputs)), 1)
            outputs = outputs.cpu().numpy()

            for count in range(inds[0].shape[0]):
                result[inds[0][count] * tilesize : (inds[0][count] + 1) * tilesize, inds[1][count] * tilesize : (inds[1][count] + 1) * tilesize] = outputs[count] if probs[count] >= decisionthresh or not labelundecisive else -1

            del inputs
            del outputs

            print(str(i) + "/" + str(int(len(dataloader.dataset) // dataloader.batch_size)))

        return result


def cluster_routine(net, dataloader, overview, tilesize, num_features, num_clusters):
    ids = InferenceDataset(overview, tilesize, tilesize, 256)
    result = ids.infer_flattened(net, True)
    result_features = np.array(list(result.values()))
    result_inds = list(result.keys())

    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(result_features)
    traf = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(result_features)
    dists = traf[range(result_features.shape[0]), kmeans.labels_]
    r = np.zeros((dists.shape)).astype(np.bool)
    for l in np.unique(kmeans.labels_):
        r[kmeans.labels_ == l] = dists[kmeans.labels_ == l] - np.mean(dists[kmeans.labels_ == l]) < 0
        #r[kmeans.labels_ == l] = True

    labeledim = np.ones((ids.tilecounty * tilesize, ids.tilecountx * tilesize))
    labeledim *= -1
    for i in range(len(r)):
        indices = result_inds[i]
        if r[i] == True: labeledim[indices[0] * tilesize : (indices[0] + 1) * tilesize, indices[1] * tilesize : (indices[1] + 1) * tilesize] = kmeans.labels_[i]


    return labeledim



from skimage.feature import local_binary_pattern


def cluster_routine_2(dataloader, overview, tilesize, num_features, num_clusters):
    with torch.no_grad():
        if (len(overview.shape) == 2):
            overview = overview.reshape((1, overview.shape[0], overview.shape[1]))

        isRGB = overview.shape[0] == 3

        tilecounty = int(overview.shape[1] // tilesize)
        tilecountx = int(overview.shape[2] // tilesize)

        num_features = tilesize * tilesize

        result = np.zeros((tilecounty * tilecountx,num_features))


        for i, data in enumerate(dataloader):
            inputs, inds = data

            #inputs = preprocess(inputs, isRGB).cpu().numpy()

            inputs = inputs.cpu().numpy()
            outputs = None

            for i in range(inputs.shape[0]):
                tmp = local_binary_pattern(inputs[i,0], 24, 3, 'uniform').reshape((tilesize * tilesize))
                if outputs is None:
                    outputs = np.zeros((inputs.shape[0], tmp.shape[0]))
                outputs[i] = tmp

            for count in range(inds[0].shape[0]):
                result[inds[0][count] * tilecountx + inds[1][count]] = outputs[count]

            del inputs
            del outputs

            print(str(i) + "/" + str(int(len(dataloader.dataset) // dataloader.batch_size)))


        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(result)
        traf = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(result)
        dists = traf[range(result.shape[0]), kmeans.labels_]
        r = np.zeros((dists.shape)).astype(np.bool)
        for l in np.unique(kmeans.labels_):
            r[kmeans.labels_ == l] = dists[kmeans.labels_ == l] - np.mean(dists[kmeans.labels_ == l]) < 0
            #r[kmeans.labels_ == l] = True


        labeledim = np.zeros((tilecounty, tilecountx))
        for i in range(len(r)):
            indices = (int(i // tilecountx), int(i % tilecountx))
            if r[i] == True: labeledim[indices[0], indices[1]] = kmeans.labels_[i]




        return labeledim




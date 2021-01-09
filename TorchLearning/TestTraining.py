import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from TorchLearning.PretrainedModel import preprocess
from TorchLearning.PretrainedModel import preprocess_for_custom
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


def cluster_routine(net, dataloader, overview, tilesize, num_features):

    with torch.no_grad():
        if (len(overview.shape) == 2):
            overview = overview.reshape((1, overview.shape[0], overview.shape[1]))

        isRGB = overview.shape[0] == 3

        tilecounty = int(overview.shape[1] // tilesize)
        tilecountx = int(overview.shape[2] // tilesize)
        result = np.zeros((num_features,tilecounty*tilesize, tilecountx*tilesize))

        net.eval()

        for i, data in enumerate(dataloader):
            inputs, inds = data

            inputs = preprocess(inputs, isRGB)

            outputs = torch.sigmoid(net(inputs)).cpu().numpy()

            for count in range(inds[0].shape[0]):
                result[:,inds[0][count] * tilesize : (inds[0][count] + 1) * tilesize, inds[1][count] * tilesize : (inds[1][count] + 1) * tilesize] = outputs[count]

            del inputs
            del outputs

            print(str(i) + "/" + str(int(len(dataloader.dataset) // dataloader.batch_size)))



        return result





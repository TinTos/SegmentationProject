import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np


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


def inference_routine(net, dataloader, overview, tilesize):
    if (len(overview.shape) == 2):
        overview = overview.reshape((1, overview.shape[0], overview.shape[1]))

    tilecounty = int(overview.shape[1] // tilesize)
    tilecountx = int(overview.shape[2] // tilesize)
    result = np.zeros((tilecounty, tilecountx))

    net.eval()

    for i, data in enumerate(dataloader):
        inputs, inds = data
        outputs = net(inputs)

        for count in range(inds[0].shape[0]):
            result[inds[0][count],inds[1][count]] = torch.sigmoid(outputs[count,0])

    return result





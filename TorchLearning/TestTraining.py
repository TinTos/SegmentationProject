import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, models, transforms

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
    with torch.no_grad():
        if (len(overview.shape) == 2):
            overview = overview.reshape((1, overview.shape[0], overview.shape[1]))

        tilecounty = int(overview.shape[1] // tilesize)
        tilecountx = int(overview.shape[2] // tilesize)
        result = np.zeros((tilecounty, tilecountx))

        net.eval()

        for i, data in enumerate(dataloader):
            inputs, inds = data

            inputs = torch.cat([inputs, inputs, inputs], dim=1)
            inputs = F.interpolate(inputs, size=224)
            inputs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs)

            outputs = torch.sigmoid(net(inputs)).cpu().numpy()

            #for count in range(inds[0].shape[0]):
            result[inds[0][:],inds[1][:]] = outputs[:,0]

            del inputs
            del outputs

        return result





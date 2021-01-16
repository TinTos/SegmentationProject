import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

class InferenceDataset():
    def __init__(self, overview, tilesize, stepsize, batchsize):
        self.tilesize = tilesize
        if len(overview.shape) == 2:
            self.channelnumber = 1
            self.overview = overview.reshape((1,overview.shape[0], overview.shape[1]))
        elif len(overview.shape) == 3:
            self.channelnumber = overview.shape[0]
            self.overview = overview

        self.tilecountx = int(overview.shape[-1] // stepsize)
        self.tilecounty = int(overview.shape[-2] // stepsize)
        while (self.tilecountx - 1) * stepsize + tilesize >= overview.shape[-1]:
            self.tilecountx -= 1
        while (self.tilecounty - 1) * stepsize + tilesize >= overview.shape[-2]:
            self.tilecounty -= 1

        self.tilesize = tilesize
        self.batchsize = batchsize
        self.stepsize = stepsize

    def __len__(self):
        return int(np.ceil(self.tilecountx * self.tilecounty / self.batchsize))

    def __getitem__(self, idx):
        bx = int((idx % (self.tilecounty * self.tilecountx)) % self.tilecountx)
        by = int((idx % (self.tilecounty * self.tilecountx)) // self.tilecountx)

        tile = self.overview[:, by * self.stepsize : by * self.stepsize + self.tilesize, bx * self.stepsize : bx * self.stepsize + self.tilesize]

        return tile.astype(np.float32), (by, bx)

    def get_batch(self, bi):
        batch = np.zeros((self.batchsize, self.channelnumber, self.tilesize, self.tilesize)).astype(np.float32)
        inds = []
        c=0
        for i in range(bi * self.batchsize, (bi+1) * self.batchsize):
            tile, ind = self[i]
            batch[c] = tile
            inds.append(ind)
            c += 1

        return batch, inds

    def preprocess(self, inputs):
        if self.channelnumber != 3: inputs = inputs.repeat(1, 3, 1, 1)
        inputs = inputs - inputs.view(inputs.shape[0], 3, inputs.shape[-1] * inputs.shape[-1]).min(axis=2)[0].reshape(
            inputs.shape[0], 3, 1, 1)
        inputs = inputs / inputs.view(inputs.shape[0], 3, inputs.shape[-1] * inputs.shape[-1]).max(axis=2)[0].reshape(
            inputs.shape[0], 3, 1, 1)
        inputs = F.interpolate(inputs, size=224)
        inputs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs)
        return inputs

    def infer(self, model, ongpu, sigmoid, thresh = 0):
        result = np.zeros((self.overview.shape[-2], self.overview.shape[-1]))
        for bi in range(len(self)):
            batch, inds = self.get_batch(bi)
            batch = torch.from_numpy(batch)
            if ongpu: batch = batch.cuda()
            batch = self.preprocess(batch)
            infers = model(batch)
            if sigmoid: infers = torch.sigmoid(infers)
            probs, labels = torch.max(infers, 1)
            labels = labels.cpu().numpy()
            c = 0
            for iy, ix in inds:
                result[iy*self.stepsize : (iy+1)*self.stepsize, ix*self.stepsize : (ix+1)*self.stepsize] = (labels[c] if probs[c] >= thresh else -1)
                c += 1

        return result




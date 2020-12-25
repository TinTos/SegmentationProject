from __future__ import print_function, division
import os
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset

class TileDataset(Dataset):
    def __init__(self, overview, tilesize, batchsize, shiftcount):
        self.overview = overview
        self.tilesize = tilesize
        self.batchsize = batchsize
        self.shiftcount = shiftcount
        self.shiftsize = batchsize // shiftcount
        if len(overview.shape) == 2:
            self.tilecountx = int(overview.shape[1] // tilesize)
            self.tilecounty = int(overview.shape[0] // tilesize)
            self.channelnumber = 1
        elif len(overview.shape) == 3:
            self.tilecountx = int(overview.shape[2] // tilesize)
            self.tilecounty = int(overview.shape[1] // tilesize)
            self.channelnumber = overview.shape[0]

    def __len__(self):
        return self.shifts * self.tilecountx * self.tilecounty

    def __getitem__(self, idx):
        bx = (idx % (self.tilecounty * self.tilecountx)) % self.tilecountx
        by = (idx % (self.tilecounty * self.tilecountx)) // self.tilecountx
        sx = (idx // (self.tilecounty * self.tilecountx))
        sy = (idx // (self.tilecounty * self.tilecountx * self.shiftcount))

        tile = self.overview[sy * self.shiftsize + by * self.tilesize: sy * self.shiftsize + (by + 1) * self.tilesize,
               sx * self.shiftsize + bx * self.tilesize: sx * self.shiftsize + (bx + 1) * self.tilesize]

        return torch.from_numpy(tile.astype(np.float32))




class TileDataset2(Dataset):
    def __init__(self, overview, tilesize):
        super(TileDataset2).__init__()
        self.tilesize = tilesize

        if len(overview.shape) == 2:
            self.tilecountx = int(overview.shape[1] // tilesize)
            self.tilecounty = int(overview.shape[0] // tilesize)
            self.channelnumber = 1
            self.overview = overview.reshape((1,overview.shape[0], overview.shape[1]))
        elif len(overview.shape) == 3:
            self.tilecountx = int(overview.shape[2] // tilesize)
            self.tilecounty = int(overview.shape[1] // tilesize)
            self.channelnumber = overview.shape[0]
            self.overview = overview

    def __len__(self):
        return self.tilecountx * self.tilecounty

    def __getitem__(self, idx):
        bx = int((idx % (self.tilecounty * self.tilecountx)) % self.tilecountx)
        by = int((idx % (self.tilecounty * self.tilecountx)) // self.tilecountx)

        tile = self.overview[:,by * self.tilesize : (by + 1) * self.tilesize, bx * self.tilesize : (bx + 1) * self.tilesize]

        return torch.from_numpy(tile.astype(np.float32)).cuda(), (by, bx)


def reshape_for_inference(overview, tilesize):
    if(len(overview.shape) == 2):
        overview = overview.reshape((1, overview.shape[0], overview.shape[1]))

    tilecounty = int(overview.shape[1] // tilesize)
    tilecountx = int(overview.shape[2] // tilesize)

    overview = overview[:,:tilecounty * tilesize,:tilecountx * tilesize]

    overview = overview.reshape((tilecountx * tilecounty, overview.shape[0], tilesize, tilesize))

    return torch.from_numpy(overview.astype(np.float32))


def reshape_from_inference(overview, tilesize, inferred):
    if(len(overview.shape) == 2):
        overview = overview.reshape((1, overview.shape[0], overview.shape[1]))

    tilecounty = int(overview.shape[1] // tilesize)
    tilecountx = int(overview.shape[2] // tilesize)

    result = inferred[:,0].reshape((tilecounty,tilecountx))

    return result




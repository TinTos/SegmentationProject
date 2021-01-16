from copy import deepcopy

import numpy as np
import torch

from Dataloading.TileSets.LabeledImageTileSet import LabeledImageTileSet


class LabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, overview, labeledImageTileSet):
        super(LabeledImageDataset, self).__init__()
        if len(overview.shape) == 2:
            self.channelnumber = 1
            self.overview = overview.reshape((1, overview.shape[0], overview.shape[1]))
        elif len(overview.shape) == 3:
            self.channelnumber = overview.shape[0]
            self.overview = overview

        self.tileset = labeledImageTileSet

    @classmethod
    def from_scratch(cls, overview, labeledimage, tilesize, shiftcount, areathresh, minlabel):
        tileset = LabeledImageTileSet(tilesize, shiftcount, areathresh, overview.shape, labeledimage, minlabel)
        return cls(overview, tileset)

    def __len__(self):
        return len(self.tileset)

    def __getitem__(self, item):
        tiledata, label = self.tileset[item]
        tile = self.overview[:, tiledata['miny']: tiledata['maxy'], tiledata['minx']: tiledata['maxx']]
        tensor_x = torch.from_numpy(tile.astype(np.float32))
        tensor_y = torch.tensor(label.astype(np.float32))

        return tensor_x, tensor_y



    def split(self, valpercent):

        shuffled = np.random.permutation(len(self.tileset))
        ind = int(valpercent * len(self.tileset))

        val = shuffled[:ind]
        train = shuffled[ind:]

        rsval = deepcopy(self.tileset)
        rstrain = deepcopy(self.tileset)

        rsval.split_train_val(val)
        rstrain.split_train_val(train)

        return LabeledImageDataset(self.overview, rstrain), LabeledImageDataset(self.overview, rsval)
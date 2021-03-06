from Dataloading.TileSets.RectTileSet import RectTileSet
import torch
import torch.utils.data
import numpy as np
from copy import deepcopy

class RectDataset(torch.utils.data.Dataset):

    @classmethod
    def from_scratch(cls, overview, rectangles, labels, tilesize, shiftcount):
        shiftlength = int(tilesize // shiftcount)
        tilesize = tilesize

        rectsets = []
        for r in rectangles:
            rectsets.append(RectTileSet(tilesize, r, shiftcount, shiftlength))

        return cls(overview, rectsets, labels)

    def __init__(self, overview, rectsets, labels):
        super(RectDataset).__init__()

        if len(overview.shape) == 2:
            self.channelnumber = 1
            self.overview = overview.reshape((1, overview.shape[0], overview.shape[1]))
        elif len(overview.shape) == 3:
            self.channelnumber = overview.shape[0]
            self.overview = overview

        self.labels = labels
        self.labelsencoded = self.one_hot_encoding()

        self.rectsets = rectsets


    def __len__(self):
        result = 0
        for r in self.rectsets:
            result += len(r)

        return result

    def __getitem__(self, item):
        rectindex = -1
        while(item >= 0):
            rectindex += 1
            item -= len(self.rectsets[rectindex])

        item += len(self.rectsets[rectindex])

        tiledata = self.rectsets[rectindex][item]

        tile = self.overview[:, tiledata['miny'] : tiledata['maxy'], tiledata['minx'] : tiledata['maxx']]

        tensor_x = torch.from_numpy(tile.astype(np.float32))
        tensor_y = torch.tensor(self.labelsencoded[rectindex].astype(np.float32))

        return tensor_x, tensor_y


    def one_hot_encoding(self):
        unique = np.unique(self.labels)

        result = []
        for l in self.labels: result.append((unique == l).astype('int'))

        return result

    def split(self, valpercent):
        rsvals = []
        rstrains = []
        for rs in self.rectsets:
            shuffled = np.random.permutation(len(rs))
            ind = int(valpercent * len(rs))

            val = shuffled[:ind]
            train = shuffled[ind:]

            rsval = deepcopy(rs)
            rstrain = deepcopy(rs)

            rsval.split_train_val(val)
            rstrain.split_train_val(train)

            rsvals.append(rsval)
            rstrains.append(rstrain)

        return RectDataset(self.overview, rstrains, self.labels), RectDataset(self.overview, rsvals, self.labels)











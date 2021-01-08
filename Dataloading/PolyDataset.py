from Dataloading.RectTileSet import RectTileSet
from Dataloading.PolyTileSet import PolyTileSet
import torch
import torch.utils.data
import numpy as np
from copy import deepcopy

class PolyDataset(torch.utils.data.Dataset):
    @classmethod
    def from_scratch_with_adapted_shiftcount(cls, overview, polygons, labels, tilesize, shiftcount, areathresh):
        tilesize = tilesize

        polysets = []
        for p in polygons:
            if cls.is_rectangle(p):
                polysets.append(RectTileSet(tilesize, p, 1))
            else:
                polysets.append(PolyTileSet(tilesize, p, 1, areathresh, overview.shape))

        ls = [len(p) for p in polysets]
        ls /= max(ls)

        polysets = []

        c = 0
        for p in polygons:
            if cls.is_rectangle(p):
                polysets.append(RectTileSet(tilesize, p, int(shiftcount /  ls[c])))
            else:
                polysets.append(PolyTileSet(tilesize, p, int(shiftcount /  ls[c]), areathresh, overview.shape))

            c += 1


        return cls(overview, polysets, labels)


    @classmethod
    def from_scratch(cls, overview, polygons, labels, tilesize, shiftcount, areathresh):
        shiftlength = int(tilesize // shiftcount)
        tilesize = tilesize

        polysets = []
        for p in polygons:
            if cls.is_rectangle(p):
                polysets.append(RectTileSet(tilesize, p, shiftcount))
            else:
                polysets.append(PolyTileSet(tilesize, p, shiftcount, areathresh, overview.shape))

        return cls(overview, polysets, labels)

    def __init__(self, overview, polysets, labels):
        super(PolyDataset).__init__()

        if len(overview.shape) == 2:
            self.channelnumber = 1
            self.overview = overview.reshape((1, overview.shape[0], overview.shape[1]))
        elif len(overview.shape) == 3:
            self.channelnumber = overview.shape[0]
            self.overview = overview

        self.labels = labels
        self.labelsencoded = self.one_hot_encoding()

        self.polysets = polysets


    def __len__(self):
        result = 0
        for r in self.polysets:
            result += len(r)

        return result

    def __getitem__(self, item):
        polyindex = -1
        while(item >= 0):
            polyindex += 1
            item -= len(self.polysets[polyindex])

        item += len(self.polysets[polyindex])

        tiledata = self.polysets[polyindex][item]

        tile = self.overview[:, tiledata['miny'] : tiledata['maxy'], tiledata['minx'] : tiledata['maxx']]

        tensor_x = torch.from_numpy(tile.astype(np.float32))
        tensor_y = torch.tensor(self.labelsencoded[polyindex].astype(np.float32))

        return tensor_x, tensor_y


    @classmethod
    def is_rectangle(cls, polygon):
        result = len(np.unique(polygon[:,1])) == 2
        result &= len(np.unique(polygon[:,0])) == 2

        return result

    def one_hot_encoding(self):
        unique = np.unique(self.labels)

        result = []
        for l in self.labels: result.append((unique == l).astype('int'))

        return result

    def split(self, valpercent):
        rsvals = []
        rstrains = []
        for rs in self.polysets:
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

        return PolyDataset(self.overview, rstrains, self.labels), PolyDataset(self.overview, rsvals, self.labels)











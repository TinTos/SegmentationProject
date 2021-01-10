from copy import deepcopy

import numpy as np
from shapely.geometry import Polygon
from Dataloading.TileSetBase import TileSetBase
import torch

class LabeledImageTileSet(TileSetBase, torch.utils.data.Dataset):
    def __init__(self, size, shiftcount, areathreshold, overviewshape, labeledImage, minlabel):
        super(LabeledImageTileSet, self).__init__(size, self.overviewbb(overviewshape), shiftcount)

        self.areathreshold = areathreshold
        self.overviewshape = overviewshape
        self.labeledImage = labeledImage
        self.minlabel = minlabel
        self.initialize()

    @classmethod
    def overviewbb(cls, ovrshape):
        return np.array([[0, 0], [ovrshape[-2], ovrshape[-1]]])

    def initialize(self):
        self.tiles = []
        for index_shift_y in range(self.shiftcount):
            for index_shift_x in range(self.shiftcount):
                for index_y in range(self.count_y):
                    for index_x in range(self.count_x):
                        indexdic = {'y': index_y,
                                    'x': index_x,
                                    'sy': index_shift_y,
                                    'sx': index_shift_x}

                        rect = self.get_tile(indexdic)

                        if rect['miny'] >= 0 and rect['minx'] >= 0 and rect['maxy'] < self.overviewshape[-2] and rect['maxx'] < self.overviewshape[-1]:
                            labeltile = self.labeledImage[rect['miny'] : rect['maxy'], rect['minx'] : rect['maxx']]
                            oc, bins = np.histogram(labeltile)
                            area = np.max(oc)
                            if area < self.areathreshold: continue
                            itemindex = np.argmax(oc)
                            label = np.ceil(bins[itemindex])

                            if label < self.minlabel: continue

                            self.tiles.append((rect, label))

    def __len__(self):
        return len(self.tiles) if self.inds is None else len(self.inds)

    def __getitem__(self, item):
        if self.inds is not None: item = self.inds[item]

        return self.tiles[item]




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

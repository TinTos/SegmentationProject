import numpy as np
from Dataloading.TileSets.TileSetBase import TileSetBase
import torch

class LabeledImageTileSet(TileSetBase, torch.utils.data.Dataset):
    def __init__(self, size, shiftcount, areathreshold, overviewshape, labeledImage, minlabel):
        super(LabeledImageTileSet, self).__init__(size, self.overviewbb(overviewshape), shiftcount)

        self.areathreshold = areathreshold
        self.overviewshape = overviewshape
        self.labeledImage = labeledImage
        self.minlabel = minlabel
        self.labels = np.sort(np.unique(labeledImage))
        self.labels = self.labels[self.labels >= minlabel]
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

                            label = (self.labels == label).astype('int')
                            self.tiles.append((rect, label))

    def __len__(self):
        return len(self.tiles) if self.inds is None else len(self.inds)

    def __getitem__(self, item):
        if self.inds is not None: item = self.inds[item]

        return self.tiles[item]





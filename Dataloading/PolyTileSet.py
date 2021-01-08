import numpy as np
from shapely.geometry import Polygon
from Dataloading.TileSetBase import TileSetBase

class PolyTileSet(TileSetBase):
    def __init__(self, size, points, shiftcount, areathreshold, overviewshape):
        super(PolyTileSet, self).__init__(size, points, shiftcount)
        self.polygon = Polygon(points)
        self.areathreshold = areathreshold
        self.overviewshape = overviewshape
        self.initialize()

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
                        rectpolygon = Polygon([(rect['miny'],rect['minx']), (rect['miny'],rect['maxx']), (rect['maxy'],rect['maxx']), (rect['maxy'],rect['minx'])])
                        intersectionarea = rectpolygon.intersection(self.polygon).area
                        fraction = intersectionarea / rectpolygon.area
                        if fraction > self.areathreshold and rect['miny'] >= 0 and rect['minx'] >= 0 and rect['maxy'] < self.overviewshape[-2] and rect['maxx'] < self.overviewshape[-1]:
                            self.tiles.append(rect)

    def __len__(self):
        return len(self.tiles) if self.inds is None else len(self.inds)

    def __getitem__(self, item):
        if self.inds is not None: item = self.inds[item]

        return self.tiles[item]




import numpy as np
from shapely.geometry import Polygon

class PolyTileSet:
    def __init__(self, size, points, shiftcount, shiftlength, areathreshold):
        self.size = size
        self.points = points
        self.polygon = Polygon(points)
        self.areathreshold = areathreshold

        self.miny = int(np.min(self.points[:,0]))
        self.maxy = int(np.max(self.points[:,0]))

        self.minx = int(np.min(self.points[:,1]))
        self.maxx = int(np.max(self.points[:,1]))

        self.shiftcount = shiftcount
        self.shiftlength = shiftlength

        self.initialize()

        self.inds = None


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
                        if fraction > self.areathreshold:
                            self.tiles.append(rect)


    @property
    def count_x(self):
        return int((self.maxx - self.minx) // self.size)

    @property
    def count_y(self):
        return int((self.maxy - self.miny) // self.size)

    def __len__(self):
        return len(self.tiles) if self.inds is None else len(self.inds)

    def __getitem__(self, item):
        if self.inds is not None: item = self.inds[item]

        return self.tiles[item]

    def get_tile(self, indexdic):
        miny = indexdic['y'] * self.size
        maxy = (indexdic['y'] + 1) * self.size
        miny += indexdic['sy'] * self.shiftlength
        maxy += indexdic['sy'] * self.shiftlength

        minx = indexdic['x'] * self.size
        maxx = (indexdic['x'] + 1) * self.size
        minx += indexdic['sx'] * self.shiftlength
        maxx += indexdic['sx'] * self.shiftlength

        result = {'miny' : miny + self.miny,
                  'maxy' : maxy + self.miny,
                  'minx' : minx + self.minx,
                  'maxx' : maxx + self.minx }

        return result


    def split_train_val(self, inds):
        if(len(inds) > self.__len__()):
            Warning('Wrong indices for splitting rectangular set')
            return
        self.inds = inds





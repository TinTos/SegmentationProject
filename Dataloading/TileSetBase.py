import numpy as np

class TileSetBase:
    def __init__(self, size, points, shiftcount):
        self.size = size
        self.points = points

        (miny, minx), (maxy, maxx) = self.get_ordered_bbox(points)
        self.minx = int(minx)
        self.maxx = int(maxx)
        self.miny = int(miny)
        self.maxy = int(maxy)

        self.shiftcount = shiftcount
        self.shiftlength = int(size // shiftcount)
        self.inds = None

    @property
    def count_x(self):
        return int((self.maxx - self.minx) // self.size)

    @property
    def count_y(self):
        return int((self.maxy - self.miny) // self.size)

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

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

    @classmethod
    def get_ordered_bbox(cls, rectangle):
        minx = min(rectangle[:, 1])
        maxx = max(rectangle[:, 1])
        miny = min(rectangle[:, 0])
        maxy = max(rectangle[:, 0])

        return (miny, minx), (maxy, maxx)





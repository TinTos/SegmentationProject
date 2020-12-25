import numpy as np

class RectTileSet:
    def __init__(self, size, miny, minx, maxy, maxx, shiftcount, shiftlength):
        self.size = size
        self.minx = int(minx)
        self.maxx = int(maxx)
        self.miny = int(miny)
        self.maxy = int(maxy)
        self.shiftcount = shiftcount
        self.shiftlength = shiftlength
        self.inds = None

    @property
    def count_x(self):
        return int((self.maxx - self.minx) // self.size - 1)

    @property
    def count_y(self):
        return int((self.maxy - self.miny) // self.size - 1)

    def __len__(self):
        return int(self.shiftcount * self.shiftcount * self.count_x * self.count_y) if self.inds is None else len(self.inds)

    def __getitem__(self, item):
        if self.inds is not None: item = self.inds[item]
        index_x = int(item % self.count_y)
        index_y = int((item % (self.count_x * self.count_y)) // self.count_x)
        index_shift_x = int(item // (self.count_x * self.count_y))
        index_shift_y = int(item // (self.count_x * self.count_y * self.shiftcount))

        indexdic ={'y' : index_y,
         'x' : index_x,
         'sy' : index_shift_y,
         'sx' : index_shift_x}

        result = self.get_tile(indexdic)

        return result

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





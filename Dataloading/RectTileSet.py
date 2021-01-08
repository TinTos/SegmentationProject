import numpy as np
from Dataloading.TileSetBase import TileSetBase

class RectTileSet(TileSetBase):
    def __init__(self, size, points, shiftcount):
        super(RectTileSet, self).__init__(size, points, shiftcount)

    @property
    def count_x(self):
        return super(RectTileSet, self).count_x - 1

    @property
    def count_y(self):
        return super(RectTileSet, self).count_y - 1

    def __len__(self):
        return int(self.shiftcount * self.shiftcount * self.count_x * self.count_y) if self.inds is None else len(self.inds)

    def __getitem__(self, item):
        if self.inds is not None: item = self.inds[item]
        index_x = int(item % self.count_x)
        index_y = int((item % (self.count_x * self.count_y)) // self.count_x)
        index_shift_x = int(int(item // (self.count_x * self.count_y)) % self.shiftcount)
        index_shift_y = int(item // (self.count_x * self.count_y * self.shiftcount))

        indexdic ={'y' : index_y,
         'x' : index_x,
         'sy' : index_shift_y,
         'sx' : index_shift_x}

        result = self.get_tile(indexdic)

        return result
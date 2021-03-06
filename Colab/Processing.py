import numpy as np
from PIL import Image

from Dataloading.TileSets.PolyTileSet import PolyTileSet


def vote_filter(segmask):
    for y in range(1, segmask.shape[0] - 1):
        for x in range(1, segmask.shape[1] - 1):
            vs = [segmask[y - 1, x - 1], segmask[y - 1, x], segmask[y - 1, x + 1], segmask[y, x - 1],
                  segmask[y, x], segmask[y, x + 1], segmask[y + 1, x - 1], segmask[y + 1, x],
                  segmask[y + 1, x + 1]]
            un = np.unique(vs)
            sums = []
            for i in un:
                sums.append(np.sum(vs == i))
            itemindex = np.where(sums == np.max(sums))

            if itemindex.shape[0] == 1 and itemindex.shape[1] == 1: segmask[y, x] = un[itemindex[0][0]]


def scale_with_limits(overview, min_contrast, max_contrast):
    ovrscaled = overview.copy()
    ovrscaled[ovrscaled < min_contrast] = min_contrast
    ovrscaled[ovrscaled > max_contrast] = max_contrast
    ovrscaled -= min_contrast
    ovrscaled /= (max_contrast - min_contrast)
    return ovrscaled


def LoadSingleChannelPng(path):
    # ovr=np.load('Outputdata/overview.npy')[0:4000,0:4000]
    ImageRaw = np.array(Image.open(path))
    ovr = np.zeros((ImageRaw.shape[0], ImageRaw.shape[1]))
    # ovr[0,:,:] = ImageRaw[:,:,0]
    ovr[:, :] = ImageRaw[:, :, 1]
    # ovr[2,:,:] = ImageRaw[:,:,2]
    del ImageRaw

    return ovr

def get_unary_potential(labels, polys, ts, overviewshape, areathresh):
    labelsunique = np.unique(labels)
    nchan = labelsunique.shape[0]
    result = np.zeros((nchan, overviewshape[-2] // ts, overviewshape[-1] // ts))
    c = 0
    for l in labels:
      pts = PolyTileSet(ts, polys[c], 1, areathresh, overviewshape)
      ch = np.where(labelsunique == l)
      for t in pts.tiles:
        result[ch, t['maxy'] // ts, t['maxx'] // ts] = 1
      c += 1

    return result
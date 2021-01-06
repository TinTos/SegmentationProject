import os
import numpy as np

def export(overview, polys, labels, path):
    try:
        os.mkdir(path)
        print("Directory ", path, " Created ")
    except FileExistsError:
        print("Directory ", path, " already exists")

    np.save(path + '//polys', polys)
    np.save(path + '//labels', labels)
    np.save(path + '//overview', overview)

def importdata(path):
    polys = np.load(path + '//polys.npy')
    labels = np.load(path + '//labels')
    overview = np.load(path + '//overview')

    return overview, polys, labels


def importdatafromfolder(directory):
    polys = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            polys.append(np.load(directory + '//' + filename))
            labels.append(filename.split('.')[0])
        else:
            continue

    return polys, labels
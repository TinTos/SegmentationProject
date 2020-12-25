import gzip
import numpy as np
import struct
import imageio
from zipfile import ZipFile


def get_batch(file_name, batchindex, batchsize):
    archive = ZipFile(file_name, 'r')
    infolist = archive.infolist()
    index = batchindex * batchsize + 1

    firstImage = imageio.imread(archive.open(infolist[index].filename).read())

    result = np.zeros((batchsize, firstImage.shape[0], firstImage.shape[1]))
    result[0, :, :] = firstImage
    for i in range(index + 1, index + batchsize):
        image = imageio.imread(archive.open(infolist[i].filename).read())
        result[i - index, :, :] = image

    return result


def get_data(file_name):
    archive = ZipFile(file_name, 'r')
    infolist = archive.infolist()

    firstImage = imageio.imread(archive.open(infolist[1].filename).read())

    result = np.zeros((len(infolist) - 1, firstImage.shape[0] * firstImage.shape[1]))
    result[0, :] = firstImage.flatten()
    for i in range(2, len(infolist)):
        image = imageio.imread(archive.open(infolist[i].filename).read())
        result[i - 1, :] = image.flatten()

    return result


def get_label(file_name):
    archive = ZipFile(file_name, 'r')
    infolist = archive.infolist()

    result = np.zeros((len(infolist) - 1))

    for i in range(1, len(infolist)):
        image = imageio.imread(archive.open(infolist[i].filename).read())
        result[i - 1] = int(np.min((1, np.max(image))))

    return result


def open_zip_directory(file_name):
    # opening the zip file in READ mode
    with ZipFile(file_name, 'r') as zip:
        # printing all the contents of the zip file

        # extracting all the files
        print('Extracting all the files now...')
        # zip.extractall()
        print('Done!')

    # load compressed MNIST gz files and return numpy arrays


def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res


# one-hot encode a 1-D array
def one_hot_encode(array, num_of_classes):
    return np.eye(num_of_classes)[array.reshape(-1)]
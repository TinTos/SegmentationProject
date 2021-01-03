# instantiate Qt GUI
import sys
from _thread import start_new_thread
import napari
from napari.layers.shapes.shapes import Shapes
import pytorch_lightning
from PySide2.QtCore import QObject
from PySide2.QtCore import Signal
from PySide2.QtWidgets import QApplication, QPushButton
import numpy as np
from Dataloading.PolyDataset import PolyDataset
import torch
from TorchLearning.TestTraining import inference_routine
from Dataloading.Dataset import TileDataset2
from TorchLearning.LightningModule import LitModel
from pytorch_lightning import Trainer
from PIL import Image

def start_training():
    start_new_thread(train_and_infer)


def update_viewer():
    try: viewer.add_image(mask)
    except: print('Something wen wrong')


def train_and_infer():
    #preprocess
    min_contrast = viewer.layers[0].contrast_limits[0]
    max_contrast = viewer.layers[0].contrast_limits[1]
    ovrscaled = scale_with_limits(ovr, min_contrast, max_contrast)

    #get data
    labels, rects = get_rects_and_labels()
    numclasses = len(np.unique(labels))
    dstrain, dsval = PolyDataset.from_scratch(ovrscaled, rects, labels, 64, 3, 0.75).split(0.1)
    dltrain = torch.utils.data.DataLoader(dstrain, batch_size = 64, shuffle = True)
    dlval = torch.utils.data.DataLoader(dsval, batch_size = 64, shuffle = True)

    litmodel = LitModel(numclasses, 0.1, len(dstrain), 64)
    trainer = Trainer(gpus=1, auto_lr_find=True, max_epochs=7)

    trainer.tune(litmodel, train_dataloader=dltrain, val_dataloaders=dlval)
    trainer.fit(litmodel, dltrain, dlval)
    torch.save(litmodel.classifier.state_dict(), 'LastModel')

    #infer
    ds2 = TileDataset2(ovrscaled, 64)
    inference_batchsize = 256
    inference_batchsize_found = False
    #while(not inference_batchsize_found):
        #try:
    #torch.cuda.empty_cache()
    dl2 = torch.utils.data.DataLoader(ds2, batch_size=int(inference_batchsize))
    inferred = inference_routine(litmodel.classifier, dl2, ovrscaled, 64)
    #inferred = inference_routine(model_ft, dl2, ovrscaled, 64)
    inference_batchsize_found = True
        #except:
            #print("OS error: {0}".format(err))
            #inference_batchsize /= 2

    np.save('mask', inferred)

    #signal
    global mask
    mask = inferred
    f.finished.emit()


def get_rects_and_labels():
    rects = []
    labels = []
    for i in range(1, len(viewer.layers)):
        layer = viewer.layers[i]
        # if layer.data.shape[1] != 4: continue
        rects += layer.data
        for l in range(len(layer.data)):
            labels.append(layer.name)
    return labels, rects


def scale_with_limits(overview, min_contrast, max_contrast):
    ovrscaled = overview.copy()
    ovrscaled[ovrscaled < min_contrast] = min_contrast
    ovrscaled[ovrscaled > max_contrast] = max_contrast
    ovrscaled -= min_contrast
    ovrscaled /= (max_contrast - min_contrast)
    return ovrscaled


class Communicate(QObject):
    finished = Signal()


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


if __name__ =='__main__':
    # create the viewer and display the image
    app = QApplication.instance()
    if app == None:
        app = QApplication([])

    #ovr=np.load('Outputdata/overview.npy')[0:4000,0:4000]
    ImageRaw = np.array(Image.open("C:\\Users\\mtoss\\Documents\\DTCleanup\\SmartCyteAlt\\Probe1_DL.png"))

    ovr = np.zeros((ImageRaw.shape[0],ImageRaw.shape[1]))
    #ovr[0,:,:] = ImageRaw[:,:,0]
    ovr[:,:] = ImageRaw[:,:,1]
    #ovr[2,:,:] = ImageRaw[:,:,2]

    del ImageRaw

    viewer = napari.view_image(ovr, rgb=False)
    viewer.layers.append(Shapes(np.load('1.npy', allow_pickle=True), name = '1', shape_type='polygon', face_color='blue'))
    viewer.layers.append(Shapes(np.load('2.npy', allow_pickle=True), name = '2', shape_type='polygon', face_color='red'))
    viewer.layers.append(Shapes(np.load('3.npy', allow_pickle=True), name = '3', shape_type='polygon', face_color='green'))
    viewer.layers.append(Shapes(np.load('4.npy', allow_pickle=True), name = '4', shape_type='polygon', face_color='cyan'))
    viewer.layers.append(Shapes(np.load('5.npy', allow_pickle=True), name = '5', shape_type='polygon', face_color='blue'))
    viewer.layers.append(Shapes(np.load('6.npy', allow_pickle=True), name = '6', shape_type='polygon', face_color='pink'))
    viewer.layers.append(Shapes(np.load('7.npy', allow_pickle=True), name = '7', shape_type='polygon', face_color='blue'))
    viewer.layers.append(Shapes(np.load('8.npy', allow_pickle=True), name = '8', shape_type='polygon', face_color='blue'))

    button = QPushButton("Start!")
    button.clicked.connect(start_training)
    button.show()

    f = Communicate()
    f.finished.connect(update_viewer)

    sys.exit(app.exec_())








# instantiate Qt GUI
import sys
from _thread import start_new_thread

import napari
from PySide2.QtCore import QObject
from PySide2.QtCore import Signal
from skimage.data import astronaut
from skimage.transform import rescale, resize, downscale_local_mean
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton
import scipy.misc
import numpy as np
from Dataloading.RectDataset import RectDataset
import torch
from TorchLearning.TestTraining import inference_routine
from Dataloading.Dataset import TileDataset2
from TorchLearning.PretrainedModel import train_model
from TorchLearning.PretrainedModel import get_pretrained_model_criterion_optimizer_scheduler
import queue

#somewhere accessible to both:

def resize_label_large(input, shape):
    r=resize(input, shape, 0)
    r[r<np.mean(r)]=0
    r[r>0]=1
    return r.astype('uint')

def start_training():
    start_new_thread(get_data)


def update_viewer():
    try: viewer.add_image(mask)
    except: print('Something wen wrong')


def get_data():
    min_contrast = viewer.layers[0].contrast_limits[0]
    max_contrast = viewer.layers[0].contrast_limits[1]
    ovrscaled = ovr.copy()
    ovrscaled[ovrscaled < min_contrast] = min_contrast
    ovrscaled[ovrscaled > max_contrast] = max_contrast
    ovrscaled -= min_contrast
    ovrscaled /= (max_contrast - min_contrast)

    min_debug = np.min(ovrscaled)
    max_debug = np.max(ovrscaled)


    rects = []
    labels = []
    for i in range(1, len(viewer.layers)):
        layer = viewer.layers[i]
        #if layer.data.shape[1] != 4: continue
        rects += layer.data
        for l in range(len(layer.data)):
            labels.append(layer.name)

    dstrain, dsval = RectDataset.from_scratch(viewer.layers['Image'].data, rects, labels, 64, 2).split(0.1)

    dls = {'train' : torch.utils.data.DataLoader(dstrain, batch_size = 64, shuffle = True), 'val' : torch.utils.data.DataLoader(dsval, batch_size = 64, shuffle = True)}

    model_ft, criterion, optimizer_ft, exp_lr_scheduler = get_pretrained_model_criterion_optimizer_scheduler()
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dls)

    #net = Net(64, np.unique(labels).shape[0])

    #training_routine(net, dl)

    del dls
    del criterion
    del optimizer_ft
    del exp_lr_scheduler
    del rects
    del labels
    del dstrain
    del dsval

    ds2 = TileDataset2(ovrscaled, 64)

    inference_batchsize = 256
    inference_batchsize_found = False
    #while(not inference_batchsize_found):
        #try:
    torch.cuda.empty_cache()
    dl2 = torch.utils.data.DataLoader(ds2, batch_size=int(inference_batchsize))
    inferred = inference_routine(model_ft, dl2, ovrscaled, 64)
    inference_batchsize_found = True
        #except:
            #print("OS error: {0}".format(err))
            #inference_batchsize /= 2

    global mask

    mask = inferred

    f.finished.emit()


def from_dummy_thread(func_to_call_from_main_thread):
    callback_queue.put(func_to_call_from_main_thread)

def from_main_thread_blocking():
    callback = callback_queue.get() #blocks until an item is available
    callback()

def from_main_thread_nonblocking():
    while True:
        try:
            callback = callback_queue.get(block=False) #doesn't block
        except queue.Empty: #raised when queue is empty
            continue
        callback()

class Communicate(QObject):
    finished = Signal()

if __name__ =='__main__':
    callback_queue = queue.Queue()

    # create the viewer and display the image
    app = QApplication.instance()
    if app == None:
        app = QApplication([])


    ovr=np.load('Outputdata/overview.npy')[0:4000,0:4000]

    #lb=np.load('Outputdata/lable.npy')
    #image_resized = resize(ovr, (ovr.shape[0] // 16, ovr.shape[1] // 16), anti_aliasing=True)

    #lbrs = np.load('Outputdata/label_2000_4000_fullres.npy')
    #lbrs = resize_large(lb, ovr.shape)

    viewer = napari.view_image(ovr, rgb=False)

    button = QPushButton("Click me")
    button.clicked.connect(start_training)
    button.show()

    f = Communicate()
    f.finished.connect(update_viewer)

    #while(True):
    #viewer.add_layer(lb)
    #viewer.layers[1].editable=True
    #viewer.add_labels(lbrs)

    sys.exit(app.exec_())








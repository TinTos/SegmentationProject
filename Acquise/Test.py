# instantiate Qt GUI
import sys
import napari
from skimage.data import astronaut
from skimage.transform import rescale, resize, downscale_local_mean
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton
import scipy.misc
import numpy as np
from Dataloading.RectDataset import RectDataset
import torch
from TorchLearning.TestModel import Net
from TorchLearning.TestTraining import training_routine
from TorchLearning.TestTraining import inference_routine
from Dataloading.Dataset import reshape_for_inference, TileDataset2
from Dataloading.Dataset import reshape_from_inference
from TorchLearning.PretrainedModel import train_model
from TorchLearning.PretrainedModel import get_pretrained_model_criterion_optimizer_scheduler

def resize_label_large(input, shape):
    r=resize(input, shape, 0)
    r[r<np.mean(r)]=0
    r[r>0]=1
    return r.astype('uint')

def get_data():
    rects = []
    labels = []
    for i in range(1, len(viewer.layers)):
        layer = viewer.layers[i]
        rects += layer.data
        for l in range(len(layer.data)):
            labels.append(layer.name)

    dstrain, dsval = RectDataset.from_scratch(viewer.layers['Image'].data, rects, labels, 64, 2).split(0.1)



    dls = {'train' : torch.utils.data.DataLoader(dstrain, batch_size = 64, shuffle = True), 'val' : torch.utils.data.DataLoader(dsval, batch_size = 64, shuffle = True)}

    model_ft, criterion, optimizer_ft, exp_lr_scheduler = get_pretrained_model_criterion_optimizer_scheduler()
    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dls)

    net = Net(64, np.unique(labels).shape[0])

    #training_routine(net, dl)

    ds2 = TileDataset2(ovr, 64)

    dl2 = torch.utils.data.DataLoader(ds2, batch_size = 32)

    inferred = inference_routine(model_ft, dl2, ovr, 64)

    viewer.add_image(inferred)


    #return ds

if __name__ =='__main__':
    # create the viewer and display the image
    app = QApplication.instance()
    if app == None:
        app = QApplication([])

    ovr=np.load('Outputdata/overview.npy')
    lb=np.load('Outputdata/lable.npy')
    #image_resized = resize(ovr, (ovr.shape[0] // 16, ovr.shape[1] // 16), anti_aliasing=True)

    lbrs = np.load('Outputdata/label_2000_4000_fullres.npy')
    #lbrs = resize_large(lb, ovr.shape)


    viewer = napari.view_image(ovr, rgb=False)

    button = QPushButton("Click me")
    button.clicked.connect(get_data)
    button.show()


    #viewer.add_layer(lb)
    #viewer.layers[1].editable=True
    #viewer.add_labels(lbrs)

    sys.exit(app.exec_())








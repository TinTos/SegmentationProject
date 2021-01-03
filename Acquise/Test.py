# instantiate Qt GUI
import sys
from _thread import start_new_thread
import napari
from napari.layers.shapes.shapes import Shapes
from PySide2.QtCore import QObject
from PySide2.QtCore import Signal
from PySide2.QtWidgets import QApplication, QPushButton
import numpy as np
from Acquise.Processing import scale_with_limits, LoadSingleChannelPng
from Acquise.SegmentationRoutine import segment

def start_training():
    start_new_thread(train_and_infer)


def update_viewer():
    try: viewer.add_image(mask)
    except: print('Something wen wrong')


def train_and_infer(viewer):
    #preprocess
    min_contrast = viewer.layers[0].contrast_limits[0]
    max_contrast = viewer.layers[0].contrast_limits[1]
    ovrscaled = scale_with_limits(ovr, min_contrast, max_contrast)

    #get data
    labels, rects = get_rects_and_labels(viewer)
    inferred = segment(labels, ovrscaled, rects)

    #signal
    global mask
    mask = inferred
    f.finished.emit()


def get_rects_and_labels(viewer):
    rects = []
    labels = []
    for i in range(1, len(viewer.layers)):
        layer = viewer.layers[i]
        # if layer.data.shape[1] != 4: continue
        rects += layer.data
        for l in range(len(layer.data)):
            labels.append(layer.name)
    return labels, rects


class Communicate(QObject):
    finished = Signal()


if __name__ =='__main__':
    # create the viewer and display the image
    app = QApplication.instance()
    if app == None:
        app = QApplication([])

    ovr = LoadSingleChannelPng("C:\\Users\\mtoss\\Documents\\DTCleanup\\SmartCyteAlt\\Probe1_DL.png")

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








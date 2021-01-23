import numpy as np
import torch
from pytorch_lightning import Trainer
from torchvision import models
from Dataloading.Datasets.PolyDataset import PolyDataset
from Dataloading.Datasets.LabeledImageDataset import LabeledImageDataset
from TorchLearning.LightningModule import LitModel
from Colab.KmeansClustering import cluster_routine
from Dataloading.Datasets.InferenceDataset import InferenceDataset


def segment(overview, polys, labels, tilesize=64, shiftcount=3, coveringthreshold=0.75, epochs = 2, model = None, labelundecisive = False, decisionthresh = 0.5, adaptiveshiftcount = False):
    numclasses = len(np.unique(labels))

    dstrain, dsval = PolyDataset.from_scratch(overview, polys, labels, tilesize, shiftcount, coveringthreshold).split(0.1) if not adaptiveshiftcount else PolyDataset.from_scratch_with_adapted_shiftcount(overview, polys, labels, tilesize, shiftcount, coveringthreshold).split(0.1)

    dltrain = torch.utils.data.DataLoader(dstrain, batch_size=64, shuffle=True)
    dlval = torch.utils.data.DataLoader(dsval, batch_size=64, shuffle=False)

    litmodel = LitModel(numclasses, 0.1, len(dstrain), 64, model)
    fit_model(dltrain, dlval, epochs, litmodel)

    # infer
    inference_batchsize = 256
    ds2 = InferenceDataset(overview, tilesize, tilesize, inference_batchsize)
    resultdic = ds2.infer_flattened(litmodel.classifier, True)

    inferred = ds2.label_overview(resultdic)

    np.save('mask', inferred)
    return inferred


def fit_model(dltrain, dlval, epochs, litmodel):
    trainer = Trainer(gpus=1, auto_lr_find=True, max_epochs=epochs)
    trainer.tune(litmodel, train_dataloader=dltrain, val_dataloaders=dlval)
    trainer2 = Trainer(gpus=1, auto_lr_find=True, max_epochs=epochs)
    trainer2.fit(litmodel, dltrain, dlval)
    torch.save(litmodel.classifier.state_dict(), 'LastModel')


def segment_unsupervised(overview, num_classes = 8, tilesize_learn = 64, tilesize_cluster = 128, shiftcount = 2, epochs = 3, model = None, labelsureonly = True, decisionthresh = 0.5):
    classifier = models.resnet18(pretrained=True).cuda()
    labeledimage = cluster_routine(classifier, overview, tilesize_cluster, num_classes, 256, labelsureonly)

    dstrain, dsval = LabeledImageDataset.from_scratch(overview, labeledimage, tilesize_learn, shiftcount, 0.75, 0).split(0.1)

    dltrain = torch.utils.data.DataLoader(dstrain, batch_size=64, shuffle=True)
    dlval = torch.utils.data.DataLoader(dsval, batch_size=64, shuffle=False)

    litmodel = LitModel(num_classes, 0.1, len(dstrain), 64, model)
    fit_model(dltrain, dlval, epochs, litmodel)

    # infer
    ds2 = InferenceDataset(overview, tilesize_learn, tilesize_learn, 256)

    inf_flat = ds2.infer_flattened(litmodel.classifier.cuda(), True)
    inferred = ds2.label_overview(inf_flat)
    np.save('mask', inferred)
    return inferred

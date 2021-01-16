import numpy as np
import torch
from pytorch_lightning import Trainer
from torchvision import models

from Dataloading.Dataset import TileDataset2
from Dataloading.Datasets.PolyDataset import PolyDataset
from Dataloading.Datasets.LabeledImageDataset import LabeledImageDataset
from TorchLearning.LightningModule import LitModel
from TorchLearning.TestTraining import inference_routine, cluster_routine
from Dataloading.Datasets.InferenceDataset import InferenceDataset


def segment(overview, polys, labels, tilesize=64, shiftcount=3, coveringthreshold=0.75, epochs = 6, model = None, labelundecisive = False, decisionthresh = 0.5, adaptiveshiftcount = False):
    numclasses = len(np.unique(labels))

    dstrain, dsval = PolyDataset.from_scratch(overview, polys, labels, tilesize, shiftcount, coveringthreshold).split(0.1) if not adaptiveshiftcount else PolyDataset.from_scratch_with_adapted_shiftcount(overview, polys, labels, tilesize, shiftcount, coveringthreshold).split(0.1)

    dltrain = torch.utils.data.DataLoader(dstrain, batch_size=64, shuffle=True)
    dlval = torch.utils.data.DataLoader(dsval, batch_size=64, shuffle=False)

    litmodel = LitModel(numclasses, 0.1, len(dstrain), 64, model)
    fit_model(dltrain, dlval, epochs, litmodel)

    # infer
    ds2 = TileDataset2(overview, tilesize)
    inference_batchsize = 256
    dl2 = torch.utils.data.DataLoader(ds2, batch_size=int(inference_batchsize))
    inferred = inference_routine(litmodel.classifier, dl2, overview, tilesize, labelundecisive, decisionthresh)

    np.save('mask', inferred)
    return inferred


def segmenttest(overview, polys, labels, tilesize=64, shiftcount=3, coveringthreshold=0.75, epochs = 4, model = None, labelundecisive = False, decisionthresh = 0.5, adaptiveshiftcount = False):
    numclasses = len(np.unique(labels))

    dstrain, dsval = PolyDataset.from_scratch(overview, polys, labels, tilesize, shiftcount, coveringthreshold).split(0.1) if not adaptiveshiftcount else PolyDataset.from_scratch_with_adapted_shiftcount(overview, polys, labels, tilesize, shiftcount, coveringthreshold).split(0.1)

    dltrain = torch.utils.data.DataLoader(dstrain, batch_size=64, shuffle=True)
    dlval = torch.utils.data.DataLoader(dsval, batch_size=64, shuffle=False)

    litmodel = LitModel(numclasses, 0.1, len(dstrain), 64, model)
    fit_model(dltrain, dlval, epochs, litmodel)

    # infer
    ids = InferenceDataset(overview, tilesize, tilesize, 32)
    inferred = ids.infer(litmodel, True, True)

    np.save('mask', inferred)
    return inferred


def fit_model(dltrain, dlval, epochs, litmodel):
    trainer = Trainer(gpus=1, auto_lr_find=True, max_epochs=epochs)
    trainer.tune(litmodel, train_dataloader=dltrain, val_dataloaders=dlval)
    trainer.fit(litmodel, dltrain, dlval)
    torch.save(litmodel.classifier.state_dict(), 'LastModel')


def segment_unsupervised(overview, num_classes = 8, tilesize_learn = 64, tilesize_cluster = 128, shiftcount = 2, epochs = 3, model = None, labelundecisive = False, decisionthresh = 0.5):
    ds2 = TileDataset2(overview, tilesize_cluster)
    inference_batchsize = 256
    dl2 = torch.utils.data.DataLoader(ds2, batch_size=int(inference_batchsize))

    classifier = models.resnet18(pretrained=True).cuda()
    labeledimage = cluster_routine(classifier, dl2, overview, tilesize_cluster, 1000, num_classes)

    return labeledimage
    dstrain, dsval = LabeledImageDataset.from_scratch(overview, labeledimage, tilesize_learn, shiftcount, 0.75, 0).split(0.1)

    dltrain = torch.utils.data.DataLoader(dstrain, batch_size=64, shuffle=True)
    dlval = torch.utils.data.DataLoader(dsval, batch_size=64, shuffle=False)

    litmodel = LitModel(num_classes, 0.1, len(dstrain), 64, model)
    fit_model(dltrain, dlval, epochs, litmodel)

    # infer
    ds2 = TileDataset2(overview, tilesize_learn)
    inference_batchsize = 256
    dl2 = torch.utils.data.DataLoader(ds2, batch_size=int(inference_batchsize))
    inferred = inference_routine(litmodel.classifier, dl2, overview, tilesize_learn, labelundecisive, decisionthresh)

    np.save('mask', inferred)
    return inferred



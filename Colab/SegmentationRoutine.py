import numpy as np
import torch
from pytorch_lightning import Trainer

from Dataloading.Dataset import TileDataset2
from Dataloading.PolyDataset import PolyDataset
from TorchLearning.LightningModule import LitModel
from TorchLearning.TestTraining import inference_routine


def segment(overview, polys, labels, tilesize=64, shiftcount=3, coveringthreshold=0.75, epochs = 6, model = None, labelundecisive = False, decisionthresh = 0.5, adaptiveshiftcount = False):
    numclasses = len(np.unique(labels))

    dstrain, dsval = PolyDataset.from_scratch(overview, polys, labels, tilesize, shiftcount, coveringthreshold).split(0.1) if not adaptiveshiftcount else PolyDataset.from_scratch_with_adapted_shiftcount(overview, polys, labels, tilesize, shiftcount, coveringthreshold).split(0.1)

    dltrain = torch.utils.data.DataLoader(dstrain, batch_size=64, shuffle=True)
    dlval = torch.utils.data.DataLoader(dsval, batch_size=64, shuffle=True)

    litmodel = LitModel(numclasses, 0.1, len(dstrain), 64, model)
    trainer = Trainer(gpus=1, auto_lr_find=True, max_epochs=epochs)
    trainer.tune(litmodel, train_dataloader=dltrain, val_dataloaders=dlval)
    trainer.fit(litmodel, dltrain, dlval)
    torch.save(litmodel.classifier.state_dict(), 'LastModel')

    # infer
    ds2 = TileDataset2(overview, 64)
    inference_batchsize = 256
    dl2 = torch.utils.data.DataLoader(ds2, batch_size=int(inference_batchsize))
    inferred = inference_routine(litmodel.classifier, dl2, overview, 64, labelundecisive, decisionthresh)

    np.save('mask', inferred)
    return inferred
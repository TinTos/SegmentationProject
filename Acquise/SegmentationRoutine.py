import numpy as np
import torch
from pytorch_lightning import Trainer

from Dataloading.Dataset import TileDataset2
from Dataloading.PolyDataset import PolyDataset
from TorchLearning.LightningModule import LitModel
from TorchLearning.TestTraining import inference_routine


def segment(overview, polys, labels):
    numclasses = len(np.unique(labels))

    dstrain, dsval = PolyDataset.from_scratch(overview, polys, labels, 64, 3, 0.75).split(0.1)

    dltrain = torch.utils.data.DataLoader(dstrain, batch_size=64, shuffle=True)
    dlval = torch.utils.data.DataLoader(dsval, batch_size=64, shuffle=True)

    litmodel = LitModel(numclasses, 0.1, len(dstrain), 64)
    trainer = Trainer(gpus=1, auto_lr_find=True, max_epochs=7)
    trainer.tune(litmodel, train_dataloader=dltrain, val_dataloaders=dlval)
    trainer.fit(litmodel, dltrain, dlval)
    torch.save(litmodel.classifier.state_dict(), 'LastModel')

    # infer
    ds2 = TileDataset2(overview, 64)
    inference_batchsize = 256
    dl2 = torch.utils.data.DataLoader(ds2, batch_size=int(inference_batchsize))
    inferred = inference_routine(litmodel.classifier, dl2, overview, 64)

    np.save('mask', inferred)
    return inferred
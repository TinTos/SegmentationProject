import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms, models

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, overview, tilesize, stepsize, batchsize):
        super(InferenceDataset, self).__init__()
        self.tilesize = tilesize
        if len(overview.shape) == 2:
            self.channelnumber = 1
            self.overview = overview.reshape((1,overview.shape[0], overview.shape[1]))
        elif len(overview.shape) == 3:
            self.channelnumber = overview.shape[0]
            self.overview = overview

        self.tilecountx = int(overview.shape[-1] // stepsize)
        self.tilecounty = int(overview.shape[-2] // stepsize)
        while (self.tilecountx - 1) * stepsize + tilesize >= overview.shape[-1]:
            self.tilecountx -= 1
        while (self.tilecounty - 1) * stepsize + tilesize >= overview.shape[-2]:
            self.tilecounty -= 1

        self.tilesize = tilesize
        self.batchsize = batchsize
        self.stepsize = stepsize

    def __len__(self):
        return self.tilecountx * self.tilecounty

    @property
    def batchcount(self):
        return int(np.ceil(len(self) / self.batchsize))

    def __getitem__(self, idx):
        bx = int((idx % (self.tilecounty * self.tilecountx)) % self.tilecountx)
        by = int((idx % (self.tilecounty * self.tilecountx)) // self.tilecountx)

        tile = self.overview[:, by * self.stepsize : by * self.stepsize + self.tilesize, bx * self.stepsize : bx * self.stepsize + self.tilesize]

        return tile.astype(np.float32), (by, bx)

    def get_batch(self, bi):
        batch = np.zeros((self.batchsize, self.channelnumber, self.tilesize, self.tilesize)).astype(np.float32)
        inds = []
        c=0
        for i in range(bi * self.batchsize, (bi+1) * self.batchsize):
            tile, ind = self[i]
            batch[c] = tile
            inds.append(ind)
            c += 1

        return batch, inds

    def preprocess(self, inputs, doRGB = True, onGPU = True):
        if type(inputs) is np.ndarray: inputs = torch.from_numpy(inputs.astype(np.float32))
        if onGPU: inputs = inputs.cuda()
        if self.channelnumber == 1 and doRGB: inputs = inputs.repeat(1, 3, 1, 1)
        inputs = inputs - inputs.view(inputs.shape[0], 3, inputs.shape[-1] * inputs.shape[-1]).min(axis=2)[0].reshape(
            inputs.shape[0], 3, 1, 1)
        inputs = inputs / inputs.view(inputs.shape[0], 3, inputs.shape[-1] * inputs.shape[-1]).max(axis=2)[0].reshape(
            inputs.shape[0], 3, 1, 1)
        inputs = F.interpolate(inputs, size=224)
        inputs = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(inputs)
        return inputs

    def label_overview(self, resultdic, thresh = 0):
        result = np.zeros((self.overview.shape[-2], self.overview.shape[-1]))

        for i in resultdic:
            probs = resultdic[i]
            maxprob = np.max(probs)
            l = np.argmax(probs)
            result[i[0]*self.stepsize : (i[0]+1)*self.stepsize, i[1]*self.stepsize : (i[1]+1)*self.stepsize] = (l if maxprob >= thresh else -1)

        return result

    def label_overview_custom(self, model, ongpu, sigmoid, thresh = 0, doRGB = True):
        with torch.no_grad():
            result = np.zeros((self.overview.shape[-2], self.overview.shape[-1]))
            for bi in range(self.batchcount):
                batch, inds = self.get_batch(bi)
                batch = torch.from_numpy(batch)
                if ongpu: batch = batch.cuda()
                batch = self.preprocess(batch, doRGB)
                infers = model(batch)
                if sigmoid: infers = torch.sigmoid(infers)
                probs, labels = torch.max(infers, 1)
                labels = labels.cpu().numpy()
                c = 0
                for iy, ix in inds:
                    result[iy*self.stepsize : (iy+1)*self.stepsize, ix*self.stepsize : (ix+1)*self.stepsize] = (labels[c] if probs[c] >= thresh else -1)
                    c += 1

            return result



    def infer_flattened(self, model, doRGB, sigmoid = True):
        dataloader = torch.utils.data.DataLoader(self, batch_size=self.batchsize, shuffle=False)
        result = {}
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, inds = data
                inputs = self.preprocess(inputs, doRGB)
                indsy = inds[0].cpu().numpy()
                indsx = inds[1].cpu().numpy()

                outputs = model(inputs)
                if sigmoid: outputs = torch.sigmoid(outputs)
                outputs = outputs.cpu().numpy()

                outputs[np.isnan(outputs)] = 0

                for count in range(inds[0].shape[0]):
                    result[(indsy[count], indsx[count])] = outputs[count]

                del inputs
                del outputs

                print(str(i) + "/" + str(int(len(dataloader.dataset) // dataloader.batch_size)))

            return result



if __name__ == "__main__":
    ov = np.zeros((1024,1024))
    ids = InferenceDataset(ov, 64, 64, 128)
    m = models.resnet18(True)
    r = ids.infer_flattened(m, True)
    print(r)

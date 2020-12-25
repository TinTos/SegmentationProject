from skimage.transform import rescale, resize, downscale_local_mean
from PySide2.QtWidgets import QApplication, QMainWindow
import scipy.misc
import numpy as np
from skimage.io import imsave, imread
import matplotlib.pyplot as plt

if __name__ =='__main__':
    # create the viewer and display the image
    #lb=np.load('label_2000_4000_fullres.npy')
    #lbAugmented=np.zeros((2000,4000))
    #lbAugmented[0:2000,2000:4000]=lb
    data=np.load('Outputdata/overview.npy')#[2000:4000,0:4000]

    #split=lbAugmented[0:(31*64),0:(62*64)].reshape(31*62,64,64)

    #test=split.reshape(31*64,62*64)
    #print(split.shape)

    #plt.imshow(split[30,:,:])
    #plt.show()



    batchsize = 64
    batchcountx=int(data.shape[1]//batchsize)
    batchcounty=int(data.shape[0]//batchsize)
    shift=4
    shiftCount=1

    dataSplit = np.zeros((batchcounty-1, batchcountx-1, batchsize, batchsize))
    #labelSplit = np.zeros((shiftCount, shiftCount, batchcounty-1, batchcountx-1, batchsize, batchsize))

    #labelTensor=np.zeros((shiftCount,shiftCount,batchcounty-1,batchcountx-1))

    for sx in range(shiftCount):
        for sy in range(shiftCount):
            for y in range(batchcounty - 1):
                for x in range(batchcountx - 1):
                    #imsave('DataNew\\shift_' + str(sx) + '_' + str(sy) + '_loc_' + str(i) +'_'+str(j) +'.png',data[sx*shift + i*batchsize : sx*shift + (i+1)*batchsize, sy*shift +j*batchsize : sy*shift +(j+1)*batchsize])
                    #label=lbAugmented[sy*shift + y*batchsize : sy*shift + (y+1)*batchsize, sx*shift + x*batchsize : sx*shift +(x+1)*batchsize]
                    date=data[sy*shift + y*batchsize : sy*shift + (y+1)*batchsize, sx*shift + x*batchsize : sx*shift +(x+1)*batchsize]
                    dataSplit[y,x,:,:]=date
                    #labelSplit[sy,sx,y,x,:,:]=label
                    #max=np.max(label)
                    #if max > 0:
                    #    labelTensor[sx,sy,y,x]=1
                    #else:
                    #    labelTensor[sx, sy, y, x] = 0
                    #imsave('LabelNew\\shift_' + str(sx) + '_' + str(sy) + '_loc_' + str(i) +'_'+str(j) + '.png',label)

                    print('sx')
                    print(sx)
                    print('sy')
                    print(sy)
                    print('y')
                    print(y)
                    print('x')
                    print(x)




    #np.save('LabelNew\\LabelTensor',labelTensor)
    #np.save('LabelNew\\LabelImages',labelSplit)
    #np.save('DataNew\\DataImages',dataSplit)
    np.save('DataNew\\DataTest',dataSplit)


def split_into_tiles(images, shift, shiftCount, tileSize, label):
    batchcountx = int(images[0].shape[1] // tileSize)
    batchcounty = int(images[0].shape[0] // tileSize)
    results = list()

    for i in range(len(images)):
        if(label[i]):
            results.Add(np.zeros((shiftCount, shiftCount, batchcounty-1, batchcountx-1, 1, 1)))
        else:
            results.Add(np.zeros((shiftCount, shiftCount, batchcounty-1, batchcountx-1, tileSize, tileSize)))
        data = images[i]
        for sx in range(shiftCount):
            for sy in range(shiftCount):
                for y in range(batchcounty - 1):
                    for x in range(batchcountx - 1):
                            date=data[sy*shift + y*batchsize : sy*shift + (y+1)*batchsize, sx*shift + x*batchsize : sx*shift +(x+1)*batchsize]
                            results[i][sy,sx,y,x,:,:]=date
                            if(label[i]):
                                max=np.max(label)
                                if max > 0:
                                   results[i][sx,sy,y,x]=1
                                else:
                                   results[i][sx, sy, y, x] = 0

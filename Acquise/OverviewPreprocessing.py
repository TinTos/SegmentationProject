# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:27:51 2019

@author: MartinT
"""

from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np

def read_tiff(path, n_images):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
    img = Image.open(path)
    images = []
    for i in range(n_images):
        try:
            img.seek(i)
            slice_ = np.zeros((img.height, img.width))
            for j in range(slice_.shape[0]):
                for k in range(slice_.shape[1]):
                    slice_[j,k] = img.getpixel((j, k))

            images.append(slice_)

        except EOFError:
            # Not enough frames in img
            break
    
    np.mean(images[-1])

    return np.array(images)

def calc_brenner(path, n_images):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
    
    img = Image.open(path)
    brenner = []
    for i in range(n_images):
        try:
            img.seek(i)
            slice_ = np.zeros((img.height, img.width))
            
            for j in range(slice_.shape[1]):
                for k in range(slice_.shape[0]):                    
                    slice_[k,j] = img.getpixel((j, k))
                    
            
            b=np.sum(np.abs(slice_[0:slice_.shape[0] - 1,:] - slice_[1:slice_.shape[0],:])) + np.sum(np.abs(slice_[:,0:slice_.shape[1] - 1] - slice_[: , 1:slice_.shape[1]]))

            brenner.append(b)
            print(i)

        except EOFError:
            # Not enough frames in img
            break
        
        
  
    
    
def plane_iterator_xy(index, planecountx):
    y=round(np.floor(index/planecountx))
    x=index%planecountx
    if(y%2==1):
        x=planecountx-1-x
    return x,y

    
def plane_iterator_yx(index, planecounty):
    x=round(np.floor(index/planecounty))
    y=index%planecounty
    if(x%2==1):
        y=planecounty-1-y
    return x,y
        
    
def plane_iterator(index, planecountx, planecounty, xor, yor, acquisitionorderxy): 
    x=0
    y=0
    if(acquisitionorderxy):
        x,y=plane_iterator_xy(index,planecountx)
    else:
        x,y=plane_iterator_yx(index,planecounty)
    if(xor==-1):
        x=planecountx-1-x
    if(yor==1):
        y=planecounty-1-y
    
    return x,y
     
        
        
    
def get_pyramid_overview(path, n_images, xor, yor, planecountx, planecounty, acquisitionorderxy):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """    
    img = Image.open(path)
    overview = np.zeros(1)
    for i in range(n_images):
        try:
            img.seek(i)
            slice_ = np.zeros((img.height, img.width))
            
            if(i==0):
                overview = np.zeros((planecounty*img.height,  planecountx*img.width))
                
            for j in range(slice_.shape[1]):
                for k in range(slice_.shape[0]):                    
                    slice_[k,j] = img.getpixel((j, k))
                    
            
            xindex,yindex=plane_iterator(i, planecountx, planecounty, xor, yor, acquisitionorderxy)        
                        
            overview[int(yindex)*img.height:(int(yindex+1))*img.height,int(xindex)*img.width:(int(xindex+1))*img.width]=slice_/np.mean(slice_)          
            
            print(i)

        except EOFError:
            # Not enough frames in img
            break
        
    return overview
        

def read_nth_tiff(path, n):
    """
    path - Path to the multipage-tiff file
    n_images - Number of pages in the tiff file
    """
    img = Image.open(path)
    images = []
   
    
    img.seek(n)
    slice_ = np.zeros((img.height, img.width))
    for j in range(slice_.shape[1]):
        for k in range(slice_.shape[0]):
            slice_[k,j] = img.getpixel((j, k))

    images.append(slice_)

   

    return np.array(images)


def get_pixel_size(metdataxml):
    pixelpath='{http://till-id.com/SIAM}BandMetaData/{http://till-id.com/SIAM}BandMetaData/{http://till-id.com/SIAM}ResultMetaData/{http://till-id.com/SIAM}ResultMetaData/{http://till-id.com/SIAM}PixelRoiOnCameraChip'
    x0path='{http://till-id.com/SIAM}X0'
    x1path='{http://till-id.com/SIAM}X1'
    y0path='{http://till-id.com/SIAM}Y0'
    y1path='{http://till-id.com/SIAM}Y1'
    
    pixelroi=metdataxml.findall(pixelpath)[1]
    x=int(pixelroi.findall(x1path)[0].text)-int(pixelroi.findall(x0path)[0].text)
    y=int(pixelroi.findall(y1path)[0].text)-int(pixelroi.findall(y0path)[0].text)
    
    return x,y

def get_tile_rect(metdataxml):
    imageregionpath='{http://till-id.com/SIAM}ImageRegion'
    imagex0path='{http://till-id.com/SIAM}X1'
    imagex1path='{http://till-id.com/SIAM}X2'
    imagey0path='{http://till-id.com/SIAM}Y1'
    imagey1path='{http://till-id.com/SIAM}Y2'
    
    imageregion=metdataxml.findall(imageregionpath)[0]
    imagex=float(imageregion.findall(imagex1path)[0].text)-float(imageregion.findall(imagex0path)[0].text)
    imagey=float(imageregion.findall(imagey1path)[0].text)-float(imageregion.findall(imagey0path)[0].text)
    
    planeregionpath='{http://till-id.com/SIAM}PlaneRegions/{http://till-id.com/SIAM}PlaneRegion'
    planex0path='{http://till-id.com/SIAM}X1'
    planex1path='{http://till-id.com/SIAM}X2'
    planey0path='{http://till-id.com/SIAM}Y1'
    planey1path='{http://till-id.com/SIAM}Y2'
    
    imageregion=metdataxml.findall(planeregionpath)[0]
    planex=float(imageregion.findall(planex1path)[0].text)-float(imageregion.findall(planex0path)[0].text)
    planey=float(imageregion.findall(planey1path)[0].text)-float(imageregion.findall(planey0path)[0].text)
    
    return imagex/planex,imagey/planey

def get_overview_orientation(metadataxml):
    acquisitionorderpath='{http://till-id.com/SIAM}AcquisitionOrder'
    planeregionpath='{http://till-id.com/SIAM}PlaneRegions/{http://till-id.com/SIAM}PlaneRegion'
    planex0path='{http://till-id.com/SIAM}X1'
    planey0path='{http://till-id.com/SIAM}Y1'
    
      
        
    firstregion=metadataxml.findall(planeregionpath)[0]
    secondregion=metadataxml.findall(planeregionpath)[3]
    lastregion=metadataxml.findall(planeregionpath)[-1]
    
    firstplanex=float(firstregion.findall(planex0path)[0].text)
    secondplanex=float(secondregion.findall(planex0path)[0].text)
    lastplanex=float(lastregion.findall(planex0path)[0].text)
    
    firstplaney=float(firstregion.findall(planey0path)[0].text)
    secondplaney=float(secondregion.findall(planey0path)[0].text)
    lastplaney=float(lastregion.findall(planey0path)[0].text)
    
    xor=0
    yor=0
    
    print(firstplanex)
    print(lastplanex)
    print(firstplaney)
    print(lastplaney)
    print(secondplanex)
    print(secondplaney)
    
    if(metadataxml.findall(acquisitionorderpath)[0].text=='ZYXT'):
        
        if(firstplanex < lastplanex):
            xor=1
        else:
            xor=-1
            
        firstplaney=float(firstregion.findall(planey0path)[0].text)
        lastplaney=float(lastregion.findall(planey0path)[0].text)
        
        
        if(firstplaney < secondplaney):
            yor=1
        else:
            yor=-1
            
    if(metadataxml.findall(acquisitionorderpath)[0].text=='ZXYT'):
        
        if(firstplanex < secondplanex):
            xor=1
        else:
            xor=-1
            
        firstplaney=float(firstregion.findall(planey0path)[0].text)
        lastplaney=float(lastregion.findall(planey0path)[0].text)
        
        
        if(firstplaney < lastplaney):
            yor=1
        else:
            yor=-1
            
    return xor,yor, metadataxml.findall(acquisitionorderpath)[0].text=='ZXYT' 


def reshape_brennerlist(brennerlist, planecountx, planecounty):
    meander=max(planecounty, planecountx)
    straight=min(planecounty, planecountx)
    result=np.zeros((straight,meander))
    for i in range(straight):
        a=brennerlist[i*meander:(i+1)*meander]
        if(i % 2 == 0):
            result[i,:]=a
        else:
            result[i,:]=np.flip(a)
    return result


if __name__ == '__main__':
    #filenamestained = 'T:\\Thomas\\SmartCyteSchnitteMitPaTePo\\Trans20191211115055354_11.12.2019_14.47.15\\Images\\107_Overview_TRANS_Red+TRANS_Green+TRANS_Blue\\Pyramid_0_TRANS_Red_L3_.tif'
    filenamephase = 'C:\\Users\\mtoss\\Desktop\\Unstained\\Pyramid_0_TRANS_Red_L2_.tif'
    #filenamemetadatastained = 'T:\\Thomas\SmartCyteSchnitteMitPaTePo\\Trans20191211115055354_11.12.2019_14.47.15\\Images\\107_Overview_TRANS_Red+TRANS_Green+TRANS_Blue\\MetaDataImage.xml'
    filenamemetadataphase = 'C:\\Users\\mtoss\\Desktop\\Unstained\\MetaDataImage.xml'

    #mdstained = ET.parse(filenamemetadatastained).getroot()
    mdphase = ET.parse(filenamemetadataphase).getroot()

    xphase, yphase = get_tile_rect(mdphase)
    #xstained, ystained = get_tile_rect(mdstained)

    #xorstained,yorstained,acxystained=get_overview_orientation(mdstained)
    xorphase,yorphase,acxyphase=get_overview_orientation(mdphase)
    ov=get_pyramid_overview(filenamephase, round(xphase)* round(yphase), xorphase, yorphase, round(xphase),round(yphase), acxyphase)
    np.save('overviewnew2test', ov)



#brennerphase = calc_brenner(filenamephase, round(xphase)*round(yphase))
#brennerstained = calc_brenner(filenamestained,  round(xstained)*round(ystained))


#mapstained = reshape_brennerlist(brennerstained,xstained,ystained)
#mapphase = reshape_brennerlist(brennerphase,xphase,yphase)


# there are lots of ways of parsing XML in Pyhton, for example....
#elemList = []




#for elem in root.iter():
#  elemList.append(elem.tag) # indent this by tab, not two spaces as I did here

# now I remove duplicities - by convertion to set and back to list
#elemList = list(set(elemList))

# Just printing out the result
#print(elemList)


   
    #for type_tag in metdata.findall('{http://till-id.com/SIAM}BandMetaData/{http://till-id.com/SIAM}BandMetaData/{http://till-id.com/SIAM}ResultMetaData/{http://till-id.com/SIAM}ResultMetaData/{http://till-id.com/SIAM}PixelRoiOnCameraChip')[1].iter():
    #    value = type_tag.tag
    #    t=type_tag.text
    #    if(value)
    


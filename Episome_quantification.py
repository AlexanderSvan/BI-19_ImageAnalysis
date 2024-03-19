import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from skimage.filters import threshold_otsu, gaussian
import pandas as pd
from skimage.morphology import area_closing, area_opening, binary_dilation, diamond
from skimage.measure import regionprops
from cellpose import models
import cv2
from scipy import ndimage as ndi
from skimage import io, img_as_ubyte

mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap']='gray'
mpl.rcParams['xtick.bottom']=False
mpl.rcParams['xtick.labelbottom']=False
mpl.rcParams['ytick.left']=False
mpl.rcParams['ytick.labelleft']=False
mpl.rcParams['savefig.pad_inches']=0.0
plt.rcParams['figure.dpi'] = 300

class LoadImages:
    
    def __init__(self, src):
        self.src=src
        self.IndexImages()
        
    def IndexImages(self):
        self.dict={}
        for file in os.listdir(self.src):
            self.dict[file]={}
            for img in os.listdir(self.src+"\\"+file):
                if os.path.isfile(self.src+"\\"+file+"\\"+img):
                    if "t0c0" in img:
                        self.dict[file]['LEL']=self.src+"\\"+file+"\\"+img
                    elif "t0c1" in img:
                        self.dict[file]['CMV']=self.src+"\\"+file+"\\"+img
                    elif "t0c2" in img:
                        self.dict[file]['DAPI']=self.src+"\\"+file+"\\"+img
    
    def GetImage(self, fname):
        b=io.imread(self.dict[fname]['DAPI'])
        g=io.imread(self.dict[fname]['LEL'])
        r=io.imread(self.dict[fname]['CMV'])
        return b,g,r
    
    def SaveMasks(self, fname, lectin_mask, nuclei_mask, cmv_mask, vas, ext):
        mask_src=self.src+"\\"+fname+"\\Masks"
        if not os.path.exists(mask_src): 
            os.makedirs(mask_src) 
        io.imsave(mask_src+"\\LEL.mask.png", img_as_ubyte(lectin_mask), cmap='gray')
        io.imsave(mask_src+"\\nuclei.mask.png", img_as_ubyte(nuclei_mask), cmap='gray')
        io.imsave(mask_src+"\\CMV.mask.png", img_as_ubyte(cmv_mask), cmap='gray')
        io.imsave(mask_src+"\\vascular_nuclei.mask.png", img_as_ubyte(vas), cmap='gray')
        io.imsave(mask_src+"\\extra_nuclei.mask.png", img_as_ubyte(ext), cmap='gray')
        
    def LoadMasks(self, fname):
        mask_src=self.src+"\\"+fname+"\\Masks"
        LEL_mask=(io.imread(mask_src+"\\LEL.mask.png")>0)
        CMV_mask=(io.imread(mask_src+"\\CMV.mask.png")>0)
        nuclei_mask=(io.imread(mask_src+"\\nuclei.mask.png")>0)
        return LEL_mask, CMV_mask, nuclei_mask
    
class IdentifyVascularNuclei:
    
    def __init__(self, vessles, nuclei, vessle_mask=None, nuclei_mask=None):
        self.vessle_int=vessles
        self.nuclei_int=nuclei
        if vessle_mask is not None:
            self.vessle_mask=vessle_mask
        else:
            self.vessle_mask=self.get_vessle_mask()
        if nuclei_mask is not None:
            self.nuclei_mask=nuclei_mask
        else:
            self.nuclei_mask=self.get_nuclei_mask(self.nuclei_int)
        self.find_vascular_nuclei()
    
    def get_vessle_mask(self, blur_sigma=8, thr_offset=0.8, opening_area=499, closing_area=499):
        blur=gaussian(self.vessle_int, sigma=blur_sigma)
        mask=blur>threshold_otsu(blur)*thr_offset
        rm_small=area_opening(mask, area_threshold=opening_area)
        return area_closing(rm_small, area_threshold=closing_area)
    
    def get_nuclei_mask(self, img):
        nuc_model = models.Cellpose(gpu=False, model_type='nuclei')
        nuc_mask, flows, styles, diams = nuc_model.eval([img], diameter=60, channels=[0,0],
                                                 flow_threshold=0.4, do_3D=False)
        return nuc_mask[0]>0
        
    def find_vascular_nuclei(self):
        objs=regionprops(ndi.label(self.nuclei_mask)[0], self.vessle_mask.astype(int))
        self.vascular_nuclei=[]
        self.extravascular_nuclei=[]
        for obj in objs:
            if np.count_nonzero(obj.image_intensity)/obj.area >0.9:
                self.vascular_nuclei.append(obj)
            else:
                self.extravascular_nuclei.append(obj)
                
    def get_sorted_masks(self):
        vasular=np.isin(ndi.label(self.nuclei_mask)[0],[obj.label for obj in self.vascular_nuclei])
        extravasular=np.isin(ndi.label(self.nuclei_mask)[0],[obj.label for obj in self.extravascular_nuclei])
        return vasular, extravasular
    
    def print_masks(self, export=False, fname=None):
        if export:
            os.makedirs('./masks', mode=0o777, exist_ok=True)
        
        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3, dpi=600, figsize=(12,8))
        ax1.set_title('Nuclei')
        ax1.imshow(self.nuclei_int)
        ax4.set_title('Vessles')
        ax4.imshow(self.vessle_int)
        ax2.set_title('Nuclei Mask')
        ax2.imshow(self.nuclei_mask>0)
        ax5.set_title('Vessle Mask')
        ax5.imshow(self.vessle_mask)
        ax3.set_title('Masked Nuclei')
        ax3.imshow(self.nuclei_int*np.invert(self.nuclei_mask>0))
        ax6.set_title('Masked Vessles')
        ax6.imshow(self.vessle_int*np.invert(self.vessle_mask))
        plt.tight_layout()
        if export:
            plt.savefig(f'./masks/{fname}.png')
        else:
            plt.show()
            
    def show_status(self):
        fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8,4))
        ax1.imshow(self.vessle_mask)
        ax2.imshow(self.nuclei_mask>0)
        ax3.imshow(cv2.merge((self.vessle_mask.astype(int)*255, (self.nuclei_mask>0).astype(int)*255, np.zeros_like(self.vessle_mask).astype(int))))
        plt.tight_layout()
           
#%% Generate all masks
        
src=r'E:\Tiffany GBA CMV\MIPs'
os.chdir(src)
images=LoadImages(src+"\SingleChannel")

for num, i in enumerate(list(images.dict.keys())[6:]):
    print(num,"/",len(list(images.dict.keys())[6:]))
    # print("processing: ", i)
    b,g,r=images.GetImage(i)
        
    vessle_analysis=IdentifyVascularNuclei(g,b)
    vascular, extra = vessle_analysis.get_sorted_masks()
    images.SaveMasks(i,
                     vessle_analysis.vessle_mask, 
                     vessle_analysis.nuclei_mask, 
                     r>5750, 
                     vascular, 
                     extra)


#%% Segmentation of nuclear mask

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import math

class NucleiSegmentation:
    
    def __init__(self, nuc_mask):
        self.ori_nuc=nuc_mask
        self.objs=regionprops(ndi.label(self.ori_nuc)[0])
        
    def SegmentRoundness(self):
        seg_out=np.zeros_like(self.ori_nuc).astype(int)
        for obj in self.objs:
                roundness=(obj.perimeter*obj.perimeter)/(math.pi*4*obj.area)
                if roundness > 1.4:
                    segmented=self.WatershedSegmentation(obj)
                    labels=ndi.label(segmented)[0]
                    sl=obj.slice
                    seg_out[sl[0].start:sl[0].stop,sl[1].start:sl[1].stop]+=labels
                else:
                    sl=obj.slice
                    seg_out[sl[0].start:sl[0].stop,sl[1].start:sl[1].stop]+=obj.image_filled
        return seg_out>0

    def FilterArea(self, thr):
        self.objs=[obj for obj in self.objs if obj.area>thr]
    
    def WatershedSegmentation(self, obj):
        distance = ndi.distance_transform_edt(obj.image_filled)
        coords = peak_local_max(distance, footprint=np.ones((20, 20)), labels=obj.image_filled, exclude_border=False, min_distance=15)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=obj.image_filled, watershed_line=True)
        return labels

i=list(images.dict.keys())[1]
b,g,r=images.GetImage(i)
ves, cmv, nuc = images.LoadMasks(i)

        
seg=NucleiSegmentation(nuc)
mask_seg=seg.SegmentRoundness()
plt.imshow(mask_seg[3000:,3000:])

#%% Analysis on generated masks

src=r'E:\Tiffany GBA CMV\MIPs'
os.chdir(src)
images=LoadImages(src+"\SingleChannel")

results={}

for num, i in enumerate(list(images.dict.keys())):
    results[i]={}
   
    print("sample nr: ", num+1)
    print("processing: ", i)

    b,g,r=images.GetImage(i)
    ves, cmv, nuc = images.LoadMasks(i)
    nuc=NucleiSegmentation(nuc).SegmentRoundness()
    vessle_analysis2=IdentifyVascularNuclei(g,b, vessle_mask=ves, nuclei_mask=nuc)
    vascular, extra = vessle_analysis2.get_sorted_masks()
    
    results[i]['pct_extravascular_cmv']=np.count_nonzero(cmv*extra)/(np.count_nonzero(cmv*(vascular+extra)))*100
    results[i]['pct_nuclear_cmv']=(np.count_nonzero(cmv*(binary_dilation(vascular+extra, footprint=diamond(3))))/np.count_nonzero(cmv))*100
    results[i]['total_cmv_area']=np.count_nonzero(cmv)
    
df=pd.DataFrame(results).T

df['treatment']=df.index.str.split("_").str[1]
df['region']=df.index.str.split("_").str[7]
df['animal']=df.index.str.split("_").str[2]

mean=df.groupby(['treatment','region','animal']).mean().reset_index()

highdose2=mean[np.isin(mean['treatment'],['HighDose3weeks','WTUninjected'])]
highdose2.columns=['Treatment','Region','Animal',
                  '% extravascular CMV signal',
                  '% transduced vascular cells',
                  '% transduced parenchymal cells']
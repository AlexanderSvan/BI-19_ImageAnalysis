from aicsimageio.readers import CziReader
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib as mpl
from skimage import io, exposure

os.chdir(r'E:/Tiffany GBA CMV/')

mpl.rcParams['image.interpolation'] = 'none'
mpl.rcParams['image.cmap']='gray'
mpl.rcParams['xtick.bottom']=False
mpl.rcParams['xtick.labelbottom']=False
mpl.rcParams['ytick.left']=False
mpl.rcParams['ytick.labelleft']=False
mpl.rcParams['savefig.pad_inches']=0.0
plt.rcParams['figure.dpi'] = 300


def CZI_to_MIP(path, export_single=False):
    
    if not os.path.exists(f"./MIPs/SingleChannel/{path}"): 
        os.makedirs(f"./MIPs/SingleChannel/{path}") 
    reader = CziReader(path)
    reader.set_scene(0)
    
    g,r,b=reader.data[0]
    
    b_max=np.max(b, axis=0)
    g_max=np.max(g, axis=0)
    r_max=np.max(r, axis=0)

    if export_single:
        io.imsave(f'./MIPs/SingleChannel/{path}/t0c0.Green.tiff', g_max)
        io.imsave(f'./MIPs/SingleChannel/{path}/t0c1.Red.tiff', r_max)
        io.imsave(f'./MIPs/SingleChannel/{path}/t0c2.Blue.tiff', b_max)

    return cv2.merge([r_max,g_max,b_max])

files=[img for img in os.listdir() if '.czi' in img]

for file in files:
    merged=CZI_to_MIP(file, export_single=True)
    io.imsave(f'./MIPs/{file}.tiff', merged)






img_scaled = cv2.normalize(merged, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

plt.figure(dpi=300)
plt.imshow(img_scaled)
import itertools
import os
import shutil
import cv2
import os
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision import transforms, models
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import matplotlib as mpl
from PIL import Image
from TinyUNetArch import TinyUNet
import time
from monodepth_utils import lin_interp

def convert_array_to_pil(depth_map):
    # Input: depth_map -> HxW numpy array with depth values 
    # Output: colormapped_im -> HxW numpy array with colorcoded depth values
    mask = depth_map!=0
    disp_map = 1/depth_map
    vmax = np.percentile(disp_map[mask], 95)
    vmin = np.percentile(disp_map[mask], 5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask] = 255
    return colormapped_im

def interpolate_depth(image_path):
    
    depth_map = np.asarray(Image.open(image_path)) / 256
    #print(np.max(depth_map))

    # Get location (xy) for valid pixeles
    y, x = np.where(depth_map > 0)
    # Get depth values for valid pixeles
    d = depth_map[depth_map != 0]

    #Generate an array Nx3
    xyd = np.stack((x,y,d)).T

    gt = lin_interp(depth_map.shape, xyd)/100

    return(gt)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


model = TinyUNet(in_channels=4, num_classes=1).to(device)
model.load_state_dict(torch.load('tiny_sliding.mod'))

model.eval()

color=cv2.imread('tester.png')[-224:,29:1213]
h,w=color.shape[:2]
datum=np.zeros((h,w,4))

image = color/255

sparse_og=np.asarray(Image.open('tester_sparse.png'))[-224:,29:1213]
print(sparse_og.shape)
sparse=interpolate_depth('tester_sparse.png')[-224:,29:1213]
datum[:,:,:3]=image
datum[:,:,3]=sparse


image_tensor=torch.unsqueeze(torch.from_numpy(datum.transpose((2, 0, 1))),0).float().to(device)
recon_batch=model(image_tensor)

pred = recon_batch[0].cpu().detach().numpy()[0]
pred[pred < 0] = 0
pred[pred > 1] = 1
pred1=255*pred
pred=256*pred*100

pred=np.where(sparse_og==0,pred,sparse_og)

pilmage=convert_array_to_pil(pred)
colored=np.array(pilmage)
colored=cv2.medianBlur(colored, 7)

cv2.imshow('color',color)
cv2.imshow('pred',pred1.astype(np.uint8))
cv2.imshow('cmap',colored[:,:,::-1])






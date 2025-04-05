import PIL.Image as Image
from monodepth_utils import lin_interp
import numpy as np
import cv2
import largestinteriorrectangle as lir
import os
import random


kitti_dir=r'D:\KITTI\train\color'

filenames=os.listdir(kitti_dir)

l=0

def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T

gradient=get_gradient_2d(200,10,1184,224,False)
for filename in filenames[:1000]:
    l+=1
    print(l)
    imname=kitti_dir+'/'+filename

    image = cv2.imread(imname)
    color_scale=image.copy()
    #cv2.imshow('Original',image)

    blur_factor=2*random.randint(2,4)+1
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_img = cv2.medianBlur(hsv_img, blur_factor)
    hsv_img=hsv_img.astype(float)

    h,w=hsv_img.shape[:2]
    #print(h,w)

    color_back=np.zeros((h,w,3))
    gray_back=128*np.ones((h,w))

    sx=random.uniform(0.95,1.06)
    #print(sx)
    sy=sx*random.uniform(0.95,1.05)

    sx=int(sx*w)
    sy=int(sy*h)
           

    color_scale=cv2.resize(color_scale,(sx,sy))
    hsv_img=cv2.resize(hsv_img,(sx,sy))
    gray = 0.5*hsv_img[:,:,2]+0.5*hsv_img[:,:,random.randint(0,1)]

    px_start=0
    px_end=w

    x_start=0
    x_end=sx

    py_start=0
    py_end=h

    y_start=0
    y_end=sy

    if sx>w:
        x_start=random.randint(0,sx-w)
        x_end=x_start+w
        px_end=w
    else:
        px_start=random.randint(0,w-sx)
        px_end=px_start+sx
        
    if sy>h:
        y_start=random.randint(0,sy-h)
        y_end=y_start+h
        py_end=h
    else:
        py_start=random.randint(0,h-sy)
        py_end=py_start+sy

    gray_back[py_start:py_end,px_start:px_end]=gray[y_start:y_end,x_start:x_end]
    color_back[py_start:py_end,px_start:px_end,:]=color_scale[y_start:y_end,x_start:x_end,:]

    
    gray_back=0.6*gradient+0.4*gray_back
            
    gray_back=gray_back/255
    if random.randint(0,2)>1:
        gray_back=1-gray_back

    cv2.imwrite(imname.replace('color','syn_color'),color_back)
    np.save(imname.replace('color','syn_sparse').replace('png','npy'),gray_back)
cv2.imshow('gray',gray_back)

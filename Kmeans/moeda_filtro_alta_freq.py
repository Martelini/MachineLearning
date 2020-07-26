#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 23:58:11 2020

@author: mateus
"""

#f = cv2.cvtColor(cv2.imread("/home/mateus/Downloads/iCloud_Photos/imagem4.JPEG"), cv2.COLOR_BGR2RGB)

import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

img = cv2.imread("/home/mateus/Downloads/iCloud_Photos/imagem4.JPEG",0)
f = np.fft.fft2(np.float32(img))
fshift = np.fft.fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift)+1)
c = 255/np.amax(magnitude_spectrum)
magnitude_spectrum = c*magnitude_spectrum

#plt.subplot(121),plt.imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#plt.show()

#%%

w, h = img.shape

n = 1
filtro = np.zeros([w,h])
D0 = 300
cx = int(np.floor(w/2)+1)
cy = int(np.floor(h/2)+1)
for x in range(w):
    for y in range(h):
        D = math.sqrt((x-cx)**2+(y-cy)**2)
        filtro[x,y] = 1/(1+(D0/(D+0.000001))**(2*n))
            
#%%
    
img_filtro = np.uint8(filtro*255)

#plt.figure('Filtro')
#plt.imshow(img_filtro, 'gray')
        
#%%

f1 = np.multiply(fshift,filtro)
#f1[f1<0] = 0
magnitude_spectrum1 = c*np.log(np.abs(f1)+1)


#plt.figure('Imagem filtrada - Fourier')
#plt.imshow(magnitude_spectrum1,'gray')

#%%


f_ishift = np.fft.ifftshift(f1)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
#img_back = img_back - img_back.min()
#img_back = img_back*255 / img_back.max()
#img_back = img_back.astype(np.uint8)

img_back = np.uint8(img_back)

#%%

#plt.figure('Imagem filtrada final')
#plt.imshow(img_back,'gray')

#%%


ret2,th2 = cv2.threshold(img_back,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#plt.imshow(th2, 'gray')

#%%

kernel = np.ones([2,2])

morph = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

plt.imshow(morph,'gray')

#%%

filled = ndimage.binary_fill_holes(morph)

plt.imshow(filled, 'gray')


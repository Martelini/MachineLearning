#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:37:27 2020

@author: Mateus Martelini Souza
"""
# Importing the libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
#import math

#from sklearn.decomposition import PCA
#from sklearn.decomposition import FastICA
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.preprocessing import normalize
#from sklearn.model_selection import StratifiedKFold

#import pyautogui
import time

if os.name == "windows":
    # folder for windows
    folder = "C:\\Users\Admin\Documents\Mateus"
elif os.name == "posix":
    # folder for linux
    folder = '/home/mateus/TCC/'

os.chdir(folder)
import imfun

#%% Carrega a imagem multiespectral

folder = '/home/mateus/TCC/Aprendendo/Wide Field Image/12) 12.11.20 - Voluntário Saudável 1'

# Carrega os comprimentos de onda

wl1 = [int(name[-7:-4]) for name in os.listdir(folder)]

# Carrega todas as imagens

I1 = imfun.load_gray_images(folder, None)

#%% Detecção do círculo

circles = cv2.HoughCircles(I1[5], method=cv2.HOUGH_GRADIENT, dp=1.2,minDist=10)

output = I1[5].copy

if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    # show the output image

    cv2.imshow("output", np.hstack([I1[5], output]))
    cv2.waitKey(0)

#%% Cortar imagem

ret, points = imfun.crop_image(I1[5], False, None)

wl = []
I = []

for i, value in enumerate(wl1):
    if value % 10 == 0:
        if np.mean(I1[i][points[0][0]:points[1][0], points[0][1]:points[1]][1]) > 30:
            wl.append(value)
            #I.append(cv2.equalizeHist(I1[i] & poly))
            I.append(I1[i][points[0][0]:points[1][0], points[0][1]:points[1]][1])
#%% Redução de imagens

wl = []
I = []

for i, value in enumerate(wl1):
    if value % 10 == 0:
        if np.mean(I1[i]) > 30:
            wl.append(value)
            I.append(cv2.equalizeHist(I1[i]))


#%% Cria imagem RGB e BGR

bgr_index = [wl1.index(490), wl1.index(570), wl1.index(640)]

bgr = np.dstack([I1[i] for i in bgr_index])
rgb = bgr[:,:,::-1]

plt.figure()
plt.imshow(rgb)
plt.title('Imagem RGB não alinhada')
plt.show








#%% Alinhamento das imagens usando Feature Based

Iref = I[0].copy()
I_al = [Iref]

for i in range(1,len(I)):
    I_r, warp = imfun.align_features(I[i], Iref, False)
    I_al.append(I_r)



#I_r, warp = imfun.align_features(cv2.equalizeHist(I1[bgr_index[2]]), cv2.equalizeHist(I1[bgr_index[0]]), True)



#%% Alinhamento das imagens usando ECC

start = time.time()

bgr_al, _ = imfun.align_ECC(bgr, cv2.MOTION_AFFINE)

end = time.time()
print(end-start)

#%%

bgr_index1 = [wl.index(490), wl.index(570), wl.index(630)]
bgr_al = np.dstack([I_al[i] for i in bgr_index1])
rgb_al = bgr_al[:,:,::-1]

plt.figure()
plt.imshow(rgb_al)
plt.title('Imagem RGB alinhada')
plt.show

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
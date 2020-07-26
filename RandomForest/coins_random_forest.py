#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:19:23 2020

@author: mateus
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

import os

from sklearn.ensemble import RandomForestClassifier

def dados(img, label):
    w, h, d = img.shape
    img_data = np.reshape(img, [w*h, d])

    img_x = np.int8(img_data)
    img_y = np.ones(w*h)*label
    
    return img_x, img_y

#%%

folder1 = '/home/mateus/Downloads/iCloud_Photos/moedas/'
folder2 = '/home/mateus/Downloads/iCloud_Photos/base/'

data_base_length = 0
n_labels = 2

coins = []
base = []
for folder in (folder1,folder2):
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
            data_base_length += img.shape[0]
            if folder == folder1:
                coins.append(img)
            elif folder == folder2:
                base.append(img)
            

data = np.zeros([data_base_length, 4])

l=0
for i in (coins, base):
    for j in range(len(i)):
        for k in range(i[j].shape[0]):
            data[l,0:3] = i[j][k]
            if i == coins:
                data[l,3] = 1
            elif i == base:
                data[l,3] = 0
            l += 1

#%%

folder = '/home/mateus/Downloads/iCloud_Photos/'
filename = 'imagem8.jpeg'
analise = cv2.imread(os.path.join(folder,filename))

analise_data = analise.reshape([analise.shape[0]*analise.shape[1], analise.shape[2]])

#%%

random_forest = RandomForestClassifier(max_depth=12, random_state=0)

random_forest.fit(data[:,0:3],data[:,3])

#%%

result = np.int8(random_forest.predict(analise_data))


#%%

codebook = np.linspace(0,1,n_labels)

#codebook = shuffle(colors, random_state=0)[:n_labels]

w, h = analise.shape[0:2]
image_class = np.zeros([w,h])
label_idx = 0
for i in range(w):
    for j in range(h):
        image_class[i,j] = codebook[result[label_idx]]
        label_idx += 1

image_class = np.uint8(image_class*255)


#%%
'''
plt.subplot(121),plt.imshow(cv2.cvtColor(analise, cv2.COLOR_BGR2GRAY), cmap = 'gray')
plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(image_class, cmap = 'gray')
plt.title('Imagem classificada'), plt.xticks([]), plt.yticks([])
plt.show()
'''

#%%

'''
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))

morph = cv2.morphologyEx(image_class, cv2.MORPH_OPEN, kernel, iterations = 2)

sure_bg = cv2.dilate(morph,kernel,iterations=3)

dist_transform = cv2.distanceTransform(morph,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

plt.figure('eds2f')
plt.subplot(121)
plt.title('Limpa ruido')
plt.imshow(np.uint8(morph), 'gray')

plt.subplot(122)
plt.title('Dilatou - sure background')
plt.imshow(np.uint8(sure_bg), 'gray')

plt.figure('sadas')
plt.subplot(121)
plt.imshow(np.uint8(sure_fg), 'gray')
plt.title('Distance transform + threshold - Sure foreground')
plt.subplot(122)
plt.imshow(np.uint8(unknown), 'gray')
plt.title('Diferença')
'''
#%%
filled = ndimage.binary_fill_holes(image_class)

filled[filled==True] = 255
filled[filled==False] = 0

filled = np.int8(filled)

#plt.imshow(np.uint8(filled), 'gray')

#%%

kernel1 = np.ones([35,35])
#kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
morph = cv2.erode(filled*255, kernel1, iterations = 2)


#kernel2 = np.ones([2,2])
#morph = cv2.erode(morph, kernel2, iterations = 1)

#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
#morph = cv2.morphologyEx(image_class, cv2.MORPH_OPEN, kernel, iterations = 1)
#plt.imshow(morph, 'gray')



#%%

#ret,thresh1 = cv2.threshold(morph,127,255,cv2.THRESH_BINARY)

ret, labels = cv2.connectedComponents(np.uint8(morph))
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

#%%

plt.subplot(121),plt.imshow(cv2.cvtColor(analise, cv2.COLOR_BGR2GRAY), cmap = 'gray')
plt.title('Imagem original'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(labeled_img, cmap = 'gray')
plt.title('Imagem classificada'), plt.xticks([]), plt.yticks([])
plt.show()

print('O número de moedas contadas é:', ret-1)


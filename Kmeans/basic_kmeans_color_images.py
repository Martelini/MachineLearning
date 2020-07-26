#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:37:53 2020

@author: mateus
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#%%

f = cv2.cvtColor(cv2.imread("/home/mateus/Downloads/iCloud_Photos/imagem4.JPEG"), cv2.COLOR_BGR2RGB)

image = f.astype(np.int16)/255

w, h, d = original_shape = image.shape
image_array = np.reshape(image, (w*h, d))

# Training of Kmeans with 1000 random pixels

n_cluster = 2

image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(image_array_sample)

# Predict the whole image

labels = kmeans.predict(image_array)

image_rec = np.uint8(labels.reshape(w, h))


#%%

colors = np.random.rand(100,3)

codebook = shuffle(colors, random_state=0)[:n_cluster]
#codebook = np.array([[255, 0, 0],[0, 255, 0],[0, 0, 255]], dtype = np.uint8)

#labels_random = pairwise_distances_argmin(codebook, image_array, axis=0)

image = np.zeros((w, h, d))
label_idx = 0
for i in range(w):
    for j in range(h):
        image[i][j] = codebook[labels[label_idx]]
        label_idx += 1

#%%

plt.figure('Imagem Original')
plt.imshow(f)

plt.figure('Imagem classificada')
plt.imshow(image, 'gray')

#%%

codebook = np.array([[1, 1, 1], [0, 0, 0]])

image = np.zeros((w, h, d))
label_idx = 0
for i in range(w):
    for j in range(h):
        image[i][j] = codebook[labels[label_idx]]
        label_idx += 1




#%%

cv2.imwrite('Moeda.jpg', np.uint8(image*255))
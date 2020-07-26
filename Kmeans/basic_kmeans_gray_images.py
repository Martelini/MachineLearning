#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:08:15 2020

@author: mateus
"""
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

f = cv2.imread("/home/mateus/Downloads/iCloud_Photos/imagem7.JPEG", 0)

img_eq = cv2.equalizeHist(f)

image_array = np.int16(img_eq)/255

w, h = image_array.shape

data = np.zeros([w*h])

index = 0
for i in range(w):
    for j in range(h):
        #data[index,0] = index
        data[index] = image_array[i,j]
        index += 1

data = data.reshape(-1,1)

#%%

n_cluster = 3

image_array_sample = shuffle(data, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(image_array_sample)

# Predict the whole image

labels = kmeans.predict(data)

#image_class = np.uint8(labels*255)


#%%

colors = np.random.rand(100)

codebook = shuffle(colors, random_state=0)[:n_cluster]
#codebook = np.array([[255, 0, 0],[0, 255, 0],[0, 0, 255]], dtype = np.uint8)

#labels_random = pairwise_distances_argmin(codebook, image_array, axis=0)

image_class = np.zeros([w, h])
label_idx = 0
for i in range(w):
    for j in range(h):
        image_class[i][j] = codebook[labels[label_idx]]
        label_idx += 1

image_class = np.uint8(image_class*255)

#%%

plt.figure('Imagem Original')
plt.imshow(f)

plt.figure('Imagem classificada')
plt.imshow(image_class, 'gray')

#%%

#codebook = np.array([[1, 1, 1], [0, 0, 0]])
'''
image = np.zeros([w, h])
label_idx = 0
for i in range(w):
    for j in range(h):
        image[i][j] = codebook[labels[label_idx]]
        label_idx += 1


'''

#%%

#cv2.imwrite('Moeda.jpg', np.uint8(image*255))


#%%

kernel = np.ones([3,3])

#morph = cv2.dilate(image_class,kernel, iterations = 1)
morph = cv2.morphologyEx(image_class, cv2.MORPH_CLOSE, kernel)

plt.imshow(morph,'gray')
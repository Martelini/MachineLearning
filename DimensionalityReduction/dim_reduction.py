#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:37:27 2020

@author: mateus
"""
# Import the libraries

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2



from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
#from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

import pyautogui
import time

folder = '/home/mateus/TCC/Aprendendo/DimensionalityReduction/'
os.chdir(folder)
import imfun

#%% Function to reorganize the result image

colors = np.array([[200,200,0],[128,0,0],[0,128,0],[0,0,128],[128,128,128],[0,128,128],[128,128,0],[0,200,200],[0,0,200],[0,200,0],[200,0,0]])

def reorganize(img, codebook, w, h, z):
    i = 0
    image = np.zeros([w,h,z])
    for x in range(w):
        for y in range(h):
            image[x,y] = codebook[int(img[i]),:]
            i += 1
    return np.uint8(image)

#%% Load the images

folder = '/home/mateus/TCC/Aprendendo/DimensionalityReduction/04) 21.05.20 - Microscopio Nikon, Slide 3B'
os.chdir(folder)
names = os.listdir(os.curdir)

wavelength = [410, 430, 450, 470, 490, 510, 530, 540, 550, 570, 590, 610, 630, 650, 670, 690, 710]

images = []
for i in range(len(wavelength)):
    images.append(cv2.imread(names[i], cv2.IMREAD_GRAYSCALE))

#%% RGB image


blue = wavelength.index(470) 
green = wavelength.index(570)
red = wavelength.index(630) 

bgr = np.dstack((images[blue],images[green],images[red]))
rgb = np.dstack((images[red],images[green],images[blue]))

w, h = bgr.shape[0:2]
d = len(images)

#%% Hyperspectral image

hyper_image = np.zeros([w, h, d])

for x in range(w):
    for y in range(h):
        for z in range(d):
            hyper_image[x,y,z] = images[z][x,y]


#%% Crop image and select region

label = []
data = []

pyautogui.alert('Nesse momento as regiões serão selecionadas', "Regiões")
label.append(pyautogui.prompt('Escreva o nome da primeira região:'))

i = 0
end = 'Sim'

while end == 'Sim':
    zoom = pyautogui.confirm(text='Deseja dar zoom para selecionar a região?', title='Zoom', buttons=['Sim', 'Não'])
    if zoom == 'Sim':
        pyautogui.alert('Escolha um local da imagem para dar zoom', "Zoom")
        time.sleep(0.1)
        ret, points = imfun.crop_image(bgr, False, None)
    else:
        ret = bgr
        points = [(0,0),(0,0)]
    pyautogui.alert('Selecione um polígono na imagem', "Seleção")
    time.sleep(0.1)
    poly, points1 = imfun.polyroi(ret)#, window_name=('Selecione a região'+label[0]))
    poly = poly[:,:,0]
    index = np.asarray(np.where(poly==255)).T + [min(points[0][1], points[1][1]),min(points[0][0], points[1][0])]
    for x, y in index:
        data.append(np.hstack((hyper_image[x,y,:], i)))
    keep = pyautogui.confirm(text='Deseja selecionar mais pedaços dessa região?', title='Continuar', buttons=['Sim', 'Não'])
    if keep == 'Não':
        end = pyautogui.confirm(text='Deseja selecionar outra região?', title='Adicionar outra região', buttons=['Sim', 'Não'])
        if end == 'Sim':
            label.append(pyautogui.prompt('Escreva o nome da próxima região:'))
            i += 1

pyautogui.alert('Aguarde o treinamento e a predição da imagem', "Aguarde")

data = np.asarray(data)
#%% Organizing data

print('Fitting PCA and ICA')

# Normalization

data_x = data[:,:d]
data_x_norm = data_x/data_x.max()

# Applying feature extraction algorithms

#explained_variance = 0.95
n_components = 2

pca = PCA(n_components=n_components) # It's necessary just one component for this variance
pca_data = pca.fit_transform(data_x_norm)


ica = FastICA(n_components=n_components)
ica_data = ica.fit_transform(data_x_norm)

# Too slow

#lda = LatentDirichletAllocation(n_components=n_components)
#lda.fit(data_x_norm)

#%% Random Forest fitting

print('Random Forest fitting')

data_y = data[:,-1]
n_trees = 50

rf_pca = RandomForestClassifier(n_trees)
rf_ica = RandomForestClassifier(n_trees)

rf_pca.fit(pca_data, data_y)
rf_ica.fit(ica_data, data_y)

# Preparing the image to be predicted

print('Preparing the image to be predicted')

hyper_image_data = hyper_image.reshape([w*h, d])

hyper_image_norm = hyper_image_data/hyper_image_data.max()

pca_image = pca.fit_transform(hyper_image_norm)

ica_image = ica.fit_transform(hyper_image_norm)

# Predicting image

print('Predicting image')

result_pca = rf_pca.predict(pca_image)
result_ica = rf_ica.predict(ica_image)

# Reshaping image

codebook = colors[:len(label),:]

final_pca = reorganize(result_pca, codebook, w, h, 3)
final_ica = reorganize(result_ica, codebook, w, h, 3)

# Show the resulting images

plt.figure("Figura original e Random Forest")
plt.subplot(131)
plt.imshow(rgb)
plt.title("Imagem original")

plt.subplot(132)
plt.imshow(final_pca)
plt.title("Random Forest with PCA")

plt.subplot(133)
plt.imshow(final_ica)
plt.title("Random Forest with ICA")
plt.show

#%% Steps


# Carregar as imagens dos espectros - ok
# Juntar todas as imagens para formar uma imagem multiespectral - not necessary
# Fazer uma imagem RGB para usar na polyroi e crop image - ok

# Perguntar o label - ok

# Fazer o crop image - ok
# Fazer o polyroi - ok
# Reshape the image para deixar os pixels em uma única dimensão - ok
# Empilhar a imagem - ok

# Perguntar se continuará no mesmo label - ok
# Se não, criar novo label ou finalizar - ok



# Normalizar os dados - ok
# Fazer PCA, ICA e LDA - ok
# Aplicar Random Forest nos três - ok
# Predizer a imagem inteira com os três algoritmos - ok
# Reshape a imagem para voltar ao tamanho original - ok
# Apresentar as três imagens - ok

# Implementar filtro passa-alta
# Tentar outros algoritmos de extração de features




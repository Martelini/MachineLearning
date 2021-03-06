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
import math

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
#from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedShuffleSplit

import pyautogui
import time

# folder for windows
folder = "C:\\Users\Admin\Documents\Mateus"

# folder for linux
#folder = '/home/mateus/TCC/'

os.chdir(folder)
import imfun

del folder
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

#%% Definition of the filter

def filter_freq(img, tipo, freq):
    w, h = img.shape[0:2]

    a = np.fft.fft2(img)#, [2*w, 2*h])
    a = np.fft.fftshift(a)

    filtro = np.zeros([w, h])
    n=1

    cx = int(np.floor(w/2)+1)
    cy = int(np.floor(h/2)+1)
    for x in range(w):
        for y in range(h):
            D = math.sqrt((x-cx)**2+(y-cy)**2)
            if tipo == 'high':
                filtro[x,y] = 1/(1+(freq/(D+0.000001))**(2*n))
            elif tipo == 'low':
                filtro[x,y] = 1/(1+((D+0.000001)**(2*n)/freq))
            else:
                print('Não há esse filtro')

    b = np.multiply(a,filtro)

    c = np.fft.ifftshift(b)
    c = np.fft.ifft2(c)#, [w,h])
    c= np.uint8(np.abs(c))
    
    return c


#%% Funções

def load_hyper_image(folder):
    names = os.listdir(os.curdir)
    print("Loading hyperspectral image...")
    os.chdir(folder)
    images = []
    for i in range(len(names)):
        images.append(cv2.imread(names[i], cv2.IMREAD_GRAYSCALE))
    w, h = np.shape(images[0])
    d = len(images)
    hyper_image = np.zeros([w, h, d])
    for x in range(w):
        for y in range(h):
            for z in range(d):
                hyper_image[x,y,z] = images[z][x,y]
    return hyper_image

def create_RGB_image(wavelength, hyper_image):
    blue = wavelength.index(470) 
    green = wavelength.index(570)
    red = wavelength.index(630)

    bgr = np.uint8(np.dstack((hyper_image[:,:,blue],hyper_image[:,:,green],hyper_image[:,:,red])))
    rgb = np.uint8(np.dstack((hyper_image[:,:,red],hyper_image[:,:,green],hyper_image[:,:,blue])))
    
    rgb_index = (red, green, blue)
    
    return bgr, rgb, rgb_index
#%% Load the hyperspectral image

# folder for windows
folder = 'C:\\Users\Admin\Documents\Mateus\MachineLearning-master\DimensionalityReduction\MicroscopioNikonSlide3B'

# folder for linux
#folder = '/home/mateus/TCC/Aprendendo/DimensionalityReduction/04) 21.05.20 - Microscopio Nikon, Slide 3B'

#wavelength = [410, 430, 450, 470, 490, 510, 530, 540, 550, 570, 590, 610, 630, 650, 670, 690, 710]
wavelength = [410, 430, 450, 470, 490, 510, 530, 540, 550, 570, 590, 610, 630, 650, 670, 690, 710]


hyper_image = load_hyper_image(folder)

#%% RGB image

bgr, rgb, col = create_RGB_image(wavelength, hyper_image)

w, h = bgr.shape[0:2]
d = len(wavelength)

#%% Crop image and select region
'''
label = []
index = []

pyautogui.alert('Nesse momento as regiões serão selecionadas', "Regiões")

glass_ex = pyautogui.confirm(text='Existe a região do vidro?', title='Vidro', buttons=['Sim', 'Não'])

if glass_ex == "Sim":
      label.append(pyautogui.prompt('Escreva o nome da região correspondente ao vidro:'))

else: 
      label.append(pyautogui.prompt('Escreva o nome da primeira região:'))

i = 0
end = 'Sim'

while end == 'Sim':
    zoom = pyautogui.confirm(text='Deseja dar zoom para selecionar a região?', title='Zoom', buttons=['Sim', 'Não'])
    if zoom == 'Sim':
        pyautogui.alert('Escolha um local da imagem para dar zoom', "Zoom")
        time.sleep(0.3)
        ret, points = imfun.crop_image(bgr, False, None)
    else:
        ret = bgr
        points = [(0,0),(0,0)]
    poly, points1 = imfun.polyroi(ret)#, window_name=('Selecione a região'+label[0]))
    poly = poly[:,:,0]
    temp = np.asarray(np.where(poly==255)).T + [min(points[0][1], points[1][1]),min(points[0][0], points[1][0])]
    temp = np.hstack((temp,i*np.ones([temp.shape[0],1])))
    index.append(temp)
    keep = pyautogui.confirm(text='Deseja selecionar mais pedaços dessa região?', title='Continuar', buttons=['Sim', 'Não'])
    if keep == 'Não':
        end = pyautogui.confirm(text='Deseja selecionar outra região?', title='Adicionar outra região', buttons=['Sim', 'Não'])
        if end == 'Sim':
            label.append(pyautogui.prompt('Escreva o nome da próxima região:'))
            i += 1
'''
#%%

glass_index = []

if glass_ex == "Sim":
      for i in range(len(index)):
            if index[i][0,2] == 0:
                  glass_index.append(index[i])

del ret

#%% High-pass filter

def apply_filter(hyper_image, col):

    f_image = hyper_image

    aplicar_filtro = pyautogui.confirm(text='Deseja aplicar algum filtro?', title='Filtro', buttons=['Sim', 'Não'])

    while aplicar_filtro == "Sim":
        tipo = pyautogui.prompt("Tipo do filtro: (low, high, median)")
        print("Applying the filter...")
        comp = pyautogui.confirm(text='Quantas componentes?', title='Filtro', buttons=['RGB', 'Todas'])
        if tipo == "low" or tipo == "high":
            frequency = int(pyautogui.prompt('Digite a frequência de corte em pixels:'))
            if comp == "RGB":
                filtered_hyper_image = np.zeros([w,h,3])
                for i in range(3):
                       filtered_hyper_image[:,:,i] = filter_freq(hyper_image[:,:,col[i]], tipo, frequency)
            elif comp == "Todas":
                filtered_hyper_image = np.zeros([w,h,d])
                for i in range(d):
                    filtered_hyper_image[:,:,i] = filter_freq(hyper_image[:,:,i], tipo, frequency)
        elif tipo == "median":
            size = int(pyautogui.prompt("Digite o tamanho do filtro"))
            if comp == "RGB":
                filtered_hyper_image = np.zeros([w,h,3])
                for i in range(3):
                    filtered_hyper_image[:,:,i] = cv2.medianBlur(np.uint8(hyper_image[:,:,col[i]]), size)
            elif comp == "Todas":
                filtered_hyper_image = np.zeros([w,h,d])
                for i in range(d):
                    filtered_hyper_image[:,:,i] = cv2.medianBlur(np.uint8(hyper_image[:,:,i]), size)
        f_image = np.dstack((f_image,filtered_hyper_image))
        aplicar_filtro = pyautogui.confirm(text='Deseja aplicar mais algum filtro?', title='Filtro', buttons=['Sim', 'Não'])
    return f_image, aplicar_filtro

final_image, aplicar_filtro = apply_filter(hyper_image, col)

# #%% Show filtered image

# filtered_rgb = np.uint8(np.dstack((filtered_hyper_image[:,:,0],filtered_hyper_image[:,:,1],filtered_hyper_image[:,:,2])))

# plt.figure("Filtered Image")
# plt.subplot(121)
# plt.imshow(rgb)
# plt.title("Original Image")

# plt.subplot(122)
# plt.imshow(filtered_rgb)
# plt.title("Filtered Image")

#%%
#del hyper_image
#%% White normalization
   
if glass_ex == "Sim" and aplicar_filtro == "Não":
      glass_values = []

      if glass_ex == "Sim":
            for i in range(len(index)):
                  if index[i][0,2] == 0:
                        for x, y in index[i][:,0:2]:  
                              glass_values.append(hyper_image[int(x),int(y),:])

      glass_values = np.asarray(glass_values)
      
      glass_mean = np.mean(glass_values, axis=0)

      for z in range(d):
            final_image[:,:,z] = final_image[:,:,z]/glass_mean[z]
      


#%%

d = np.shape(final_image)[2]

data = []

for i in range(len(index)):
    for x, y, l in index[i]:
        data.append(np.hstack((final_image[int(x),int(y),:], l)))

data = np.asarray(data)

#%% Split training and test sets

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)

sss.get_n_splits(data[:,:d])

for i, j in sss.split(data[:,:d], data[:,-1]):
    train_index = i
    test_index = j

#%% Organizing data

apply_pca = pyautogui.confirm(text='Deseja aplicar PCA?', title='PCA', buttons=['Sim', 'Não'])

if apply_pca == 'Sim':
    n_comp = int(pyautogui.prompt("Digite o número de componentes ou a variância explicada:"))
    print('Fitting PCA...')
    data_x_norm = normalize(data[train_index][:,:d])
    pca = PCA(n_components=n_comp)
    pca_data = pca.fit_transform(data_x_norm)
    del data_x_norm
    
apply_ica = pyautogui.confirm(text='Deseja aplicar ICA?', title='ICA', buttons=['Sim', 'Não'])

if apply_ica == 'Sim':
    n_comp1 = int(pyautogui.prompt("Digite o número de componentes:"))
    print('Fitting ICA...')
    data_x_norm = normalize(data[train_index][:,:d])
    ica = FastICA(n_components=n_comp1)
    ica_data = ica.fit_transform(data_x_norm)
    del data_x_norm


# Too slow

#lda = LatentDirichletAllocation(n_components=n_components)
#lda.fit(data_x_norm)

#%% Random Forest fitting using train set

n_trees = int(pyautogui.prompt("Digite o número de árvores:"))

apply_rf = pyautogui.confirm(text='Deseja aplicar Random Forest nos dados brutos?', title='Random Forest', buttons=['Sim', 'Não'])

if apply_rf == "Sim":
    print('Fitting Random Forest...')
    rf = RandomForestClassifier(n_trees)
    rf.fit(data[train_index][:,:d], data[train_index][:,-1])

if apply_pca == "Sim":
    print('Fitting Random Forest with PCA...')
    rf_pca = RandomForestClassifier(n_trees)
    rf_pca.fit(pca_data, data[train_index][:,-1])
    
if apply_ica == "Sim":
    print('Fitting Random Forest with ICA...')
    rf_ica = RandomForestClassifier(n_trees)
    rf_ica.fit(ica_data, data[train_index][:,-1])
    

#%% Predicting the whole image

hyper_image_data = final_image.reshape([w*h, d])

if apply_rf == "Sim":
    print('Predicting image with Random Forest...')
    result = rf.predict(hyper_image_data)

if apply_pca == "Sim":
    print('Predicting image with Random Forest and PCA...')
    hyper_image_norm = normalize(hyper_image_data)
    pca_image = pca.transform(hyper_image_norm)
    result_pca = rf_pca.predict(pca_image)
      
if apply_ica == "Sim":
    print('Predicting image with Random Forest and ICA...')
    hyper_image_norm = normalize(hyper_image_data)
    ica_image = ica.transform(hyper_image_norm)
    result_ica = rf_ica.predict(ica_image)


#%% Reshaping image

codebook = colors[:len(label),:]

if apply_rf == "Sim":
    final = reorganize(result, codebook, w, h, 3)

if apply_pca == "Sim":
    final_pca = reorganize(result_pca, codebook, w, h, 3)
    
if apply_ica == "Sim":
    final_ica = reorganize(result_ica, codebook, w, h, 3)


#%% Show the resulting images

i = 1

total = int(apply_rf == "Sim") + int(apply_pca == "Sim") + int(apply_ica == "Sim")
if total == 1:
    sub = [1,2]
elif total == 2:
    sub = [1,3]
elif total == 3:
    sub = [2,2]


plt.figure("Resultado final")

plt.subplot(sub[0],sub[1],i)
i += 1
plt.imshow(rgb)
plt.title("Imagem original")
plt.show

if apply_rf == "Sim":
    plt.subplot(sub[0],sub[1],i)
    i += 1
    plt.imshow(final)
    plt.title("Random Forest")
    plt.show

if apply_pca == "Sim":
    plt.subplot(sub[0],sub[1],i)
    i += 1
    plt.imshow(final_pca)
    plt.title("Random Forest with PCA")
    plt.show
    
if apply_ica == "Sim":
    plt.subplot(sub[0],sub[1],i)
    i += 1
    plt.imshow(final_ica)
    plt.title("Random Forest with ICA")
    plt.show
    
#%% Applying Random Forest on test set to get the score

if apply_rf == "Sim":
    print('Calculating the score with only Random Forest...')
    score_rf = rf.score(data[test_index][:,:d], data[test_index][:,-1])

if apply_pca == "Sim":
    print('Calculating the score with Random Forest and PCA...')
    test_set_norm = normalize(data[test_index][:,:d])
    test_pca = pca.transform(test_set_norm)
    score_pca = rf_pca.score(test_pca,data[test_index][:,-1])
    
if apply_ica == "Sim":
    print('Calculating the score with Random Forest and ICA...')
    test_set_norm = normalize(data[test_index][:,:d])
    test_ica = ica.transform(test_set_norm)
    score_ica = rf_ica.score(test_ica[:,:d],data[test_index][:,-1])


print("Os scores foram:")
if apply_rf == "Sim":
    print("Dados brutos:", score_rf)

if apply_pca == "Sim":
    print("PCA:", score_pca)
    
if apply_ica == "Sim":
    print("ICA:", score_ica)





















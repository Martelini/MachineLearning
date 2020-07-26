#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 22:31:39 2020

@author: mateus
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#%%

casos = pd.read_csv("/home/mateus/TCC/Aprendendo/LinearRegression/covid19/Dados-covid-19-estado.csv", delimiter=';', encoding = 'iso8859_13',na_values="")

x = np.linspace(1, len(casos['Data']), len(casos['Data']))
y = casos['Total de casos']
z = casos['Casos por dia']

plt.scatter(x,y)
plt.xlabel('Casos acumulados [dias]')
plt.ylabel('Número de casos')


#%%

degree = 3
pol_reg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

pol_reg.fit(np.reshape(x, [-1,1]),y)

x_plot = np.linspace(0,len(x), 1000)

y_plot = pol_reg.predict(np.reshape(x_plot,[-1,1]))

plt.plot(x_plot, y_plot, 'r')
plt.title('Acumulado de casos de COVID-19')
plt.show

print("Os parâmetros encontrados são:", pol_reg._final_estimator.coef_[0],pol_reg._final_estimator.coef_[1],pol_reg._final_estimator.coef_[2],pol_reg._final_estimator.coef_[3])

#%%

exp_reg = LinearRegression()

exp_reg.fit(np.reshape(x, [-1,1]),np.log2(y+0.0001))

theta0 = exp_reg.intercept_
theta1 = exp_reg.coef_[0]

x_plot_exp = np.linspace(0,len(x), 1000)

y_plot_exp = 2**theta0+2**np.multiply(theta1,x_plot_exp)

plt.plot(x_plot_exp, y_plot_exp, 'g')
plt.show

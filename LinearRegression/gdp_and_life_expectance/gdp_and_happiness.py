# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load the data

oecd = pd.read_csv("/home/mateus/TCC/Aprendendo/LinearRegression/gdp_and_life_expectance/oecd_bli_2015.csv", thousands = ',')
gdp = pd.read_csv("/home/mateus/TCC/Aprendendo/LinearRegression/gdp_and_life_expectance/gdp_per_capita.csv", thousands = ',', delimiter='\t',
encoding='latin1', na_values="n/a")

#print(oecd.head(n=5))
#print(gdp.head(n=5))

# Prepare the data

countries_gdp = np.asarray(gdp[['Country', '2015']])
countries_oecd = np.asarray(oecd[['Country', 'Value', 'Indicator', 'Inequality']])

n = min(countries_gdp.shape[0],countries_oecd.shape[0])                  

data = np.zeros([n,2])

k = 0
for i in range(len(countries_gdp)):
    for j in range(len(countries_oecd)):
        if countries_gdp[i][0] == countries_oecd[j][0]:
            if countries_oecd[j][2] == 'Life satisfaction':
                if countries_oecd[j][3] == 'Total':
                    data[k,0] = countries_gdp[i][1]
                    data[k,1] = countries_oecd[j][1]
                    k += 1 

data = data[0:k,:]

x = data[:,0].reshape(-1,1)
y = data[:,1].reshape(-1,1)

# Visualize the data

plt.plot(x, y, 'ro')
plt.xlim(0,np.amax(x)+3000)
plt.ylim(0,10)
plt.xlabel("GDP")
plt.ylabel("Life satisfaction")


# Select a linear model

lin_reg_model = LinearRegression()

lin_reg_model.fit(x, y)


# Make a prediction

x_new = np.asarray([32134,62728]).reshape(-1,1)
y_new = lin_reg_model.predict(x_new)

# Plot the line and prediction

x_line = np.linspace(0,np.amax(x)+3000,1000).reshape(-1,1)
y_line = lin_reg_model.predict(x_line)

plt.plot(x_new, y_new, 'bo')
plt.plot(x_line, y_line, 'g')
plt.show()

print("Os coeficientes são:" )
print("Inclinação:", lin_reg_model.coef_[0][0])
print("Intercepto:", lin_reg_model.intercept_[0])

#%%

pol_reg_model = make_pipeline(PolynomialFeatures(2), LinearRegression())

pol_reg_model.fit(x,y)

plt.scatter(x,y)

y_line1 = pol_reg_model.predict(x_line)

plt.plot(x_line, y_line1, 'g')
plt.show

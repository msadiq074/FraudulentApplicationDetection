# -*- coding: utf-8 -*-
"""
Created on Thu May 14 00:38:44 2020

@author: msadi
"""

# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Feature Scaling 
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

# training the SOM
from minisom import MiniSom
som = MiniSom(x=10,y=10,input_len=15)
som.random_weights_init(X)
som.train_random(X,10000)

# visualiziing the results 
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,markers[Y[i]],markeredgecolor=colors[Y[i]],markerfacecolor='None',markersize=10,markeredgewidth=2)
show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perovskite Material Discovery Experimental Design
Created on Mon Jul 31 13:51:36 2023

@author: me-tcoons
"""

# %% import necessary tools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
import pandas as pd

#%% import data
df = pd.read_csv('20230628_Area_transfer.csv')  
x = x=df.iloc[:,1:4]
y = df.loc[:,"Percent area Transfer (%)"]
x_train = np.array(x)[:80,:]
y_train = np.array(y)[:80]

#%% pre process data
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

#%%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
# %% train GP
kernel = 1 * RBF(length_scale=1.0)
gaussian_process = GaussianProcessRegressor(kernel=kernel)
gaussian_process.fit(x_train, y_train)
gaussian_process.kernel_

# %%
x_all = scaler.transform(np.array(x))
mean_prediction, std_prediction = gaussian_process.predict(x_all, return_std=True)

# %%

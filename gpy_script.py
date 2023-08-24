#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perovskite Material Discovery Experimental Design
Created on Aug 2 2023

@author: me-tcoons
"""

# %% import necessary tools
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import pandas as pd

#%% import data
nTrain = 20
df = pd.read_csv('20230628_Area_transfer.csv')  
noises = df.loc[:,"Stdev (%)"]
x = df.iloc[:,1:4]
y = df.loc[:,"Percent area Transfer (%)"]
x_train = torch.tensor(np.array(x)[:nTrain,:])
#y_train = torch.tensor(np.log(np.array(y)[:nTrain]))
y_train = torch.tensor(np.array(y)[:nTrain])
noises_train = torch.tensor(np.array(noises)[:nTrain])


#%% pre process data
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = torch.tensor(scaler.transform(x_train))
#y_scaler = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
#y_train = torch.tensor(y_scaler.transform(y_train.reshape(-1,1))).squeeze()
noises_train[np.isnan(noises_train)]=np.average(noises_train)

#%%
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises_train,learn_additional_noise=True)
model = ExactGPModel(x_train, y_train, likelihood)

# %%
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
training_iter=50

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(x_train)
    # Calc loss and backprop gradients
    loss = -mll(output, y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
# %%

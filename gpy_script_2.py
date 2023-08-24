#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %% import necessary tools
print("import necessary tools")
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import pandas as pd


# In[1]:


#%% import data
print("import data")
nTrain = 50
df = pd.read_csv('20230628_Area_transfer.csv')  
df = df.sample(frac=1)
noises = df.loc[:,"Stdev (%)"]
x = df.iloc[:,1:4]
y = df.loc[:,"Percent area Transfer (%)"]
x_train = torch.tensor(np.array(x)[:nTrain,:])
y_train = torch.tensor(np.array(y)[:nTrain])
noises_train = torch.tensor(np.array(noises)[:nTrain])


# In[179]:


#%% pre process data
from sklearn import preprocessing

# need to scale noises only by dividing by variance of inputs
noises_train = noises_train/torch.std(x_train)
noises_train[np.isnan(noises_train)]=np.average(noises_train[np.isnan(noises_train)==False])

# scale x training data per standard scaler to N(0,1)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = torch.tensor(scaler.transform(x_train))

# apply log to y training data and then scale to N(0,1)
y_train = torch.log(y_train)
y_scaler = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
y_train = torch.tensor(y_scaler.transform(y_train.reshape(-1,1))).squeeze()
print(x_train)
print(noises_train)
print(y_train)
#noise_scaler = preprocessing.StandardScaler().fit(noises_train.reshape(-1,1))
#noises_train = noise_scaler.transform(noises_train.reshape(-1,1))
#noises_train[np.isnan(noises_train)]=np.average(noises_train[np.isnan(noises_train)==False])
#print(noises_train)


# In[180]:


#%%
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises_train,learn_additional_noise=True)
model = ExactGPModel(x_train, y_train, likelihood)


# In[181]:


# %%
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


# In[182]:


training_iter=200

print('Starting GP parameter tuning...')
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(x_train)
    # Calc loss and backprop gradients
    loss = -mll(output, y_train)
    loss.backward()
    if np.mod(i,20)==0:
        #print('Iter %d/%d - Loss: %.3f   lengthscale 1: %.3f   lengthscale 2: %.3f   lengthscale 3: %.3f' % (
         #   i + 1, training_iter, loss.item(),
          #  model.covar_module.base_kernel.lengthscale[0][0].item(),
           # model.covar_module.base_kernel.lengthscale[0][1].item(),
            #model.covar_module.base_kernel.lengthscale[0][1].item(),
        #))
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item()
        ))
    optimizer.step()


# In[183]:


x_test = torch.tensor(np.array(x)[nTrain:,:])
x_test = torch.tensor(scaler.transform(x_test))
y_test = torch.log(torch.tensor(np.array(y)[nTrain:]))
y_test = torch.tensor(y_scaler.transform(y_test.reshape(-1,1))).squeeze()
print(y_test)


# In[184]:


model.eval()
with torch.no_grad():
    trained_pred_dist = likelihood(model(x_test))
    predictive_mean = trained_pred_dist.mean
    lower, upper = trained_pred_dist.confidence_region()


# In[185]:


final_nlpd = gpytorch.metrics.negative_log_predictive_density(trained_pred_dist, y_test)

print(f'nTrained model NLPD: {final_nlpd:.2f}')


# In[186]:


print(y_test)
print(predictive_mean)
print(lower)
print(upper)


# In[198]:


test_size=y_test.size(dim=0)
testInds = np.linspace(0,int(test_size))
plt.figure()
plt.plot(testInds,y_test,'o',label = 'training data')
plt.plot(testInds,predictive_mean,'x',label='predicted mean')
plt.plot(testInds,lower,'-',label = 'lower confidence interval')
plt.plot(testInds,upper,'-',label = 'upper confidence interval')
plt.legend()


# In[193]:


print(y_test.size(dim=0))


# In[116]:





# In[125]:





# In[ ]:





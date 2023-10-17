# % [markdown]
# # Gaussian Process Tutorial for Perovskite Characterization
# Thomas Coons 2023

# %% import necessary tools
import sys
print(sys.executable)

print("import necessary tools")
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import pandas as pd
import tqdm as notebook_tqdm
from scipy.stats import norm
from matplotlib import colors
import matplotlib.pyplot as plt

#%%
def gridspace_points(output_file_path, output_file_name, column_names, x_range, y_range, graph_levels):
    # Create an empty DataFrame with the desired column names
    
    mesh_list = []
    #df_mesh_pts_flat = pd.DataFrame(columns=column_names)
    
    for i, level in enumerate(graph_levels):
        
        # Reshaping the data for the heatmap
        
        P = np.linspace(x_range[0], x_range[1])
        T = np.linspace(y_range[0], y_range[1])
        
        
        P, T = np.meshgrid(P, T)  # 2D grid for interpolation
        
        P_flat = P.flatten(order='C')
        T_flat = T.flatten(order = 'C')
        t = np.full(len(T_flat), level)
        
        flattened_meshpts = np.hstack((t.reshape(-1, 1), P_flat.reshape(-1, 1), T_flat.reshape(-1, 1)))
        
        # Create a DataFrame from the combined array
        #df_flat_t = pd.DataFrame(flattened_meshpts)
        mesh_list.append(flattened_meshpts)
        #df_flat_t.columns = column_names
        #df_mesh_pts_flat = pd.concat(df, pd.DataFrame([df_flat_t]), ignore_index=True)

    #df_mesh_pts_flat = pd.DataFrame(mesh_list,columns=column_names)
    df_mesh_pts_flat = pd.concat((pd.DataFrame(mesh_list[0]),pd.DataFrame(mesh_list[1]),pd.DataFrame(mesh_list[2])))  
    df_mesh_pts_flat.columns = column_names  
    df_mesh_pts_flat.to_csv(os.path.join(output_file_path, output_file_name + '.csv'), index=False)
    return df_mesh_pts_flat
    #return mesh_list

#df_test = gridspace_points('./','gridspace_new', ['Time_min','Pressure_MPA', 'Temperature_C'],[0, 8.27],[70, 150],[5,10,15],)

# %% import data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
print("import data")
df = pd.read_csv('Final_GPs/AT_final.csv')  
# Uncomment the line below to get a fresh split!
#df = df.sample(frac=1)
x = df.iloc[:,1:4]
y = df.loc[:,"Percent area Transfer (%)"]
noises = df.loc[:,"Stdev_eff"]
nData = len(x)
x = np.array(x); y=np.array(y); noises = np.array(noises)
# creating upper/lower confidence intervals
quantile = 2
y_upper = y+(quantile/2)*noises; y_lower = y-(quantile/2)*noises

# defining priors
prior_bounds = np.array([1, 99])

#%% Leave-One-Out (LOO) Test Metric Loop
final_nlpd = torch.zeros(nData)
final_msll = torch.zeros(nData)
final_mse = torch.zeros(nData)
final_mae = torch.zeros(nData)
loo_means = np.zeros((nData,))
loo_stdevs = np.zeros((nData,))
loo_uppers = np.zeros((nData,))
loo_lowers = np.zeros((nData,))
loo_lowers_noninverse = np.zeros((nData,))

for i in range(nData): # i indicates the data index to leave out as a test dataset
    selector = [index for index in range(nData) if index != i] # all but index i

    # selecting LOO datasets
    x_train = x[selector,:]
    y_train = y[selector]; y_upper_train = y_upper[selector]; y_lower_train = y_lower[selector]
    noises_train = noises[selector]

    x_test = x[i,:]
    y_test = y[i]; y_upper_test = y_upper[i]; y_lower_test = y_lower[i]
    noises_test = noises[i]

    # converting these objects from np arrays to torch tensors in the cringiest way possible
    noises_test = torch.tensor(np.array(noises_test)).reshape(-1,1)
    x_test = torch.tensor(np.array(x_test)).reshape(1,-1)
    y_test = torch.tensor(np.array(y_test)).reshape(-1,1)
    y_lower_test = torch.tensor(np.array(y_lower_test)).reshape(-1,1)
    y_upper_test = torch.tensor(np.array(y_upper_test)).reshape(-1,1)

    # scale x data per scikit standard scaler to N(0,1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = torch.tensor(scaler.transform(x_train)).double()
    x_test = torch.tensor(scaler.transform(x_test)).double()

    # scale y data to U(0,1)
    y_scaler = preprocessing.QuantileTransformer(n_quantiles=nData-1)
    # y_scaler.fit(y_scaler_norm.transform(y_train.reshape(-1,1))) #y_scaler.fit(y_train.reshape(-1,1))
    # y_train = y_scaler.transform(y_scaler_norm.transform(y_train.reshape(-1,1))) * (1-1e-8) + 1e-16
    # y_lower_train = y_scaler.transform(y_scaler_norm.transform(y_lower_train.reshape(-1,1))) * (1-1e-8) + 1e-16
    # y_upper_train = y_scaler.transform(y_scaler_norm.transform(y_upper_train.reshape(-1,1))) * (1-1e-8) + 1e-16
    # y_test = y_scaler.transform(y_scaler_norm.transform(y_test.reshape(-1,1))) * (1-1e-8) + 1e-16
    # y_lower_test = y_scaler.transform(y_scaler_norm.transform(y_lower_test.reshape(-1,1))) * (1-1e-8) + 1e-16
    # y_upper_test = y_scaler.transform(y_scaler_norm.transform(y_upper_test.reshape(-1,1))) * (1-1e-8) + 1e-16
    # y_bounds = y_scaler.transform(y_scaler_norm.transform(prior_bounds.reshape(-1,1))) * (1-1e-8) + 1e-16

    y_scaler.fit(y_train.reshape(-1,1))
    y_train = y_scaler.transform(y_train.reshape(-1,1)) * (1-1e-8) + 1e-16
    y_lower_train = y_scaler.transform(y_lower_train.reshape(-1,1)) * (1-1e-8) + 1e-16
    y_upper_train = y_scaler.transform(y_upper_train.reshape(-1,1)) * (1-1e-8) + 1e-16
    y_test = y_scaler.transform(y_test.reshape(-1,1)) * (1-1e-8) + 1e-16
    y_lower_test = y_scaler.transform(y_lower_test.reshape(-1,1)) * (1-1e-8) + 1e-16
    y_upper_test = y_scaler.transform(y_upper_test.reshape(-1,1)) * (1-1e-8) + 1e-16
    y_bounds = y_scaler.transform(prior_bounds.reshape(-1,1)) * (1-1e-8) + 1e-16

    # apply Gaussian inverse CDF (PPF in scipy lingo) to unscaled data
    # y_train = norm.ppf(y_train)
    # y_lower_train = norm.ppf(y_lower_train)
    # y_upper_train = norm.ppf(y_upper_train)
    # y_test = norm.ppf(y_test)
    # y_lower_test = norm.ppf(y_lower_test)
    # y_upper_test = norm.ppf(y_upper_test)
    # y_bounds = norm.ppf(y_bounds)

    y_train = torch.tensor(y_train).squeeze().double()
    y_lower_train = torch.tensor(y_lower_train).squeeze().double()
    y_upper_train = torch.tensor(y_upper_train).squeeze().double()
    y_test = torch.tensor(y_test).double()
    y_lower_test = torch.tensor(y_lower_test).double()
    y_upper_test = torch.tensor(y_upper_test).double()
    y_bounds = torch.tensor(y_bounds).double()

    # recalculate upper/lower CIs to find noises in new scaled space
    noises_train = ((y_upper_train - y_lower_train)/quantile)
    noises_test = ((y_upper_test - y_lower_test)/quantile)
    y_prior_std = ((y_bounds[1]-y_bounds[0])/4)**2
    nTrain = y_train.size(dim=0)

    # defining model
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.LinearMean(input_size=3)
            self.covar_module = gpytorch.kernels.ScaleKernel(\
                gpytorch.kernels.MaternKernel(ard_num_dims=3))#, \
                #outputscale_prior = gpytorch.priors.NormalPrior(y_prior_std,y_prior_std/4))
            #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))
            self.covar_module.outputscale=y_prior_std

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises_train**2,learn_additional_noise=False)
    model = ExactGPModel(x_train, y_train, likelihood)
    model.double()
    likelihood.double()

    # ## Step Three: tuning the kernel (hyper) paramaters

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.25)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter=100

    print("Starting GP parameter tuning... for LOO index i="+str(i))
    for ii in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        #if i==0:
            #if np.mod(ii,50)==0:
                #print('Iter %d/%d - Loss: %.3f   lengthscale 1: %.3f   lengthscale 2: %.3f   lengthscale 3: %.3f' % (
                #    ii + 1, training_iter, loss.item(),
                #    model.covar_module.base_kernel.lengthscale[0][0].item(),
                #    model.covar_module.base_kernel.lengthscale[0][1].item(),
                #    model.covar_module.base_kernel.lengthscale[0][1].item(),
                #))  
            #print('Iter %d/%d - Loss: %.3f %' (
            #    i + 1, training_iter, loss.item()
                #model.covar_module.base_kernel.lengthscale,
                #model.covar_module.base_kernel.lengthscale[1],
                #model.covar_module.base_kernel.lengthscale[2]
            #))
            #torch.print("lengthscale: "+str(model.covar_module.base_kernel.lengthscale))
        optimizer.step()
        model.covar_module.outputscale = y_prior_std

    # % [markdown]
    # ## Step Four: Testing our model
    # 
    model.eval()
    with torch.no_grad():
        trained_pred_dist = likelihood(model(x_test),noise=noises_test**2)#model(x_test)
        predictive_mean = trained_pred_dist.mean
        lower, upper = trained_pred_dist.confidence_region()


    #predictive_mean_inverse = y_scaler_norm.inverse_transform(y_scaler.inverse_transform(predictive_mean.reshape(-1,1)))
    #lower_inverse = y_scaler_norm.inverse_transform(y_scaler.inverse_transform(lower.reshape(-1,1)))
    #upper_inverse = y_scaler_norm.inverse_transform(y_scaler.inverse_transform(upper.reshape(-1,1)))

    # scaling back input/output data by first applying CDF 
    # y_test_inverse = norm.cdf(y_test)
    # lower_inverse = norm.cdf(lower)
    # upper_inverse = norm.cdf(upper)
    # predictive_mean_inverse = norm.cdf(predictive_mean)

    # then scaling back by applying inverse tranform (from scikit standard scaler)
    y_test_inverse = y_scaler.inverse_transform(y_test)
    x_test_inverse = scaler.inverse_transform(x_test)
    predictive_mean_inverse = y_scaler.inverse_transform(predictive_mean.reshape(-1,1))
    lower_inverse = y_scaler.inverse_transform(lower.reshape(-1,1))
    upper_inverse = y_scaler.inverse_transform(upper.reshape(-1,1))
    loo_means[i] = predictive_mean_inverse[0][0]
    loo_uppers[i] = upper_inverse[0][0]
    loo_lowers[i] = lower_inverse[0][0]
    loo_stdevs[i] = (upper_inverse[0][0]-lower_inverse[0][0])/4
    loo_lowers_noninverse[i] = lower[0]

    # calculating LOO test metrics
    final_nlpd[i] = gpytorch.metrics.negative_log_predictive_density(trained_pred_dist, y_test)
    final_msll[i] = gpytorch.metrics.mean_standardized_log_loss(trained_pred_dist, y_test)
    final_mse[i] = gpytorch.metrics.mean_squared_error(trained_pred_dist, y_test, squared=True)
    final_mae[i] = gpytorch.metrics.mean_absolute_error(trained_pred_dist, y_test)

    # # delete between this ****
    # model.eval()
    # with torch.no_grad():
    #     trained_pred_dist = likelihood(model(x_train))
    #     predictive_mean = trained_pred_dist.mean
    #     lower, upper = trained_pred_dist.confidence_region()
    # #(predictive_mean)

    # # scaling back input/output data by first applying CDF 
    # y_test_inverse = norm.cdf(y_test)
    # lower_inverse = norm.cdf(lower)
    # upper_inverse = norm.cdf(upper)
    # predictive_mean_inverse = norm.cdf(predictive_mean)

    # # then scaling back by applying inverse tranform (from scikit standard scaler)
    # y_test_inverse = y_scaler.inverse_transform(y_test_inverse)
    # x_test_inverse = scaler.inverse_transform(x_test)
    # predictive_mean_inverse = y_scaler.inverse_transform(predictive_mean_inverse.reshape(-1,1))
    # lower_inverse = y_scaler.inverse_transform(lower_inverse.reshape(-1,1))
    # upper_inverse = y_scaler.inverse_transform(upper_inverse.reshape(-1,1))
    # yerrs = np.hstack(((predictive_mean_inverse-lower_inverse),(upper_inverse-predictive_mean_inverse))).transpose()

    # fig, ax = plt.subplots(dpi=1000)
    # #plt.errorbar(y, loo_means,yerr=(loo_uppers-loo_lowers)/2,fmt="_",color='orange',elinewidth=0.75,zorder=1,label='$+/-\sigma$ Confidence Interval')
    # plt.errorbar(y[selector], predictive_mean_inverse.flatten(),yerr=yerrs,fmt="_",color='orange',elinewidth=0.75,zorder=1,label='$+/-2\sigma$ Confidence Interval')
    # plt.scatter(y[selector], predictive_mean_inverse, c='crimson',zorder=2,label='Mean Predictions')
    # p1 = max(max(upper_inverse)[0], max(y_train.numpy()))
    # p2 = min(min(lower_inverse)[0], min(y_train.numpy()))
    # plt.plot([p1, p2], [p1, p2], 'b-')
    # plt.xlabel('True Values [%]', fontsize=15)
    # plt.ylabel('LOO Predictions [%]', fontsize=15)
    # plt.axis('equal')
    # plt.legend()
    # plt.title('Training Data using LOO, Percent Area Transfer vs. Experimental Data')

    # # and this ****

print("Median NLPD: "+str(torch.median(final_nlpd)))
print("Median MSE: "+str(torch.median(final_mse)))
print("Correlation coefficient: "+str(np.corrcoef(np.array(y),loo_means)[0,1]))

# %% plotting LOO predictions vs training set data

yerrs = np.vstack(((loo_means-loo_lowers),(loo_uppers-loo_means)))

fig, ax = plt.subplots(dpi=1000)
#plt.errorbar(y, loo_means,yerr=(loo_uppers-loo_lowers)/2,fmt="_",color='orange',elinewidth=0.75,zorder=1,label='$+/-\sigma$ Confidence Interval')
plt.errorbar(y, loo_means,yerr=yerrs,fmt="_",color='orange',elinewidth=0.75,zorder=1,label='$+/-2\sigma$ Confidence Interval')
plt.scatter(y, loo_means, c='crimson',zorder=2,label='Mean Predictions')
p1 = max(max(loo_uppers), max(y))
p2 = min(min(loo_lowers), min(y))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values [%]', fontsize=15)
plt.ylabel('LOO Predictions [%]', fontsize=15)
plt.axis('equal')
plt.legend()
plt.title('LOO Predictions for Percent Area Transfer vs. Experimental Data')

#%% Now creating final model using full dataset
x_train = x[:,:]
y_train = y[:]; y_upper_train = y_upper[:]; y_lower_train = y_lower[:]
noises_train = noises[:]
nTrain = np.shape(y_train)[0]

# % pre process data

# scale x data per standard scaler to N(0,1)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = torch.tensor(scaler.transform(x_train)).double()

# scale y data to U(0,1)
y_scaler = preprocessing.QuantileTransformer(n_quantiles=nData-1).fit(y_train.reshape(-1,1))
y_train = y_scaler.transform(y_train.reshape(-1,1)) * (1-1e-8)  + 1e-16
y_lower_train = y_scaler.transform(y_lower_train.reshape(-1,1)) * (1-1e-8)  + 1e-16
y_upper_train = y_scaler.transform(y_upper_train.reshape(-1,1)) * (1-1e-8)  + 1e-16
y_bounds = y_scaler.transform(prior_bounds.reshape(-1,1)) * (1-1e-8)  + 1e-16

# apply inverse CDF to y data
# y_train = norm.ppf(y_train)
# y_lower_train = norm.ppf(y_lower_train)
# y_upper_train = norm.ppf(y_upper_train)
# y_bounds = norm.ppf(y_bounds)

# converting data arrays into torch tensors
y_train = torch.tensor(y_train).squeeze().double()
y_lower_train = torch.tensor(y_lower_train).squeeze().double()
y_upper_train = torch.tensor(y_upper_train).squeeze().double()
noises_train = torch.tensor(noises_train).squeeze().double()
y_bounds = torch.tensor(y_bounds).double()

noises_train = (y_upper_train - y_lower_train)/2
y_prior_std = ((y_bounds[1]-y_bounds[0])/4)**2

# defining model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(input_size=3)
        self.covar_module = gpytorch.kernels.ScaleKernel(\
                gpytorch.kernels.MaternKernel(ard_num_dims=3))
        self.covar_module.outputscale=y_prior_std

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
fullLikelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises_train**2,learn_additional_noise=False)
fullModel = ExactGPModel(x_train, y_train, fullLikelihood)
fullModel.double()
fullLikelihood.double()

# ## Step Three: tuning the kernel (hyper) parameters

fullModel.train()
fullLikelihood.train()

# Use the adam optimizer
#optimizer = torch.optim.Adam(fullModel.parameters(), lr=0.25)  # Includes GaussianLikelihood parameters
optimizer = torch.optim.Adam(list(fullModel.parameters()), lr=0.25)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(fullLikelihood, fullModel)

training_iter=100

print('Starting GP parameter tuning for full dataset...')
for ii in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = fullModel(x_train)
    # Calc loss and backprop gradients
    loss = -mll(output, y_train)
    loss.backward()
    if np.mod(ii,50)==0:
        print('Iter %d/%d - Loss: %.3f   lengthscale 1: %.3f   lengthscale 2: %.3f   lengthscale 3: %.3f   outputscale: %.3f  ' % (
            ii + 1, training_iter, loss.item(),
            fullModel.covar_module.base_kernel.lengthscale[0][0].item(),
            fullModel.covar_module.base_kernel.lengthscale[0][1].item(),
            fullModel.covar_module.base_kernel.lengthscale[0][1].item(),
            fullModel.covar_module.outputscale.item()
        ))
        #print('Iter %d/%d - Loss: %.3f %' (
        #    i + 1, training_iter, loss.item()
            #model.covar_module.base_kernel.lengthscale,
            #model.covar_module.base_kernel.lengthscale[1],
            #model.covar_module.base_kernel.lengthscale[2]
        #))
        #torch.print("lengthscale: "+str(model.covar_module.base_kernel.lengthscale))
    optimizer.step()
    fullModel.covar_module.outputscale = y_prior_std

#%%
fullModel.eval()
with torch.no_grad():
    trained_pred_dist = fullLikelihood(fullModel(x_train),noise=noises_train**2)
    #trained_pred_dist = fullModel(x_train)
    predictive_mean = trained_pred_dist.mean
    lower, upper = trained_pred_dist.confidence_region()

# scaling back y data by first applying CDF 
# lower_inverse = norm.cdf(lower)
# upper_inverse = norm.cdf(upper)
# predictive_mean_inverse = norm.cdf(predictive_mean)

# scaling back input/output data
predictive_mean_inverse = y_scaler.inverse_transform(predictive_mean.reshape(-1,1))
lower_inverse = y_scaler.inverse_transform(lower.reshape(-1,1))
upper_inverse = y_scaler.inverse_transform(upper.reshape(-1,1))

print("Correlation coefficient: "+str(np.corrcoef(y,predictive_mean_inverse.squeeze())[0,1]))

# %% plotting predictions vs training set data

fig, ax = plt.subplots(dpi=1000)
yerrs = np.vstack((predictive_mean.numpy()-lower.numpy(),upper.numpy()-predictive_mean.numpy()))
plt.errorbar(y_train, predictive_mean,yerr=yerrs,fmt="_",color='orange',elinewidth=0.75,zorder=1)
plt.scatter(y_train, predictive_mean, c='crimson',zorder=2,label='Mean Predictions')
p1 = max(max(predictive_mean).item(), max(y_train))
p2 = min(min(predictive_mean).item(), min(y_train))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.legend()
plt.title('Normalized Predictions vs. Training Data')

fig1, ax1 = plt.subplots(dpi=1000)
yerrs_inverse = np.hstack((predictive_mean_inverse-lower_inverse,upper_inverse-predictive_mean_inverse)).transpose()
ax1.errorbar(y, predictive_mean_inverse.squeeze(),yerr=yerrs_inverse,fmt="_",color='orange',elinewidth=0.75,zorder=1,label='$+/-2\sigma$ Confidence Interval')
ax1.scatter(y, predictive_mean_inverse, c='crimson',zorder=2,label='Mean Predictions')
#plt.scatter(y, lower_inverse, c='green')
#plt.scatter(y, upper_inverse, c='purple')
p1 = max(max(predictive_mean_inverse)[0], max(y))
p2 = min(min(predictive_mean_inverse)[0], min(y))
ax1.plot([p1, p2], [p1, p2], 'b-')
ax1.set_xlabel('True Values', fontsize=15)
ax1.set_ylabel('Predictions', fontsize=15)
plt.axis('equal')
ax1.legend()
ax1.set_title('Real-Valued Predictions vs. Training Data')

#%% plotting heatmaps
x_grids = pd.read_csv('gridspace_new.csv')
x_pred = torch.tensor(np.array(x_grids))

x_pred = torch.tensor(scaler.transform(x_pred))

fullModel.eval()
with torch.no_grad():
    trained_pred_dist = fullLikelihood(fullModel(x_pred))
    predictive_mean = trained_pred_dist.mean
    lower, upper = trained_pred_dist.confidence_region()

# scaling back x data
x_pred_inverse = scaler.inverse_transform(x_pred)

# scaling back y data
# lower_inverse = norm.cdf(lower)
# upper_inverse = norm.cdf(upper)
# predictive_mean_inverse = norm.cdf(predictive_mean)
predictive_mean_inverse = np.array(y_scaler.inverse_transform(predictive_mean.reshape(-1,1)))
upper_inverse = np.array(y_scaler.inverse_transform(upper.reshape(-1,1)))
lower_inverse = np.array(y_scaler.inverse_transform(lower.reshape(-1,1)))
predictive_stdev = (upper_inverse-lower_inverse)/4

mean_pred_array = predictive_mean_inverse
mean_pred_results = np.concatenate((x_pred_inverse, predictive_mean_inverse,predictive_stdev), axis=1)
df_results = pd.DataFrame(mean_pred_results, columns = ['Time (sec)', 'Pressure (MPa)','Temp (C)', 'Mean', 'StDev'])
df_results.to_csv('at_GP_heatmap_preds.csv', index=False)

levelsi=np.arange(100)
for test_time in [5,10,15]:
    li = int((test_time/5-1)*2500)
    ui = int(li+2500)
    
    # mean plots
    fig, ax = plt.subplots(dpi=1000)
    x=np.array(x)
    y=np.array(y)
    trueYs = y[np.where(x[:,0]==test_time)].reshape(-1,1)
    trueXs = x[np.where(x[:,0]==test_time)][:,1:3]
    normi = colors.Normalize(vmin=0,vmax=100)
    tcf = ax.tricontourf(x_pred_inverse[li:ui,1],x_pred_inverse[li:ui,2],mean_pred_array[li:ui,:].squeeze(),norm=normi,levels=levelsi)
    ax.scatter(trueXs[:,0], trueXs[:,1],c=trueYs.squeeze(),marker='*',norm=normi)#,marker='*',label='Training Data')
    plt.colorbar(tcf, ax=ax)
    ax.set_title("Model Mean Predictions at Time = "+str(test_time)+" min")
    
    # lower CI plots
    fig1, ax1 = plt.subplots(dpi=1000)
    mi = np.min((min(lower_inverse[li:ui,:]), min(trueYs)))
    ma = np.max((max(lower_inverse[li:ui,:]), max(trueYs)))
    normi = colors.Normalize(vmin=0,vmax=100)
    tcf = ax1.tricontourf(x_pred_inverse[li:ui,1],x_pred_inverse[li:ui,2],lower_inverse[li:ui,:].squeeze(),norm=normi,levels=levelsi)
    ax1.scatter(trueXs[:,0], trueXs[:,1],c=trueYs.squeeze(),marker='*',norm=normi)#,marker='*',label='Training Data')
    fig1.colorbar(tcf, ax=ax1)
    ax1.set_title("Lower CI Predictions at Time = "+str(test_time)+" min")
    
    # upper CI plots
    fig2, ax2 = plt.subplots(dpi=1000)
    mi = np.min(((min(upper_inverse[li:ui,:]), min(trueYs))))
    ma = np.max(((max(upper_inverse[li:ui,:]), max(trueYs))))
    normi = colors.Normalize(vmin=0,vmax=100)
    tcf = ax2.tricontourf(x_pred_inverse[li:ui,1],x_pred_inverse[li:ui,2],upper_inverse[li:ui,:].squeeze(),norm=normi,levels=levelsi)
    ax2.scatter(trueXs[:,0], trueXs[:,1],c=trueYs.squeeze(),marker='*',norm=normi)#,marker='*',label='Training Data')
    fig2.colorbar(tcf, ax=ax2)
    ax2.set_title("Upper CI Predictions at Time = "+str(test_time)+" min")

    # St Dev plots
    fig3, ax3 = plt.subplots(dpi=1000)
    x=np.array(x)
    y=np.array(y)
    noises=np.array(noises)
    trueNoises = noises[np.where(x[:,0]==test_time)].reshape(-1,1)/(2*torch.std(y_train))
    trueXs = (x[np.where(x[:,0]==test_time)])[:,1:3]
    #mi = np.min(((min(predictive_stdev[li:ui,:])[0], min(trueNoises))))
    #ma = np.max(((max(predictive_stdev[li:ui,:])[0], max(trueNoises))))
    #normi = matplotlib.colors.Normalize(vmin=mi,vmax=ma)
    tcf = plt.tricontourf(x_pred_inverse[li:ui,1],x_pred_inverse[li:ui,2],predictive_stdev[li:ui].squeeze(),levels=100)#,norm=norm)
    #plt.scatter(trueXs[:,0], trueXs[:,1],c=trueNoises.squeeze(),marker='*',norm=norm)#,label='Input StDevs')
    fig3.colorbar(tcf, ax=ax3)
    plt.title("Model StDev Predictions at Time = "+str(test_time)+" min")


# %%

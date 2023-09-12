# % [markdown]
# # Gaussian Process Tutorial for Perovskite Characterization
# Thomas Coons 2023
# 
# In this tutorial, we'll walk through the steps I took to train the GP for interfacial toughness, then we will also go through how to use the model to make predictions. My goal is to help you build some intuition about the model, understand the specific steps I took (which may not necessarily be the best - your feedback is helpful since you understand the application better than I do!), and to be able to use my code to generate plots and results of your choosing.
# 
# First, we import some helpful packages, the most important of which is GPyTorch, a flexible GP code base that is built on BOTorch.

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
    df_mesh_pts_flat = pd.concat((pd.DataFrame(mesh_list[0]),pd.DataFrame(mesh_list[0]),pd.DataFrame(mesh_list[0])))  
    df_mesh_pts_flat.columns = column_names  
    df_mesh_pts_flat.to_csv(os.path.join(output_file_path, output_file_name + '.csv'), index=False)
    return df_mesh_pts_flat
    #return mesh_list

df_test = gridspace_points('./','gridspace_new', ['Pressure_MPA', 'Temperature_C', 'Time_min'],[0, 8.27],[70, 150],[5,10,15],)


# %%
def eval_gp(model,x_test,likelihood,x_scaler=[],inputScaled=True,y_scaler=[],outputScaled=False):
# function to help encapsulate the prediction process
# inputs: model (of type gpytorch.models.ExactGPModel)
#         x_test (input vector to test)
#         inputScaled [default: False] (Bool True if inputs are already scaled)
#         outputScaled [default: False] (Bool True if desired outputs should be scaled)
#         y_scaler, x_scaler (sklearn.preprocessing.StandardScaler(), required if in/outputs are to be scaled)
    if inputScaled==False:
        x_test=torch.tensor(x_scaler.transform(x_test)).double()

    model.eval()
    with torch.no_grad():
        trained_pred_dist = likelihood(model(x_test))
        predictive_mean = trained_pred_dist.mean
        lower, upper = trained_pred_dist.confidence_region()
        predictive_stdev = (upper-lower)/4
    
    if outputScaled==True:
        predictive_mean = torch.exp(y_scaler.transform(predictive_mean))
        lower = torch.exp(y_scaler.transform(lower))
        upper = torch.exp(y_scaler.transform(upper))
        predictive_stdev = (upper-lower)/4

    
    return predictive_mean, predictive_stdev, lower, upper

# % [markdown]
# # Step One: Importing and Formatting Data
# 
# There are several steps that are usually taken when fitting a data-driven model: scaling the data, and dividing the test/train/validation split.
# 
# ## Why apply a logarithm to the output data?
# 
# In this whole process, we are assuming that the true interfacial toughness $y$ can be represented by a true underlying mapping $G(x)$ from the input space $x \in \mathbb{R}^{3}$ (the temperature, pressure, and time inputs) to the output space $y \in \mathbb{R}$, with some additive uncertainty $\epsilon$. Symbolically, we are assuming:
# \begin{equation}
# y=G(x)+\epsilon.
# \end{equation}
# 
# Ideally, we would model the underlying truth $y$ via a surrogate, say $\hat{G}(x)$. However, since we plan to use a Gaussian Process to predict outputs $y \geq 0$, whose predictions will be Gaussian (and Gaussian distributions have no strict upper/lower bounds, i.e. they may have samples of $y$ that are negative), we are actually interested in finding a $\hat{G}: x \rightarrow \ln{(y)}$ since a Gaussian distribution on $\ln{(y)}$ produces strictly positive predictions when exponentiated. Therefore, we seek:
# \begin{equation}
# \ln{(y)}=\hat{G}(x) + \eta
# \end{equation}
# where the measurements of $\ln{(y)}$ are themselves Gaussian random variables (and therefore $y$ is assumed to be a log-normal random variable).

# %% import data
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
print("import data")
df = pd.read_csv('20230628_Interfacial-Toughness.csv')  
# Uncomment the line below to get a fresh split!
#df = df.sample(frac=1)
x = df.iloc[:,1:4]
y = df.loc[:,"Toughness (J/m^2)"]
noises = df.loc[:,"Stdev (J/m^2)"]
nData = len(x)
x = np.array(x); y=np.array(y); noises = np.array(noises)

#%% Leave-One-Out (LOO) Test Metric Loop
final_nlpd = torch.zeros(nData)
final_msll = torch.zeros(nData)
final_mse = torch.zeros(nData)
final_mae = torch.zeros(nData)
loo_means = np.zeros((nData,))
loo_stdevs = np.zeros((nData,))

for i in range(nData):#range(nData): # i indicates the data index to leave out as a test dataset
    selector = [index for index in range(nData) if index != i] # all but index i

    x_train = x[selector,:]
    y_train = y[selector]
    noises_train = noises[selector]

    x_test = x[i,:]
    y_test = y[i]
    noises_test = noises[i]

    # converting these objects from pandas dataframe to tensor in the cringiest way possible
    x_train = torch.tensor(np.array(x_train))
    y_train = torch.tensor(np.array(y_train))
    noises_train = torch.tensor(np.array(noises_train))
    noises_test = torch.tensor(np.array(noises_test)).reshape(-1,1)
    x_test = torch.tensor(np.array(x_test)).reshape(1,-1)
    y_test = torch.tensor(np.array(y_test)).reshape(-1,1)

    # % [markdown]
    # ## Why scale the data?
    # 
    # For almost all data-driven models, normalizing the input and output data can dramatically improve model performance. One intuitive way to think about this is to consider the case where one input or output ranges from say 0 to 1000 while another ranges from 0 to 1. When the model is updated and optimized according to say MSE, will it take the two outputs into equal consideration? Probably not, because the MSE of the output vector is dominated by the larger component, and the model may not consider all of the outputs equally during training.
    # 
    # Here, we center and scale the inputs and outputs so that they roughly match a standard normal distribution, $N(0,1)$. This is a best practice for GPs.
    # 
    # In scikit-learn, there is a StandardScaler tool that does exactly this. To transform "physical" quantities of interfacial toughness to the scaled output, use $\mathtt{y\_scaler.transform(y\_unscaled)}$, and to do the inverse transformation to get scaled output from the GP into the units J/m^2, use $\mathtt{y\_scaler.inverse\_transform(y\_scaled)}$.
    # 
    # Also note that the $\mathtt{noises}$ (I will explain how they are used more later on) need to also be normalized by dividing by the standard deviation of the output data. We do not use the standard scaler here because we do not want these values to be centered, just scaled the same way we scaled the outputs $y$. These noises represent the uncertainty (standard deviations) of each observation, uncertainty information that the GP can be trained on rather than taking the data to be exact.

    # %
    # % pre process data

    # scale x data per standard scaler to N(0,1)
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = torch.tensor(scaler.transform(x_train)).double()
    x_test = torch.tensor(scaler.transform(x_test)).double()

    # need to scale noises only by dividing by stdev of (scaled and log'd) inputs
    noises_train = noises_train/torch.std(y_train)
    noises_train = noises_train/2 # divide by sqrt(n) (this may be wrong if n_samples =/= 4!!!!)
    noises_train = noises_train.double()

    # scale y data to N(0,1)
    y_test = torch.log(y_test).double()
    y_train = torch.log(y_train).double()
    y_scaler = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
    y_train = torch.tensor(y_scaler.transform(y_train.reshape(-1,1))).squeeze().double()
    y_test = torch.tensor(y_scaler.transform(y_test)).double()
    nTrain = y_train.size(dim=0)

    # checking dimensions of test/training set
    #print(noises_train.dtype)
    #print(x_train.size())
    #print(y_train.size())
    #print(x_test.size())
    #print(y_test.size())
    #print(noises_train.size())


    #% [markdown]
    # ## Step Two: Defining our GP Model
    # 
    # A Gaussian Process model embeds the assumption that any finite number of random variables produced by the model are jointly Gaussian. In other words, any outputs predicted by the GP are related by a multivariate Gaussian distribution, which can either be sampled from (producing scalar outputs) or taken as random variables (producing RV outputs). To make these predictions, the GP is defined by a mean function $m(x)$ and a kernel or covariance function $k(x,x')$:
    # \begin{equation}
    #     f(x) \sim GP\left( m(x), k(x,x') \right),
    # \end{equation}
    # with outputs associated with an input $x_{i} \in \mathbb{R}^{3}$:
    # \begin{equation}
    #     y_{i} = N\left(m(x_{i}), k(x_{i},x') \right).
    # \end{equation}
    # 
    # In our case, the mean function $m(x)$ is a linear 

    #%
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.LinearMean(input_size=3)
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=3))
           # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises_train,learn_additional_noise=True)
    #likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #print(x_train.dtype)
    model = ExactGPModel(x_train, y_train, likelihood)
    model.double()
    likelihood.double()

    # % [markdown]
    # ## Step Three: tuning the kernel (hyper) paramaters

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.25)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter=300

    print("Starting GP parameter tuning... for LOO index i="+str(i))
    for ii in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
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

    # % [markdown]
    # ## Step Four: Testing our model
    # 
    model.eval()
    with torch.no_grad():
        trained_pred_dist = likelihood(model(x_test))
        predictive_mean = trained_pred_dist.mean
        lower, upper = trained_pred_dist.confidence_region()
    #(predictive_mean)

    # scaling back input/output data
    y_test_inverse = y_scaler.inverse_transform(y_test)
    x_test_inverse = scaler.inverse_transform(x_test)
    predictive_mean_inverse = y_scaler.inverse_transform(predictive_mean.reshape(-1,1))
    lower_inverse = y_scaler.inverse_transform(lower.reshape(-1,1))
    upper_inverse = y_scaler.inverse_transform(upper.reshape(-1,1))
    loo_means[i] = predictive_mean_inverse[0][0]
    loo_stdevs[i] = (upper_inverse[0][0]-lower_inverse[0][0])/4


    # calculating LOO test metrics
    final_nlpd[i] = gpytorch.metrics.negative_log_predictive_density(trained_pred_dist, y_test)
    final_msll[i] = gpytorch.metrics.mean_standardized_log_loss(trained_pred_dist, y_test)
    final_mse[i] = gpytorch.metrics.mean_squared_error(trained_pred_dist, y_test, squared=True)
    final_mae[i] = gpytorch.metrics.mean_absolute_error(trained_pred_dist, y_test)

print("Median NLPD: "+str(torch.median(final_nlpd)))
print("Median MSE: "+str(torch.median(final_mse)))
print("Correlation coefficient: "+str(np.corrcoef(np.array(y),loo_means)[0,1]))

#%% Now creating final model using full dataset
x_train = x[:,:]
y_train = y[:]
noises_train = noises[:]

# converting these objects from pandas dataframe to tensor in the cringiest way possible
x_train = torch.tensor(np.array(x_train))
y_train = torch.tensor(np.array(y_train))
noises_train = torch.tensor(np.array(noises_train))

# % pre process data

# scale x data per standard scaler to N(0,1)
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = torch.tensor(scaler.transform(x_train)).double()

# need to scale noises only by dividing by stdev of (scaled and log'd) inputs
noises_train = noises_train/torch.std(y_train)
noises_train = noises_train/2 # divide by sqrt(n) (this may be wrong if n_samples =/= 4!!!!)
noises_train = noises_train.double()

# scale y data to N(0,1)
y_train = torch.log(y_train).double()
y_scaler = preprocessing.StandardScaler().fit(y_train.reshape(-1,1))
y_train = torch.tensor(y_scaler.transform(y_train.reshape(-1,1))).squeeze().double()
nTrain = y_train.size(dim=0)
#%
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(input_size=3)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=3))
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noises_train,learn_additional_noise=True)
#likelihood = gpytorch.likelihoods.GaussianLikelihood()
print(x_train.dtype)
model = ExactGPModel(x_train, y_train, likelihood)
model.double()
likelihood.double()

# need to find way to set prior variance (constant in front of RBF kernel)

# % [markdown]
# ## Step Three: tuning the kernel (hyper) parameters

model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.25)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter=300

print('Starting GP parameter tuning for full dataset...')
for ii in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(x_train)
    # Calc loss and backprop gradients
    loss = -mll(output, y_train)
    loss.backward()
    if np.mod(ii,50)==0:
        print('Iter %d/%d - Loss: %.3f   lengthscale 1: %.3f   lengthscale 2: %.3f   lengthscale 3: %.3f   outputscale: %.3f' % (
            ii + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale[0][0].item(),
            model.covar_module.base_kernel.lengthscale[0][1].item(),
            model.covar_module.base_kernel.lengthscale[0][1].item(),
            model.covar_module.outputscale.item()
        ))
        #print('Iter %d/%d - Loss: %.3f %' (
        #    i + 1, training_iter, loss.item()
            #model.covar_module.base_kernel.lengthscale,
            #model.covar_module.base_kernel.lengthscale[1],
            #model.covar_module.base_kernel.lengthscale[2]
        #))
        #torch.print("lengthscale: "+str(model.covar_module.base_kernel.lengthscale))
    optimizer.step()
#%%
model.eval()
with torch.no_grad():
    trained_pred_dist = likelihood(model(x_train))
    predictive_mean = trained_pred_dist.mean
    lower, upper = trained_pred_dist.confidence_region()

# scaling back input/output data
predictive_mean_inverse = y_scaler.inverse_transform(predictive_mean.reshape(-1,1))
lower_inverse = y_scaler.inverse_transform(lower.reshape(-1,1))
upper_inverse = y_scaler.inverse_transform(upper.reshape(-1,1))
predictive_mean_inverse = np.array(np.exp(y_scaler.inverse_transform(predictive_mean.reshape(-1,1))))
upper_inverse = np.array(np.exp(y_scaler.inverse_transform(upper.reshape(-1,1))))
lower_inverse = np.array(np.exp(y_scaler.inverse_transform(lower.reshape(-1,1))))

print("Correlation coefficient: "+str(np.corrcoef(y,predictive_mean_inverse.squeeze())[0,1]))

# %% plotting predictions vs training set data

fig, ax = plt.subplots(dpi=1000)
plt.errorbar(y_train, predictive_mean,yerr=(upper-lower)/2,fmt="_",color='orange',elinewidth=0.75,zorder=1)
plt.scatter(y_train, predictive_mean, c='crimson',zorder=2)
p1 = max(max(predictive_mean), max(y_train))
p2 = min(min(predictive_mean), min(y_train))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.title('Normalized Predictions vs. Training Data')

fig1, ax1 = plt.subplots(dpi=1000)
yerr = (upper_inverse-lower_inverse).squeeze()/2
ax1.errorbar(y, predictive_mean_inverse.squeeze(),yerr=yerr,fmt="_",color='orange',elinewidth=0.75,zorder=1)
ax1.scatter(y, predictive_mean_inverse, c='crimson',zorder=2)
p1 = max(max(predictive_mean_inverse), max(y))
p2 = min(min(predictive_mean_inverse), min(y))
ax1.plot([p1, p2], [p1, p2], 'b-')
ax1.set_xlabel('True Values', fontsize=15)
ax1.set_ylabel('Predictions', fontsize=15)
plt.axis('equal')
ax1.set_title('Real-Valued Predictions vs. Training Data')

#%% plotting heatmaps
x_grids = pd.read_csv('gridspace_new.csv')
x_pred = torch.tensor(np.array(x_grids))

x_pred = torch.tensor(scaler.transform(x_pred))

model.eval()
with torch.no_grad():
    trained_pred_dist = likelihood(model(x_pred))
    predictive_mean = trained_pred_dist.mean
    lower, upper = trained_pred_dist.confidence_region()
# scaling back input/output data
x_pred_inverse = scaler.inverse_transform(x_pred)
predictive_mean_inverse = np.array(np.exp(y_scaler.inverse_transform(predictive_mean.reshape(-1,1))))
upper_inverse = np.array(np.exp(y_scaler.inverse_transform(upper.reshape(-1,1))))
lower_inverse = np.array(np.exp(y_scaler.inverse_transform(lower.reshape(-1,1))))
predictive_stdev = (upper_inverse-lower_inverse)/4
print('x shape', x_pred_inverse.shape)
print('predicted mean shape: ', predictive_mean_inverse.shape)
mean_pred_array = np.array(predictive_mean_inverse)
mean_pred_results = np.concatenate((x_pred_inverse, mean_pred_array), axis=1)
df_results = pd.DataFrame(mean_pred_results)
df_results.to_csv('pred_mean_results_full.csv', index=False)

for test_time in [5,10,15]:
    li = int((test_time/5-1)*2500)
    ui = int(li+2500)
    
    # mean plots
    fig, ax = plt.subplots(dpi=1000)
    tcf = ax.tricontourf(x_pred_inverse[li:ui,1],x_pred_inverse[li:ui,2],mean_pred_array[li:ui,:].squeeze())
    x=np.array(x)
    y=np.array(y)
    trueYs = y[np.where(x[:,0]==test_time)].reshape(-1,1)
    trueXs = x[np.where(x[:,0]==test_time)][:,1:3]
    ax.scatter(trueXs[:,0], trueXs[:,1],c=trueYs.squeeze(),marker='*')#,marker='*',label='Training Data')
    plt.colorbar(tcf, ax=ax)
    ax.set_title("Model Mean Predictions at Time = "+str(test_time)+" min")
    
    # lower CI plots
    fig1, ax1 = plt.subplots(dpi=1000)
    tcf = ax1.tricontourf(x_pred_inverse[li:ui,1],x_pred_inverse[li:ui,2],lower_inverse[li:ui,:].squeeze())
    x=np.array(x)
    y=np.array(y)
    trueYs = y[np.where(x[:,0]==test_time)].reshape(-1,1)
    trueXs = x[np.where(x[:,0]==test_time)][:,1:3]
    ax1.scatter(trueXs[:,0], trueXs[:,1],c=trueYs.squeeze(),marker='*')#,marker='*',label='Training Data')
    plt.colorbar(tcf, ax=ax1)
    ax1.set_title("Lower CI Predictions at Time = "+str(test_time)+" min")
    
    # upper CI plots
    fig2, ax2 = plt.subplots(dpi=1000)
    tcf = ax2.tricontourf(x_pred_inverse[li:ui,1],x_pred_inverse[li:ui,2],upper_inverse[li:ui,:].squeeze())
    x=np.array(x)
    y=np.array(y)
    trueYs = y[np.where(x[:,0]==test_time)].reshape(-1,1)
    trueXs = x[np.where(x[:,0]==test_time)][:,1:3]
    ax2.scatter(trueXs[:,0], trueXs[:,1],c=trueYs.squeeze(),marker='*')#,marker='*',label='Training Data')
    plt.colorbar(tcf, ax=ax2)
    ax2.set_title("Upper CI Predictions at Time = "+str(test_time)+" min")

    # St Dev plots
    li = int((test_time/5-1)*2500)
    ui = int(li+2500)
    print(li)
    print(ui)
    plt.figure()
    plt.tricontourf(x_pred_inverse[li:ui,1],x_pred_inverse[li:ui,2],predictive_stdev[li:ui].squeeze())
    x=np.array(x)
    y=np.array(y)
    noises=np.array(noises)
    trueNoises = noises[np.where(x[:,0]==test_time)].reshape(-1,1)/(2*torch.std(y_train))
    trueXs = (x[np.where(x[:,0]==test_time)])[:,1:3]
    plt.scatter(trueXs[:,0], trueXs[:,1],c=trueNoises.squeeze(),marker='*')#,label='Input StDevs')
    plt.colorbar()
    plt.title("Model StDev Predictions at Time = "+str(test_time)+" min")


# %%

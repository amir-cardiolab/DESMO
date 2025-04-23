#!/usr/bin/env python
# coding: utf-8

# In[1]:


import vtk
import numpy as np
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
from vtk.numpy_interface import dataset_adapter as dsa
import time
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
from torchvision import datasets, transforms
# import pysr
from scipy.ndimage import uniform_filter1d
np.bool = np.bool_
from pysindy import SINDy
import scipy
import pyvista as pv
import math
import datetime
import itertools
import pyvista as pv
import math
import datetime
import itertools
import os


# In[2]:


##########################################################################
# Function definitions

def read_velocity_data(input_dir, filename, reader, t_1, t_n):
# Read velocity data from file
# Inputs:
# input_dir - input directory location
# filename - velocity timeseries filename 
# reader - vtk reader
# t_1 - first timestep to read
# t_n - last timestep to read
# Outputs:
# X - data matrix containing the velocity data
# mesh - mesh object containing the mesh

    print('Reading velocity data and mesh from:', input_dir + filename)

    velocity_list = []
    for i in range(t_1,t_n,1):
        reader.SetFileName(input_dir+filename+str(i)+'.vtu')
        reader.Update()
        output = reader.GetOutput()
        # f_18 is the name of the velocity vector dataset assigned by FEniCS for this case
        velocity_dataset = output.GetPointData().GetArray("velocity")
        velocity = VN.vtk_to_numpy(velocity_dataset)
        velocity_vec = np.reshape(velocity,(-1,1))
        velocity_list.append(velocity_vec)

    # arrange the velocity data into a big data matrix
    X = np.asarray(velocity_list)
    X = X.flatten('F')

    X = np.reshape(X,(-1,t_n-t_1))
    # rows of X correspond to velocity components at spatial locations
    # columns of X correspond to timesteps
    #     t_1 t_2.  .  t_end
    # X = [u  u  .  .  .]  (x_1,y_1)
    #     [v  v  .  .  .]  (x_1,y_1)
    #     [w  w  .  .  .]  (x_1,y_1)
    #     [u  u  .  .  .]  (x_2,y_2)
    #     [v  v  .  .  .]  (x_2,y_2) 
    #     [w  w  .  .  .]  (x_2,y_2)
    #     [.  .  .  .  .]   .
    #     [.  .  .  .  .]   .
    #     [.  .  .  .  .]   .

    # read the mesh for later visualization and saving data
    mesh = reader.GetOutput()

    return X, mesh


def convert3Dto2D_data(X):    
# If the problem is 2D, the w component of the velocity will be all zeros
# These can be deleted to have a smaller data matrix in size
# Input:
# X - velocity data matrix with 3 velocity components
# Output:
# X2D - velocity data matrix with 2 velocity components
#
#       t_1 t_2.  .  t_end
# X2D = [u  u  .  .  .]  (x_1,y_1)
#       [v  v  .  .  .]  (x_1,y_1)
#       [u  u  .  .  .]  (x_2,y_2)
#       [v  v  .  .  .]  (x_2,y_2) 
#       [.  .  .  .  .]   .
#       [.  .  .  .  .]   .
#       [.  .  .  .  .]   . 

    X2D = np.delete(X, list(range(2,X.shape[0],3)),axis = 0)
    return X2D


def convertToMagnitude(X,d):
# Use velocity magnitude instead of the vector   
# Input:
# X - original data matrix with velocity vector
# d- 2 or 3 Dimensional data
# Output:
# X_mag - velocity data matrix containing velocity magnitude 
#     t_1   t_2  .  .  t_end
# X_mag = [|u|  |u|  .  .  .]  (x_1,y_1)
#         [|u|  |u|  .  .  .]  (x_2,y_2)
#         [.      .  .  .  .]   .
#         [.      .  .  .  .]   .
#         [.      .  .  .  .]   .

    n = X.shape[0]
    m = X.shape[1]
    X_mag = np.zeros((int(n/d),m))

    for i in range(0,m):
        Ui = X[:,i]
        Ui = np.reshape(Ui,(-1,d))
        Ui_mag = np.sqrt(np.sum(np.square(Ui),1))
        X_mag[:,i] = Ui_mag

    return X_mag


def subtract_mean(X):
# subtract the temporal mean of the data set
# Input:
# X - original data matrix
# Output:
# X - data matrix with temporal mean subtracted
# X_mean - temporal mean of the data
    n = X.shape[0]
    m = X.shape[1]  
    X_mean = np.mean(X,1)
    for i in range(0,n):
        X[i,:] = X[i,:]-X_mean[i]

    return X, X_mean


# In[3]:


############################################################################

input_dir = "/scratch/general/nfs1/u1447794/Phase4/cylinder_data/"
filename = 'velocity_'
reader = vtk.vtkXMLUnstructuredGridReader()

t_transient = 999
t_end = 2000

X, mesh = read_velocity_data(input_dir, filename, reader, t_transient, t_end)


# In[4]:


# convert from vector to magnitude
convertToMagnitude_flag = True
dim = 2
if convertToMagnitude_flag:
    if dim == 3:
        X = convertToMagnitude(X,dim)
    elif dim == 2:
        X = convert3Dto2D_data(X)
        X = convertToMagnitude(X,dim)
else:
    X = RearrangeDataForTranspose(X)
    
    
# substract temporal mean
subtract_mean_flag = True

if subtract_mean_flag:
    X, X_mean = subtract_mean(X)
    
n = X.shape[0]
m = X.shape[1]
print("Data matrix X is n by m:", n, "x", m)


# In[9]:


def POD_analysis(X, plot_flag = True, x_range = 30, y_range = 30, z_range = 30, r = 4, plane_to_plot = 15):
    # do SVD for POD modes
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # energy content
    energy_content = S**2 / np.sum(S**2)
    cumulative_energy = np.cumsum(energy_content)
    
    POD_modes = U[:, :r]  # POD modes (spatial modes)
    temporal_coeffs = Vt[:r, :]  # Temporal coefficients

    # reconstruct the original matrix using r modes:
    X_approx = POD_modes @ np.diag(S[:r]) @ Vt[:r, :]
    
    err_POD  = np.linalg.norm(X-X_approx)/np.linalg.norm(X)
    print("POD error with ",r," modes:",err_POD)
    
    
    if plot_flag:
        plt.figure(figsize=(9, 3))
        plt.subplot(1, 3, 1)
        plt.plot(S)
        plt.yscale('log')
        plt.xlabel('modes')
        plt.ylabel('magnitude')
        plt.title('Singular Values')

        plt.subplot(1, 3, 2)
        plt.plot(cumulative_energy,'o-')
        plt.title("Cumulative energy")
        plt.xlabel("modes")
        
        plt.subplot(1, 3, 3)
        plt.plot(temporal_coeffs.T)
        plt.legend(['1','2','3','4'])
        plt.xlabel('time')
        plt.title("POD - Temporal coefficients")
        # plt.show()
        plt.close()
        print("#################################################################")
        # Plot mean flow
        pv_mesh = pv.wrap(mesh)
        window_size = (600, 400)
        # Add the scalar data (X[:, 0]) to the mesh as a scalar field
        # Make sure the length of X[:, 0] matches the number of points in the mesh
        pv_mesh.point_data["mean v"] = X_mean

        # Now plot the mesh with the scalar values
        plotter = pv.Plotter(shape=(2, 2), window_size=(600, 600))
        plotter.subplot(0, 0)  # Position (0, 0)
        plotter.add_mesh(pv_mesh, scalars="mean v", cmap="turbo", show_edges=False)
        plotter.add_title("Mean v")
        # Set predefined views
        plotter.view_xy()  # Top-down view
        plotter.camera.zoom(1.0)  # Zoom in by a factor of 1.5

        pv_mesh1 = pv.wrap(mesh)
        pv_mesh1.point_data["POD1"] = U[:,0]  # Set mean flow data
        plotter.subplot(0, 1)  # Position (0, 0)
        plotter.add_mesh(pv_mesh1, scalars="POD1", cmap="turbo", show_edges=False)
        plotter.add_title("POD1")
        plotter.view_xy()  # Top-down view

        pv_mesh2 = pv.wrap(mesh)
        pv_mesh2.point_data["POD2"] = U[:,1]  # Set mean flow data
        plotter.subplot(1, 0)  # Position (0, 0)
        plotter.add_mesh(pv_mesh2, scalars="POD2", cmap="turbo", show_edges=False)
        plotter.add_title("POD2")
        plotter.view_xy()  # Top-down view

        pv_mesh3 = pv.wrap(mesh)
        pv_mesh3.point_data["POD3"] = U[:,2]   # Set mean flow data
        plotter.subplot(1, 1)  # Position (0, 0)
        plotter.add_mesh(pv_mesh3, scalars="POD3", cmap="turbo", show_edges=False)
        plotter.add_title("POD3")
        plotter.view_xy()  # Top-down view

        plotter.save_graphic(save_modes_path+'POD.pdf')
        plotter.close()
        print("#################################################################")
        
        # plot reconstruction
        plotter = pv.Plotter(shape=(1, 2), window_size=(600, 300))
        ttime = 300
        pv_mesh1 = pv.wrap(mesh)
        pv_mesh2 = pv.wrap(mesh)
        pv_mesh1.point_data["velocity"] = X[:,ttime]  # Set mean flow data
        pv_mesh2.point_data["POD approx"] = X_approx[:,ttime]  # Set mean flow data
        plotter.subplot(0, 0)  # Position (0, 0)
        plotter.add_mesh(pv_mesh1, scalars="velocity", cmap="turbo", show_edges=False)
        plotter.add_title("True", font_size=10)
        plotter.view_xy()  # Top-down view

        plotter.subplot(0, 1)  # Position (0, 0)
        plotter.add_mesh(pv_mesh2, scalars="POD approx", cmap="turbo", show_edges=False)
        plotter.add_title("POD Predicted", font_size=10)
        plotter.view_xy()  # Top-down view

#         plotter.show()
        plotter.save_graphic(save_modes_path+'POD_rec.pdf')
        plotter.close()

    
    return X_approx, POD_modes, temporal_coeffs, S[:r]


# ## POD ##

# In[10]:


# pv_mesh = pv.wrap(mesh)
# window_size = (600, 400)
# # Add the scalar data (X[:, 0]) to the mesh as a scalar field
# # Make sure the length of X[:, 0] matches the number of points in the mesh
# pv_mesh.point_data["velocity"] = X[:, 300]#+X_mean

# # Now plot the mesh with the scalar values
# plotter = pv.Plotter(window_size=window_size)
# plotter.add_mesh(pv_mesh, scalars="velocity", cmap="viridis", show_edges=False)
# # Set predefined views
# plotter.view_xy()  # Top-down view
# plotter.camera.zoom(1.5)  # Zoom in by a factor of 1.5
# plotter.show()
# # plotter.save_graphic("mean.pdf")
# plotter.close()


# In[13]:


plot_POD_flag = True
# data size in each direction if it's voxelized data
x_range = 30
y_range = 30
z_range = 30

r = 4
r_DESMO = 4
save_modes_path = f'./figures/DESMO_Cylinder_r{r_DESMO}/modes/'
os.makedirs(os.path.dirname(save_modes_path), exist_ok=True)
X_approx, POD_modes, temporal_coeffs, S = POD_analysis(X,plot_POD_flag, x_range, y_range, z_range, r)


# ## DEcomposed Sparse Modal Optimization (DESMO) ##

# In[14]:


# Check if GPU can be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Running on GPU',flush=True)
else: print('Running on CPU',flush=True)


# In[15]:


# Prepare dataset for pyTorch
X_tensor = torch.from_numpy(X.T)
dataset = torch.utils.data.TensorDataset(X_tensor)
batchsize = m
# Set seed for reproducible results
seed = 42
torch.manual_seed(seed)
#shuffle data manually and save indices
shuffle_flag = False
if shuffle_flag:
    index_list = torch.randperm(len(dataset)).tolist()
    shuffled_dataset = torch.utils.data.Subset(dataset, index_list)
    data_loader = torch.utils.data.DataLoader(shuffled_dataset, batch_size = batchsize, shuffle = False)
    
else:
    data_loader = torch.utils.data.DataLoader(dataset, batch_size = batchsize, shuffle = False)


# In[16]:


def POOL_DATA(yin, nVars, polyorder):
    n = yin.shape[0]
    yout = torch.zeros((n, 1)).to(device)

    # poly order 0
    yout[:, 0] = torch.ones(n)
    # poly order 1
    for i in range(nVars):
        yout = torch.cat((yout, yin[:, i].reshape((yin.shape[0], 1))), dim=1).to(device)

    # poly order 2
    if polyorder >= 2:
        for i in range(nVars):
            for j in range(i, nVars):
                yout = torch.cat((yout, (yin[:, i] * yin[:, j]).reshape((yin.shape[0], 1))), dim=1).to(device)

    if polyorder >= 3:
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j,nVars):
                    yout = torch.cat((yout, (yin[:, i] * yin[:, j] * yin[:,k]).reshape((yin.shape[0], 1))), dim=1).to(device)
           
    if polyorder >= 4:
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j,nVars):
                    for l in range(k,nVars):
                        yout = torch.cat((yout, (yin[:, i] * yin[:, j] * yin[:,k] * yin[:,l]).reshape((yin.shape[0], 1))), dim=1).to(device)
                
    if polyorder >= 5:
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j,nVars):
                    for l in range(k,nVars):
                        for ii in range(l,nVars):
                            yout = torch.cat((yout, (yin[:, i] * yin[:, j] * yin[:,k] * yin[:,l] * yin[:,ii]).reshape((yin.shape[0], 1))), dim=1).to(device)
                            
                            
    if polyorder >= 6:
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j,nVars):
                    for l in range(k,nVars):
                        for ii in range(l,nVars):
                            for jj in range(ii,nVars):
                                yout = torch.cat((yout, (yin[:, i] * yin[:, j] * yin[:,k] * yin[:,l] * yin[:,ii] * yin[:,jj]).reshape((yin.shape[0], 1))), dim=1).to(device)
                                
                                
    if polyorder >= 7:
        for i in range(nVars):
            for j in range(i, nVars):
                for k in range(j,nVars):
                    for l in range(k,nVars):
                        for ii in range(l,nVars):
                            for jj in range(ii,nVars):
                                for kk in range(jj,nVars):
                                    yout = torch.cat((yout, (yin[:, i] * yin[:, j] * yin[:,k] * yin[:,l] * yin[:,ii] * yin[:,jj] * yin[:,kk]).reshape((yin.shape[0], 1))), dim=1).to(device)                                    
        
    return yout


# In[17]:


def binomial_coefficient(n, k):
    """
    Compute the binomial coefficient "n choose k".
    """
    if k > n:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

def calculate_number_of_terms(nVars, polyorder):
    """
    Calculate the number of terms in the polynomial expansion of given nVars and polyorder.
    """
    num_terms = 0
    for k in range(polyorder + 1):  # Iterate over each order from 0 to polyorder
        num_terms += binomial_coefficient(nVars + k - 1, k)
    return num_terms


# In[18]:


def outer_product_matrix(A, B):
    """
    Function to compute outer products between all columns of matrix A (900x10)
    and all columns of matrix B (1000x10), and returns a result of size 900x1000x100.

    Parameters:
    A (numpy.ndarray): A matrix of size 900x10.
    B (numpy.ndarray): A matrix of size 1000x10.

    Returns:
    numpy.ndarray: An array of size 900x1000x100 where each slice is an outer product
                   of a column from A and a column from B.
    """
    # Get the number of rows and columns from A and B
    num_cols_A = A.shape[1]
    num_cols_B = B.shape[1]
    
    # We expect num_cols_A == num_cols_B == 10
    if num_cols_A !=  num_cols_B:
        raise ValueError("Both matrices A and B must have exactly same number of columns.")
    
    # Initialize the output array of size 900x1000x100
    output = torch.zeros((A.shape[0], B.shape[0], num_cols_A * num_cols_B)).to(device)
    
    # Fill in the output array with the outer products of columns from A and B
    for i in range(num_cols_A):
        for j in range(num_cols_B):
            # Compute the outer product of the i-th column of A and the j-th column of B
            outer_product = torch.outer(A[:, i], B[:, j]).to(device)
            
            # Store it in the correct slice of the output array
            output[:, :, i * num_cols_B + j] = outer_product.to(device)
    
    return output


# In[19]:


class DESMO(nn.Module):
    def __init__(self, n, m, polyorder,r_DESMO, omega_init = 10000):
        super(DESMO, self).__init__()
        
        # optimizable modes
        
        self.phi_list = nn.ParameterList([nn.Parameter(torch.ones(n)) for _ in range(r_DESMO)])
        
        # calculate number of polynomial terms for r variables of order polyorder
        num_terms = calculate_number_of_terms(r_DESMO,polyorder)
        print('Number of terms in polynomial library:',num_terms)
        
        # create vector of optimizable coefficients for sparsity
        self.c_coef = nn.Parameter(torch.ones(num_terms))

        # otpimizable temporal coefficients for polynomial terms
        self.z_list = nn.ParameterList([nn.Parameter(torch.ones(m)) for _ in range(num_terms)])

        # Temporal coefficients for sin/cos/tanh terms
        self.zsin_list = nn.ParameterList([nn.Parameter(torch.ones(m)) for _ in range(r_DESMO)])
        self.zcos_list = nn.ParameterList([nn.Parameter(torch.ones(m)) for _ in range(r_DESMO)])
        self.ztanh_list = nn.ParameterList([nn.Parameter(torch.ones(m)) for _ in range(r_DESMO)]) 

        # Sparsity coefficients for sin/cos/tanh terms
        self.sin_coef_list = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(r_DESMO)])
        self.cos_coef_list = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(r_DESMO)])
        self.tanh_coef_list = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(r_DESMO)])

        
        # Define optimizable frequencies for sin, cos, and tanh terms
        self.omega_list = nn.ParameterList([nn.Parameter(torch.tensor(1.0) * omega_init) for _ in range(3 * r_DESMO)]) 
        

               
        
    def forward(self, X):

        # Initialize with POD modes and coefficients
        phi_list = [
            phi * torch.from_numpy(POD_modes[:, i]).type(torch.FloatTensor).to(device)
            for i, phi in enumerate(self.phi_list)
        ]



        latent_spatial = torch.stack(phi_list, dim=1)
    
        # create candidate library for spatial modes
        theta_phi = self.c_coef * POOL_DATA(latent_spatial,r_DESMO,polyorder)

        z_values = torch.stack([z for z in self.z_list],dim=0)
        
        
        # Fourier terms: sin, cos, and tanh contributions
        fourier_contrib = 0
        for i in range(len(self.phi_list)):
            phi = phi_list[i]
            zsin = self.zsin_list[i]
            zcos = self.zcos_list[i]
            ztanh = self.ztanh_list[i]

            omega_sin = self.omega_list[3 * i]
            omega_cos = self.omega_list[3 * i + 1]
            omega_tanh = self.omega_list[3 * i + 2]

            sin_term = self.sin_coef_list[i] * zsin.view(-1, 1) @ torch.sin(omega_sin * phi.T).view(1, -1)
            cos_term = self.cos_coef_list[i] * zcos.view(-1, 1) @ torch.cos(omega_cos * phi.T).view(1, -1)
            tanh_term = self.tanh_coef_list[i] * ztanh.view(-1, 1) @ torch.tanh(omega_tanh * phi.T).view(1, -1)

            fourier_contrib = fourier_contrib + sin_term + cos_term + tanh_term


        final_reconstruction = theta_phi @ z_values + fourier_contrib.T



        return final_reconstruction.T, latent_spatial, z_values


# In[20]:



polyorder = 3
omega_init = 10000
model_desmo = DESMO(n, m, polyorder,r_DESMO,omega_init).to(device)


# In[21]:


# Define parameter groups with specific learning rates
optimizer = torch.optim.Adamax([
    # Sparsity coefficients group
    {'params': [model_desmo.c_coef] + 
               [param for param in model_desmo.sin_coef_list] +
               [param for param in model_desmo.cos_coef_list] +
               [param for param in model_desmo.tanh_coef_list],
     'lr': 1e-2},

    # Phi parameters group
    {'params': [param for param in model_desmo.phi_list], 'lr': 1e-3},

    # Z parameters group
    {'params': [param for param in model_desmo.z_list] +
               [param for param in model_desmo.zsin_list] +
               [param for param in model_desmo.zcos_list] +
               [param for param in model_desmo.ztanh_list],
     'lr': 1e-2},

    # Omega parameters group
    {'params': [param for param in model_desmo.omega_list], 'lr': 1e3},
], weight_decay=0.0)

scheduler_temporal = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1000, factor=0.1, verbose=True,min_lr=1e-6)
criterion = nn.MSELoss()
# Model info
pytorch_total_params = sum(p.numel() for p in model_desmo.parameters())
print("Total number of parameters in DESMO:", pytorch_total_params)


# In[22]:


def poly_norm(c_coef,z,phi):
    # calculate the norm of the polynomial terms
    phi_list = [param.data for param in phi]
    phis=torch.stack(phi_list,dim=1)
    phi_lib = POOL_DATA(phis,r_DESMO,polyorder)
    
    z_list = [param.data for param in z]
    zs=torch.stack(z_list,dim=1)
    
    norms = []
        # Iterate through model_desmo.c_coef from index 0 to the end
    for i, c in enumerate(c_coef):
        # Update slices for phi_lib and zs to i:i+1
        phi_slice = phi_lib[:, i:i+1]
        zs_slice = zs[:, i:i+1]

        # Compute the norm for the current index
        result = torch.norm(c * (phi_slice @ zs_slice.T), p=2)

        # Print the result
#         print(f"Poly Library {i}: Norm = {result}")
        norms.append(result)
    
    return norms


# In[23]:


def nonlinear_norm(sin_coef,cos_coef,tanh_coef,zsin,zcos,ztanh,phi,omega):
    # calculate the norm of the nonlinear terms
    
    phi_list = [param.data for param in phi]
    phis=torch.stack(phi_list,dim=1)
    
    zsin_list = [param.data for param in zsin]

    zcos_list = [param.data for param in zcos]

    ztanh_list = [param.data for param in ztanh]

    norms = []

    #iterate through phi's
    for i in range(phis.shape[1]):
        phi = phi_list[i]
        zsin = zsin_list[i]
        zcos = zcos_list[i]
        ztanh = ztanh_list[i]

        omega_sin = omega[3 * i]
        omega_cos = omega[3 * i + 1]
        omega_tanh = omega[3 * i + 2]

        sin_term = sin_coef[i] * zsin.view(-1, 1) @ torch.sin(omega_sin * phi.T).view(1, -1)
        cos_term = cos_coef[i] * zcos.view(-1, 1) @ torch.cos(omega_cos * phi.T).view(1, -1)
        tanh_term = tanh_coef[i] * ztanh.view(-1, 1) @ torch.tanh(omega_tanh * phi.T).view(1, -1)

        # Compute the norm for the current index
        result_sin = torch.norm(sin_term, p=2)
        result_cos = torch.norm(cos_term, p=2)
        result_tanh = torch.norm(tanh_term, p=2)

        norms.append(result_sin)
        norms.append(result_cos)
        norms.append(result_tanh)
        
        
    return norms


# In[24]:


# Training loop remains similar
num_epochs = 100000
beta = 1e-3 # orthogonal loss
l1_lambda = 1e-4 #sparsity loss
time_to_plot = 300
outputs_dual = []
loss_list_dual = []
plot_DESMO_flag = False
for epoch in range(num_epochs):
    for x in data_loader:  # Iterate over batches for spatial data
        snapshot = x[0].type(torch.FloatTensor).to(device)


        recon_combined, latent_spatial, latent_temporal = model_desmo(snapshot)

        # Orthogonal loss calculation
        ortho_loss_spatial = 0
        num_latents = latent_spatial.size(1)
        
        # Compute pairwise dot products of latent vectors and sum the results
        for i in range(num_latents):
            for j in range(i + 1, num_latents):
                ortho_loss_spatial += torch.norm(latent_spatial[:, i] @ latent_spatial[:, j].T, p='fro')
        # reconstruction loss
        loss = criterion(recon_combined, snapshot)
        
        # L1 sparsity loss: sum over all coefficients in sin_coef_list, cos_coef_list, and tanh_coef_list
        l1_loss = torch.norm(model_desmo.c_coef, p=1)  # Regular sparsity loss for c_coef
        for sin_coef in model_desmo.sin_coef_list:
            l1_loss = l1_loss + torch.norm(sin_coef, p=1)  # Sparsity loss for each sin_coef in sin_coef_list
        for cos_coef in model_desmo.cos_coef_list:
            l1_loss = l1_loss + torch.norm(cos_coef, p=1)  # Sparsity loss for each cos_coef in cos_coef_list
        for tanh_coef in model_desmo.tanh_coef_list:
            l1_loss = l1_loss + torch.norm(tanh_coef, p=1) 

        total_loss = loss +  beta * (ortho_loss_spatial)+ l1_lambda*l1_loss
        
        # Plot mean flow (assuming X_mean is reshaped to (30, 30, 30))
        if epoch % 100 == 0 and plot_DESMO_flag:
            ttime = time_to_plot
            plotter = pv.Plotter(shape=(1, 3), window_size=(900, 300))
            pv_mesh_d = pv.wrap(mesh)
            pv_mesh_AE = pv.wrap(mesh)
            pv_mesh_POD = pv.wrap(mesh)
            pv_mesh_d.point_data["velocity"] = snapshot[ttime,:].detach().cpu().numpy()  # Set mean flow data
            pv_mesh_AE.point_data["AE approx"] = recon_combined[ttime,:].detach().cpu().numpy()  # Set mean flow data
            pv_mesh_POD.point_data["POD approx"] = X_approx[:,ttime]
            plotter.subplot(0, 0)  # Position (0, 0)
            plotter.add_mesh(pv_mesh_d, scalars="velocity", cmap="viridis", show_edges=False,scalar_bar_args={'n_labels': 2})
            plotter.add_title("True")
            plotter.view_xy()  # Top-down view

            plotter.subplot(0, 1)  # Position (0, 0)
            plotter.add_mesh(pv_mesh_AE, scalars="AE approx", cmap="viridis", show_edges=False,scalar_bar_args={'n_labels': 2})
            plotter.add_title("AE Predicted")
            plotter.view_xy()  # Top-down view


            plotter.subplot(0, 2)  # Position (0, 0)
            plotter.add_mesh(pv_mesh_POD, scalars="POD approx", cmap="viridis", show_edges=False,scalar_bar_args={'n_labels': 2})
            plotter.add_title("POD Predicted")
            plotter.view_xy()  # Top-down view
            
            # Show the plot
            plotter.show()
            plotter.close()
            
        optimizer.zero_grad()
        total_loss.backward()

        optimizer.step()
        loss_list_dual.append((epoch, loss.item()))
        
        if epoch == num_epochs-1 or epoch % 100 ==0:
            outputs_dual = []
            outputs_dual.append((epoch+1, snapshot, latent_spatial.detach(), latent_temporal.detach(), 
                                 recon_combined.detach())) #, prim1.detach(),prim2.detach()
            
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Rec Loss: {loss.item():.12f}, Spatial ortho loss: {ortho_loss_spatial.item():.8f}, L1 loss: {l1_loss.item():.4f} ', flush = True)
        scheduler_temporal.step(total_loss)

    
    if epoch % 2000 == 0:
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'./models/DESMO_Cylinder_r{r_DESMO}_epoch{epoch+1}_{current_time}.pt'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model_desmo.state_dict(), save_path)
        print(f"Model saved at epoch {epoch+1} to {save_path}")


# In[25]:


epochs, losses = zip(*loss_list_dual)

plt.plot(epochs,losses)
plt.xlabel("epochs")
plt.ylabel("rec loss")
plt.yscale('log')
os.makedirs(os.path.dirname(f'./figures/DESMO_Cylinder_r{r_DESMO}'), exist_ok=True)
plt.savefig(f'./figures/DESMO_Cylinder_r{r_DESMO}/Cylinder_loss.png')
plt.show()
plt.close()
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f'./models/DESMO_r{r_DESMO}_final_{current_datetime}.pt'
torch.save(model_desmo.state_dict(), filename)
print(f"Model saved to {filename}")


# In[26]:


out_x = outputs_dual[0][1].detach().cpu().numpy()
latent_spatial = outputs_dual[0][2].detach().cpu().numpy()
latent_temporal = outputs_dual[0][3].detach().cpu().numpy()
recon_dual = outputs_dual[0][4].detach().cpu().numpy()


# In[27]:


def plot_poly_modes(phi, c_coef, save_modes_path):
    phi = torch.from_numpy(phi).to(device)
    phi_lib = POOL_DATA(phi,r_DESMO,polyorder)

    for i, c in enumerate(c_coef):
        # Update slices for phi_lib and zs to i:i+1
        phi_slice = phi_lib[:, i:i+1].detach().cpu().numpy()

        plotter2 = pv.Plotter(shape=(1, 1), window_size=(800, 800))

        pv_mesh = pv.wrap(mesh)
        pv_mesh.point_data["DESMO1"] = phi_slice 
        plotter2.add_mesh(
            pv_mesh,
            scalars="DESMO1",
            cmap="turbo",
            show_edges=False,
            scalar_bar_args={'title': f'DESMOTerm{i}', 'n_labels': 2}
        )
        plotter2.add_title(f"DESMO modeTerm{i}", font_size=10)
        plotter2.view_xy()  # Top-down view
        modeid = f'PolymodeTerm{i}.pdf'
        plotter2.save_graphic(save_modes_path + modeid)
        plotter2.close()


# In[28]:


plot_poly_modes(latent_spatial,model_desmo.c_coef,save_modes_path)


# In[29]:


def plot_nonlinear_modes(phis,omega,save_modes_path):

    #iterate through phi's
    for i in range(phis.shape[1]):
        phi = phis[:,i]

        omega_sin = omega[3 * i].detach().cpu().numpy()
        omega_cos = omega[3 * i + 1].detach().cpu().numpy()
        omega_tanh = omega[3 * i + 2].detach().cpu().numpy()

        sin_term =  np.sin(omega_sin * phi.T)
        cos_term =  np.cos(omega_cos * phi.T)
        tanh_term =  np.tanh(omega_tanh * phi.T)

        plotter2 = pv.Plotter(shape=(1, 1), window_size=(800, 800))

        pv_mesh = pv.wrap(mesh)
        pv_mesh.point_data["DESMO1"] = sin_term 
        plotter2.add_mesh(
            pv_mesh,
            scalars="DESMO1",
            cmap="turbo",
            show_edges=False,
            scalar_bar_args={'title': f'sinDESMOTerm{i}', 'n_labels': 2}
        )
        plotter2.add_title(f"DESMO modeTerm{i}", font_size=10)
        plotter2.view_xy()  # Top-down view
        modeid = f'SinmodeTerm{i}.pdf'
        plotter2.save_graphic(save_modes_path + modeid)
        plotter2.close()
        
        plotter2 = pv.Plotter(shape=(1, 1), window_size=(800, 800))

        pv_mesh = pv.wrap(mesh)
        pv_mesh.point_data["DESMO1"] = cos_term 
        plotter2.add_mesh(
            pv_mesh,
            scalars="DESMO1",
            cmap="turbo",
            show_edges=False,
            scalar_bar_args={'title': f'cosDESMOTerm{i}', 'n_labels': 2}
        )
        plotter2.add_title(f"DESMO modeTerm{i}", font_size=10)
        plotter2.view_xy()  # Top-down view
        modeid = f'CosmodeTerm{i}.pdf'
        plotter2.save_graphic(save_modes_path + modeid)
        plotter2.close()
        
        
        plotter2 = pv.Plotter(shape=(1, 1), window_size=(800, 800))

        pv_mesh = pv.wrap(mesh)
        pv_mesh.point_data["DESMO1"] = tanh_term 
        plotter2.add_mesh(
            pv_mesh,
            scalars="DESMO1",
            cmap="turbo",
            show_edges=False,
            scalar_bar_args={'title': f'tanhDESMOTerm{i}', 'n_labels': 2}
        )
        plotter2.add_title(f"DESMO modeTerm{i}", font_size=10)
        plotter2.view_xy()  # Top-down view
        modeid = f'TanhmodeTerm{i}.pdf'
        plotter2.save_graphic(save_modes_path + modeid)
        plotter2.close()


# In[30]:


plot_nonlinear_modes(latent_spatial,model_desmo.omega_list,save_modes_path)


# In[31]:


if plot_DESMO_flag:
    ttime = 200
    plotter2 = pv.Plotter(shape=(3, 2), window_size=(800, 800))

    # First subplot: True Velocity
    pv_mesh1 = pv.wrap(mesh)  # Create a new mesh for the first subplot
    pv_mesh1.point_data["velocity1"] = out_x[ttime, :]  # Set true velocity data
    plotter2.subplot(0, 0)  # Position (0, 0)
    plotter2.add_mesh(
        pv_mesh1,
        scalars="velocity1",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={'title': 'Velocity', 'n_labels': 2}
    )
    plotter2.add_title("True", font_size=10)
    plotter2.view_xy()  # Top-down view

    # Second subplot: AE approximation
    pv_mesh2 = pv.wrap(mesh)  # Create a new mesh for the second subplot
    pv_mesh2.point_data["AE approx"] = recon_dual[ttime, :]  # Set AE approximation data
    plotter2.subplot(0, 1)  # Position (0, 1)
    plotter2.add_mesh(
        pv_mesh2,
        scalars="AE approx",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={'title': 'AE rec', 'n_labels': 2}
    )
    plotter2.add_title("Predicted", font_size=10)
    plotter2.view_xy()  # Top-down view

    # Third subplot: AE mode 1
    pv_mesh3 = pv.wrap(mesh)  # Create a new mesh for the third subplot
    pv_mesh3.point_data["AE1"] = latent_spatial[:, 0]  # Set AE mode 1 data
    plotter2.subplot(1, 0)  # Position (1, 0)
    plotter2.add_mesh(
        pv_mesh3,
        scalars="AE1",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={'title': 'AE1', 'n_labels': 2}
    )
    plotter2.add_title("AE mode1", font_size=10)
    plotter2.view_xy()  # Top-down view

    # Fourth subplot: AE mode 2
    pv_mesh4 = pv.wrap(mesh)  # Create a new mesh for the fourth subplot
    pv_mesh4.point_data["AE2"] = latent_spatial[:, 1]  # Set AE mode 2 data
    plotter2.subplot(1, 1)  # Position (1, 1)
    plotter2.add_mesh(
        pv_mesh4,
        scalars="AE2",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={'title': 'AE2', 'n_labels': 2}
    )
    plotter2.add_title("AE mode2", font_size=10)
    plotter2.view_xy()  # Top-down view


    # Third subplot: AE mode 1
    pv_mesh5 = pv.wrap(mesh)  # Create a new mesh for the third subplot
    pv_mesh5.point_data["sinAE1"] = np.sin(model_desmo.omega_list[0].detach().cpu().numpy() * latent_spatial[:, 0])  # Set AE mode 1 data
    plotter2.subplot(2, 0)  # Position (1, 0)
    plotter2.add_mesh(
        pv_mesh5,
        scalars="sinAE1",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={'title': 'sinAE1', 'n_labels': 2}
    )
    plotter2.add_title("sinAE mode1", font_size=10)
    plotter2.view_xy()  # Top-down view

    # Fourth subplot: AE mode 2
    pv_mesh6 = pv.wrap(mesh)  # Create a new mesh for the fourth subplot
    pv_mesh6.point_data["sinAE2"] = np.sin(model_desmo.omega_list[3].detach().cpu().numpy() * latent_spatial[:, 1])  # Set AE mode 2 data
    plotter2.subplot(2, 1)  # Position (1, 1)
    plotter2.add_mesh(
        pv_mesh6,
        scalars="sinAE2",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={'title': 'sinAE2', 'n_labels': 2}
    )
    plotter2.add_title("sinAE mode2", font_size=10)
    plotter2.view_xy()  # Top-down view

    # Show the plot
#     plotter2.show()
    plotter2.save_graphic(save_modes_path+'modes.pdf')
    plotter2.close()


# In[32]:


err_DESMO  = np.linalg.norm(X-recon_dual.T)/np.linalg.norm(X)
print("DESMO error with ",r_DESMO," modes:",err_DESMO)


# In[33]:


plot_POD_flag = False
r = r_DESMO
X_approx, POD_modes, temporal_coeffs, S = POD_analysis(X,plot_POD_flag, x_range, y_range, z_range, r)
r = r_DESMO*2
X_approx, POD_modes, temporal_coeffs, S = POD_analysis(X,plot_POD_flag, x_range, y_range, z_range, r)




ttime = 200
plotter2 = pv.Plotter(shape=(4, 2), window_size=(800, 800))

# Third subplot: AE mode 1
pv_mesh3 = pv.wrap(mesh)  # Create a new mesh for the third subplot
pv_mesh3.point_data["DESMO1"] = latent_spatial[:, 0]  # Set AE mode 1 data
plotter2.subplot(0, 0)  # Position (1, 0)
plotter2.add_mesh(
    pv_mesh3,
    scalars="DESMO1",
    cmap="turbo",
    show_edges=False,
    scalar_bar_args={'title': 'DESMO1', 'n_labels': 2}
)
plotter2.add_title("DESMO mode1", font_size=10)
plotter2.view_xy()  # Top-down view

# Fourth subplot: AE mode 2
pv_mesh4 = pv.wrap(mesh)  # Create a new mesh for the fourth subplot
pv_mesh4.point_data["DESMO2"] = latent_spatial[:, 1]  # Set AE mode 2 data
plotter2.subplot(0, 1)  # Position (1, 1)
plotter2.add_mesh(
    pv_mesh4,
    scalars="DESMO2",
    cmap="turbo",
    show_edges=False,
    scalar_bar_args={'title': 'DESMO2', 'n_labels': 2}
)
plotter2.add_title("DESMO mode2", font_size=10)
plotter2.view_xy()  # Top-down view


# Third subplot: AE mode 1
pv_mesh5 = pv.wrap(mesh)  # Create a new mesh for the third subplot
pv_mesh5.point_data["sinAE1"] = np.sin(model_desmo.omega_list[0].detach().cpu().numpy() * latent_spatial[:, 0])  # Set AE mode 1 data
plotter2.subplot(1, 0)  # Position (1, 0)
plotter2.add_mesh(
    pv_mesh5,
    scalars="sinAE1",
    cmap="turbo",
    show_edges=False,
    scalar_bar_args={'title': 'sinAE1', 'n_labels': 2}
)
plotter2.add_title("sinAE mode1", font_size=10)
plotter2.view_xy()  # Top-down view

# Fourth subplot: AE mode 2
pv_mesh6 = pv.wrap(mesh)  # Create a new mesh for the fourth subplot
pv_mesh6.point_data["sinAE2"] = np.sin(model_desmo.omega_list[3].detach().cpu().numpy() * latent_spatial[:, 1])  # Set AE mode 2 data
plotter2.subplot(1, 1)  # Position (1, 1)
plotter2.add_mesh(
    pv_mesh6,
    scalars="sinAE2",
    cmap="turbo",
    show_edges=False,
    scalar_bar_args={'title': 'sinAE2', 'n_labels': 2}
)
plotter2.add_title("sinAE mode2", font_size=10)
plotter2.view_xy()  # Top-down view


# Third subplot: AE mode 1
pv_mesh1 = pv.wrap(mesh)  # Create a new mesh for the third subplot
pv_mesh1.point_data["cosDESMO1"] = np.cos(model_desmo.omega_list[1].detach().cpu().numpy() * latent_spatial[:, 0])  # Set AE mode 1 data
plotter2.subplot(2, 0)  # Position (1, 0)
plotter2.add_mesh(
    pv_mesh1,
    scalars="cosDESMO1",
    cmap="turbo",
    show_edges=False,
    scalar_bar_args={'title': 'cosDESMO1', 'n_labels': 2}
)
plotter2.add_title("cosDESMO mode1", font_size=10)
plotter2.view_xy()  # Top-down view

# Fourth subplot: AE mode 2
pv_mesh2 = pv.wrap(mesh)  # Create a new mesh for the fourth subplot
pv_mesh2.point_data["cosDESMO2"] = np.cos(model_desmo.omega_list[4].detach().cpu().numpy() * latent_spatial[:, 1])  # Set AE mode 2 data
plotter2.subplot(2, 1)  # Position (1, 1)
plotter2.add_mesh(
    pv_mesh2,
    scalars="cosDESMO2",
    cmap="turbo",
    show_edges=False,
    scalar_bar_args={'title': 'cosDESMO2', 'n_labels': 2}
)
plotter2.add_title("cosDESMO mode2", font_size=10)
plotter2.view_xy()  # Top-down view


pv_mesh7 = pv.wrap(mesh)  # Create a new mesh for the third subplot
pv_mesh7.point_data["tanhDESMO1"] = np.tanh(model_desmo.omega_list[2].detach().cpu().numpy() * latent_spatial[:, 0])  # Set AE mode 1 data
plotter2.subplot(3, 0)  # Position (1, 0)
plotter2.add_mesh(
    pv_mesh7,
    scalars="tanhDESMO1",
    cmap="turbo",
    show_edges=False,
    scalar_bar_args={'title': 'tanhDESMO1', 'n_labels': 2}
)
plotter2.add_title("tanhDESMO mode1", font_size=10)
plotter2.view_xy()  # Top-down view


pv_mesh8 = pv.wrap(mesh)  # Create a new mesh for the fourth subplot
pv_mesh8.point_data["tanhDESMO2"] = np.tanh(model_desmo.omega_list[5].detach().cpu().numpy() * latent_spatial[:, 1])  # Set AE mode 2 data
plotter2.subplot(3, 1)  # Position (1, 1)
plotter2.add_mesh(
    pv_mesh8,
    scalars="tanhDESMO2",
    cmap="turbo",
    show_edges=False,
    scalar_bar_args={'title': 'tanhDESMO2', 'n_labels': 2}
)
plotter2.add_title("tanhDESMO mode2", font_size=10)
plotter2.view_xy()  # Top-down view

# Show the plot
# plotter2.show()
plotter2.save_graphic(save_modes_path+'/more_modes.pdf')
plotter2.close()


# In[36]:



plt.plot(latent_temporal.T,label='z')
plt.xlabel('timesteps')
plt.ylabel('z')
plt.legend()
plt.savefig(save_modes_path+'/temporal.png')
plt.close()


# In[37]:


# Save the original parameters before thresholding
original_c_coef = model_desmo.c_coef.clone()
original_sin_coef_list = [sin_coef.clone() for sin_coef in model_desmo.sin_coef_list]
original_cos_coef_list = [cos_coef.clone() for cos_coef in model_desmo.cos_coef_list]
original_tanh_coef_list = [tanh_coef.clone() for tanh_coef in model_desmo.tanh_coef_list]


# In[39]:


# sparsity loss
polynorms = torch.stack(poly_norm(model_desmo.c_coef,model_desmo.z_list,model_desmo.phi_list))
nlnorms = torch.stack(nonlinear_norm(model_desmo.sin_coef_list,model_desmo.cos_coef_list,model_desmo.tanh_coef_list,
            model_desmo.zsin_list,model_desmo.zcos_list,model_desmo.ztanh_list,model_desmo.phi_list,model_desmo.omega_list))



# In[42]:


print('####################################################################################')
print("Poly norms: ",polynorms.detach().cpu().numpy())
print("Nonlinear terms norms:",nlnorms.detach().cpu().numpy())
print('####################################################################################')


# In[43]:


# Define threshold values from 10^-8 to 10^-1
threshold_values = [pow(10,-i) for i in np.arange(4, -3, -0.5)]

# Iterate over threshold values
results = []
for threshold in threshold_values:
    # Restore original parameters before applying a new threshold
    model_desmo.c_coef.data = original_c_coef.clone()
    for i, sin_coef in enumerate(model_desmo.sin_coef_list):
        model_desmo.sin_coef_list[i].data = original_sin_coef_list[i].clone()
    for i, cos_coef in enumerate(model_desmo.cos_coef_list):
        model_desmo.cos_coef_list[i].data = original_cos_coef_list[i].clone()
    for i, tanh_coef in enumerate(model_desmo.tanh_coef_list):
        model_desmo.tanh_coef_list[i].data = original_tanh_coef_list[i].clone()
    
    # Apply thresholding with no_grad to prevent gradient tracking
    with torch.no_grad():
        # Threshold for `c_coef`
        model_desmo.c_coef.data[torch.abs(polynorms) < threshold] = 0

        # Threshold for sin, cos, and tanh coefficient lists
        for i,sin_coef in enumerate(model_desmo.sin_coef_list):
            sin_coef.data[torch.abs(nlnorms[i*3]) < threshold] = 0
        for i,cos_coef in enumerate(model_desmo.cos_coef_list):
            cos_coef.data[torch.abs(nlnorms[i*3+1]) < threshold] = 0
        for i,tanh_coef in enumerate(model_desmo.tanh_coef_list):
            tanh_coef.data[torch.abs(nlnorms[i*3+2]) < threshold] = 0

    # Set the model to inference mode
    model_desmo.eval()

    # Run inference and calculate error
    outputs_dual = []
    for x in data_loader:  # Iterate over batches for spatial data
        snapshot = x[0].type(torch.FloatTensor).to(device)
        recon_combined, latent_spatial, latent_temporal = model_desmo(snapshot)

        outputs_dual = []
        outputs_dual.append((snapshot, latent_spatial.detach(), latent_temporal.detach(), 
                             recon_combined.detach()))
    out_x = outputs_dual[0][0].detach().cpu().numpy()
    latent_spatial = outputs_dual[0][1].detach().cpu().numpy()
    latent_temporal = outputs_dual[0][2].detach().cpu().numpy()
    recon_dual = outputs_dual[0][3].detach().cpu().numpy()

    err_DESMO = np.linalg.norm(X - recon_dual.T) / np.linalg.norm(X)

    # Count total number of nonzero terms across all coefficients
    nonzero_terms = (
        torch.sum(model_desmo.c_coef != 0).item() +
        sum(torch.sum(sin_coef != 0).item() for sin_coef in model_desmo.sin_coef_list) +
        sum(torch.sum(cos_coef != 0).item() for cos_coef in model_desmo.cos_coef_list) +
        sum(torch.sum(tanh_coef != 0).item() for tanh_coef in model_desmo.tanh_coef_list)
    )

    # Save results for the current threshold
    results.append((threshold, err_DESMO, nonzero_terms))

    print(f"Threshold: {threshold}, Error: {err_DESMO:.6f}, Nonzero Terms: {nonzero_terms}")
    
    plotter2 = pv.Plotter(shape=(1, 2), window_size=(1600, 800))

    # First subplot: True Velocity
    pv_mesh1 = pv.wrap(mesh)  # Create a new mesh for the first subplot
    pv_mesh1.point_data["velocity1"] = out_x[ttime, :]  # Set true velocity data
    plotter2.subplot(0, 0)  # Position (0, 0)
    plotter2.add_mesh(
        pv_mesh1,
        scalars="velocity1",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={'title': 'Velocity', 'n_labels': 2}
    )
    plotter2.add_title("True", font_size=10)
    plotter2.view_xy()  # Top-down view

    # Second subplot: AE approximation
    pv_mesh2 = pv.wrap(mesh)  # Create a new mesh for the second subplot
    pv_mesh2.point_data["DESMO approx"] = recon_dual[ttime, :]  # Set AE approximation data
    plotter2.subplot(0, 1)  # Position (0, 1)
    plotter2.add_mesh(
        pv_mesh2,
        scalars="DESMO approx",
        cmap="turbo",
        show_edges=False,
        scalar_bar_args={'title': 'DESMO rec', 'n_labels': 2}
    )
    plotter2.add_title(f"DESMO Predicted, {nonzero_terms} terms", font_size=10)
    plotter2.view_xy()  # Top-down view
    plotter2.save_graphic(save_modes_path + f'reconstruction_threshold{threshold}.pdf')
    plotter2.close()
# Print final results in a table format
print("\nSummary of Results:")
print(f"{'Threshold':<10} {'Error':<15} {'Nonzero Terms':<15}")
for threshold, error, nonzero_terms in results:
    print(f"{threshold:<10.1e} {error:<15.6f} {nonzero_terms:<15}")


# In[ ]:





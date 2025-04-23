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


def RearrangeDataForTranspose(X):
# Reshape data matrix for temporal reduction
# Each row contains both u and v for a given spatial location
# Each two columns contain a snapshot of u and of v
# The rows of the matrix will be taken as different data points and will be compared to each other
# Therefore, it is not fair to comapre u with v, this necessitates this reshaping
# Input:
# X - original data matrix
# Output:
# X_new - new data matrix, arranged as:
# X_new = [u  v  u  v  .]  (x_1,y_1)
#         [u  v  u  v  .]  (x_2,y_2)
#         [u  v  u  v  .]  (x_3,y_3)
#         [u  v  u  v  .]  (x_4,y_4)
#         [.  .  .  .  .]   .
#         [.  .  .  .  .]   .
#         [.  .  .  .  .]   .
#         t1 t1 t2 t2  .

    u = X[0::2,:]
    v = X[1::2,:]

    n = X.shape[0]
    m = X.shape[1]

    X_new = np.zeros((int(n/2),int(m*2)))
    for i in range(m):
        X_new[:,2*i] = u[:,i]
        X_new[:,2*i+1] = v[:,i]

    return X_new

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

input_dir = "/home/hunor/PhD/flow_over_cylinder/data_moretimesteps/Re100/"
#input_dir = "../../../CFD_results/navier_stokes_cylinder_/"
filename = 'velocity_'
reader = vtk.vtkXMLUnstructuredGridReader()

t_transient = 999
t_end = 2000

X, mesh = read_velocity_data(input_dir, filename, reader, t_transient, t_end)

# X = convert3Dto2D_data(X)


# In[4]:


# convertToMagnitude_flag 
#                   if True: velocity magnitude will be used |u|
#                   if False: velocity vector will be used [u v]
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

    
subtract_mean_flag = True


if subtract_mean_flag:
    X, X_mean = subtract_mean(X)


# In[5]:


# Assuming 'mesh' is your vtkPolyData or vtkUnstructuredGrid object
points = mesh.GetPoints()

# Get the number of points in the mesh
num_points = points.GetNumberOfPoints()

# Extract the x, y, z coordinates of each point
coords = []
for i in range(num_points):
    x, y, z = points.GetPoint(i)
    coords.append((x, y, z))

# Convert to separate x, y arrays (or numpy arrays if you want)
x_coords = [coord[0] for coord in coords]
y_coords = [coord[1] for coord in coords]
z_coords = [coord[2] for coord in coords]


# In[6]:


pv_mesh = pv.wrap(mesh)
window_size = (600, 400)
# Add the scalar data (X[:, 0]) to the mesh as a scalar field
# Make sure the length of X[:, 0] matches the number of points in the mesh
pv_mesh.point_data["velocity"] = X[:, 300]+X_mean

# Now plot the mesh with the scalar values
plotter = pv.Plotter(window_size=window_size)
plotter.add_mesh(pv_mesh, scalars="velocity", cmap="viridis", show_edges=False)
# Set predefined views
plotter.view_xy()  # Top-down view
plotter.camera.zoom(1.5)  # Zoom in by a factor of 1.5
plotter.show()
plotter.save_graphic("mean.pdf")
plotter.close()


# ## POD ##

# In[7]:


# Add the scalar data (X[:, 0]) to the mesh as a scalar field
# Make sure the length of X[:, 0] matches the number of points in the mesh
pv_mesh.point_data["mean velocity"] = X_mean

# Now plot the mesh with the scalar values
plotter = pv.Plotter(window_size=window_size)
plotter.add_mesh(pv_mesh, scalars="mean velocity", cmap="viridis", show_edges=False)
# Set predefined views
plotter.view_xy()  # Top-down view
plotter.camera.zoom(1.5)  # Zoom in by a factor of 1.5
plotter.show()


# In[8]:


U, S, Vt = np.linalg.svd(X, full_matrices=False)


# In[9]:


energy_content = S**2 / np.sum(S**2)
cumulative_energy = np.cumsum(energy_content)
num_modes = np.argmax(cumulative_energy >= 0.90) + 1  # Number of modes for 90% energy

print(f"Number of modes capturing 90% of energy: {num_modes}")


# In[10]:


plt.plot(S)
plt.yscale('log')
plt.xlabel('modes')
plt.ylabel('magnitude')


# In[11]:


plt.plot(cumulative_energy,'o-')
plt.title("Cumulative energy")
plt.xlabel("modes")
plt.show()


# In[12]:


plotter = pv.Plotter(shape=(2, 2), window_size=(800, 800))

# # Plot mean flow
# pv_mesh1 = pv.wrap(mesh)
# pv_mesh1.point_data["Mean Flow"] = X_mean  # Set mean flow data
# plotter.subplot(0, 0)  # Position (0, 0)
# plotter.add_mesh(pv_mesh1, scalars="Mean Flow", cmap="viridis", show_edges=False)
# plotter.add_title("Mean Flow")
# plotter.view_xy()  # Top-down view

pv_mesh1 = pv.wrap(mesh)
pv_mesh1.point_data["POD1"] = U[:,0]  # Set mean flow data
plotter.subplot(0, 0)  # Position (0, 0)
plotter.add_mesh(pv_mesh1, scalars="POD1", cmap="viridis", show_edges=False)
plotter.add_title("POD1")
plotter.view_xy()  # Top-down view

pv_mesh1 = pv.wrap(mesh)
pv_mesh1.point_data["POD2"] = U[:,1]  # Set mean flow data
plotter.subplot(0, 1)  # Position (0, 0)
plotter.add_mesh(pv_mesh1, scalars="POD2", cmap="viridis", show_edges=False)
plotter.add_title("POD2")
plotter.view_xy()  # Top-down view

pv_mesh1 = pv.wrap(mesh)
pv_mesh1.point_data["POD3"] = U[:,2]   # Set mean flow data
plotter.subplot(1, 0)  # Position (0, 0)
plotter.add_mesh(pv_mesh1, scalars="POD3", cmap="viridis", show_edges=False)
plotter.add_title("POD3")
plotter.view_xy()  # Top-down view


pv_mesh1 = pv.wrap(mesh)
pv_mesh1.point_data["POD4"] = U[:,3]   # Set mean flow data
plotter.subplot(1, 1)  # Position (0, 0)
plotter.add_mesh(pv_mesh1, scalars="POD4", cmap="viridis", show_edges=False)
plotter.add_title("POD4")
plotter.view_xy()  # Top-down view

plotter.show()

# Show the plot
plotter.close()


# In[13]:


max_modes = 20  # Maximum number of modes to consider
errors = []

# Calculate reconstruction error for different k
for k in range(1, max_modes + 1):
    POD_modes = U[:, :k]  # Keep the first k modes
    temporal_coeffs = Vt[:k, :]  # Keep the first k temporal coefficients

    # Reconstruct the original matrix using k modes
    X_approx = POD_modes @ np.diag(S[:k]) @ Vt[:k, :]

    # Calculate the reconstruction error
    err_POD = np.linalg.norm(X - X_approx) / np.linalg.norm(X)
    errors.append(err_POD)

# Plotting the reconstruction error
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_modes + 1), errors, marker='o', linestyle='-', color='b')
plt.title('Reconstruction Error vs Number of Modes (k)')
plt.xlabel('Number of Modes (k)')
plt.ylabel('Reconstruction Error')
plt.yscale('log')  # Use logarithmic scale for better visibility
plt.grid(True)
plt.xticks(range(1, max_modes + 1))
y_ticks = np.logspace(-2, 0, num = 7)  # Change the range and number of ticks as needed
plt.yticks(y_ticks)

plt.show()


# In[14]:


k = 2  # For example, keep the first 10 modes
POD_modes = U[:, :k]  # POD modes (spatial modes)
temporal_coeffs = Vt[:k, :]  # Temporal coefficients

# If needed, reconstruct the original matrix using k modes:
X_approx = POD_modes @ np.diag(S[:k]) @ Vt[:k, :]

err_POD  = np.linalg.norm(X-X_approx)/np.linalg.norm(X)
print("POD error with ",k," modes:",err_POD)


# In[15]:


mse_loss = np.mean((X - X_approx) ** 2)
print("MSE loss between X and X_approx:", mse_loss)


# In[16]:


plt.plot(X_approx[500,:],label="POD ")
plt.plot(X[500,:],label="True")
plt.legend()
plt.xlabel('time')
plt.show()


# In[17]:


plt.plot(temporal_coeffs.T)
plt.legend(['1','2','3'])
plt.xlabel('time')
plt.title("POD - Temporal coefficients")
plt.show()


# ## Autoencoder ##

# In[18]:


# #normalize everything to [0,1]
# u=1
# l=0
# Xmax = np.max(X)
# Xmin = np.min(X)
# X = (X-Xmin)/(Xmax-Xmin)*(u-l)+l

n = X.shape[0]
m = X.shape[1]
print("Data matrix X is n by m:", n, "x", m)


# In[19]:


# Check if GPU can be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print('Running on GPU')
else: print('Running on CPU')


# In[20]:


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


# In[21]:


def plot_latent_space(latent, epoch):
    latent = latent.detach().cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.scatter(latent[:, 0], latent[:, 1], c=np.arange(len(latent)), marker='o', alpha=0.5)
    plt.title(f'Latent Space at Epoch {epoch}')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
#     plt.xlim(-2, 2)  # Adjust based on expected range
#     plt.ylim(-2, 2)  # Adjust based on expected range
    plt.grid(True)
    plt.show()


# ## Dual Autoencoder ##

# In[22]:


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


# In[23]:


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


# In[24]:


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


# In[25]:


class Autoencoder_Linear_Temporal(nn.Module):
    def __init__(self,m):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(m, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
        
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, m)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, encoded_input=False):
         if encoded_input:
             # If the input is already encoded, run it through the decoder only
            decoded = self.decoder(x)
            return decoded
         else:
            # Otherwise, run the input through the encoder and then the decoder
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded


# In[26]:

# DESMO-Autoencoder class
class SINDyAutoencoder(nn.Module):
    def __init__(self, n, m, polyorder,r):
        super(SINDyAutoencoder, self).__init__()
        
# 
#         self.phi1 = nn.Parameter(torch.ones(n))
#         self.phi2 = nn.Parameter(torch.ones(n))
        
        self.temporal_ae = Autoencoder_Linear_Temporal(m)
        
        
        # calculate number of polynomial terms for r variables of order polyorder
        num_terms = calculate_number_of_terms(r,polyorder)
        print('Number of terms in polynomial library:',num_terms)
        # create vector of optimizable coefficients
        self.c_coef = nn.Parameter(torch.ones(num_terms))

        
        self.z_list = nn.ParameterList([nn.Parameter(torch.ones(m)) for _ in range(num_terms)])
        
        
        
        self.zcos_coef_1 = nn.Parameter(torch.ones(m))  
        self.zcos_coef_2 = nn.Parameter(torch.ones(m))  
        self.zsin_coef_1 = nn.Parameter(torch.ones(m))  
        self.zsin_coef_2 = nn.Parameter(torch.ones(m))  
        
        self.ztanh_coef_1 = nn.Parameter(torch.ones(m))  
        self.ztanh_coef_2 = nn.Parameter(torch.ones(m))  

        
        self.cos_coef_1 = nn.Parameter(torch.tensor(1.0))  
        self.cos_coef_2 = nn.Parameter(torch.tensor(1.0))  
        self.sin_coef_1 = nn.Parameter(torch.tensor(1.0))  
        self.sin_coef_2 = nn.Parameter(torch.tensor(1.0))  
        self.tanh_coef_1 = nn.Parameter(torch.tensor(1.0))  
        self.tanh_coef_2 = nn.Parameter(torch.tensor(1.0))  

        
        self.omega_phi1 = nn.Parameter(torch.tensor(10000.0))  # Frequency for sin phi1
        self.omega_phi2 = nn.Parameter(torch.tensor(1000.0))  # Frequency for cos phi1
        self.omega_phi3 = nn.Parameter(torch.tensor(10000.0))  # Frequency for tanh phi1
        self.omega_phi4 = nn.Parameter(torch.tensor(1000.0))  # Frequency for sin phi2
        self.omega_phi5 = nn.Parameter(torch.tensor(100.0))  # Frequency for cos phi2
        self.omega_phi6 = nn.Parameter(torch.tensor(100.0))  # Frequency for tanh phi2
        

               
        
    def forward(self, X):

        
        latent_spatial, ae_rec = self.temporal_ae(X.T)  # phi components
        phi1 = latent_spatial[:, 0].unsqueeze(1)  # First latent spatial vector
        phi2 = latent_spatial[:, 1].unsqueeze(1)  # Second latent spatial vector
        latent_spatial = torch.stack([phi1,phi2],dim=1)
    

        # create candidate library for spatial modes
        theta_phi = self.c_coef * POOL_DATA(latent_spatial,r,polyorder)

        z_values = torch.stack([z for z in self.z_list],dim=0)
        


    
        sin_z1 = self.sin_coef_1 * self.zsin_coef_1.view(-1,1) @ torch.sin(self.omega_phi1 * phi1.T).view(1,-1)
        cos_z1 = self.cos_coef_1 * self.zcos_coef_1.view(-1,1) @ torch.cos(self.omega_phi2 * phi1.T).view(1,-1)
        sin_z2 = self.sin_coef_2 * self.zsin_coef_2.view(-1,1) @ torch.sin(self.omega_phi3 * phi2.T).view(1,-1)
        cos_z2 = self.cos_coef_2 * self.zcos_coef_2.view(-1,1) @ torch.cos(self.omega_phi4 * phi2.T).view(1,-1)
        tanh1 = self.tanh_coef_1 * self.ztanh_coef_1.view(-1,1) @ torch.tanh(self.omega_phi5 * phi1.T).view(1,-1)
        tanh2 = self.tanh_coef_2 * self.ztanh_coef_2.view(-1,1) @ torch.tanh(self.omega_phi6 * phi2.T).view(1,-1)
        
        fourier_contrib = sin_z1 + sin_z2 + cos_z1 + cos_z2# + tanh1 + tanh2


        final_reconstruction = theta_phi @ z_values + fourier_contrib.T #



        return final_reconstruction.T, latent_spatial, z_values, ae_rec.T


# In[27]:


# Initialize the dual autoencoder
polyorder = 2
r = 2
model_dualAE = SINDyAutoencoder(n, m, polyorder,r).to(device)


# In[28]:


# Define the names of parameters with different learning rates
f_g_param_names = [
    'c_coef',
    'sin_coef_1', 'tanh_coef_1', 'tanh_coef_2','ztanh_coef_1', 'ztanh_coef_2',
    'cos_coef_1','sin_coef_2','cos_coef_2', 'zsin_coef_1','zcos_coef_1','zsin_coef_2','zcos_coef_2',
    
]

omega_phi_param_names = [
    'omega_phi1', 'omega_phi2', 'omega_phi3', 'omega_phi4', 'omega_phi5', 'omega_phi6'
]
# Separate parameter groups
# Define the optimizer with separate parameter groups
optimizer = torch.optim.Adamax([
    # Custom learning rate for omega_phi parameters
    {'params': [param for name, param in model_dualAE.named_parameters() if name in omega_phi_param_names], 'lr': 1e2},
    
    # Custom learning rate for f and g coefficients (excluding omega_phi)
    {'params': [param for name, param in model_dualAE.named_parameters() 
                if name in f_g_param_names and name not in omega_phi_param_names], 'lr': 1e-2},
    
    # Default learning rate for all other parameters
    {'params': [param for name, param in model_dualAE.named_parameters() 
                if name not in f_g_param_names and name not in omega_phi_param_names]}
], lr=1e-2, weight_decay=0.0)

# Optionally, define a learning rate scheduler
# scheduler_temporal = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)
scheduler_temporal = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5000, factor=0.1, verbose=True,min_lr=1e-6)
criterion = nn.MSELoss()
# Model info
pytorch_total_params = sum(p.numel() for p in model_dualAE.parameters())
print("Total number of parameters in Dual AE:", pytorch_total_params)


# In[29]:


import random
def set_seed(seed):
    # Set seed for Python random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU environments
        
    # For determinism on GPU operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example: Set seed to 42
set_seed(43)


# In[30]:


# Training loop remains similar
num_epochs = 100000
beta = 1e-3 # orthogonal loss
l1_lambda = 1e-6 #sparsity loss
ae_beta = 1e-3
outputs_dual = []
loss_list_dual = []
for epoch in range(num_epochs):
    for x in data_loader:  # Iterate over batches for spatial data
        snapshot = x[0].type(torch.FloatTensor).to(device)

        recon_combined, latent_spatial, latent_temporal, ae_rec = model_dualAE(snapshot)

        ortho_loss_spatial = (criterion(latent_spatial[:,0] @ latent_spatial[:,1].T,torch.zeros(1).to(device)))

        # Compute loss (compare combined reconstruction with original data)
        loss = criterion(recon_combined, snapshot)
        ae_loss = criterion(ae_rec, snapshot)

        l1_loss = (torch.norm(model_dualAE.c_coef, p=1) + torch.norm(model_dualAE.cos_coef_1,p=1) + 
                  torch.norm(model_dualAE.cos_coef_2,p=1) + torch.norm(model_dualAE.sin_coef_1,p=1) + 
                  torch.norm(model_dualAE.sin_coef_2,p=1))

        total_loss = loss +  beta * (ortho_loss_spatial)+ l1_lambda*l1_loss + ae_beta * ae_loss
        
        # Plot mean flow (assuming X_mean is reshaped to (30, 30, 30))
        if epoch % 100 == 0:
            ttime = 300
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
                                 recon_combined.detach(), ae_rec.detach())) #, prim1.detach(),prim2.detach()
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Rec Loss: {loss.item():.8f}, Spatial ortho loss: {ortho_loss_spatial.item():.8f}, L1 loss: {l1_loss.item():.4f}, AE loss: {ae_loss.item():.8f} ')
    scheduler_temporal.step(total_loss)


# In[31]:


epochs_dual, losses_dual = zip(*loss_list_dual)

plt.plot(epochs_dual,losses_dual)
plt.xlabel("epochs")
plt.ylabel("rec loss")
plt.yscale('log') 
plt.show()


# In[32]:


torch.save(model_dualAE.state_dict(),'./models/DESMO_AE_cylinder_r2_100kepochs.pt')


# In[54]:


out_x = outputs_dual[0][1].detach().cpu().numpy()
latent_spatial = outputs_dual[0][2].detach().cpu().numpy()
latent_temporal = outputs_dual[0][3].detach().cpu().numpy()
recon_dual = outputs_dual[0][4].detach().cpu().numpy()
ae_recon =  outputs_dual[0][5].detach().cpu().numpy()


# In[55]:


ttime = 200
plotter2 = pv.Plotter(shape=(4, 2), window_size=(800, 800))

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
pv_mesh5.point_data["sinAE1"] = np.sin(model_dualAE.omega_phi1.detach().cpu().numpy() * latent_spatial[:, 0])  # Set AE mode 1 data
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
pv_mesh6.point_data["sinAE2"] = np.sin(model_dualAE.omega_phi3.detach().cpu().numpy() * latent_spatial[:, 1])  # Set AE mode 2 data
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


# Third subplot: AE mode 1
pv_mesh6 = pv.wrap(mesh)  # Create a new mesh for the third subplot
pv_mesh6.point_data["cosAE1"] = np.cos(model_dualAE.omega_phi2.detach().cpu().numpy() * latent_spatial[:, 0])  # Set AE mode 1 data
plotter2.subplot(3, 0)  # Position (1, 0)
plotter2.add_mesh(
    pv_mesh6,
    scalars="cosAE1",
    cmap="turbo",
    show_edges=False,
    scalar_bar_args={'title': 'cosAE1', 'n_labels': 2}
)
plotter2.add_title("cosAE mode1", font_size=10)
plotter2.view_xy()  # Top-down view

# Fourth subplot: AE mode 2
pv_mesh7 = pv.wrap(mesh)  # Create a new mesh for the fourth subplot
pv_mesh7.point_data["cosAE2"] = np.cos(model_dualAE.omega_phi4.detach().cpu().numpy() * latent_spatial[:, 1])  # Set AE mode 2 data
plotter2.subplot(3, 1)  # Position (1, 1)
plotter2.add_mesh(
    pv_mesh7,
    scalars="cosAE2",
    cmap="turbo",
    show_edges=False,
    scalar_bar_args={'title': 'cosAE2', 'n_labels': 2}
)
plotter2.add_title("cosAE mode2", font_size=10)
plotter2.view_xy()  # Top-down view


# Show the plot
plotter2.show()


plotter = pv.Plotter(shape=(1, 2), window_size=(600, 300))
ttime = 200
pv_mesh1 = pv.wrap(mesh)
pv_mesh2 = pv.wrap(mesh)
pv_mesh1.point_data["velocity"] = X[:,ttime]  # Set mean flow data
pv_mesh2.point_data["AE approx"] = ae_recon[ttime,:]  # Set mean flow data
plotter.subplot(0, 0)  # Position (0, 0)
plotter.add_mesh(pv_mesh1, scalars="velocity", cmap="turbo", show_edges=False)
plotter.add_title("True", font_size=10)
plotter.view_xy()  # Top-down view
                  
plotter.subplot(0, 1)  # Position (0, 0)
plotter.add_mesh(pv_mesh2, scalars="AE approx", cmap="turbo", show_edges=False)
plotter.add_title("AE Predicted", font_size=10)
plotter.view_xy()  # Top-down view

plotter.show()


# In[62]:


plt.plot(latent_temporal.T)
# plt.plot(Vt[0:1,:].T)
# plt.plot(Vt[1:2,:].T)
# plt.legend(["AE1","AE2","POD1","POD2"])
plt.show()


# In[39]:


err_AEDual  = np.linalg.norm(X-recon_dual.T)/np.linalg.norm(X)
print("AEDual error with ",r," modes:",err_AEDual)


# In[40]:


U, S, Vt = np.linalg.svd(X, full_matrices=False)
k = 3  # For example, keep the first 10 modes
POD_modes = U[:, :k]  # POD modes (spatial modes)
temporal_coeffs = Vt[:k, :]  # Temporal coefficients

# If needed, reconstruct the original matrix using k modes:
X_approx = POD_modes @ np.diag(S[:k]) @ Vt[:k, :]

err_POD  = np.linalg.norm(X-X_approx)/np.linalg.norm(X)
print("POD error with ",k," modes:",err_POD)


# In[42]:


plotter = pv.Plotter(shape=(1, 2), window_size=(600, 300))

# Plot mean flow
ttime = 200
pv_mesh1 = pv.wrap(mesh)
pv_mesh2 = pv.wrap(mesh)
pv_mesh1.point_data["velocity"] = X[:,ttime]  # Set mean flow data
pv_mesh2.point_data["POD approx"] = X_approx.T[ttime,:]  # Set mean flow data
plotter.subplot(0, 0)  # Position (0, 0)
plotter.add_mesh(pv_mesh1, scalars="velocity", cmap="turbo", show_edges=False)
plotter.add_title("True", font_size=10)
plotter.view_xy()  # Top-down view
                  
plotter.subplot(0, 1)  # Position (0, 0)
plotter.add_mesh(pv_mesh2, scalars="POD approx", cmap="turbo", show_edges=False)
plotter.add_title("Predicted", font_size=10)
plotter.view_xy()  # Top-down view


# Show the plot
plotter.show()


# ## Threshold coefficient ##

# In[231]:


# After training is complete
threshold = 2e-4
copy_coeffs = model_dualAE.c_coef.clone()
print(copy_coeffs)
with torch.no_grad():
    model_dualAE.c_coef.data[torch.abs(model_dualAE.c_coef.data) < threshold] = 0
    model_dualAE.cos_coef_1.data[torch.abs(model_dualAE.cos_coef_1.data) < threshold] = 0
    model_dualAE.cos_coef_2.data[torch.abs(model_dualAE.cos_coef_2.data) < threshold] = 0
    model_dualAE.sin_coef_1.data[torch.abs(model_dualAE.sin_coef_1.data) < threshold] = 0
    model_dualAE.sin_coef_2.data[torch.abs(model_dualAE.sin_coef_2.data) < threshold] = 0
print("Updated coefficients in c_coef with small values set to zero.")

# Set the model to inference mode
model_dualAE.eval()
print("Model is set to inference mode.")

print(model_dualAE.c_coef[model_dualAE.c_coef != 0])


# In[46]:


outputs_dual = []
for x in data_loader:  # Iterate over batches for spatial data
    snapshot = x[0].type(torch.FloatTensor).to(device)
#         snapshot_noisy = snapshot + torch.randn_like(snapshot) * noise_factor


    recon_combined, latent_spatial, latent_temporal, AE_rec = model_dualAE(snapshot)


    # Plot mean flow (assuming X_mean is reshaped to (30, 30, 30))
    ttime = 200
    plotter = pv.Plotter(shape=(1, 4), window_size=(900, 300))
    pv_mesh_d = pv.wrap(mesh)
    pv_mesh_DESMO = pv.wrap(mesh)
    pv_mesh_POD = pv.wrap(mesh)
    pv_mesh_AE = pv.wrap(mesh)
    pv_mesh_d.point_data["velocity"] = snapshot[ttime,:].detach().cpu().numpy()  # Set mean flow data
    pv_mesh_DESMO.point_data["DESMO approx"] = recon_combined[ttime,:].detach().cpu().numpy()  # Set mean flow data
    pv_mesh_POD.point_data["POD approx"] = X_approx[:,ttime]
    pv_mesh_AE.point_data["AE approx"] = AE_rec[ttime,:].detach().cpu().numpy() 
    plotter.subplot(0, 0)  # Position (0, 0)
    plotter.add_mesh(pv_mesh_d, scalars="velocity", cmap="turbo", show_edges=False,scalar_bar_args={'n_labels': 2})
    plotter.add_title("True", font_size=10)
    plotter.view_xy()  # Top-down view

    plotter.subplot(0, 1)  # Position (0, 0)
    plotter.add_mesh(pv_mesh_AE, scalars="AE approx", cmap="turbo", show_edges=False,scalar_bar_args={'n_labels': 2})
    plotter.add_title("AE Predicted", font_size=10)
    plotter.view_xy()  # Top-down view


    plotter.subplot(0, 2)  # Position (0, 0)
    plotter.add_mesh(pv_mesh_POD, scalars="POD approx", cmap="turbo", show_edges=False,scalar_bar_args={'n_labels': 2})
    plotter.add_title("POD Predicted", font_size=10)
    plotter.view_xy()  # Top-down view

    plotter.subplot(0, 3)  # Position (0, 0)
    plotter.add_mesh(pv_mesh_DESMO, scalars="DESMO approx", cmap="turbo", show_edges=False,scalar_bar_args={'n_labels': 2})
    plotter.add_title("DESMO Predicted", font_size=10)
    plotter.view_xy()  # Top-down view
    
    # Show the plot
    plotter.show()
    plotter.close()

    outputs_dual = []
    outputs_dual.append((epoch+1, snapshot, latent_spatial.detach(), latent_temporal.detach(), 
                             recon_combined.detach()))


# In[236]:


out_x = outputs_dual[0][1].detach().cpu().numpy()
latent_spatial = outputs_dual[0][2].detach().cpu().numpy()
latent_temporal = outputs_dual[0][3].detach().cpu().numpy()
recon_dual = outputs_dual[0][4].detach().cpu().numpy()


# In[237]:


err_AEDual  = np.linalg.norm(X-recon_dual.T)/np.linalg.norm(X)
print("AEDual error with ",r," modes:",err_AEDual)


# In[238]:


print(model_dualAE.c_coef[model_dualAE.c_coef != 0])


# In[239]:


ttime = 200
plotter2 = pv.Plotter(shape=(4, 2), window_size=(800, 800))

# First subplot: True Velocity
pv_mesh1 = pv.wrap(mesh)  # Create a new mesh for the first subplot
pv_mesh1.point_data["velocity1"] = out_x[ttime, :]  # Set true velocity data
plotter2.subplot(0, 0)  # Position (0, 0)
plotter2.add_mesh(
    pv_mesh1,
    scalars="velocity1",
    cmap="viridis",
    show_edges=False,
    scalar_bar_args={'title': 'Velocity', 'n_labels': 2}
)
plotter2.add_title("True")
plotter2.view_xy()  # Top-down view

# Second subplot: AE approximation
pv_mesh2 = pv.wrap(mesh)  # Create a new mesh for the second subplot
pv_mesh2.point_data["AE approx"] = recon_dual[ttime, :]  # Set AE approximation data
plotter2.subplot(0, 1)  # Position (0, 1)
plotter2.add_mesh(
    pv_mesh2,
    scalars="AE approx",
    cmap="viridis",
    show_edges=False,
    scalar_bar_args={'title': 'AE rec', 'n_labels': 2}
)
plotter2.add_title("Predicted")
plotter2.view_xy()  # Top-down view

# Third subplot: AE mode 1
pv_mesh3 = pv.wrap(mesh)  # Create a new mesh for the third subplot
pv_mesh3.point_data["AE1"] = latent_spatial[:, 0]  # Set AE mode 1 data
plotter2.subplot(1, 0)  # Position (1, 0)
plotter2.add_mesh(
    pv_mesh3,
    scalars="AE1",
    cmap="viridis",
    show_edges=False,
    scalar_bar_args={'title': 'AE1', 'n_labels': 2}
)
plotter2.add_title("AE mode1")
plotter2.view_xy()  # Top-down view

# Fourth subplot: AE mode 2
pv_mesh4 = pv.wrap(mesh)  # Create a new mesh for the fourth subplot
pv_mesh4.point_data["AE2"] = latent_spatial[:, 1]  # Set AE mode 2 data
plotter2.subplot(1, 1)  # Position (1, 1)
plotter2.add_mesh(
    pv_mesh4,
    scalars="AE2",
    cmap="viridis",
    show_edges=False,
    scalar_bar_args={'title': 'AE2', 'n_labels': 2}
)
plotter2.add_title("AE mode2")
plotter2.view_xy()  # Top-down view


# Third subplot: AE mode 1
pv_mesh5 = pv.wrap(mesh)  # Create a new mesh for the third subplot
pv_mesh5.point_data["sinAE1"] = np.sin(model_dualAE.omega_phi1.detach().cpu().numpy() * latent_spatial[:, 0])  # Set AE mode 1 data
plotter2.subplot(2, 0)  # Position (1, 0)
plotter2.add_mesh(
    pv_mesh5,
    scalars="sinAE1",
    cmap="viridis",
    show_edges=False,
    scalar_bar_args={'title': 'sinAE1', 'n_labels': 2}
)
plotter2.add_title("sinAE mode1")
plotter2.view_xy()  # Top-down view

# Fourth subplot: AE mode 2
pv_mesh6 = pv.wrap(mesh)  # Create a new mesh for the fourth subplot
pv_mesh6.point_data["sinAE2"] = np.sin(model_dualAE.omega_phi3.detach().cpu().numpy() * latent_spatial[:, 1])  # Set AE mode 2 data
plotter2.subplot(2, 1)  # Position (1, 1)
plotter2.add_mesh(
    pv_mesh6,
    scalars="sinAE2",
    cmap="viridis",
    show_edges=False,
    scalar_bar_args={'title': 'sinAE2', 'n_labels': 2}
)

plotter2.add_title("sinAE mode2")
plotter2.view_xy()  # Top-down view

# Third subplot: AE mode 1
pv_mesh7 = pv.wrap(mesh)  # Create a new mesh for the third subplot
# pv_mesh7.point_data["cosAE1"] = np.cos(model_dualAE.omega_phi2.detach().cpu().numpy() * latent_spatial[:, 0])  # Set AE mode 1 data
pv_mesh7.point_data["sinPOD1"] = np.sin(model_dualAE.omega_phi1.detach().cpu().numpy() * POD_modes[:,0])
plotter2.subplot(3, 0)  # Position (1, 0)
plotter2.add_mesh(
    pv_mesh7,
    scalars="sinPOD1",
    cmap="viridis",
    show_edges=False,
    scalar_bar_args={'title': 'sinPOD1', 'n_labels': 2}
)
plotter2.add_title("sin POD mode1", font_size=10)
plotter2.view_xy()  # Top-down view

# Fourth subplot: AE mode 2
pv_mesh8 = pv.wrap(mesh)  # Create a new mesh for the fourth subplot
# pv_mesh8.point_data["cosAE2"] = np.cos(model_dualAE.omega_phi4.detach().cpu().numpy() * latent_spatial[:, 1])  # Set AE mode 2 data
pv_mesh8.point_data["sinPOD2"] = np.sin(model_dualAE.omega_phi3.detach().cpu().numpy() * POD_modes[:, 1])  # 
plotter2.subplot(3, 1)  # Position (1, 1)
plotter2.add_mesh(
    pv_mesh8,
    scalars="sinPOD2",
    cmap="viridis",
    show_edges=False,
    scalar_bar_args={'title': 'sinPOD2', 'n_labels': 2}
)

plotter2.add_title("sin POD mode2", font_size=10)
plotter2.view_xy()  # Top-down view

# Show the plot
plotter2.show()


# In[240]:


plt.plot(model_dualAE.zsin_coef_1.detach().cpu().numpy())
plt.plot(model_dualAE.zsin_coef_2.detach().cpu().numpy())
plt.legend(["z_sin1", "z_sin2"])
plt.show()


# In[241]:


# plt.plot(latent_temporal.detach().cpu().numpy().T[:,1:3])
plt.plot(Vt[2:4,:].T)
# plt.plot(Vt[1:2,:].T)
plt.legend(["POD3","POD4"])
plt.show()


# In[573]:


plt.plot(latent_temporal.T[:,1:3])
# plt.plot(Vt[0:1,:].T)
# plt.plot(Vt[1:2,:].T)
plt.legend(["z1","z2"])
plt.xlabel('time')
plt.ylabel('z')
plt.show()


# In[ ]:





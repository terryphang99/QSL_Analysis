# Basic imports
import numpy as np
import pandas as pd

# Data Formatting
import xarray as xr
import netCDF4 as nc4
from pathlib import Path

#Plotting
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter

# Machine Learning and Data Analysis
import pyUSID as usid
import pycroscopy as px
from pycroscopy.viz import cluster_utils
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import subprocess
import sys

#------------------------------------------------------------------------------------------#
# Uses xarray to get B and k values
def xarr(path):
    rawD = xr.open_dataset(path);
    B = xr.open_dataset(path).coords['B'];
    k = xr.open_dataset(path).coords['k'];
    D = xr.open_dataset(path).to_array()[0];
    return rawD,B,k,D


# Processes data by removing text column and converting to numpy. Returns (0) df and (1) Y
def dp(data):
    df = data;
    df.columns.values[0] = 'IndexField'
    Y = df.drop(columns = ['IndexField']).iloc[:,:1601].to_numpy()
    return df,Y

# Transposes and truncates input data by wavenumber. Returns (0) Y
def tt(data,lower_bound_wavenumber,upper_bound_wavenumber):
    df = data.iloc[0:][::1].T.iloc[1:]#.astype(float)
    df.index = df.index.astype(float)
    df = df.loc[240:350]
    Y = df.T;
    return Y


#--------------------------------------------------------------------------------------#
# Performs NMF and returns (0) Y, (1) W, (2) H, (3) froY, (4) error, (5) relerr
def nmf(path, num_comps, a = 0, bl = 'frobenius', i = 'nndsvd', 
        l1 = 0, mi = 10000, rs = 0, reg = 'both', s = 'False', sol = 'cd',
        t = 0.001, ver = 0):
    Y = xarr(path)[3];
    
    if np.min(Y) < 0:
        Y -= np.min(Y);
        
    model = NMF(alpha = a, 
                beta_loss = bl, 
                init= i, 
                l1_ratio = l1,
                max_iter = mi,
                n_components= num_comps, 
                random_state= rs,
                regularization = reg,
                shuffle = s,
                solver = sol,
                tol = t,
                verbose = ver
               );
    
    W = model.fit_transform(Y)
    H = model.components_
    froY = np.linalg.norm(Y, 'fro');
    error = model.reconstruction_err_;
    relerr = error/froY;
#    print("Parameters =", model.get_params(W.all()));
    return Y,W,H,froY,error,relerr

# Performs NMF with transposed data W and H and returns (0) Y, (1) W, (2) H, (3) froY, (4) error, (5) relerr
def nmfT(data, num_comps, a = 0, bl = 'frobenius', i = 'nndsvd', 
        l1 = 0, mi = 10000, rs = 0, reg = 'both', s = 'False', sol = 'cd',
        t = 0.001, ver = 0):
    Y = dp(data)[1].T;
    
    if np.min(Y) < 0:
        Y -= np.min(Y);
        
    model = NMF(alpha = a, 
                beta_loss = bl, 
                init= i, 
                l1_ratio = l1,
                max_iter = mi,
                n_components= num_comps, 
                random_state= rs,
                regularization = reg,
                shuffle = s,
                solver = sol,
                tol = t,
                verbose = ver
               );
    
    W = model.fit_transform(Y).T
    H = model.components_.T
    froY = np.linalg.norm(Y, 'fro');
    error = model.reconstruction_err_;
    relerr = error/froY;
#    print("Parameters =", model.get_params(W.all()));
    return Y,W,H,froY,error,relerr

# Returns all stats
def stats(path,num_comps):
    print("Matrices:")
    print("Original Spectrum = ",nmf(path,num_comps)[0]);
    print("Components = ",nmf(path,num_comps)[2]);
    print("Weights =", nmf(path,num_comps)[1]);
    print("Approximated Spectrum =", np.matmul(nmf(path,num_comps)[1],nmf(path,num_comps)[2]));
    print("----------------------------------------------------------");
    print("Error:");
    print("Frobenius norm of original spectrum =",nmf(path,num_comps)[3]);
    print("Reconstruction error =", nmf(path,num_comps)[4]);
    print("Relative error =", nmf(path,num_comps)[5]);
    
    
    
#-----------------------------------------------------------------------------------------#
# Plots the Components vs. Wavenumber
def comps(path,num_comps):
    # df = xarr(path)[0];
    W = nmf(path,num_comps)[1];
    H = nmf(path,num_comps)[2];
    plt.figure();
    for i in range(0,W.shape[1]):
        plt.plot(xarr(path)[2], H[i],'-',label = 'component {}'.format(i));
        plt.xlabel('Raman Shift (cm$^{-1}$)');
        plt.ylabel('Intensity')
        # plt.xlim([0,1000])
        # plt.xticks([]);
        plt.legend(frameon = True);
    plt.show();
    
# Plots the weights vs. Field
def compwt(path, num_comps):
    W = nmf(path,num_comps)[1];
    plt.figure();
    for i in range(0,W.shape[1]):
        plt.plot(xarr(path)[1], W.T[i],'o-',label = 'component {}'.format(i));
        plt.xlabel('B(T)');
        plt.ylabel('Component Weight');
        plt.title('{} components'.format(num_comps))
        # plt.xticks([]);
        plt.legend(frameon = True, bbox_to_anchor=(1.05, 1), loc='upper left');
    plt.show();

# Plots relative error and returns (0) arrrelerr, (1) avgerr
def relerr(path,maxcomps):
    compguesses = list(range(1,maxcomps+1));
    arrrelerr = [];
    for i in range(0,len(compguesses)):
        relerr = nmf(path,compguesses[i])[5];
        arrrelerr.append(relerr);
#     avgerr = np.average(arrrelerr)
    plt.figure();
    plt.plot(range(1,maxcomps+1),arrrelerr,'o-');
#     plt.plot(range(1,maxcomps+1),[avgerr]*(maxcomps),'-')
    plt.xlabel('# of Components Used');
    plt.ylabel('Relative Error');
    plt.title('Relative Error vs. # Components')
    plt.show()
    return arrrelerr        
    


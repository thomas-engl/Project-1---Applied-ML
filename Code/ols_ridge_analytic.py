# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 20:20:12 2025

@author: Philipp Brückelt
"""

""" Part a) and b) """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.colors as colors


### random seed
np.random.seed(123)

# Runge function
def runge(x): return 1 / (1 + 25*x**2)


# create feature matrix 

def feature_matrix(x, deg):
    mat = np.zeros((len(x), deg + 1))
    mat[:, 0] = 1
    for i in range(1, deg + 1):
        mat[:, i] = x**i
    return mat

def variance_array(arr):
    n = len(arr)
    m = np.mean(arr)
    return 1/(n-1) * np.sum((arr - m)**2)

def get_scaling(matrix):
    n, p = np.shape(matrix)
    mean = np.array([np.mean(matrix[:,i]) for i in range(p)])
    std = np.array([np.sqrt(variance_array(matrix[:,i])) for i in range(p)])
    std[std==0] = 1
    return mean, std

def get_scaled_data(x, y, deg):
    X = feature_matrix(x, deg)
    # train-test split
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y)
    # scale data
    mean, std = get_scaling(X_train_)
    # for every column, subtract the mean and divide by the standard deviation
    # finally, remove the first column which is just 0
    X_train = ((X_train_ - mean) / std)[:,1:]
    X_test = ((X_test_ - mean) / std)[:,1:]
    y_m = y_train_.mean()
    y_train = y_train_ - y_m
    y_test = y_test_ - y_m
    # we also return y_m to be able to rescale the data later
    return X_train, X_test, y_train, y_test, y_m

# ordinary least squares

def ols(feature_matrix, y):
    ols_mat = np.transpose(feature_matrix) @ feature_matrix
    # moore-penrose pseudo inverse
    mp = np.linalg.pinv(ols_mat)
    return mp @ np.dot(np.transpose(feature_matrix), y)

# Ridge regression

def ridge(feature_matrix, y, hyper_param):
    _, p = np.shape(feature_matrix)
    m = np.linalg.pinv(np.transpose(feature_matrix) @ feature_matrix + 
                       hyper_param * np.eye(p))
    return m @ np.dot(np.transpose(feature_matrix), y)

# mean squared error

def mse(y_est, y_true):
    return 1 / len(y_est) * np.sum((y_est - y_true)**2)

# R^2 score

def r2(y_est, y_true):
    y_mean = 1/len(y_true) * np.sum(y_true)
    denominator = np.sum((y_true - y_mean)**2)
    nominator = np.sum((y_true - y_est)**2)
    return 1 - nominator / denominator 
    
# plot results

def plot_results(x_values, results_ols, results_ridge, variances, markers, 
                 quantity='MSE'):
    """ plots results for OLS and Ridge for different variances.
    The order of the entries of results_ols and results_ridge must be adapted
    to the order of the variances
    markers is a list of the markers used in the plots
    quantity is either 'MSE' or 'R2'
    """
    plt.figure(dpi=150)
    for i in range(len(results_ols)):
        plt.plot(x_values, results_ols[i], marker=markers[i], color='red',
                 label=r'OLS, $\sigma^2 = {}$'.format(variances[i]))
        plt.plot(x_values, results_ridge[i], marker=markers[i], color='blue',
                 label=r'Ridge, $\sigma^2 = {}$'.format(variances[i]))
    # place the legend in the top left corner since the values there are not
    # as interesting as on the right hand side
    plt.legend(loc='upper left')
    plt.grid()
    plt.xlabel("polynomial degree")
    plt.ylabel('MSE' if quantity=='MSE' else r'$R^2$ score')
    
### create heat map for ridge regression    

def heat_map_ridge(params, degs, values, quantity='MSE'):
    # quantity is again either 'MSE' or 'R2'
    k, l = len(params), len(degs)
    z = np.array(values).reshape((k, l))
    fig, ax = plt.subplots(dpi=200)
    im = ax.imshow(z, cmap=cm.jet)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(l), labels=degs)
    y_labels = np.array(['10^{}'.format(int(i)) for i in np.log10(params)])
    ax.set_yticks(range(k), labels=y_labels)
    ax.set_ylabel(r'$\lambda$')
    ax.set_xlabel('polynomial degree')
    # make colorbar fit to size of the heat map
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    fig.tight_layout()
    plt.show()

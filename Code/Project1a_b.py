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

def scale_matrix(matrix):
    n, p = np.shape(matrix)
    mean = np.array([np.mean(matrix[:,i]) for i in range(p)])
    std = np.array([np.sqrt(variance_array(matrix[:,i])) for i in range(p)])
    std[std==0] = 1
    # for every column, subtract the mean and divide by the standard deviation
    # finally, remove the first column which is just 0
    return ((matrix - mean) / std)[:,1:]

def get_scaled_data(x, y, deg):
    X = feature_matrix(x, deg)
    # scale data
    X_norm = scale_matrix(X)
    y_centered = y - np.mean(y)
    # return feature matrix and targets splitted in test and train data
    return train_test_split(X_norm, y_centered)

"""
def get_scaled_data(x, y, deg):
    X = feature_matrix(x, y, deg)
    # scale data
    y_centered = y - np.mean(y)
    X_norm = 
"""

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
def plot_results(x_values, results_ols, results_ridge, quantity='MSE'):
    """ plots results for OLS and Ridge, quantity is either 'MSE' or 'R2' """
    plt.figure(dpi=150)
    plt.plot(x_values, results_ols, '-D', color='red', label='OLS')
    plt.plot(x_values, results_ridge, '-D', color='blue', label='Ridge')
    plt.legend()
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
    
def polynomial(coefficients, x, intercept=True):
    if intercept:
        return sum(coefficients[i] * x**i for i in range(len(coefficients)))
    else:
        return sum(coefficients[i] * x**(i+1) for i in range(len(coefficients)))

# vectorize the function
polynomial_v = np.vectorize(polynomial, excluded={0, 2})


# In[1]:

"""
Numerical Experiment 1
===============================================================================
compute MSE and R^2 score for OLS and Ridge regression using polynomials of 
degree up to 15
"""

num_pts = 1000
# uniform distributed random data points
x_pts = np.random.uniform(-1, 1, num_pts)
# noisy data, normal distributed with mean 0 and standard deviation 0.1
y_pts = runge(x_pts) + np.random.normal(0, 0.1, num_pts)
max_polynomial_deg = 15
degs = np.arange(1, max_polynomial_deg + 1)
lam = 0.1                                           # hyper paramter for Ridge
mses_ols = []
mses_ridge = []
r2_ols = []
r2_ridge = []
betas_ols = []

for deg in degs:
    X_train, X_test, y_train, y_test = get_scaled_data(x_pts, y_pts, deg)
    # compute approximation using OLS
    beta_ols = ols(X_train, y_train)
    betas_ols.append(beta_ols)
    y_predict = X_test @ beta_ols
    mses_ols.append(mse(y_predict, y_test))
    r2_ols.append(r2(y_predict, y_test))
    # compute approximation using Ridge regression
    beta_ridge = ridge(X_train, y_train, lam)
    y_predict_ridge = X_test @ beta_ridge
    mses_ridge.append(mse(y_predict_ridge, y_test))
    r2_ridge.append(r2(y_predict_ridge, y_test))

""" Plot the data """

# figure for MSEs
plot_results(degs, np.array(mses_ols), np.array(mses_ridge))
# figures for R^2 scores
plot_results(degs, np.array(r2_ols), np.array(r2_ridge), quantity='R2')

""" Plot coefficients """

plt.figure(dpi=150)
colors = ['blue', 'red', 'green']
for (i, beta) in enumerate(betas_ols[4::5]):
    markerline, stemlines, baseline = plt.stem(
        np.arange(1, len(beta) + 1) + i/4, np.array(beta), linefmt=colors[i], 
        bottom=0, label=r'$\deg = {}$'.format(5*(i+1)))
    markerline.set_markerfacecolor(colors[i])
    # hide baseline in the for loop and create one common base line later
    baseline.set_visible(False)

# maximal coefficient - used for y-limit
m = np.max(np.array([np.max(np.abs(i)) for i in betas_ols[4::5]]))

plt.axhline(0, color="grey", linestyle="--", linewidth=1)
plt.grid()
plt.xlabel(r'$i$')
plt.xticks(np.arange(1, 16))
plt.ylabel(r'$\beta_i$')
plt.yscale('symlog', linthresh=1)
plt.ylim(-1.5*m, 1.5*m)
plt.legend()
plt.show()

""" Plot the approximating polynomial for deg 5, 10, 15 """

plt.figure(dpi=150)
x = np.linspace(-1, 1, 101)
for i, beta in enumerate(betas_ols[4::5]):
    plt.plot(x, polynomial_v(beta, x, intercept=False), 
             label=r'$\deg = {}$'.format(5*(i+1)))
# plt.plot(x, runge(x) - np.mean(runge(x)), label=r'$f(x)$')
plt.grid()
plt.legend()
plt.show()


# In[2]:

"""
Numerical Experiment 2
===============================================================================
MSE and R^2 score for Ridge using different hyper parameters
"""

mses = []
r2_scores = []
lambdas = np.logspace(-5, 0, 6)

for param in lambdas:
    lst_mse = []
    lst_r2 = []
    for deg in degs:
        X_train, X_test, y_train, y_test = get_scaled_data(x_pts, y_pts, deg)
        beta = ridge(X_train, y_train, param)
        y_predict = X_test @ beta
        lst_mse.append(mse(y_predict, y_test))
        lst_r2.append(r2(y_predict, y_test))
    mses.append(lst_mse)
    r2_scores.append(lst_r2)
    
""" Plot the MSE and R^2 score for Ridge regression using different hyper
parameters """
    
heat_map_ridge(lambdas, degs, mses)
heat_map_ridge(lambdas, degs, r2_scores, quantity='R2')

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


# Runge function
def runge(x): return 1 / (1 + 25*x**2)


# create feature matrix 

def feature_matrix(x, y, deg):
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
    return (matrix - mean) / std

def get_scaled_data(x, y, deg):
    X = feature_matrix(x_pts, y_pts, deg)
    # scale data
    X_norm = scale_matrix(X)
    y_centered = y_pts - np.mean(y_pts)
    # return feature matrix and targets splitted in test and train data
    return train_test_split(X_norm, y_centered)

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
    plt.grid()
    plt.xlabel("polynomial degree")
    plt.ylabel('MSE' if quantity=='MSE' else r'$R^2$ score')
    plt.legend()
    plt.show()
    
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
    
def polynomial(coefficients, x):
    return sum(coefficients[i] * x**i for i in range(len(coefficients)))

# vectorize the function
polynomial_v = np.vectorize(polynomial, excluded={0, 'coefficients'})

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
for beta in betas_ols:
    plt.scatter(np.arange(len(beta)), np.array(beta))
plt.xlabel(r'$i$')
plt.ylabel(r'$\beta_i$')
plt.show()

""" Plot the approximating polynomial for deg 5, 10, 15 """

plt.figure(dpi=150)
x = np.linspace(-1, 1, 101)
for i, beta in enumerate(betas_ols[4::5]):
    plt.plot(x, polynomial_v(beta, x), label=r'$\deg = {}$'.format(5*(i+1)))
# plt.plot(x, runge(x), label=r'$f(x)$')
plt.grid()
plt.legend()
plt.show()


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

# Created on Tue Jul 25 11:20 2023

# Author: Chenyang Wang (16chenyang.wang@tum.de)
#         Baha Zarrouki (baha.zarrouki@tum.de)            
import itertools
import numpy as np
from casadi import *
import math
import chaospy
import copy

"""
Source: [1] Zarrouki, Baha, Chenyang Wang, and Johannes Betz. 
        "A stochastic nonlinear model predictive control with an uncertainty propagation horizon 
        for autonomous vehicle motion control." arXiv preprint arXiv:2310.18753 (2023).
"""
def hermiteGeneration(x,n):
    # generate hermite polynomials as orthogonal polynomial basis
    # represent \phi_{\alpha_{k}}(w) in the line under Eq.12 in [1]
    if n == 0:
        return (1.0 + 0.0 * x) / math.sqrt(math.factorial(n))
    elif n == 1:
        return x / math.sqrt(math.factorial(n))
    else:
        return (x * hermiteGeneration(x, n-1) - (n - 1) * hermiteGeneration(x, n-2)) / math.sqrt(math.factorial(n))
    
def alphaGeneration(n_rand, n):
    # generate alphas for multivariate hermite polynomials
    # each single alpha represents \alpha_{i,k} in the line above Eq.2 in [1]
    # alphas represent the matrix of single alpha, represent the subscript character \alpha_k of \mathbf{\phi} in eq.2

    # generate alphas whose sum is less or equal than degree n
    alphas = np.array([i for i in itertools.product(range(0, n + 1), repeat = n_rand)])
    alphas = alphas[np.sum(alphas,axis=1) <= n]
    # sort alphas based upon their row sums
    alphas = alphas[np.argsort(alphas.sum(axis=1))[::-1],:]
    alphas = alphas[::-1]
    return alphas

def polyChaosExpansion(x,alphas,disturbance_type):

    # The result represents \boldsymbol{\Phi}(\boldsymbol{\Tilde{w}}^{( i)}) in Eq.12 [1]
    # x : input vector of size 8 * 1
    # n_rand: number of random variables
    # n: degree of Expansion Polynomial
    row, col = alphas.shape
    result = []
    if disturbance_type == 'gaussian':
        for i in range(row):
            tmp = 1
            for j in range(col):
                tmp *= hermiteGeneration(x[j], alphas[i][j])
            result.append(tmp)
        return np.array(result)
       
def computeSamplesAndAmatrix(n_samples,alphas,num_poly_terms,x0_std):
    active_index = np.nonzero(x0_std)[0]
    n_vars = len(active_index)
    distribution = [chaospy.Normal(0, 1) for i in range(n_vars)]
    # define external disturbances
    w_distribution = chaospy.J(*distribution)
    # sample external disturbances
    w_samples = w_distribution.sample(n_samples,rule = 'hammersley')
    # initialize Phi_tilde and initial state samples
    # Phi_tilde is \boldsymbol{\tilde{\Phi}} in Eq.11 in [1]
    Phi_tilde = np.zeros((n_samples,num_poly_terms))
    for index in range(0,n_samples):
        w_sample = w_samples[:,index]
        # pce:  \boldsymbol{\Phi}(\boldsymbol{\Tilde{w}}^{( i)}) in Eq.12 [1]
        pce = polyChaosExpansion(w_sample,alphas,'gaussian')
        Phi_tilde[index,:] = pce.T
    # PCE matrix is A in Eq.8 in [1]
    # Here, we use simple least square method not weighted least square
    A = np.linalg.inv(Phi_tilde.T @ Phi_tilde) @ Phi_tilde.T

    return w_samples, A

def compute_x0dist(x0,w_samples,n_samples,x0_std):
    active_index = np.nonzero(x0_std)[0]
    n_vars = len(active_index)
    x0_samples = np.zeros((n_samples+1,8))
    for index in range(0,n_samples):
        w_sample = w_samples[:,index]
        # post process of sampled disturbances and add them to initial state to get one sample of x0 
        w_sample = np.array([x0_std[i] for i in active_index]) * w_sample
        x = copy.copy(x0)
        for i in range(n_vars):
            x[active_index[i]] += w_sample[i]
        x0_samples[index+1,:] = x
    x0_samples[0,:] = x0
    return x0_samples
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import invwishart
from scipy.linalg import inv


def bayes_ml(Y, Z, bandwidths_list, tau):
    
    # number of iterations
    n_iterations = len(bandwidths_list)
    # sample size
    n = Y.shape[0]
    # dimension of the ambient space
    D = Y.shape[1]
    
    # pairwise distances
    dist = pairwise_distances(Y)
    
    if Z is not None:
        print('before first iteration:', np.linalg.norm(Y - Z, ord='fro')**2 / n)
    
    for k in range(n_iterations):
        
        # bandwidth
        h = bandwidths_list[k]
        
        # compute weights and sum of weights
        if (k==0):
            W = np.exp(-dist**2/h**2) * (dist < tau)
            N = np.sum(W, axis=1)
        else:
            W = np.exp(-a_dist**2 * h**2) * (dist < tau)
            N = np.sum(W, axis=1)
        
        # compute adjusted distances
        a_dist = np.empty((n, n))
        for i in range(n):
            Y_diff = Y - Y[i, :]   #(n, D)
            
            # compute the weighted sample covariance
            Sigma = np.cov(Y_diff.T, aweights=W[i])
            # sample a precision matrix from the inverse Wishart distribution

            Omega = invwishart.rvs(N[i] + D + 2, inv(h**2 * np.identity(D) + Sigma), size=1)

            Q = Y_diff @ Omega 
            a_dist[i,:] = np.linalg.norm(Q, axis = 1)
            
        # adjusted Nadaraya-Watson estimate

        X = W.dot(Y) / np.tile(N.reshape(-1, 1), (1, D))
        
        # compare with the true values
        if Z is not None:
            print('iteration', k+1, ':', np.linalg.norm(X - Z, ord='fro')**2 / n)
                
    return X

def not_bayes_ml(Y, Z, bandwidths_list, tau):
    
    # number of iterations
    n_iterations = len(bandwidths_list)
    # sample size
    n = Y.shape[0]
    # dimension of the ambient space
    D = Y.shape[1]
    
    # pairwise distances
    dist = pairwise_distances(Y)
    
    if Z is not None:
        print('before first iteration:', np.linalg.norm(Y - Z, ord='fro')**2 / n)
    
    for k in range(n_iterations):
        
        # bandwidth
        h = bandwidths_list[k]
        
        # compute weights and sum of weights
        if (k==0):
            W = np.exp(-dist**2/h**2) * (dist < tau)
            N = np.sum(W, axis=1)
        else:
            W = np.exp(-a_dist**2 * h**2) * (dist < tau)
            N = np.sum(W, axis=1)
        
        # compute adjusted distances
        a_dist = np.empty((n, n))
        for i in range(n):
            Y_diff = Y - Y[i, :]   #(n, D)
            
            # compute the weighted sample covariance
            Sigma = np.cov(Y_diff.T, aweights=W[i])
            # sample a precision matrix from the inverse Wishart distribution

#             Omega = invwishart.rvs(N[i] + D + 2, inv(h**2 * np.identity(D) + Sigma), size=1)
            Omega = inv(h**2 * np.identity(D) + Sigma)

            Q = Y_diff @ Omega 
            a_dist[i,:] = np.linalg.norm(Q, axis = 1)
            
        # adjusted Nadaraya-Watson estimate

        X = W.dot(Y) / np.tile(N.reshape(-1, 1), (1, D))
        
        # compare with the true values
        if Z is not None:
            print('iteration', k+1, ':', np.linalg.norm(X - Z, ord='fro')**2 / n)
                
    return X

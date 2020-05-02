# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 13:47:44 2020

@author: Anirudh Raghavan
"""
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Question 1 - Perform Linear Regression and identify maximum coefficient

def LR_LargCoef (input_data, target):
    lr = LinearRegression()
    fit_model = lr.fit(input_data,target)
    coef_model = fit_model.coef_
    coef_model = np.absolute(coef_model)
    result = {}
    result['Max Coef'] = coef_model.max()
    result['Max Coef - Index'] = np.where(coef_model == coef_model.max())[0][0]
    
    return result


# Question 2 - Perform K Means clustering and identify ideal number of clusters
    
def Kmeans_choose (input_X, K):
    # k means determine k
    distortions = []
    
    for k in range(1,K):
        kmeanModel = KMeans(n_clusters=k).fit(input_X)
        distortions.append(sum(np.min(cdist(input_X, kmeanModel.cluster_centers_, 'euclidean'), 
                                      axis=1)) / input_X.shape[0])
    
    # Plot the elbow
    plt.plot(range(1,K), distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    return plt.show()

if __name__ == "__main__":
    boston = datasets.load_boston(return_X_y=True)
    result = LR_LargCoef(boston[0],boston[1])
    print(result)
    
    
    iris = datasets.load_iris(return_X_y=True)
    result_2 = Kmeans_choose(iris[0], 10)
    

# The elbow heuristic method is used to determine the optimum number of clusters.
# The idea is to compute the intra cluster variance at each number of clusters and
# determine the level at which the rate of decrease of variance flattens.
    
# From the above, graph we can see that the curve significantly flattens post 3
# clusters, thus we can say that 3 clusters is the ideal number of clusters for 
# this data.



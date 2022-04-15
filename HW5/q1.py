import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# k_means function definition
def k_means(data, k):
    # Randomly asigns centroids
    center = np.empty([k, data.shape[1]])

    for i in range(k):
        rand_val = np.random.randint(data.shape[0])
        center[i] = data[rand_val, :]
    
    # calculates pairwise distances between center and data
    dist = cdist(data,center)

    # Finds minimum distance between each random center and data point
    min_dist = np.empty(data.shape[0])

    for i in range(len(dist)):
        min_dist[i] = np.argmin(dist[i]) 

    # While loop
    while(True): 
        cent_temp = np.empty(center.shape)

        # Recomputes new center point by taking mean of cluster
        for i in range(k):
            cent_temp[i] = np.mean(data[min_dist==i], axis=0) 
 
        # Checks if old center and new center are the same, terminates if True
        if(np.allclose(center, cent_temp)):
            break

        # Assigns old center to new center
        center = cent_temp
         
        # Recomputes new minimum distances from new center
        dist = cdist(data,center)

        min_dist = np.empty(data.shape[0])

        for i in range(len(dist)):
            min_dist[i] = np.argmin(dist[i]) 
         
    return min_dist, center

# Generates random bivariate data
mu = [3, 3]
cov = np.identity(2)

pat = np.random.multivariate_normal(mu, cov, 10000)

cluster_points, cluster_cent = k_means(pat,10)

# Plotting code: from https://www.askpython.com/python/examples/k-means-clustering-from-scratch#:~:text=K%2DMeans%20is%20a%20very,scratch%20using%20the%20Numpy%20module.
cluster_unique = np.unique(cluster_points)
for i in cluster_unique:
    plt.scatter(pat[cluster_points == i , 0] , pat[cluster_points == i , 1] , label = i, marker =".")

plt.title("K-means Clustering")
plt.legend()
plt.show()


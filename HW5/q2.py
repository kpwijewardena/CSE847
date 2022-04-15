import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

data = np.array([[0, 0],[-1, 2],[-3, 6],[1, -2],[3, -6]])

for elem in data:
    plt.scatter(elem[0], elem[1], label='pat1')

plt.xlabel('Data Values')
plt.ylabel('Distribution')
plt.title('Data')
# plt.legend()
plt.show()

# Loading USPS data
data = loadmat('USPS.mat')['A']

# Number of principal components
p = 10

# Centered data
data_center =  data - np.mean(data , axis=0)

# Implementing SVD on centralized data
u, s, v = np.linalg.svd(data_center) 

print(v.shape)
components  = v

basis_data_coordinates = np.matmul(u, s)

print(components)

print(data)
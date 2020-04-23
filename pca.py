"""  Developed by Bayram Baris Sari
*   E-mail: bayrambariss@gmail.com
*   Tel No: +90 539 593 7501    
*
*   This is an implementation of Principal
*   Component Analysis.
*   
*   It decreases 64 dimensions to 2 dimensions
*   It finds 2 biggest eigenvector matrices
"""

import numpy as np
import matplotlib.pyplot as plt


# Parse the data, there are 64 dimensions.x holds this features and y holds the classes
with open("data.txt", 'r') as file:
	data = [line.split(',') for line in file.read().split("\n") if line]

number_of_rows = len(data)
# Arrange the data type
data = np.array(data)
X, Y = data[:, :64], data[:, -1]
X = X.astype(np.float32)
Y = Y.astype(np.int)

# Calculate covarience matrix
cov = np.cov(X, rowvar=False)
# print(cov)

# Find eigenvalues and eigenvectors of the covariance matrix
eigenvalue, eigenvector = np.linalg.eig(cov)
# print(eigenvalue)
# print(eigenvector)

# descending sort of eigenvectors, according to their eigenvalues
index = eigenvalue.argsort()[::-1]
eigenvalue = eigenvalue[index]
eigenvector = eigenvector[:, index]

# merge 2 biggest eigenvector matrices horizontally
matrix_w = np.hstack((eigenvector[:, 0].reshape(64, 1), eigenvector[:, 1].reshape(64, 1)))
transformed = matrix_w.T.dot(X.T)
# print(matrix_w)
# print(transformed)

fig = plt.figure(figsize=(8, 4))
# plot the data points
plt.plot(transformed[0, :], transformed[1, :], 'o', markersize=1, color='blue', alpha=0.3)
# write class labels of random 200 data points
for i in range(200):
    index = np.random.randint(0, number_of_rows)
    class_ = Y[index]
    plt.text(transformed[0][index], transformed[1][index], f"{class_}", fontsize=8)

plt.xlim([40, -40])
plt.ylim([-40, 30])
plt.xlabel('First Eigenvector')
plt.ylabel('Second Eigenvector')
plt.title('Data after PCA')
plt.show()

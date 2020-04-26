"""  Developed by Bayram Baris Sari
*   E-mail: bayrambariss@gmail.com
*   Tel No: +90 539 593 7501    
*
*   This is an implementation of Principal Component Analysis.
*   
*   It decreases 64 dimensions to 2 dimensions. Then, it finds the 2 biggest eigenvector matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.txt")


def read_and_split_data():
    with open(DATA_PATH, 'r') as file:
        data = [line.split(',') for line in file.read().split("\n") if line]

    number_of_rows = len(data)
    data = np.array(data)
    x, y = data[:, :-1], data[:, -1]
    x, y = x.astype(np.float32), y.astype(np.int)
    return x, y, number_of_rows


def sort_eigenvectors(eigenvalues, eigenvectors):
    # Descending sort of eigenvectors, according to their eigenvalues
    index = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]
    return eigenvalues, eigenvectors


def plot_matrix(matrix, class_matrix, number_of_rows):
    plt.figure(figsize=(8, 4))
    # Plot the data points
    plt.plot(matrix[0, :], matrix[1, :], 'o', markersize=1, color='blue', alpha=0.3)
    # Write class labels for 200 data points that are randomly selected
    for i in range(200):
        index = np.random.randint(0, number_of_rows)
        class_ = class_matrix[index]
        plt.text(matrix[0][index], matrix[1][index], f"{class_}", fontsize=8)

    plt.xlim([40, -40])
    plt.ylim([-40, 30])
    plt.xlabel('First Eigenvector')
    plt.ylabel('Second Eigenvector')
    plt.title('Data after PCA')
    plt.show()


def main():
    x, y, number_of_rows = read_and_split_data()
    # Calculate covariance matrix
    cov = np.cov(x, rowvar=False)
    # Find eigenvalues and eigenvectors of the covariance matrix
    eigenvalue, eigenvector = np.linalg.eig(cov)
    eigenvalue, eigenvector = sort_eigenvectors(eigenvalue, eigenvector)
    # Merge 2 biggest eigenvector matrices horizontally
    matrix_w = np.hstack((eigenvector[:, 0].reshape(64, 1), eigenvector[:, 1].reshape(64, 1)))
    transformed = matrix_w.T.dot(x.T)
    # Plot the transformed data
    plot_matrix(transformed, y, number_of_rows)


if __name__ == "__main__":
    main()

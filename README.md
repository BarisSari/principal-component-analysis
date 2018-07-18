# Principal Component Analysis

Principal component analysis (PCA) is a mathematical technique that transforms large
datasets with high dimensions, 64 for this project, to lower dimension datasets with minimum data
losses. When the number of dimensions are reduced, it means that the datasets are compressed by
finding the general properties of the oversized data. Thus, that’s the basic logic behind the PCA.
The lost datasets while applying PCA are not included the fundamental characteristics of them.
We can evaluate the performance of a dimensionality reduction algorithm by computing the
data loss, i. e. how much our data has changed with this algorithm.

![alt text](https://github.com/BarisSari/principal-component-analysis/tree/master/images/original-data.jpg)

When we check the figure above, it has seen obviously that PCA does not work very well
for this data. Since there are only two dimensions in original data, it is not necessary to reduce the
dimension. There are two different classes and it’s very easy to label this two classes in original
data. However, it’s impossible to do that in transformed data.
PCA may have many drawbacks. For instance, one of them is its performance in noisy
datasets. It might give a bad results because it might choose the noisy variables to cover the all data.
Implementation
1. Parse the data, create two arrays, one has the features, e. g. dimensions, the other one
has classes.
2. Compute Covariance Matrix. I have used numpy built-in function for this step.
3. Find eigenvectors and eigenvalues of the computed covariance matrix. I have again used
numpy built-in function for this step.
4. Sort the eigenvectors according to their eigenvalues by descending order.
5. Choose 2 of the eigenvectors with largest eigenvalues and merge them.
6. Multiply the data with these merged matrix.
7. Plot the data and annotate random 200 data points with their class labels.

The result of the implemented PCA is as follows:

![alt text](https://github.com/BarisSari/principal-component-analysis/tree/master/images/data-after-pca.jpg)

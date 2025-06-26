![image](https://github.com/user-attachments/assets/14b29e8b-3731-411d-8591-c693bef9fe85)


# PCA 

#  Principal Component Analysis (PCA) — Feature Projection

We start with a feature vector:

$$
\mathbf{x} = [x_1, x_2, \dots, x_d], \quad \mathbf{x} \in \mathbb{R}^d
$$

We apply a transformation using a projection matrix:

$$
\mathbf{W} \in \mathbb{R}^{d \times k}
$$

Then the transformed output vector is:

$$
\mathbf{z} = \mathbf{x} \mathbf{W}, \quad \mathbf{z} \in \mathbb{R}^k
$$

That is,

$$
\mathbf{z} = [z_1, z_2, \dots, z_k]
$$

---

##  Explanation

- **x**: The original feature vector with $d$ dimensions.
- **W**: A transformation matrix whose columns are the **top-k eigenvectors**.
- **z**: The resulting low-dimensional vector in a $k$ -dimensional subspace.

---

##  Key Concepts

- The transformation $\mathbf{xW}$ projects the original feature vector into a new feature space.
- This new space captures the directions of **maximum variance** (the principal components).
- The matrix $\mathbf{W}$ is built from the **eigenvectors** of the **covariance matrix** of the dataset.
- The corresponding **eigenvalues** indicate the **amount of variance** in each direction.

---

##  Mathematical Steps of PCA

1. Standardize the dataset.
2. Construct the covariance matrix $\Sigma$:
   
$$
   \Sigma = \frac{1}{n - 1} X^\top X
$$


$$
\sigma_{jk} = \frac{1}{n - 1} \sum_{i=1}^{n} \left(x_j^{(i)} - \mu_j\right)\left(x_k^{(i)} - \mu_k\right) 
$$ 


3. Decompose the covariance matrix into its eigenvectors and eigenvalues.   

$$
\Sigma \mathbf{v} = \lambda \mathbf{v}
$$

4. Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors.
5. Select k eigenvectors, which correspond to the k largest eigenvalues, where k is the dimension qality of the new feature subspace 
6. Construct a projection matrix, W, from the “top” k eigenvectors.
7. Transform the d-dimensional input dataset, X, using the projection matrix, W, to obtain the new k-dimensional feature subspace.

## Explained Variance Ratio

The explained variance ratio tells us:

> **How much of the total information (variance) is captured by a principal component (eigenvector)?**

The formula is:

$$
\
\text{Explained variance ratio} = \frac{\lambda_j}{\sum_{j=1}^{d} \lambda_j}
\
$$ 


- \( \lambda_j \): The *j*-th eigenvalue, representing the variance along the *j*-th principal component.
- \( \sum_{j=1}^{d} \lambda_j \): The sum of all *d* eigenvalues, which equals the total variance of the data.

告诉我们需要几个eigenvector， 方差达到95% 


 # Linear Discriminant Analysi 

 1. Standardize the d-dimensional dataset (d is the number of features).
 2. For each class, compute the d-dimensional mean vector.
 3. Construct the between-class scatter matrix, S_b, and the within-class scatter matrix, S_w
 4. Compute the eigenvectors and corresponding eigenvalues of the matrix,
 5. Sort the eigenvalues by decreasing order to rank the corresponding eigenvectors.
 6. Choose the k eigenvectors that correspond to the k largest eigenvalues to construct a d×k-dimensional transofrmation matrix W; the eigenvectors  are the columns of this matrix
 7. Project the examples onto the new feature subspace using the transformation matrix, W.




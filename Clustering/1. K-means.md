# K-Means 

use K-means algorithm by following four steps: 

1. **Randomly pick** `k` centroids from the examples as the initial cluster centers.
2. **Assign each example** to the nearest centroid, $\mu^{(j)}, j \in \{1, \dots, k\}$.
4. **Update the centroids** by computing the mean of all examples assigned to each centroid.
5. **Repeat Steps 2 and 3** until:
   - The cluster assignments do not change, or
   - A user-defined tolerance or maximum number of iterations is reached.

## How do we measure similarity between objects 

$$
d(\mathbf{x}, \mathbf{y})^2 = \sum_{j=1}^{m} (x_j - y_j)^2 = \||\mathbf{x} - \mathbf{y}\||_2^2
$$

* the index *j* refers to the jth dimension (feature columns) of the example inputs *x* and *y* 
* superscripts *i* and *j* to refer to the index of the example and cluster index

### K-means Objective: Minimize the Sum of Squared Errors (SSE)

The goal of the K-means algorithm is to make all points as close as possible to their assigned cluster center. This is achieved by minimizing the **Sum of Squared Errors (SSE)**:


$$
SSE = \sum_{i=1}^{n} \sum_{j=1}^{k} w^{(i,j)} \left\|| \mathbf{x}^{(i)} - \{\mu}^{(j)} \right\||_2^2
$$


Here, 


$$
w^{(i,j)} =
\begin{cases}
1, & \text{if } \mathbf{x}^{(i)} \in j \\
0, & \text{otherwise}
\end{cases}
$$

w(i, j) = 1 if the example , x(i) is in cluster j , or 0 otherwise. <br> 

## Feature Scaling
K-Means use Euclidean distance, so you have to use feature scaling <br> 



## K-Means++ Initialization Steps

The initialization in k-means++ can be summarized as follows:

1. **Initialize** an empty set, **M**, to store the *k* centroids being selected.

2. **Randomly choose** the first centroid, $\( \mu^{(j)} \)$, from the input examples and assign it to **M**.

3. For each example, $\( x^{(i)} \)$, that is **not in M**, find the **minimum squared distance**, $\( d(x^{(i)}, M)^2 \)$, to any of the centroids in **M**.

4. To randomly select the next centroid, $\( \mu^{(p)} \)$, use a weighted probability distribution equal to:


$$
   \frac{d(\mu^{(p)}, M)^2}{\sum_i d(x^{(i)}, M)^2}
$$


   For instance, we collect all points in an array and choose a weighted random sampling, such that the **larger the squared distance**, the **more likely a point gets chosen** as the centroid.

5. **Repeat steps 3 and 4** until *k* centroids are chosen.

6. **Proceed with the classic k-means algorithm**.

set `init='k-means++'` 

# Soft Clustering (FCM) 

In hard clustering, we think each point can only belong to one cluster,and we get *x* with a sparse vector of binary values: 


$$
\
\left[
\begin{array}{l}
x \in \mu^{(1)} \quad \rightarrow \quad w^{(i,j)} = 0 \\
x \in \mu^{(2)} \quad \rightarrow \quad w^{(i,j)} = 1 \\
x \in \mu^{(3)} \quad \rightarrow \quad w^{(i,j)} = 0
\end{array}
\right]
\
$$ 

For soft clustering, 


$$
\
\left[
\begin{array}{l}
x \in \mu^{(1)} \quad \rightarrow \quad w^{(i,j)} = 0.1 \\
x \in \mu^{(2)} \quad \rightarrow \quad w^{(i,j)} = 0.85 \\
x \in \mu^{(3)} \quad \rightarrow \quad w^{(i,j)} = 0.05
\end{array}
\right]
\
$$

Here, each value falls in the range [0, 1] and represents a probability of membership of the respective cluster centroid. <br>
The sum of the memberships for a given example is equal to 1. As with the k-means algorithm, we can summarize the FCM algorithm in four key steps:
1. Specify the number of k centroids and randomly assign the cluster memberships for each point
2. Compute the cluster centroids, $𝝁^{(𝑗)} , 𝑗\in_{(1, ... , 𝑘)}$
4. Update the cluster memberships for each point
5. Repeat steps 2 and 3 until the membership coefficients do not change or a user-defined tolerance
or maximum number of iterations is reached

The objective function :

## Fuzzy C-Means Objective Function

The objective function of fuzzy c-means (FCM) is:

$$
J_m = \sum_{i=1}^{n} \sum_{j=1}^{k} {w^{(i,j)}}^m \left\| x^{(i)} - \mu^{(j)} \right\|^2_2
$$

- $w_{i,j}$ represents the **membership degree** of sample $x^{(i)}$ in cluster $j$
- $m$ is the **fuzzifier**, controlling the level of cluster fuzziness (usually $m = 2$)
- $\mu^{(j)}$ is the centroid of cluster $j$
- The distance is the **squared Euclidean distance**

---

### Membership Calculation Formula

The membership degree $w_{(i,j)}$ is computed using the following formula:

$$
w_{(i,j)} =
\left[
\sum_{c=1}^{k}
\left(
\frac{
\left\| x^{(i)} - \mu^{(j)} \right\|_2
}{
\left\| x^{(i)} - \mu^{(c)} \right\|_2
}
\right)^{\frac{2}{m-1}}
\right]^{-1}
$$

---

### Example (Three Clusters)

If we have 3 clusters and want to compute $w_{i,j}$, the membership of sample $x^{(i)}$ to centroid $\mu^{(j)}$ is:

$$
w_{(i,j)} =
\left[
\left(
\frac{\left\| x^{(i)} - \mu^{(j)} \right\|_2}
     {\left\| x^{(i)} - \mu^{(1)} \right\|_2}
\right)^{\frac{2}{m-1}}
+
\left(
\frac{\left\| x^{(i)} - \mu^{(j)} \right\|_2}
     {\left\| x^{(i)} - \mu^{(2)} \right\|_2}
\right)^{\frac{2}{m-1}}
+
\left(
\frac{\left\| x^{(i)} - \mu^{(j)} \right\|_2}
     {\left\| x^{(i)} - \mu^{(3)} \right\|_2}
\right)^{\frac{2}{m-1}}
\right]^{-1}
$$

---

## Notes

- If $m=1$, FCM reduces to hard clustering like K-Means.
- A higher $m$ causes the membership values to become more **uniform** (i.e., fuzzier).
- A lower $m$ (closer to 1) makes clusters more **crisp**.

### Cluster Center Calculation in FCM

The center, $\ \mu^{(j)} \$, of a cluster itself is calculated as the mean of all examples weighted by the degree to which each example belongs to that cluster $\(w_{i,j})^m \$:

```math
\mu^{(j)} = \frac{\sum_{i=1}^{n} w_{i,j}^m \, x^{(i)}}{\sum_{i=1}^{n} w_{i,j}^m}
```

# Quantifying the Quality of Clustering via Silhouette Plots

One intrinsic metric to evaluate the quality of a clustering is **silhouette analysis**, which can also be applied to clustering algorithms other than k-means. Silhouette analysis can be used as a graphical tool to plot a measure of how tightly grouped the examples in the clusters are.

To calculate the **silhouette coefficient** of a single example in our dataset, we can apply the following three steps:

1. **Calculate the cluster cohesion**, $\ a^{(i)} \$, as the average distance between an example, $\ x^{(i)} \$, and all other points in the same cluster.

2. **Calculate the cluster separation**, $\ b^{(i)} \$, from the next closest cluster as the average distance between the example, $\ x^{(i)} \$, and all examples in the nearest cluster.

3. **Calculate the silhouette**, $\ s^{(i)} \$, as the difference between cluster cohesion and separation divided by the greater of the two, as shown here:

$$
\[
s^{(i)} = \frac{b^{(i)} - a^{(i)}}{\max(b^{(i)}, a^{(i)})}
\]
$$

---

The silhouette coefficient is bounded in the range $\(-1\) to \(1\)$. Based on the preceding equation, <br>
we can see that the silhouette coefficient is 0 if the cluster separation and cohesion are equal $\(b^{(i)} = a^{(i)})\$. <br>
Furthermore, we get close to an ideal silhouette coefficient of 1 if $(b^{(i)} \gg a^{(i)}\)$, since $b^{(i)}\$ quantifies how dissimilar an example is from other clusters, <br>
and $a^{(i)}\$ tells us how similar it is to the other examples in its own cluster.


# Introduction 

This project implements a **K-Nearest Neighbors (KNN)** algorithm from scratch (without using sklearn’s KNN model), 
and uses it to classify flowers in the classic **Iris dataset**.

The core idea is to predict the label of a new data point based on the majority label among its *k* nearest neighbors in the training set, using **Euclidean distance**.

# Dataset: Iris Flower Dataset
This project uses the well-known Iris flower dataset, which is loaded using `sklearn.datasets.load_iris()`.

The dataset includes:

`150` samples of `iris flowers`

Each sample has` 4 numeric features`:

* Sepal length (cm)
* Sepal width (cm)
* Petal length (cm)
* Petal width (cm)

Each sample is labeled as one of three flower species:

* [Iris-virginica] => 0 
* [Iris-setosa] => 1 
* [Iris-versicolor] => 2

# k- Nearest Neighbors  

* Step 1: Calculate Euclidean Distance.
* Step 2: Get Nearest Neighbors.
* Step 3: Make Predictions.

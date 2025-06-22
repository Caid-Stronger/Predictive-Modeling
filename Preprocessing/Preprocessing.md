![image](https://github.com/user-attachments/assets/1020d16c-0c61-4373-9678-cafed10d3055)

The `fit` method is used to learn the parameters from the training data,  <br>
and the `transform` method uses those parameters to transfor the data. <br> 

![image](https://github.com/user-attachments/assets/152cac39-49b8-40a8-aa62-ee09ebc3cf91)

The `fit` method is used to learn the parameers of a model <br> 
make predictions about new, unlabeled data examples via the predict method <br> 

# Feature Scaling

## 1. Normalization

- Usually scales features to the range [0, 1];
- The most common form is **Min-Max Scaling**;
- Formula:
- 
$$
  \
  x' = \frac{x - \min(x)}{\max(x) - \min(x)}
  \
$$

  

- Suitable when the data **does not follow a normal distribution**.

---

## 2. Standardization

- Scales features to **mean = 0** and **standard deviation = 1**;
- Commonly used in most machine learning algorithms, especially those using gradient descent or distance-based methods;
- Formula:


$$
\
  x' = \frac{x - \mu}{\sigma}
\
$$


  where $\mu$ is the mean of the feature and $\sigma$ is the standard deviation.

---

 Definition of Standard Deviation

The **standard deviation**  $\sigma$  measures the dispersion of a set of values relative to their mean, defined as:

$$
\
\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2}
\
$$


- $x_i$: the $i$-th data point  
- $\mu$: the mean of the data  
- $n$: the number of data points

A larger $\sigma$ indicates more spread out data, while a smaller $\sigma$ indicates data points are closer to the mean.

---

Many machine learning model such as SVM will initialize the weights  to  0 or small random value close to 0. Using standadization, we center the feature columns at mean 0 with standard deviation 1 so that the feature  columns have the same parameters as a standard normal distribution, which makes it easier to learn the weights. <br> 
It also maintain useful information about outliers and make the algorithm less sensitive to them in contrast to  min-max scaling, which scales the data  to a limited range of values. <br> 

## L2 Regularization 

Find the combination of weight coefficients that minimize th loss function for the training data <br> 
regularization is adding a penalty term to the loss function to encourage smaller weights <br> 
increase the regularization strength via the regularization parameter $\lambda$ , we shrink the weights toward zero and decrease the dependency of our model on the training data <br> 

![image](https://github.com/user-attachments/assets/bec76826-4c90-4533-a5be-3bbc7da2f15d) 
![image](https://github.com/user-attachments/assets/70d08972-f5e4-4a34-9ed6-a72a9a3a9705) 

## L1 Regularization

L1 Regularization can be used as a feature  selection  <br>

## Sequential feature selection algorithms 

feature selection: we select a subset of the original features <br> 
feature extraction: we derive information from the  feature set to construct a new feature subspace <br> 

Sequential feature selection algorithms is a greedy search algorithms that are used to reduce an initial d-dimensional feature space to a k-dimensional feature subspace where k<d. <br> 
It will select  a subset of features that are most relevant to the problem <br> 

### Sequential Backward selection 
sequentially removes features from the full feature subset until the  new feature subspace contains the desired numbe of features; define the criterion function *J* that we want  to minimize. <br> 




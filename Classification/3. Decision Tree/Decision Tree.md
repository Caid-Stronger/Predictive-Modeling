![image](https://github.com/user-attachments/assets/e341fb6a-08a8-47d5-9791-471206ab8cb2)

# Maxmizing IG -- getting the most bang for your buck 


$$
IG(D_p, f) = I(D_p) - \sum_{j=1}^{m} \frac{N_j}{N_p} I(D_j)
$$

* `f` is the feature to perform the split
* `Dp` and Dj are the dataset of the parent and jth child node
* `I` is our impurity measure
* `Np` is the total number of training examples ar the parent node
* `Nj` is the number of examples in the `jth` child node
  
IG 是信息增益，I 为不纯度度量函数,； <br>
在分类后，熵增下降程度 <br>

## Binary Decision Trees 

$$
IG(D_p, f) = I(D_p) - \frac{N_{\text{left}}}{N_p} I(D_{\text{left}}) - \frac{N_{\text{right}}}{N_p} I(D_{\text{right}})
$$

Three impurtiy measures: 
* Gini impurity
* entropy
* classification error 


## Entropy
  

$$
I_H(t) = - \sum_{i=1}^{c} p(i \mid t) \log_2 p(i \mid t)
$$

Where:

- \( t \): The current node (e.g., a subset of the dataset).
- \( c \): Total number of classes (e.g., 2 classes or 3 classes).
- \( p(i \mid t) \): The proportion of samples in node \( t \) that belong to class \( i \).

This proportion is calculated as:

$$
p(i \mid t) = \frac{\text{Number of samples in class } i}{\text{Total number of samples in node } t}
$$

- If all samples in a node belong to a single class, entropy is 0 (perfectly pure).
- If the classes are evenly distributed, entropy is maximal (e.g., 1 for binary classes).

![image](https://github.com/user-attachments/assets/7522e984-940d-4d4f-a75c-3681a403699f)

---

## Gini impurity（基尼不纯度）



```math
I_G(t) = \sum_{i=1}^{c} p(i|t)(1 - p(i|t)) = 1 - \sum_{i=1}^{c} p(i|t)^2
```

- \( p(i|t) \): the probability that a sample at node \( t \) belongs to class \( i \);
- The maximum value occurs when the classes are perfectly mixed  
  (e.g., in binary classification with 50% each class);
- Example: if \( p(1|t) = 0.5 \), \( p(2|t) = 0.5 \), then:

```math
I_G(t) = 1 - (0.5^2 + 0.5^2) = 0.5
```

---

## Classification Error 

Another impurity measure is the classification error:

$$
I_E(t) = 1 - \max\{p(i|t)\}
$$

![image](https://github.com/user-attachments/assets/f405a30a-bc00-4ae1-bd7a-5f125e2aeee5)


## Ensemble of decision trees 

1. Draw a random boostrap sample of size **n** (randomly choose n examples from the training datasets with replacement)
2. Grow a decision tree from the boostrap sample. At each node:
   a. Randomly select **d** features without replacement <br>
   b. Split node using the feature that provides the best split according to the objective function, for instance, maximaizing the information again.
3. Repeat steps 1-2 k times
4. Aggregate the prediction by each tree to assign the class label by majority vote.

![image](https://github.com/user-attachments/assets/ccd117b4-3798-41a6-9679-20dd9670d49c) 

The only parameter that we need to care about in practice is the number of trees, k (stpes3) that we choose for the random forest <br> 
other parameters: 
* the size **n**  of the bootstrap sample
* the number of features **d** that are randomly chosen for each split

increasing the size of the bootstrap sample may increase the degree of overfitting. <br> 
Because the bootstrap samples, and consequently the individual decision
trees, become more similar to one another, they learn to fit the original training dataset more closely. <br> 

Decreasing the size of the bootstrap sample increases the diversity among the individual trees since <br>
the probability that a particular training example is included in the bootstrap sample is lower. <br>
Thus, shrinking the size of the bootstrap samples may increase the randomness of the random forest, and it can help to reduce the effect of overfitting.<brn_n

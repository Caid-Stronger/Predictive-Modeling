# SVM 

our goal is to maximize the margin. <br> The margin is defined as the distance between the seperating hyperplane (decision boundary) <br> 
the training examples that are closest to this hyperplane are called *support vectors* <br> 
![image](https://github.com/user-attachments/assets/75503e38-2b99-4a7d-8925-36e4d686d2ec)

## slack variable 松弛变量
soft margin classification <br> 
the linear constraints in the SVM optimization objective need to be relaxed for nonlinearly seperable data to allow convergence of the optimization in the presence of misclassifications, under appropriate loss penalization <br> 
以便在样本并非完全线性可分的情况下仍能收敛，并且在有分类错误的前提下，优化器可以容忍一定的误差 <br> 

![image](https://github.com/user-attachments/assets/cef13db4-c19d-48f0-86d1-3c75ef71fc31)

c 为L2 正则化的倒数 <br> 


##  Why Linear Models Fail on Some Data Distributions

Some data distributions are **nonlinear in nature**, which means they cannot be separated using a straight line (linear hyperplane). For example:

###  XOR Problem

- (0, 0) and (1, 1) belong to one class  
- (1, 0) and (0, 1) belong to the other class  
-  You **cannot** draw a single straight line to separate these points

###  Concentric Circles

- One class is in the inner circle, the other class is in the outer circle  
-  You **cannot** use a single straight line to separate the inner and outer rings

---

These kinds of data are **linearly inseparable in 2D space**, meaning:
> A straight line will never be a good decision boundary.

This is why we need **nonlinear models** or **kernel methods** to transform or map the data into a higher-dimensional space where a linear separator becomes possible.

##  Kernel Methods: Making Nonlinear Data Linearly Separable

To deal with linearly inseparable data, we use **kernel methods**.  
The key idea is to **construct nonlinear combinations of the original features**, projecting the data into a higher-dimensional space.

> This is done using a **mapping function** \( \phi \), where the data may become linearly separable in the new space.

As shown in *Figure 3.14*, we can transform a 2D dataset into a new 3D feature space, where the classes become separable by projecting with the following transformation:

```math
\phi(x_1, x_2) = (z_1, z_2, z_3) = (x_1, x_2, x_1^2 + x_2^2)
```

![image](https://github.com/user-attachments/assets/d5a4b84f-5843-4f3a-a690-9c06402503d7)


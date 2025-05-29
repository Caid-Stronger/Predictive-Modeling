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

To avoid explicitly computing the dot product between two points in a high-dimensional space,  
we define a **kernel function** as follows:

```math
\kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \phi(\mathbf{x}^{(i)})^T \phi(\mathbf{x}^{(j)})
```


## RBF Kernel (Radial Basis Function Kernel)

One of the most commonly used kernels is the **Radial Basis Function (RBF)** kernel,  
also known as the **Gaussian kernel**:

```math
\kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \exp\left( -\frac{ \| \mathbf{x}^{(i)} - \mathbf{x}^{(j)} \|^2 }{2\sigma^2} \right)
```

### Simplified RBF Kernel

The RBF kernel is often simplified to:

```math
\kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \exp\left( -\gamma \| \mathbf{x}^{(i)} - \mathbf{x}^{(j)} \|^2 \right)
```

Where:

```math
\gamma = \frac{1}{2\sigma^2}
```
is a hyperparameter that must be tuned. <br> 


![image](https://github.com/user-attachments/assets/fb2c8ba6-8430-4dcc-848f-f32675274a26)


##  `gamma` 在 SVM 中的含义

当你使用 **核函数（kernel function）**，比如 RBF 核（也叫 Gaussian 核）时，`gamma` 控制的是：

> 每个样本对最终模型的“影响范围”有多大。

它实际上是核函数中的一个超参数，出现在下面这个公式中：

$$
\kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) = \exp\left( -\gamma \|\mathbf{x}^{(i)} - \mathbf{x}^{(j)}\|^2 \right)
$$


###  大 gamma：**每个点只影响自己周围的小区域**
- **结果**：模型会记住训练数据，形成非常蜿蜒、**复杂的边界**。
- **优点**：能高度拟合训练集。
- **缺点**：容易过拟合，对新样本泛化能力差。

###  小 gamma：**每个点的影响范围很大**
- **结果**：模型更关注全局结构，形成平滑、**简单的边界**。
- **优点**：泛化能力好，不容易过拟合。
- **缺点**：容易欠拟合，忽略局部细节。

---

## 🎨 可视化类比

- 大 gamma ≈ 放大镜看每个点，眼里只有附近  
- 小 gamma ≈ 从远处看整体分布，细节模糊但能看到趋势

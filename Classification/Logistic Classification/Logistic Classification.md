the odds can be written as 

$$
\frac{p}{1 - p}
$$

Defin the logit function, which is simply the logarithm of the odds(log-odds): <br> 
convert the probability [0, 1] into real-number range <br>

$$
\text{logit}(p) = \log\left( \frac{p}{1 - p} \right)
$$

Under the logistic model, we assume that there is a linear relationship between the weighted inputs: 

$$
\text{logit}(p) = w_1 x_1 + \cdots + w_m x_m + b = \sum_{i=j}^{} w_j x_j + b = \mathbf{w}^T \mathbf{x} + b
$$


logistic signoid function: <br> 
converting the probability into real0number range 

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

![image](https://github.com/user-attachments/assets/9355249f-5731-4621-88ef-747f9c5fc8bf)

---

$$
z = \mathbf{w}^T \mathbf{x} + b
$$



$$
P(y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \mathbf{w}^T \mathbf{x} + b
$$

$$
P(y = 0 \mid \mathbf{x}) = 1 - \sigma(z)
$$

---

beause $y^{(i)}$ = 0 or 1 

$$
P(y^{(i)} \mid \mathbf{x}^{(i)}; \mathbf{w}, b)
= \left( \sigma(z^{(i)}) \right)^{y^{(i)}}
  \left( 1 - \sigma(z^{(i)}) \right)^{1 - y^{(i)}}
$$

- if $y^{(i)} = 1$，then：


$$
\sigma(z^{(i)})^1 \cdot (1 - \sigma(z^{(i)}))^0 = \sigma(z^{(i)})
$$


- if $y^{(i)} = 0$，then：


$$
\sigma(z^{(i)})^0 \cdot (1 - \sigma(z^{(i)}))^1 = 1 - \sigma(z^{(i)})
$$



---

##
the likelihood, ℒ ,
that we want to maximize when we build a logistic regression model

$$
\mathcal{ℒ}(\mathbf{w}, b)
= \prod_{i=1}^{n} P(y^{(i)} \mid \mathbf{x}^{(i)}; \mathbf{w}, b)
= \prod_{i=1}^{n}
  \left( \sigma(z^{(i)}) \right)^{y^{(i)}}
  \left( 1 - \sigma(z^{(i)}) \right)^{1 - y^{(i)}}
$$


## Log-likelihood

To simplify optimization, we take the natural logarithm of the entire likelihood function:

$$
\ell(\mathbf{w}, b \mid \mathbf{x}) = \log \mathcal{ℒ }(\mathbf{w}, b \mid \mathbf{x})
$$

Since logarithms convert products into sums:

$$
\log \prod_{i=1}^{n} a_i = \sum_{i=1}^{n} \log a_i
$$

So we have:

$$
\ell(\mathbf{w}, b \mid \mathbf{x}) = \sum_{i=1}^{n}
\log \left[
\left( \sigma(z^{(i)}) \right)^{y^{(i)}}
\left( 1 - \sigma(z^{(i)}) \right)^{1 - y^{(i)}}
\right]
$$

Using the logarithmic identity:

$$
\log(a^x) = x \log a
$$

We can split the expression into two terms:

$$
\ell(\mathbf{w}, b \mid \mathbf{x}) =
\sum_{i=1}^{n} \left[
y^{(i)} \log(\sigma(z^{(i)})) +
(1 - y^{(i)}) \log(1 - \sigma(z^{(i)}))
\right]
$$

---

 **Final expression for the log-likelihood function:**

$$
\ell(\mathbf{w}, b \mid \mathbf{x}) =
\sum_{i=1}^{n} \left[
y^{(i)} \log(\sigma(z^{(i)})) +
(1 - y^{(i)}) \log(1 - \sigma(z^{(i)}))
\right]
$$


## because we want to minimize the lose function: 


$$
L(\mathbf{w}, b) =
\sum_{i=1}^{n}
\left[
-y^{(i)} \log(\sigma(z^{(i)})) -
(1 - y^{(i)}) \log(1 - \sigma(z^{(i)}))
\right]
$$


$$
L(\sigma(z), y; \mathbf{w}, b) = - y \log(\sigma(z)) - (1 - y) \log(1 - \sigma(z))
$$


- If \( y = 1 \), then \( (1 - y) = 0 \), so the second term disappears.
- If \( y = 0 \), then \( y = 0 \), so the first term disappears.

This gives us the simplified conditional form:


$$
L(\sigma(z), y; \mathbf{w}, b) =
\begin{cases}
-\log(\sigma(z)) & \text{if } y = 1 \\\\
-\log(1 - \sigma(z)) & \text{if } y = 0
\end{cases}
$$


$$
L(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} \left[-
y^{(i)} \log(\sigma(z^{(i)})) -
(1 - y^{(i)}) \log(1 - \sigma(z^{(i)}))
\right]
$$


We use this to compute the loss of classifying all training examples per epoch.

##  Logistic Regression with L2 Regularization

The loss function for logistic regression can be regularized by adding a simple regularization term,  
which will shrink the weights during model training:

$$
L(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} \left[ -y^{(i)} \log(\sigma(z^{(i)})) - (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right] + \frac{\lambda}{2n} \|\mathbf{w}\|^2
$$


---

###  Gradient of the Unregularized Loss

The partial derivative of the unregularized loss is defined as:

$$
\frac{\partial L(\mathbf{w}, b)}{\partial w_j} = \left( \frac{1}{n} \sum_{i=1}^{n} (\sigma(\mathbf{w}^T \mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)} \right)
$$


---

###  Adding Regularization

Adding the regularization term to the loss changes the partial derivative to the following form:

$$
\frac{\partial L(\mathbf{w}, b)}{\partial w_j} = \left( \frac{1}{n} \sum_{i=1}^{n} (\sigma(\mathbf{w}^T \mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)} \right) + \frac{\lambda}{n} w_j
$$


# Logistic Regression Model



### Cost Function

For linear regression, we had our cost function as:
$$
J(\theta) = \frac{1}{m} \sdot \sum_{i=1}^m \frac{1}{2} \sdot (h_\theta(x^{(i)}) - y^{(i)})^2
$$
We’ll use an alternateway of writing the cost function. Instead of using the squared error term as above, we’ll use:
$$
J(\theta) = \frac{1}{m} \sdot \sum_{i=1}^m cost(h_\theta(x^{(i)}), y^{(i)})
\\
=> cost(h_\theta(x), y) = \frac{1}{2} (h_\theta(x) - y)^2
$$
If we plot this function for logistic regression, when
$$
h_\theta(x) = \frac{1}{1 + e^{-\theta^Tx}}
$$
We’ll get a plot of non-convex function like this:

![Non Convex Function](images/image01.png)

This function will have many local optima, and we cannot run gradient descent on this sort of function. Instead, we want a convex function which will guarantee that gradient descent will converge on the global minimum.

![Convex Function](images/image02.png)

We’ll therefore use,
$$
cost(h_\theta(x), y) =
\begin{cases}
-log(h_\theta(x)) & \text{if $y=1$} \\[2ex]
-log(1- h_\theta(x)) & \text{if $y=0$}
\end{cases}
$$
![Log Function y=1](images/image03.png)

If $y=1$ and $h_\theta(x) = 1$, then $cost=0$

But as $h_\theta(x) \rarr 0$, $cost \rarr \infin$

It captures intuition that if $h_\theta(x)=0$, [predict $P(y=1|x;\theta) = 0$], but $y=1$, we’ll penalize the learning algorithm by a very large cost. It’s like the probability of a patient to have a malignant tumor is 0, even though the value of $y=1$.

![Log Function y=1](images/image04.png)

Therefore,
$$
cost(h_\theta(x),y) =
\begin{cases}
0 & \text{if $h_\theta(x)=y$} \\[2ex]
\infin & \text{if $y=0$ and $h_\theta(x) \rarr 1$} \\[2ex]
\infin & \text{if $y=0$ and $h_\theta(x) \rarr 0$}
\end{cases}
$$


> In logistic regression, the cost function for our hypothesis outputting (predicting) $h_\theta(x)$ on a training example that has label $y \epsilon \{0, 1\}$ is:
> $$
> cost(h_\theta(x), y) =
> \begin{cases}
> -log(h_\theta(x)) & \text{if $y=1$} \\[2ex]
> -log(1- h_\theta(x)) & \text{if $y=0$}
> \end{cases}
> $$
> Which of the following are true? Check all that apply.
>
> - [x] If $h_\theta(x)=y$, then $cost(h_\theta(x), y) =0$ (for $y=0$ and $y=1$)
> - [x] If $y=0$, then $cost(h_\theta(x), y) \rarr \infin$ as $h_\theta(x) \rarr 1$
> - [ ] If $y=0$, then $cost(h_\theta(x), y) \rarr \infin$ as $h_\theta(x) \rarr 0$
> - [x] Regardless of whether $y=0$ or $y=1$, if $h_\theta(x)=0.5$, then $cost(h_\theta(x), y) > 0$





## Simplifying the Cost Function & Gradient Descent

We have our cost function:
$$
cost(h_\theta(x), y) =
\begin{cases}
-log(h_\theta(x)) & \text{if $y=1$} \\[2ex]
-log(1- h_\theta(x)) & \text{if $y=0$}
\end{cases}
$$
We can write the cost function in a simpler way as follows:
$$
cost(h_\theta(x), y) = -y \sdot log(h_\theta(x)) - (1-y) \sdot log(1- h_\theta(x))
$$
Therefore,
$$
\begin{align}
J(\theta) &= \frac{1}{m} \sdot \sum_{i=1}^m cost(h_\theta(x^{(i)}) - y^{(i)}) \\
&= - \frac{1}{m} \Biggl[\sum_{i=1}^m y^{(i)} \sdot log(h_\theta(x^{(i)})) + (1-y^{(i)}) \sdot log(1-h_\theta(x^{(i)})) \Biggr]
\end{align}
$$
To fit the parameters $\theta$, minimize $J(\theta)$ for $\theta$

To give a prediction from new $x$, output $h_\theta(x) = \frac{1}{1+e^{- \theta^T x}}$



#### Gradient Descent

$$
J(\theta) = - \frac{1}{m} \Biggl[\sum_{i=1}^m y^{(i)} \sdot log(h_\theta(x^{(i)})) + (1-y^{(i)}) \sdot log(1-h_\theta(x^{(i)})) \Biggr]
$$

We want to minimize $\theta$ in $J(\theta)$:

*repeat* {
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$
} [Simultaneously update all $\theta_j$]

This update rule is exactly the same as Linear Regression, except the fact that the hypothesis has changed



$\theta$ updates on every iteration:
$$
\theta_0 := \theta_0 - \alpha \sdot \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \sdot x_0^{(i)} \\
\theta_1 := \theta_1 - \alpha \sdot \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \sdot x_1^{(i)} \\
\theta_2 := \theta_2 - \alpha \sdot \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \sdot x_2^{(i)} \\
\vdots \\
\theta_n := \theta_n - \alpha \sdot \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \sdot x_n^{(i)}
$$


Vectorized implementation of $\theta$ updates (instead of using a ‘for’ loop):
$$
\theta := \theta - \alpha \sdot \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) \sdot x^{(i)} \\[2ex]
(or) \\[2ex]
\theta := \theta - \frac{\alpha}{m} \sdot X^T (g(X\theta) - \overrightarrow{y}
$$




## Advanced Optimization

Let’s revise what gradient descent actuatlly does:

- We have a cost function $J(\theta)$ and we want to minimize $\theta$.
- For minimizing $\theta$, we write code that can compute:
    - $J(\theta)$
    - $\frac{\partial}{\partial \theta_j} J(\theta)$   [For $j=0, 1, 2, \cdots,n$]
- The above gets plugged into gradient descent



Optimization algorithms which can be used:

- Gradient Descent
- Conjugate Gradient
- BFGS
- L-BFGS

The three other algorithms doesn’t need manual selection of the learning rate $\alpha$ and often converges much faster than gradient descent. However, they’re more complex.



Let’s consider an example,
$$
\theta =
\begin{bmatrix}
\theta_1 \\
\theta_2
\end{bmatrix}
\quad \quad \quad \quad
J(\theta) = (\theta_1 - 5)^2 + (\theta_2 - 5)^2 \\
\frac{\partial}{\partial \theta_1} J(\theta) = 2(\theta_1 -5) \\
\frac{\partial}{\partial \theta_2} J(\theta) = 2(\theta_2 -5)
$$
For implementation in Octave,

```octave
function [jVal, gradient] = costFunction(theta)
	jVal = (theta(1)-5)^2 + (theta(2)-5)^2;
	gradient = zeros(2,1);
	gradient(1) = 2*(theta(1)-5);
	gradient(2) = 2*(theta(2)-5);
```

Now to call the advanced optimization functions in Octave,

```octave
options = optimset('GradObj', 'on', 'MaxIter', '100');
initialTheta = zeros(2,1);
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```





## Multiclass Classification

Suppose we have classification problems such as email tagging or classifying weather, $y$ can have different values for each type, such as:

| Example           | y=1        | y=2     | y=4    | y=4        |
| ----------------- | ---------- | ------- | ------ | ---------- |
| Email Tagging     | Work       | Friends | Family | Promotions |
| Medical Diagnosis | No Disease | Cold    | Flu    |            |
| Weather           | Sunny      | Rain    | Snow   | Cloudy     |

![Multiclass vs Binary](images/image05.png)



## One vs Rest

![One vs Rest](images/image06.png)

We can have a problem where we might want to differentiate one class from the rest. We will have a hypothesis such as:
$$
h_\theta^{(i)}(x) = {(y=i \space | \space x;\theta)}
\quad \quad \quad \quad \quad \quad \quad \quad \quad
(i=1, 2, 3)
$$
Train a logistic regression classifier $h_\theta^{(i)}(x)$ for each class $i$ to predict the probability that $y=i$.

On a new input $x$, to make a prediction, pick the class $i$ that maximizes:
$$
\max_i (h_\theta^{(i)}(x))
$$

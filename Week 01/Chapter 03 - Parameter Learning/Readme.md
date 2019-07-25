# Parameter Learning



## Gradient Descent

![Gradient Descent](images/image01.PNG)

In the above representation,

- x-axis represents $\theta_0$
- y-axis represents $\theta_1$
- z-axis represents the cost function $J(\theta_0, \theta_1)$

We will know that we have succeeded when our cost function is at the very bottom of the pits in our graph (when its value is the minimum). The way we do this is by taking the derivative (the tangential line to a function) of our cost function. The slope of the tangent is the derivative at that point and it will give us a direction to move towards. We make steps down the cost function in the direction with the steepest descent. The size of each step is determined by the parameter $\alpha $, which is called the learning rate.

For example, the distance between each 'star' in the graph above represents a step determined by our parameter $\alpha $. A smaller $\alpha $ would result in a smaller step and a larger α results in a larger step. The direction in which the step is taken is determined by the partial derivative of $J(\theta_0,\theta_1)$. Depending on where one starts on the graph, one could end up at different points. The image above shows us two different starting points that end up in two different places.

#### Problem Setup:

- Have some function $J(\theta_0, \theta_1)$
- Objective is to $min_{\theta_0, \theta_1} J(\theta_0, \theta_1)$

#### Outline:

- Start with some $\theta_0, \theta_1$
- Keep changing $\theta_0, \theta_1$ to reduce $J(\theta_0, \theta_1)$ until we hopefully end up at a minimum

#### Algorithm:

Repeat until convergence for $j=0$ and $j=1$
$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j} J(\theta_0, \theta_1)
$$
And simultaneously update $\theta_0$ and $\theta_1$
$$
temp0 := \theta_0 - \alpha \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1) \\
temp1 := \theta_1 - \alpha \frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1) \\
\theta_0 := temp0 \\
\theta_1 := temp1
$$
Here,

- $\alpha$ is the learning rate

**Note:**

- Assignment Operator will be denoted by $a:=b$ (takes the value of $b$ and overwrite the value of $a$ with that value)
- Truth Assertion Operator will be denoted by $a=b$ (assert/claim that the value of $a$ is equal to the value of $b$)

Example of **incorrect** implementation of simultaneous update (can behave in strange ways):
$$
temp0 := \theta_0 - \alpha \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1) \\
\theta_0 := temp0 \\
temp1 := \theta_1 - \alpha \frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1) \\
\theta_1 := temp1
$$

> Suppose $\theta_0 = 1$ and $\theta_1=2$, and we simultaneously update $\theta_0$ and $\theta_1$ using the rule $\theta_j := \theta_j + \sqrt{\theta_0 \theta_1}$ (for $j=0$ and $j=1$). What are the resulting values of $\theta_0$ and $\theta_1$?
>
> - [ ] $\theta_0 = 1, \theta_1 = 2$
> - [x] $\theta_0 = 1 + \sqrt{2}, \theta_1 = 2 + \sqrt{2}$
> - [ ] $\theta_0 = 2 + \sqrt{2}, \theta_1 = 1 + \sqrt{2}$
> - [ ] $\theta_0 = 1 + \sqrt{2}, \theta_1 = 2 + \sqrt{(1+\sqrt{2}) * 2}$


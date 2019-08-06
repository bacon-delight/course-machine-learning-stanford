# Overfitting



There are two ways that our model can end up not performing well:

- Underfit / High Bias
- Overfit / High Variance

Overfitting occurs if we have too many features and the learnt hypothesis fits the training set too well and may even turn out to be 0 or close to 0, but fail to generalize new examples
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}*m (h_\theta(x^{(i)})-y^{(i)})^2 \approx 0
$$


> Consider the medical diagnosis problem of classifying tumours as malignant or benign. If a hypothesis has overfit the training set, it means that:
>
> - [ ] It makes accurate predictions for examples in the training set and generalizes well to make accurate predictions on new, previously unseen examples
> - [ ] It does not make accurate predictions for examples in the training set, but it does generalize well to make accurate predictions on new, previously unseen examples
> - [x] It makes accurate predictions for examples in the training set, but it does not generalize well to make accurate predictions on new, previously unseen examples
> - [ ] It does not make accurate predictions for examples in the training set and does not generalize well to make accurate predictions on new, previously unseen examples.



#### Addressing Overfitting

- Reduce Number of Features
    - Manually select which features to keep
    - Model Selection Algorithm
- Regularization
    - Keep all the features, but reduce magnitude or values of parameters $\theta$
    - Works well when we have a lot of features, each of which contributes a bit to predicting $y$





## Cost Function Intuition

![Intuition](images/image01.png)

Suppose that we penalize and make $\theta_3$ and $\theta_4$ really small,
$$
\min_\theta \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2
$$
Now, let’s modify the above objective and create a new one,
$$
\min_\theta \frac{1}{2m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2 + 1000 \sdot \theta_3^2 + 1000 \sdot \theta_4^2
$$
We’ll end up with $\theta_3 \approx 0$ and $\theta_4 \approx 0$, which will essentially be a quadratic function.



#### Regularization

If we have small values for the parameters $\theta_0, \theta_1, \cdots, \theta_n$

- We end up with a simpler hypothesis
- Our model will be less prone to overfitting

Our cost function will be as follows:
$$
J(\theta) = \frac{1}{2m} \Biggl[\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2 + \lambda \sum_{j=1}^n \theta_j^2 \Biggr]
$$

Here, $\lambda = \text{Regularization Parameter}$. It controls the trade-off between the goal of fitting the training set well and the goal of keeping the parameters small.



> What if $\lambda$ is set to an extremely large value (perhaps too large for our problem, say $\lambda=10^{10}$?
>
> - Algorithm works fine
> - Algorithm fails to eliminate overfitting
> - Algorithm results in underfitting (fails to fit even the training set)
> - Gradient descent will fail to converge



#### Gradient Descent

*repeat* {
$$
\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)}) \sdot x_0^{(i)} \\
\theta_j := \theta_j - \alpha \Biggl[\frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)}) \sdot x_j^{(i)} + \frac{\lambda}{m} \sdot \theta_j \Biggr]
\quad \quad \quad
(j=1, 2, 3, \cdots, n)
$$
}

To simplify the above, we can rewrite it as follows:

*repeat* {
$$
\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)}) \sdot x_0^{(i)} \\
\theta_j := \theta_j(1 - \alpha \frac{\lambda}{m}) - \frac{\alpha}{m} \sdot \sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)}) \sdot x_j^{(i)}
\quad \quad \quad
(j=1, 2, 3, \cdots, n)
$$
}

Here $(1 - \alpha \frac{\lambda}{m}) < 1$


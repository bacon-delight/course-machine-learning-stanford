# Bias vs Variance

![Bias vs Variance](images/image01.png)

We have,

- Training Error
    $$
    J_{train}(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)})-y^{(i)})^2
    $$

- Cross Validation Error
    $$
    J_{cv}(\theta) = \frac{1}{2m_{cv}} \sum_{i=1}^{m_{cv}} (h_\theta(x_{cv}^{(i)})-y_{cv}^{(i)})^2
    $$

![Plotting Train and CV Errors](images/image02.png)

As we increase the degree of our polynomial, we see that the error decreases. The cross validation error will start with very high error, reach its lowest point somewhere where the degree of polynomial is not too high or too low, and again increase as the degree of polynomial increases, which is basically reaching overfitting. Plotting these like the one above, can help resolve errors much quickly.



So, if the algorithm is suffering from a bias problem (underfit),

- $J_{train}(\theta)$ will be high
- $J_{cv}(\theta) \approx J_{train}(\theta)$

If the algorithm is suffering from a high variance (overfit),

- $J_{train}(\theta)$ will be low
- $J_{cv}(\theta) > J_{train}(\theta)$


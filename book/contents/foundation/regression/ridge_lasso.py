# %% [markdown]
"""
# Regularization

Another way to avoid overfitting models is to use "shrinkage methods" that constrain / regularize 
the parameter estimates. 
"""

# %% [markdown]
"""
#### Ridge

Ridge regression minimizes $RSS + \lambda \sum_{j=1}^{p}\beta_j^2$ where $\lambda$ is a "tuning parameter" and 
the term $\lambda \sum_{j=1}^{p}\beta_j^2$ is the "shrinkage penalty". 

As $\lambda$ increases, the penalty gets larger so the estimated coefficients "shrink" towards zero. 

Another formulation would be to minimize RSS subject to a constraint that $\sum_{j=1}^{p}\beta_j^2 < s$, equivalent to 
saying that the l2-norm ($\sqrt{\sum_{j=1}^{p}\beta_j^2}$) is less than some s.

#### Lasso

Unlike ridge, lasso can set some coefficients to zero (ridge regression would set coefficient to 0 only if 
$\lambda$ is infinity). Lasso minimizes $\lambda \sum_{j=1}^{p}|\beta_j|$ 

So another formulation would be to minimize RSS subject to a constraint that $\sum_{j=1}^{p}|\beta_j| < s$, equivalent to 
saying that the l2-norm ($\sum_{j=1}^{p}|\beta_j|$) is less than some s.

#### Lasso / Ridge

In both models, cross validate to estimate the shrinkage penalty.

The only difference is that Lasso uses a l1 penalty instead of l2. 

One way to visually understand why l1 can result in estimates of 0 while l2 does not is to note that 
l1 norm can be represented as a diamond shape and l2 norm can be represented as a circle. The diamond 
has corners on the axes making it more likely that the optimal estimates lie where some coefficients 
are exactly zero.

Also, to write a little about why constraining the magnitude of coefficients helps to prevent overfitting -- 
I think another way to describe an overfit model woud be to say that the decision surface is steep and the model 
can switch from predicting high probabilities to low probabilities with small changes in the predictors. 
This would not be possible with small coefficients. 
"""

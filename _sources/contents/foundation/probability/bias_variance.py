# %% [markdown]
"""
# Bias Variance

When we fit a model to data, the error can be decomposed to a bias component and a variance component. 
Often, there's atradeoff between these two components.

If the model is simple, there is likely high bias and low variance. We also call this "underfitting". 
Our estimated parameter may not be as accurate, but there is less variability in our estimate: if we 
got a new sample, we'd expect that our model would estimate similar parameters. 

If we fit a complex model with many parameters, we likely have low bias and high variance, a.k.a. "overfitting". 
We estimate a parameter that may be very accurate, but it's likely highly sensitive to this particular 
training set.
"""

# %% [markdown]
"""
## Bias-Variance Decomposition of MSE

First, define bias and variance:

$$\text{Bias} = E[\hat{\theta}] - \theta$$

$$Var(\hat{\theta}) = E[(E[\hat{\theta}] - \hat{\theta})^2]$$

Let $S = (y - \hat{y})^2 be squared loss.

Then:

$$S = (y - \hat{y})^2$$
$$ = (y - E[\hat{y}] + E[\hat{y}] - \hat{y})^2$$
$$ = (y - E[\hat{y}])^2 + (E[\hat{y}] - \hat{y})^2 + 2(y-E[\hat{y}])(E[\hat{y}] - \hat{y})$$

taking the expected value of both sides to get $E[S]$, mean squared error:

$$E[S] = E[(y - \hat{y})^2]$$
$$ = (y - E[\hat{y}])^2 + E[(E[\hat{y}] - \hat{y})^2] + 0$$
$$ = [\text{Bias}]^2 + \text{Variance}$$


## MVUE

In general, we prefer an estimator with good MSE properties, e.g. low bias and low variance. 
An unbiased parameter with minimum variance is called the MVUE (minimum variance unbiased estimate).
This is far too theoretical for discussion here, but search "Cramer-Rao bound"
"""

# %%

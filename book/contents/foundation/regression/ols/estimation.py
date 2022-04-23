# %% [markdown]
"""
# Estimation
"""

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

n = 100
np.random.seed(0)
df = pd.DataFrame({
    "x1":np.random.normal(10,2,n),
    "e":np.random.normal(0,2,n)
})
y = 3 + 8*df["x1"] + df["e"]

# %% [markdown]
"""
## Least squares solution:

$$\hat{\beta} = (X'X)^{-1}X^{-1}y$$
"""
# %%
X = np.array(sm.add_constant(df[["x1"]], prepend=True))

# %%
# Least squares solution:
np.linalg.lstsq(X,y)

# %%
# check
mod = sm.OLS(y, X)
mod.fit().summary()

# %% [markdown]
"""
## Maximum Likelihood solution:

PDF of Normal Distribution:

$$f(y | \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{\frac{-(y-\mu)^2}{2\sigma^2}}$$

The Likelihood is the product of he individual probabilities of each data point:

$$L(\mu, \sigma | y) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}e^{\frac{-(y_i-\mu)^2}{2\sigma^2}}$$

Then log likelihood after some algebra:

$$ln(L(\mu, \sigma | y)) = -\frac{n}{2} ln(2\pi) - \frac{n}{2} ln(\sigma^2) - \frac{\sum_i^n(y_i-u)^2}{2\sigma^2}$$
"""
# %%
import scipy
from scipy.stats import norm

def log_lik(par_vec, y, X):
    # If the standard deviation prameter is negative, return a large value:
    if par_vec[-1] < 0:
        return(1e8)
    # The likelihood function values:
    lik = norm.pdf(y, 
                   loc = X.dot(par_vec[0:-1]), 
                   scale = par_vec[-1])
    
    # If all logarithms are zero, return a large value
    if all(v == 0 for v in lik):
        return(1e8)

    # Logarithm of zero = -Inf
    return(-sum(np.log(lik[np.nonzero(lik)])))
# %%
res = scipy.optimize.minimize(fun=log_lik, x0=[0,0,10], method = 'BFGS', args=(y,X))
res

# %%

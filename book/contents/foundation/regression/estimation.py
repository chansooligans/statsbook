# %% [markdown]
"""
# Estimation
"""

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import patsy
sns.set(rc={'figure.figsize':(11.7,8.27)})


# %% [markdown]
"""
## OLS / WLS

OLS and WLS solutions can be directly estimated using linear algebra.
"""

# %% [markdown]
"""
#### OLS

Given an OLS model:

$$Y = X\beta + \epsilon$$

If $X'X$ is invertible (columns of X are independent), least squares solution $\hat{\beta}$ is given by:

$$\hat{\beta} = (X'X)^{-1}X^Ty$$

e.g. many different ways to show this, one is by minimizing sum of squared errors:

$$e'e = (y-X\hat{\beta})'(y-X\hat{\beta})$$

$$e'e = y'y - 2\hat{\beta}X'y + \hat{\beta}X'X\hat{\beta}$$

$$\frac{\partial(e'e)}{\partial(\hat{\beta})} = -2X'y + 2X'X\hat{\beta} = 0$$

gives us normal equations:

$$X'X\hat{\beta} = X'y$$

Then multiply by $(X'X)^{-1}$ on both sides
"""

# %% [markdown]
"""
#### WLS

WLS relaxes homoscedasticity assumption using weights to estimate a non-constant variance-covariance error matrix.
Weights are typically assumed or estimated using the data. Observations assigned larger weights are worth more, so 
the regression line would be closer to these points.

Suppose OLS with constants w:

$$Y = X\beta + \epsilon$$

$$\epsilon_i \sim N(0, \sigma^2 / w_i)$$

Let matrix W be a diagonal matrix containing these weights, w.

The weighted least squares estimate is:

$$\hat{\beta}_{WLS} = (X'WX)^{-1}X'Wy$$

e.g. minimize sum of squared errors, just like OLS:

$$e'e = \sum_i^n{w_i(y_i - x_i'\beta)^2} = (Y-X\beta)'W(Y-X\beta)$$
"""

# %% [markdown]
"""
***
## GLM: Generalized Linear Models

#### Applied Example: Logistic Regression 

Likelihood:

$$L(\beta_0,\beta) = \prod_{i=1}^{n} p(x_i)^{y_i}(1-p(x_i))^{1-y_i}$$

Log-likelihood:

$$l(\beta_0,\beta) = \sum_{i=1}^{n} y_i\log{p(x_i)} + (1-y_i)\log{(1-p(x_i))}$$

Derivative of log-likelihood with respect to $\beta$

$$\frac{\partial l}{\partial \beta_j} = \sum_{i=1}^{n}(y_i-p(x;\beta_0,\beta))x_{ij}$$

Goal is to set this derivative to zero and solve.. but there is no closed-form solution. So we 
can approximate using a variety of methods.
"""

# %% [hide-cell]
def inverse_logit(x):
    return(1 / (1 + np.exp(-x)))

n = 500
np.random.seed(0)
df = pd.DataFrame({
    "x1":np.random.normal(1,1,n),
    "x2":np.random.normal(-2,1,n),
})

df["eta"] = -3 + 2*df["x1"] - 1.5*df["x2"]
df["probs"] = inverse_logit(df["eta"])
df["y"] = np.random.binomial(size=n, n=1, p=df["probs"])


# %% [markdown]
"""
***
## Optimizers

#### Newton's Method for Numerical Optimization (Newton-Raphson):

The aim is to find where $t(x) = 0$.
The slope of $t$ at a value $x_n$ is given by:

$$t'(x_n) = \frac{t(x_{n+1})-t(x_n)}{x_{n+1}-x_n}$$

Set $t(x_{n+1}) = 0$, and re-arrange:

$$\beta_{n+1} = x_n - \frac{t(x_n)}{t'(x_n)}$$

**Generalized to higher dimensions**

$$\beta_{n+1} = \beta_n - \nabla^2t(x^n)^{-1}\nabla t(x^n)$$

where $\nabla^2t(x^n)^{-1}$ is the inverse of the Hessian matrix of t at $x^n$

#### IRLS

see: https://ocw.mit.edu/courses/mathematics/18-650-statistics-for-applications-fall-2016/lecture-slides/MIT18_650F16_GLM.pdf  
also: https://ee227c.github.io/code/lecture24.html  

$$\beta_n - \nabla^2t(x^n)^{-1}\nabla t(x^n) => \beta_n - (X^TWX)^{-1}X^T(y-p)$$

"""

# %%
# Newton's optimization:

from numpy.linalg import inv

constant = np.repeat(1,n)
X = np.vstack((constant,df["x1"],df["x2"])).T
y = df["y"]

def newton_raphson(y, X):
    beta = np.array([0,0,0])
    delta=np.array([1,1,1])
    i = 1
    while delta.max() > 0.0000001:
        i += 1
        probs = np.exp(X @ beta) / (1 + np.exp(X @ beta))
        W = np.diag(probs * (1-probs))
        grad = -X.T @ (y-probs)
        hess = (X.T @ W) @ X
        delta = inv(hess) @ grad
        beta = beta - delta
        if i == 100000:
            break
    print(beta)
    return beta

def irls(y, X):
    beta = np.array([0,0,0])
    delta=np.array([1,1,1])
    i = 1
    while delta.max() > 0.0000001:
        i += 1
        probs = np.exp(X @ beta) / (1 + np.exp(X @ beta))
        W = np.diag(probs * (1-probs))
        grad = -X.T @ (y-probs)
        hess = (X.T @ W) @ X
        delta = np.linalg.lstsq(hess, grad)[0]
        beta = beta - delta
        if i == 100000:
            break
    print(beta)
    return beta

np.round(irls(y, X)-newton_raphson(y, X),0)

# %%
# check against statsmodels
f = 'y ~ x1 + x2'
y, X = patsy.dmatrices(f, df, return_type='dataframe')
mod = sm.Logit(y, X)
mod.fit().summary()

# %% [markdown]
"""
# BFGS
"""

# %%

# %% [markdown]
"""
# Model Comparisons
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
Below, I run three models:
- model 1: intercept only
- model 2: adding predictor x1
- model 3: adding both predictors x1 and x2

How do we know which model fits the data the best?
"""

# %%
def inverse_logit(x):
    return(1 / (1 + np.exp(-x)))

n = 500
np.random.seed(0)
df = pd.DataFrame({
    "x1":np.random.normal(1,1,n),
    "x2":np.random.normal(-2,1,n),
})

df["eta"] = - 3 + 2*df["x1"] - 1.5*df["x2"]
df["probs"] = inverse_logit(df["eta"])
df["y"] = np.random.binomial(size=n, n=1, p=df["probs"])
df["y"].value_counts()

def run_logistic_model(formula):
    y, X = patsy.dmatrices(formula, df, return_type='dataframe')
    mod = sm.GLM(y, X, family=sm.families.Binomial())
    res = mod.fit()
    print(res.summary())

# %%
mod1 = run_logistic_model("y ~ 1")
# %%
mod2 = run_logistic_model("y ~ x1")
# %%
mod3 = run_logistic_model("y ~ x1 + x2")

# %%
mod4 = run_logistic_model("y ~ x1 + x2 + x1*x2")

# %% [markdown]
"""
## Likelihood Ratio Tests

The likelihood tells us the probability of seeing the data if the parameter estimates are true. 
We want to identify a model that maximizes the likelihood. 
Given a few models, we can compare their likelihoods to determine if one yields a significantly higher 
likelihood. This comparison is called a likelihood ratio (LR) test.

The LR test uses a LR test statistic that is chi-square distributed to test for significance. 
This test statistic is also called the deviance (log-likelihood ratio statistic).

$$LR = -2ln\left(\frac{L(m1)}{L(m2)}\right) = 2(loglike(m2) - loglik(m1))$$

where L(m1) and L(m2) are the likelihoods of two models.

Suppose we want to compare model 1 (intercept only) to model 2 (adds x1). 
From the model summaries above:

$$2(loglike(m2) - loglike(m1)) = 2((-227.85) - (-286.53)) = 117.36$$

The test statistic is chi-squared with 1 degree of freedom, since m2 adds 1 predictor 
to m1. If we were comparing m3 to m1, there would be 2 degrees of freedom.
"""

# %%
from scipy.stats.distributions import chi2
p = chi2.sf(117.36, 1)
print(("{:.20f}".format(p)))

# %% [markdown]
"""
Since p < 0.05 (in fact, much smaller), we can say m2 significantly improves m1.

Comparing model 4 to model 3 (the correctly specified model), the deviance test statistic:

$$2(loglike(m4) - loglike(m3)) = 2((-190.90) - (-190.93)) = .06$$

The chi-square test below shows that the new model does not offer a significantly better fit.
"""
# %%
p = chi2.sf(.06, 1)
print(("{:.20f}".format(p)))


# %% [markdown]
"""
## AIC and BIC
"""


# %%

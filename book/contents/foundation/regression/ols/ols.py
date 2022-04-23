# %% [markdown]
"""
# Demo

Ordinary Least Squares / Linear Regression
"""

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

# %%
n = 500
np.random.seed(0)
df = pd.DataFrame({
    "x1":np.random.normal(10,1,n),
    "x2":np.random.normal(2,1,n),
    "e":np.random.normal(0,1,n)
})

df["y"] = 2 + 3*df["x1"] + (-2)*df["x2"] + df["e"]

# %%
X = sm.add_constant(df[["x1","x2"]], prepend=False)

# %%
mod = sm.OLS(df["y"], X)

# %%
res = mod.fit()

# %%
print(res.summary())


# %% [markdown]
"""
## Terms:

### Goodness of Fit:  
- R-squared: proportion of variance of dependent variable explained by covariates  
- Adjusted R-squared: adjusts R-squared for the number of predictors in the model  
- F-statistic: tests null hypothesis that all coefficients are zero  
- Prob (F-statistic): low p-value means current model is more significant than intercept-only model  
- Log-Likelihood: $log(p(X|\mu\Sigma)$ log of probability that data is produced by this model   
- AIC: $-2logL + kp$ with $k=2$; lower AIC = better fit  
- BIC:  $-2logL + kp$ with $k=log(N)$; lower BIC = better fit
    - BIC penalizes model complexity more than AIC

### Tests for normal, i.i.d residuals
- Omnibus: $K^2$ statistic 
    - [D'Agostino's K-squared test](https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test)  
    - If null hypotehsis of normality is true, then $K^2$ is approximately $\chi^2$ distributed with 2 df  
- Prob(Omnibus): small p-value means reject null of normal dist
- Skew: perfect symmetry = 0  
- Kurtosis:  normal distribution = 3
- Durbin-Watson:  Tests for autocorrelation, independence of errors
    - Ideally between 1 and 2
- Jarque-Bera (JB): also tests normality of residuals
- Prob(JB): small p-value means reject null of normal dist
- Cond. No.: used to diagnose multicollinearity; 
    - it is the condition number of the design matrix of the covariates  
"""
# %%

# %% [markdown]
"""
# WLS

Use Cases:

- Using observed data to represent a larger population: Suppose you are using a sample from a survey that oversampled from some groups. We could use the survey weights in a WLS to minimize SSE
 with respect to the population instead of the sample. Let $\pi_g$ be the population share of group g. And let $h_g$ be the sample share of group G. Then define 
weight for unit i to be $w_i$ as $\frac{pi_{gi}}{h_{gi}}$.
- duplicate observations (or aggregated data): if observations with covariates {X_i} appear multiple times, 
you could include these observations just once and set $w_i$ to be the number of times they appear.
- Unequal variance: if there is a lot of heteroskeasticity, you can set weights to the inverse of the error
variance

"""

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import patsy

# %%
# create two groups for error variance, low and high
n = 300
np.random.seed(0)
df = pd.DataFrame({
    "x1":np.random.normal(10,1,n),
    "e":np.random.normal(0,1,n)
}).sort_values('x1')

sigma = 0.5
w = np.ones(n)
w = df["x1"] / 2

df["y_true"] = 2 + 3*df["x1"] 
df["y_obs"] = df["y_true"] + w*sigma*df["e"]

# %%
sns.scatterplot(df["x1"], df["y_obs"])
sns.lineplot(df["x1"], df["y_true"], color="red")

# %% [markdown]
"""
#### With known weights:
"""

# %%
f = 'y_obs ~ x1'
y, X = patsy.dmatrices(f, df, return_type='dataframe')
mod = sm.WLS(y, X, weights = 1 / (w**2))
res = mod.fit()
print(res.summary())

# %% [markdown]
"""
#### Estimating Weights

Here, we do not know the weights. To estimate them, we start by running simple OLS to get 
residuals. Next, we regress root of absolute value of residuals against X and save fitted values. 
Last, we run a weighted least squares model, using the fitted values as $\sigma$. 
Then weights are $1 / \sigma^2$

Lots of different methods, see [here for more details](https://online.stat.psu.edu/stat462/node/186/).
"""

# %%
# simple OLS without weights
f = 'y_obs ~ x1'
y, X = patsy.dmatrices(f, df, return_type='dataframe')
mod_ols = sm.OLS(y, X)
res_ols = mod_ols.fit()

# regress root of absolute deviation against X
mod_resid = sm.OLS(np.sqrt(abs(res_ols.resid)), X)
res_resid = mod_resid.fit()

# %%
# plot true weights, w against fitted values
sns.scatterplot(w, res_resid.fittedvalues)

# %%
mod_2stage = sm.WLS(y, X, 1 / (res_resid.fittedvalues**2))
print(mod_2stage.fit().summary())

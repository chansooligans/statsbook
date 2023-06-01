# %% [markdown]
"""
# Interpretation
"""

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import patsy
sns.set(rc={'figure.figsize':(11.7,8.27)})

# %%
n = 500
np.random.seed(0)
df = pd.DataFrame({
    "x1":np.random.exponential(2,n),
    "x2":np.random.normal(2,1,n),
})

eta = np.exp(2 + 0.25*df["x1"] + 0.25*df["x2"])
df["n_students_flagged"] = np.random.poisson(lam=eta)
df["n_students_flagged"].value_counts()
df["n_enrolled"] = np.round(df["n_students_flagged"],0) + np.random.binomial(200,0.8,n)

# %%
f = 'n_students_flagged ~ x1 + x2'
y, X = patsy.dmatrices(f, df, return_type='dataframe')
mod = sm.GLM(y, X, family=sm.families.Poisson())
print(mod.fit().summary())

# %% [markdown]
"""
We interpret the coefficients by exponentiating them and treating them as multiplicative effects.

$$E(Y|X) = log(\mu) = X\beta$$
$$\mu = exp(X\beta)$$
$$\mu = exp(b_0) * exp(b_1x_1) * exp(b_2x_2)$$

The coefficient of $x_1$ is the expected difference in $log(y)$ for a one unit increase in $x_1$. We can 
say that for a one unit increase in $x_1$, we expect a $exp(b_1)$ increase in y. 
"""
# %%
print(f"{round(100*(np.exp(0.2469) - 1),2)}% increase")

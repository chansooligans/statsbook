# %% [markdown]
"""
# Multivariate Distributions

Key concepts for this section:
1. Correlation / covariance
2. Contingency Table
3. Plotting joint/conditional/marginal distributions
4. Simpson's paradox, confounder?
"""

# %% tags=['hide-cell']
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

# %%
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
sns.set(rc={'figure.figsize':(9,4)})

# %%
# Fake Data:
n=100
df = pd.DataFrame({
    "a":np.random.normal(10, 1, n),
})
df["c"] = df["a"] + np.random.normal(1, 0.5, n)

# %% [markdown]
"""
## Correlation / Covariance

**Covariance** indicates the level to which two variables vary together.

$$Cov(X,Y) = \frac{1}{n}\sum_{i=1}^n(x_i - E(X))(y_i-E(Y))$$
"""

# %%
# Compute covariance manually:
products = [
    (row["a"]-df["a"].mean()) * (row["c"]-df["c"].mean())
    for i,row in df.iterrows()
]
covariance = np.sum(products)/len(df)
print(covariance)

# %%
# Using numpy:
# if bias = False, computes "sample covariance" so denominator is N-1
np.cov(df["a"], df["c"], bias=True)


# %% [markdown]
"""
**Pearson correlation coefficient** is the covariance divided by product of standard deviations

$$\rho_{X,Y} = \frac{cov(X,Y)}{\sigma_X\sigma_Y}$$

also see: [spearman's rank correlation coefficient](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
"""

# %%
corr = covariance / (np.std(df["a"]) * np.std(df["c"]))
corr

# %%
np.corrcoef(df["a"], df["c"])

# %% [markdown]
"""
## Contingency Table
"""

# %%
df_cat = pd.DataFrame({
    "colors":np.random.choice(["red","blue","green","yellow","orange"], size=30, replace=True),
    "names":np.random.choice(["jason","jorge","lisa","paul"], size=30, replace=True),
    "states":np.random.choice(["california","arizona","oregon"], size=30, replace=True),
})

# %%
pd.crosstab(
    [df_cat["states"],df_cat["names"]], 
    df_cat["colors"], 
    margins=True
)

# %%
pd.crosstab(
    [df_cat["states"],df_cat["names"]], df_cat["colors"], 
    margins=True, 
    normalize=True
)

# %% [markdown]
"""
## Joint / Conditional / Marginal Distributions
"""

# %% [markdown]
"""
joint distribution  

$$f(a,c)$$
"""
# %%
sns.jointplot(df["a"],df["c"])

# %% [markdown]
"""
conditional distribution  

$$f(a|c>0)$$
"""
# %%
sns.kdeplot(df.loc[df["c"]>0,"a"])

# %% [markdown]
"""
marginal distribution  

$$f(a)$$
"""
# %%
sns.kdeplot(df["a"])

# %% [markdown]
"""
## Simpson's Paradox
"""

# %%
df2 = pd.DataFrame({
    "a":np.random.normal(8, 1, n),
})
df2["c"] = df2["a"] + np.random.normal(8, 0.5, n)
df["group"] = 0
df2["group"] = 1
df_simpson = pd.concat([df[["a","c","group"]],df2])

# %%
sns.regplot(df_simpson["a"], df_simpson["c"], scatter=False, color="red")
sns.scatterplot(df_simpson["a"], df_simpson["c"], hue=df_simpson["group"])

# %%

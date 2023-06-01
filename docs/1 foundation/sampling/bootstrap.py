# %% [markdown]
"""
# Bootstrap

Suppose you have a sample of 100 units. You want to construct a confidence interval
of your sample mean. How can you bootstrap to construct a sampling distribution?
"""

# %% tags=['hide-cell']
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from tqdm import tqdm
sns.set(rc={'figure.figsize':(11.7,8.27)})

# %% [markdown]
"""
## Fake Data:
"""

# %%
x = np.random.chisquare(15, 100)

# %% [markdown]
"""
## Bootstrap:

Repeatededly draw samples of size X (e.g. 40) from your sample of 100, with replacement.  
For each sample, compute the mean.
"""

# %%
def bootstrap(x, n_samples = len(x), n_resamples = 9999):
    return [
        np.mean(np.random.choice(x, size=n_samples, replace=True))
        for _ in range(n_resamples)
    ]

# %%
sample_means = bootstrap(x)

# %%
sns.kdeplot(sample_means)
plt.axvline(np.mean(sample_means), 0,1, color="red")

# %%
# 95% confidence interval
print(
    (
        np.quantile(sample_means,0.025), 
        np.quantile(sample_means,0.975)
    )
)

# %% [markdown]
"""
## Bootstrap Using Scipy Stats
"""
# %%
res = scipy.stats.bootstrap((x,), np.mean, confidence_level = 0.95)
# %%
res.confidence_interval

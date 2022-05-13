# %% [markdown]
"""
# P-values
"""

# %%
import numpy as np
np.random.seed(0)
from statsmodels.stats.weightstats import ztest
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
import scipy

# one-sided hypothesis test
# NULL Hypothesis: mean = 0
# ALT Hypothesis: mean > 0

# hypothetical population distribution
pop = np.random.normal(1,2,1000) 

# what we observe:
sample_size = 40
sample = np.random.choice(pop, sample_size)
sample_mean = np.mean(sample)
sample_var = np.var(sample)

# %% [markdown]
"""
## test statistic and pvalue

let $\bar{y}$ be sample mean, then test statistic is $\frac{\bar{y}-0}{\sqrt{\sigma^2/n}}$
where population mean is 0 and the denominator is the population variance

we know the population variance bc this is fake data, but usually, we estimate the population 
variance with the sample variance, which we can do if the sample size is large enough
"""

# %%
# using statsmodels' ztest:
test_statistic, pvalue = ztest(sample, value=0, alternative='larger')
print(test_statistic, pvalue)

# %%
test_statistic = (sample_mean - 0) / np.sqrt(sample_var/(sample_size-1))
pvalue = 1-scipy.stats.norm.cdf(abs(test_statistic), 0, 1)
print(test_statistic, pvalue)

# %% [markdown]
"""
## simulations
"""

# %%
def sim(pop, sample_size = 40):
    sample = np.random.choice(pop, sample_size)
    sample_mean = np.mean(sample)
    sample_var = np.var(sample)
    test_statistic = (sample_mean - 0) / np.sqrt(sample_var/(sample_size-1))
    pvalue = 1-scipy.stats.norm.cdf(abs(test_statistic), 0, 1)
    return pvalue

# %%
n = 20
pvalues = [sim(pop, sample_size=n) for i in range(100)]
sns.histplot(pvalues)

# %%
n = 60
pvalues = [sim(pop, sample_size=n) for i in range(100)]
sns.histplot(pvalues)

# %% [markdown]
"""
## but what if null is true
"""
# %%
pop = np.random.normal(0,2,1000) 

# %%
n = 20
pvalues = [sim(pop, sample_size=n) for i in range(100)]
sns.histplot(pvalues)

# %%
n = 60
pvalues = [sim(pop, sample_size=n) for i in range(100)]
sns.histplot(pvalues)

# %% [markdown]
"""
## but what if null is slightly different
"""
# %%
pop = np.random.normal(0.01,2,1000) 

# %%
n = 20
pvalues = [sim(pop, sample_size=n) for i in range(100)]
sns.histplot(pvalues)

# %%
n = 60
pvalues = [sim(pop, sample_size=n) for i in range(100)]
sns.histplot(pvalues)

# %%
n = 20_000
pvalues = [sim(pop, sample_size=n) for i in range(100)]
sns.histplot(pvalues)

# %% [markdown]
"""
## but if population variance is very narrow
"""

# %%
pop = np.random.normal(0.01,0.025,1000) 

# %%
n = 40
pvalues = [sim(pop, sample_size=n) for i in range(100)]
sns.histplot(pvalues)


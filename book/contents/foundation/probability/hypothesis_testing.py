# %% [markdown]
"""
# Hypothesis Testing

Elements of a Statistical Test:
1. Null Hypothesis: the hypothesis to be tested  
2. Alternative Hypothesis: hypothesis to be accepted in case null hypothesis is rejected  
3. Test Statistic: a function of the sample measurements (e.g. sample mean)
4. Rejection Region: values of the test statistic for which the null hypothesis is to be rejected
"""

# %% [markdown]
"""
## Type I/II Errors

- A **type I error** is made if $H_0$ is rejected when $H_0$ is true  
    - the probability of a type I error is the "significance level" of a test, denoted by $\alpha$
- A **type II error** is made if $H_0$ is not rejected when $H_1$ is true
    - the probability of a type II error is denoted by $\beta$

Consider an example with election data where we want to know the percent of voters in support. 
Let $H_0 : p = 0.4$ be our null hypothesis and $H_a : p < 0.4$ be our alternative hypothesis. 
Suppose that we collect a sample of 15 voters and our rejection region is $y \leq 2$. That is, 
if there are fewer than 2 votes in support, we reject the null hypothesis in favor of the 
alternative.

What is alpha and beta?
"""

# %% [markdown]
"""
#### alpha

Assume the null hypothesis is true. Our rejection region states that we reject the null hypothesis if 
our test statistic $y$ (the # of voters in support) $\leq 2$. So what is the probability that $y \leq 2$, given 
$p = 0.5$ (null hypothesis)?
"""

# %%
# since the statistic is the sum of repeated binary variable, binomial distribution is appropriate
from scipy.stats import binom

alpha = sum([
    binom.pmf(k=y, n=15, p=0.5)
    for y in [0,1,2]
])
print(round(alpha,3))

# %% [markdown]
"""
#### beta

Assume the null hypothesis is false and true vote support is p = 0.2. 
Our rejection region states that we do NOT reject the null hypothesis if our test statistic 
$y > 2$. So what is the probability that we fail to reject the null when it is false? 
In other words, what is the probability that $y > 2$ given $p = 0.2$?
"""

# %%
beta = sum([
    binom.pmf(k=y, n=15, p=0.2)
    for y in range(3,16)
])
print(round(beta,3))

# %% [markdown]
"""
#### if alpha = 0.05

Typically, we define our rejection region not based on a value of the test statistic, but 
based on a value of $\alpha$, specifically $\alpha = 0.05$. If we reject the null if 
$\alpha < 0.05$, how many votes would we need in the sample of 15 to NOT reject the null?
"""

# %%
reject = binom.ppf(0.05, n=15, p=0.5)
print(reject)

# %% [markdown]
"""
So we reject the null if there are less than 4 support-votes in the sample. We do NOT reject if there
 are at least 4 votes.

What would $\beta$ be? ($\alpha$ and $\beta$ are inversely related)
"""

# %% [markdown]
"""
***
## Large Sample Tests

Case study:
```
A company claims that average tips per shift is $15.  
We reach out to 42 randomly selected workers and ask how much they made in tips per shift.  
The mean and variance of the 42 responses are 13 and 36, respectively.  
Does the evidence contradict the company's claims?
```

Couple More Definitions:
- A one-tailed test is a hypothesis test where the rejection region is only one side of the sampling distribution
- A two-tailed test is a hypothesis test that tests whether the test statistic is greater than or less 
than a certain range of values

Here:
- Null Hypothesis: $H_0: \mu = 15$
- Alternative Hypothesis: $H_1: \mu > 15$
- Test Statistic: $Z$
- Rejection region: $RR = Z > k$ for some $k$

With a sufficiently large n, we know that the sample mean $\bar{Y}$ is approximately normally distributed 
with $\mu_{\bar{Y}} = \mu$ and $\sigma_{\bar{Y}}= \frac{\sigma}{\sqrt{n}}$

So our test statistic is:  

$$Z = \frac{\bar{Y}-\mu_{0}}{\sigma/\sqrt{n}}$$

The population variance, $\sigma^2$ is unknown, but can be estimated by teh sample variance $s^2=36$, since 
the sample size is sufficiently large (n=42).

The rejection region, with $\alpha=0.05$ is given by $z < -z_{.05} = -1.645$.

The Z-score is computed below:
"""

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(0)

import scipy.stats as st

# %%
alpha = 0.05
n = 42
sample_mean = 13
sample_var = 36
z_reject = st.norm.ppf(alpha)
Z = (sample_mean - 15) / (np.sqrt(sample_var) / np.sqrt(n))
print(f"reject if less than: {z_reject}")
print(f"test statistic is: {Z}")
print(np.where(Z < z_reject,"reject","do not reject"))

# %% [markdown]
"""
Since the Z test statistic lies in the rejection region, we reject the null hypothesis.  
At the 0.05 level of significance, the evidence is sufficient to indicate that the company's 
claim is incorrect and that the average tips per shift is less than $15
"""

# %% [markdown]
"""
***
#### Large Sample Tests with two-tails

Let the Alternative Hypothesis be: $H_1: \mu != 15$
"""

# %%
alpha = 0.05
n = 42
sample_mean = 13
sample_var = 36
z_reject = abs(st.norm.ppf(alpha/2))
Z = abs((sample_mean - 15) / (np.sqrt(sample_var) / np.sqrt(n)))
print(f"reject if greater than: {z_reject}")
print(f"abs. value of test statistic is: {Z}")
print(np.where(Z > z_reject,"reject","do not reject"))


# %% [markdown]
"""
***
## Small Sample Tests

Assume same setting as the large sample test example. But now, n = 8 instead of 42.

The test statistic is now a t-statistic, which follows a t-distribution with $n-1$ degrees of freedom.
"""

# %%
# one-tail
alpha = 0.05
n = 8
sample_mean = 13
sample_var = 36
t_reject = st.t.ppf(alpha, n-1)
tstat = (sample_mean - 15) / (np.sqrt(sample_var) / np.sqrt(n))
print(f"reject if less than: {t_reject}")
print(f"test statistic is: {tstat}")
print(np.where(tstat < t_reject,"reject","do not reject"))

# %%
# two-tail
alpha = 0.05
n = 8
sample_mean = 13
sample_var = 36
t_reject = abs(st.t.ppf(alpha/2, n-1))
tstat = abs((sample_mean - 15) / (np.sqrt(sample_var) / np.sqrt(n)))
print(f"reject if less than: {t_reject}")
print(f"test statistic is: {tstat}")
print(np.where(tstat > t_reject,"reject","do not reject"))


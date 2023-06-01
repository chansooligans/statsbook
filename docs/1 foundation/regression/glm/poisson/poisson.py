# %% [markdown]
"""
# Demo

A poisson regression is used to model counts and/or rates. A count is zero or any positive integer, such as 
the number of events, the number of students, etc.

A poisson regression (log-linear model) uses a log link function to relate the linear component ($\eta$) 
to the $(0,\infty)$ count variable, e.g. (0,1,2...)

$$E(Y|X) = log(\mu) = X\beta$$

To model rates, we include an "offset" term:

$$E(Y|X) = log(\mu) = log(n) + X\beta$$

Then Y follows a Poisson distribution parameterized by $\mu$: 

$$Y \sim Poisson(\mu)$$
"""

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import patsy
sns.set(rc={'figure.figsize':(11.7,8.27)})

# %% [markdown]
"""
## Generate Fake Data

We generate two independent variables x1 and x2. We let $\eta$ be the linear component, $X\beta$. 

To get $\mu$, we exponentiate both sides of $log(\mu) = \eta$, so we have $\mu = exp(\eta)$

Then y (# of students flagged) is poisson distributed, with its single parameter: $\mu$.

To demonstrate poisson modeling with rates, we create an offset term: n_enrolled, 
which is simply n_student_flagged plus a random positive integer. 
"""

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
sns.kdeplot(df["n_students_flagged"])

# %%
g = sns.kdeplot(df["n_students_flagged"]/df["n_enrolled"])
g.set_xlabel("rate (flagged/enrolled)")

# %%
f = 'n_students_flagged ~ x1 + x2'
y, X = patsy.dmatrices(f, df, return_type='dataframe')

# %% [markdown]
"""
## Statsmodels
"""
# %%
mod = sm.GLM(y, X, family=sm.families.Poisson())
res2 = mod.fit()
print(res2.summary())

# %% [markdown]
"""
using n_enrolled as offset to model rate (flagged out of enrolled) instead of count (# enrolled)

including an offset is equivalent to including the offset as another pedictor, except with the 
coefficient fixed to 1.

$$log(\mu) = log(n) + X\beta$$

$$log(\frac{\mu}{n}) = X\beta$$
"""
# %%
mod = sm.GLM(y, X, offset=df["n_enrolled"], family=sm.families.Poisson())
res2 = mod.fit()
print(res2.summary())

# %%

# %% [markdown]
"""
# Demo

A logitic regression is used when the dependent variable is a binary outcome.

It uses a logit link function  to relate the continuous $(-\inf,\inf)$ linear component ($\eta$)
 to the $(0,1)$ dependent variable, $P(y_i=1)$. Then a bernoulli trial to convert these probabilities into 
 binary data, 0 or 1.

$$P(y=1) = logit(\eta) = X\beta$$

$$y_i \sim Binomial(1, P_i)$$

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
## Generate Fake Data

We generate two independent variables x1 and x2. We let $\eta$ be the linear component, $X\beta$, then 
 inverse_logit($\eta$) is the link function that converts $\eta$ to a probability. 

 $$P(y=1) = inv.logit(X\beta)$$

Then y is the outcome of a bernoulli trial (binomial with 1 trial) with these probabilities.
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

# %% [markdown]
"""
The plot below shows the inverse_logit relation between $\eta$ and the probs.
"""

# %%
g = sns.lineplot(df["eta"], inverse_logit(df["eta"]))
g.set_xlabel("eta")
g.set_ylabel("prob")

# %% [markdown]
"""
The plot below shows the binary classification problem, with $y \in [{0,1}]$.
"""

# %%
sns.scatterplot(df["x1"], df["x2"], hue=df["y"])

# %%
f = 'y ~ x1 + x2'
y, X = patsy.dmatrices(f, df, return_type='dataframe')

# %% [markdown]
"""
## Statsmodels

There are a couple ways to run logistic regression using statsmodels. 
Note that in both cases, in the model output, the Link function is "logit".

See the "interpretation" page for more info about interpreting regression results.
"""
# %%
mod = sm.Logit(y, X)
res = mod.fit()
print(res.summary())

# %%
# alternatively, using GLM and specifying the Binomial family for logistic regression
mod = sm.GLM(y, X, family=sm.families.Binomial())
res2 = mod.fit()
print(res2.summary())

# %% [markdown]
"""
## sklearn
"""
# %%
model = LogisticRegression(fit_intercept = False)
mdl = model.fit(X, y)
model.coef_

# %% [markdown]
"""
default setting is regularization; turn it off by setting `penalty='none'`
"""

# %%
model = LogisticRegression(fit_intercept = False, penalty='none')
mdl = model.fit(X, y)
model.coef_


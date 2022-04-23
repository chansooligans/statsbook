# %% [markdown]
"""
# Interactions
"""
# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import patsy
from matplotlib import pyplot as plt
from scipy.special import expit as invlogit

# %% [markdown]
"""
## OLS
"""
# %%
n = 500
np.random.seed(0)
df = pd.DataFrame({
    "income":np.random.normal(5,1,n),
    # "educ":np.random.choice([1,2,3,4,5], n),
    "e":np.random.normal(0,1,n)
})

df["educ"] = 1 + np.random.binomial(4,df["income"] / df["income"].max())

df["favorability"] = 2 + 1.4*df["income"] + 5*df["educ"] + -0.9*df["income"]*df["educ"] + df["e"]
# %%
sns.scatterplot(df["income"],df["educ"], hue=df["favorability"])

# %%
f = 'favorability ~ income + C(educ) + income*C(educ)'
y, X = patsy.dmatrices(f, df, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
# %%
print(res.summary())

# %% [markdown]
"""
The above example is fake data containing favorability ratings for some politician. Each row is a constituent 
and covariates measure their level of education (categorical variable from 1 to 5) and their income (in 10k).

**Interpretations**:
- The intercept represents the favorability rating for an individual with education level 1 
and 0 income.  
- The coefficient of C(educ)[T.d] represents the average difference in favorability rating between 
an individual with education level 1 and education level 2 **for an individual with income = 0**.
- The coefficient of income represents the average change in favorability given a one unit increase (10k increase)
in income **for an individual with edcuation level = 0**.   
    - If there were no interaction term, this coefficient would represent a change in favorability given a one
    unit increase in income for any individual, regardless of income. But due to interactions, this coefficient 
    can only be interpreted for an individual with education level = 0.
- The coefficient of "income:C(educ)[T.2]" represents the average difference in the slope of the relationship between 
income and favorability, when between an individual with education level 1 vs education level 2. This coefficient can 
also be interpreted as the average change in differences between an education level 1 vs level 2, for a one unit 
increase in income. Here, a coefficient of -0.5144 suggests that the slope of income on favorability is 0.1805 for education
level 1 individuals and 0.1805-0.5144=-0.3339 for education level 2. Another interpretation would be that 
for every one unit increase in income, the gap in favorability between education level 1 and education level 2 (3.6147),
decreases by -0.5144.
"""
# %%
def plot_scatter(df,educ):
    g = sns.scatterplot("income","favorability",data=df.query(f"educ=={educ}"), alpha=0.2)
    g = sns.regplot("income","favorability",data=df.query(f"educ=={educ}"), label=educ, scatter=False)
    return g

for educ in range(1,6):
    g1 = plot_scatter(df,educ)
g1.legend()
    
# %% [markdown]
"""
## Centering 

Whether fitting an OLS, GLM, or MLM, centering can help with easier interpretation when interactions are present.
"""
# %%
df["income_ctr"] = df["income"] - np.mean(df["income"])

# %%
f = 'favorability ~ income_ctr + C(educ) + income_ctr*C(educ)'
y, X = patsy.dmatrices(f, df, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
# %%
print(res.summary())

# %% [markdown]
"""
Note the change in the coefficients for the interaction terms.  
Now, the coefficient for C(educ)[T.2], 1.0560, is the change in favorability 
between an individual with educ=1 and educ=2, with average income level. 
(as opposed to income = 0)
"""

# %% [markdown]
"""
## Logistic

Now, we generate a binary "vote" column.  
Then we model the probability of vote using a logit link.
"""
# %%
df["vote"] = np.random.binomial(1, invlogit(df["favorability"]-np.mean(df["favorability"])), n)

# %%
sns.scatterplot("income","educ",hue="vote", data=df, alpha=0.5)


# %%
f = 'vote ~ income + C(educ) + income*C(educ)'
y, X = patsy.dmatrices(f, df, return_type='dataframe')
mod = sm.Logit(y, X)
res_log = mod.fit()
print(res_log.summary())

# %% [markdown]
"""
And the model after centering:
"""
# %%
f = 'vote ~ income_ctr + C(educ) + income_ctr*C(educ)'
y, X = patsy.dmatrices(f, df, return_type='dataframe')
mod = sm.Logit(y, X)
res_log = mod.fit()
print(res_log.summary())

# %% [markdown]
"""
**Interpretations**:
- The intercept coefficient is -0.8973 before centering and 1.36 after centering. Before centering, 
this means that a voter with educ=1 and income=0 has a $logit^{-1}(-0.8973) = 0.29$ probability of voting
yes. After centering, this means that the voter with educ=1 and average income has a $logit^{-1}(-0.2587) = 0.44$ probability of voting
yes.
- In the centered model, the coefficient for C(educ)[T.2] is 0.1327. For an individual with average income, 
the individual with educ=2 has $logit^{-1}(-0.2587+0.1327) = 0.46$ probability of voting, a decrease of ~0.02 from 0.44.
- In the centered model, the coefficient for C(educ)[T.2] is 0.1327. For an individual with average income, 
the individual with educ=2 has $logit^{-1}(-0.2587+0.1327) = 0.46$ probability of voting, a decrease of ~0.02 from 0.44.
"""
# %%
invlogit(-0.8973)
# %%
invlogit(-0.2587)
# %%
invlogit(0.1327)

# %%
invlogit(-0.2587+0.1327)

# %% [markdown]
"""
To compare individuals with income 0.5 below the mean, -0.5, and education = 3 vs individuals 
with income -0.5 and education = 5  

Then compare with income = 1.  

Verify in plots below.  
"""
# %%
income = -0.5
print(invlogit(-0.2587 + 0.8255 + 0.1284 * income + -1.0049*income))
print(invlogit(-0.2587 + 0.9480 + 0.1284 * income +  -2.8437*income))

# %%
income = 1
print(invlogit(-0.2587 + 0.8255 + 0.1284 * income + -1.0049*income))
print(invlogit(-0.2587 + 0.9480 + 0.1284 * income +  -2.8437*income))

# %%
df["pred"] = res_log.predict()

# %%
def plot_scatter(df,educ):
    g = sns.scatterplot("income_ctr","vote",data=df.query(f"educ=={educ}"), alpha=0.2)
    g = sns.lineplot("income_ctr","pred",data=df.query(f"educ=={educ}"), label=educ)
    return g

for educ in range(1,6):
    g1 = plot_scatter(df,educ)
g1.legend()

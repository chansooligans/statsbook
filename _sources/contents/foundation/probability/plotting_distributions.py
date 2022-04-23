# %% [markdown]
"""
# Plotting Distributions

1. Univariate
- Box Plot
- Swarm Plot
- Violin Plot
- Density Plot
- Histogram
2. Multivariate
- Scatter Plot
- Bar Plot
- Line Plot
- Reg Plot
- LM Plot
- Joint Plot
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

# %% [markdown]
"""
## Univariate
***

First, create fake dataset:
"""

# %%
np.random.seed(0)
df = pd.DataFrame({
    'time':range(1,101),
    'NY':np.random.normal(2,1,100),
    'CA':np.random.binomial(10,0.2,100),
})
df = pd.melt(df, id_vars='time', value_vars=['NY','CA'])
df = df.rename({'variable':'state'}, axis=1)
df['flag'] = np.where(df['time']<=50,1,0)

# %% [markdown]
"""
### Box Plot
"""
# %%
sns.boxplot(x='state', y='value', data=df)

# %% [markdown]
"""
### Swarm Plot
"""
# %%
sns.swarmplot(x='state', y='value', data=df)

# %%
# Show both box and swarm
sns.boxplot(x='state', y='value', data=df, saturation=0.2, width=0.2)
sns.swarmplot(x='state', y='value', data=df)

# %% [markdown]
"""
### Violin Plot
"""
# %%
sns.violinplot(x='state', y='value', data=df)
# %%
# Show both swarm and violin
sns.swarmplot(x='state', y='value', data=df, color='white')
sns.violinplot(x='state', y='value', data=df)

# %%
# Use "split" parameter to draw split violins to compare across hue variable
sns.violinplot(x='state', y='value', hue="flag", split=True, data=df)

# %% [markdown]
"""
### Density Plot
"""
# %%
sns.kdeplot(df["value"])

# %% [markdown]
"""
### Histogram
"""
# %%
sns.histplot(df["value"])
# %% [markdown]
"""
## Multivariate
***

Add three more variables
"""

# %%
from faker import Faker
fake = Faker()

df["date"] = [
    fake.date_between(start_date='-30y', end_date='today')
    for i in range(len(df))
]
p = (abs(df["value"])/df["value"].max()).values

df["group"] = np.random.binomial(
    1,
    p   
)

df["value2"] = np.random.normal(0,1,len(df))

# %% [markdown]
"""
### Scatter Plot
"""
# %%
sns.scatterplot(
    x=df["date"], 
    y=df["value"],
    hue=df["group"]
)

# %% [markdown]
"""
### Bar Plot
"""
# %%
df_grouped = df.groupby("state").agg({"value":"mean"}).reset_index()
sns.barplot(
    x=df_grouped["state"], 
    y=df_grouped["value"],
    color="lightblue"
)

# %% [markdown]
"""
### Line Plot
"""
# %%
sns.lineplot(
    x=df["date"], 
    y=df["value"],
    hue=df["group"]
)

# %% [markdown]
"""
### Reg Plot
"""
# %%
sns.regplot(
    x=df["value"], 
    y=df["value2"]
)

# %% [markdown]
"""
### LM Plot

LM plot combines Reg plot with facet grid. You can also use groups
"""

# %%
sns.lmplot(
    x="value",
    y="value2",
    hue="state",
    data=df,
    markers=["o", "x"]
)

# %% [markdown]
"""
### Joint Plot
"""

# %%
sns.jointplot(
    x="value",
    y="value2",
    hue="state",
    data=df,
    markers=["o", "x"],
)

# %% [markdown]
"""
**Bivariate KDEs**:
"""
# %%
from scipy.stats import multivariate_normal

# cluster1
means = [-2, -2]
cov = [
    [1.0, 0.4], 
    [0.4, 0.5]
]
mvn = multivariate_normal(means, cov)

# cluster2
means2 = [2, 2]
cov2 = [
    [2.0, 0.3], 
    [0.3, 0.5]
]
mvn2 = multivariate_normal(means2, cov2)

df_mvn = pd.DataFrame(
    np.append(
        mvn.rvs(size=100),
        mvn2.rvs(size=100),
        axis=0
    ),
    columns=["x","y"]
)
df_mvn["group"] = np.repeat([0,1], 100)

# %%
sns.jointplot(
    x="x", 
    y="y",
    hue="group",
    data=df_mvn,
    kind="kde",
    fill=True
)
# %%

# %%

# %% [markdown]
"""
# Confidence Intervals
"""

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(0)
import scipy.stats as st

# %% [markdown]
"""
## Intro

First, we generate a fake population data.  
We'll use a chi square distribution for no particular reason.  
It doesn't have to follow a distribution at all. The plot is shown below.
"""

# %%
# Fake Data
population = np.random.chisquare(5,10000)
sns.histplot(population)
true_mean = np.mean(population)
print(true_mean)

# %%
print(f"For this distribution the true mean is {true_mean}")

# %% [markdown]
"""
The plan is to simulate repeated samples from this population.  
For each sample, we compute a lower bound and an upper bound for the confidence interval of
a sample mean.

First, let's declare our inputs. The z-value for a 95% confidence interval is 1.96. 
"""

# %%
sample_size = 40
zvalue = 1.96

# %% [markdown]
"""
## Simulations

Next, we run the simulations:

We initialize a dictionary "intervals" to store the lower and upper bound 
of our confidence intervals. 

We run the simulation for 200 iterations. We can make this value an input too, 
but hardcoded below for simplicity.

Remember, the bounds of the confidence interval are given by:

$$\pm \ z_{\alpha/2} * \frac{s}{\sqrt{n}}$$
"""

# %%
intervals = {
    "low":[],
    "high":[],
}

for i in range(200):
    sample = np.random.choice(population, sample_size)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample)
    intervals["low"].append(sample_mean-zvalue*(sample_std/np.sqrt(sample_size)))
    intervals["high"].append(sample_mean+zvalue*(sample_std/np.sqrt(sample_size)))

# %% [markdown]
"""
We convert the results into a dataframe. And flag the intervals that do not contain 
the true mean.
"""

# %%
df = pd.DataFrame(intervals).reset_index()
df["reject"] = np.where(
    (df["low"]>true_mean) | (df["high"]<true_mean),
    "r",
    "b"
)

# %% [markdown]
"""
Let's plot results:
"""

# %%
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.scatterplot(df["low"],df["index"], hue = df["reject"])
sns.scatterplot(df["high"],df["index"], hue = df["reject"], legend=False)
ints = plt.xlabel("Intervals")

# %%
df["reject"].value_counts() / len(df)


# %%

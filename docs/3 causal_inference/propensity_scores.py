# %% [markdown]
"""
# Propensity Scores

In the `matching` section, we identified treatment and control subjects that are similar on confounders to construct twin-like pairs. 
The critical assumption was that we know the confounders and we've measured the confounders.

One problem with matching arises when you have many confounders (or even just 4+). With an increasing number of dimensions, it gets 
increasingly difficult to identify a match (curse of dimensionality). As an example, if Age and Gender are the only two confounders, 
then for a 43 year old, male treated subject, it should be easy to find a 43 year old, male control subject. But if you have, Age, 
Gender, Ethnicity, Zip Code, College-Educated, and Occupation = Dentist, then it might be much more difficult to identify a match.

Propensity scores give you a way to reduce the multiple confounders to a single score. Then, the condition of ignorability will 
still hold, even if conditioned on this single score alone!

For discussions on proving this property of propensity scores, see [here if interested](https://stats.stackexchange.com/questions/246717/proving-the-balancing-score-property-of-propensity-score): 
It only requires basic probability concepts. The intuition is that if two subjects have the same propensity score, they might not 
be similar on the confounders, but they are similar _on average_ on the confounders.

The basic idea is to define propensity score as the probability of treatment assignment, given confounders: $\text{score} = P(T=1|X)$.
Then, assuming you have all the confounders:

$$ (Y_1, Y_0) \perp\!\!\!\perp T|X $$

is equivalent to:

$$ (Y_1, Y_0) \perp\!\!\!\perp T|\text{score} $$

## Regression Comparison

How is this differnt from ordinary regression controlling for confounders? This approach handles problems with balance and overlap. Balance
 means there are equivalent counts of treatment and control subjects. Overlap means that in some region of confounders, both treated and 
 control subjects exist.

If you don't have balanced data, your estimates may be biased IF your model is incorrectly specified. In the coded example below, 
I purposefully include a squared term to demonstrate this problem. If your data are balanced, your estimates will still be unbiased 
even if model is not correctly specified. 

If you don't have overlap, your estimates may be biased if your treatment effect is not constant. If treatment effect is the same 
for everyone, you can extrapolate results where overlap exists to subjects where there is no overlap. 

## Algorithm

1. Use a (binary) classification model with X as input and T as output.  
2. Predict scores using the model for each subject (these are our propensity scores)  
3. For each treated subject, match the control subject with closest propensity score  
4. For each confounder, plot its distribution for treated and control group using the matched data. There 
should be strong balance and overlap. If not, you may return to step 1 and try a different model.
5. Finally, estimate treatment effect using matched data, e.g. regression model with treatment and confounders as covariates. 

We want a model in step 1 that is good at predictions. Overfitting is good here and lots of potential for ML tools. 
Most common model is logistic regression.
"""

# %% [markdown]
"""
## Example

Though we discussed multiple dimensions as one motivation for using propensity scores, 
we'll use just two for demonstration simplicity (age, grade)

X = confounders  
score = treatment_score  
T = treatment  
y = outcome  

True treatment effect is 10
"""

# %%
# Fake Data
import pandas as pd
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(1)
import statsmodels.api as sm
import patsy

X = pd.DataFrame({
    "age":np.random.normal(40,2,2000),
    "grade":np.random.uniform(70,100,2000),
})
X["grade2"] = X["grade"]**2

df = X.copy()
score = expit(X @ [-2.2,0.84,0.004])
df["T"] = np.random.binomial(1, score)

df["y0"] = X @ [-0.3, 0.4, 0.01] + np.random.normal(0,1,2000)
df["y1"] = X @ [-0.3, 0.4, 0.01] + np.random.normal(0,1,2000) + np.random.normal(10,1,2000)
df["y"] = df["y1"] * df["T"] + df["y0"] * (1 - df["T"])

# %%
df["T"].value_counts()

# %%
df.head()

# %% [markdown]
"""
#### Naive Estimate 

Since model is incorrectly specified, our estimate is biased 
"""

# %%
f = 'y ~ age + grade + T'
_T, _X = patsy.dmatrices(f, df, return_type='dataframe')
naiveMod = sm.OLS(_T, _X)
resnaive = naiveMod.fit()
print(resnaive.summary())

# %% [markdown]
"""
Note the imbalanced and lack of overlap between treatment and control groups.

It's not so bad for age, but problematic with grade.
"""

# %%
var = "age"
fig, ax = plt.subplots(1, 1)
sns.kdeplot(x=var, data=df.loc[df["T"]==0])
sns.kdeplot(x=var, data=df.loc[df["T"]==1])
fig.legend(labels=["Control", "Treated"])

# %%
var = "grade"
fig, ax = plt.subplots(1, 1)
sns.kdeplot(x=var, data=df.loc[df["T"]==0])
sns.kdeplot(x=var, data=df.loc[df["T"]==1])
fig.legend(labels=["Control", "Treated"])

# %%
# looking at joint distribution, shows extent of overlap issue
sns.scatterplot(x='age', y='grade', hue="T", data=df)

# %% [markdown]
"""
#### Fit logistic regression to predict scores

Notice the imbalance in scores prior to matching.
"""

# %%
f = 'T ~ age + grade + age*grade'
_T, _X = patsy.dmatrices(f, df, return_type='dataframe')
mod = sm.GLM(_T, _X, family=sm.families.Binomial())
res = mod.fit()
df["score"] = res.predict()

fig, ax = plt.subplots(1, 1)
sns.kdeplot(x="score",data=df.loc[df["T"]==0])
sns.kdeplot(x="score",data=df.loc[df["T"]==1])
fig.legend(labels=["Control", "Treated"])

# %% [markdown]
"""
#### Match:
"""

# %%
dft = df.loc[df["T"]==1].reset_index(drop=True)
dfc = df.loc[df["T"]==0].reset_index(drop=True)

# %%
cands = dfc["score"]
matched_list = {"c":[],"t":[]}
for idx,score in dft["score"].items():
    dist = abs(cands-score)
    if min(dist) < 0.1:
        matched = dist.idxmin()
        matched_list["t"].append(idx)
        matched_list["c"].append(matched)
        cands = cands.drop(matched)

# %%
df_matched = pd.concat(
    [
        dft.loc[matched_list["t"]],
        dfc.loc[matched_list["c"]],
    ]
    ,
    axis=0
)

# %% [markdown]
"""
#### Examine Balance/Overlap of Matched Data

Scores:
"""
# %%
fig, ax = plt.subplots(1, 1)
sns.kdeplot(x="score",data=df_matched.loc[df_matched["T"]==0])
sns.kdeplot(x="score",data=df_matched.loc[df_matched["T"]==1])
fig.legend(labels=["Control", "Treated"])

# %% [markdown]
"""
Age
"""

# %%
fig, ax = plt.subplots(1, 1)
sns.kdeplot(x="age",data=df_matched.loc[df_matched["T"]==0])
sns.kdeplot(x="age",data=df_matched.loc[df_matched["T"]==1])
fig.legend(labels=["Control", "Treated"])

# %% [markdown]
"""
Grade
"""

# %%
fig, ax = plt.subplots(1, 1)
sns.kdeplot(x="grade",data=df_matched.loc[df_matched["T"]==0])
sns.kdeplot(x="grade",data=df_matched.loc[df_matched["T"]==1])
fig.legend(labels=["Control", "Treated"])

# %% [markdown]
"""
Age AND Grade
"""

# %%
sns.scatterplot(x='age', y='grade', hue="T", data=df_matched)

# %% [markdown]
"""
#### Estimate Treatment Effect:

Since data are balanced, our estimate is unbiased, even with incorrect model specification.
"""

# %%
f = 'y ~ age + grade + T'
_T, _X = patsy.dmatrices(f, df_matched, return_type='dataframe')
mod = sm.OLS(_T, _X)
resOLS = mod.fit()

# %%
print(resOLS.summary())


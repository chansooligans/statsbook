# %% [markdown]
"""
# OLS: Diagnostics

Ordinary Least Squares / Linear Regression
"""
# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(0)

# %% [markdown]
"""
***
## Outliers

Below uses a fake dataset with one extreme outlier.
"""
# %%
n = 500
df = pd.DataFrame({
    "x1":np.random.normal(10,1,n),
    "x2":np.random.normal(2,1,n),
    "e":np.random.normal(0,1,n)
})

df["y"] = 2 + 3*df["x1"] + (-2*df["x2"]) + df["e"]
df.loc[500] = [11, 10, 0, 60]
sns.scatterplot(df["x1"], df["x2"], df['y'])

# %%
X = sm.add_constant(df[["x1","x2"]], prepend=False)
mod = sm.OLS(df["y"], X)
res = mod.fit()
print(res.summary())

# %% [markdown]
"""
#### Cook's Distance:

$$D_i = \frac{\sum^n_{j=1}(\hat{y_j}-\hat{y_{j(i)}})^2}{ps^2}$$

where $y_j{i}$ is the fitted value using the model trained leaving out the i-th observation and p is the # of predictors.

```
get_influence() and summary_frame prints dfbetas (the difference in each parameter estimatie with and without the
observation). Since the dataset has 501 observations, there are 501 rows in the summary_frame. Note that 
this approach may not be ideal if dataset is really big since it runs the model N times
```
"""

# %%
influence = res.get_influence()
df_inf = influence.summary_frame()
df_inf.sort_values('cooks_d', ascending=False)

# %% [markdown]
"""
#### Leverage:

OLS predicted values are represented as:  

$$\hat{y} = X(X'X)^{-1}X'y$$

Let $H=X(X'X)^{-1}X'$, then $\hat{y} = Hy$

We call "H" the hat matrix. The leverage, $h_{ii}$ quantifies the influence that the observed response $y_i$ has on its predicted 
value $\hat{y_i}$
"""
# %%
influence = res.get_influence()
# equivalent to "hat_diag" column in df_inf
leverage = influence.hat_matrix_diag
df["leverage"] = leverage
df.sort_values('leverage', ascending=False)

# %%
# leverage vs studentized residuals plot
stud_res = res.outlier_test()
sns.scatterplot(leverage, stud_res["student_resid"])

# %% [markdown]
"""
Note the changes in results after outlier removal.
"""

# %%
df = df.drop(500)
X = sm.add_constant(df[["x1","x2"]], prepend=False)
mod = sm.OLS(df["y"], X)
res = mod.fit()
print(res.summary())

# %% [markdown]
"""
***
## Normality of Residuals

#### D'Agostino's K^2 Test (Omnibus) and Jarque-Bera
In the bottom section of the OLS summary table below, the Omnibus test and the Jarque-Bera (JB) 
tests both test the null hypothesis that the residuals are nomrally distributed.

Prob(Omnibus) is 0.205 and Prob(JB) is 0.266 so using both tests, we do not reject the null hypothesis
that the residuals are normally distributed.

We plot the density of the residuals to confirm that the residuals appear to be normally distributed.
"""

# %%
n = 500
df = pd.DataFrame({
    "x1":np.random.normal(10,1,n),
    "x2":np.random.normal(2,1,n),
    "e":np.random.normal(0,1,n)
})

df["y"] = 2 + 3*df["x1"] + (-2)*df["x2"] + df["e"]

X = sm.add_constant(df[["x1","x2"]], prepend=False)
mod = sm.OLS(df["y"], X)
res = mod.fit()
print(res.summary())

# %%
sns.kdeplot(res.resid)

# %% [markdown]
"""
Suppose we generate fake data such that residuals are not normally distributed. The results
of the tests and the residual distribution plots change accordingly:
"""

# %%
df = pd.DataFrame({
    "x1":np.random.normal(10,1,n),
    "x2":np.random.normal(2,1,n),
    "e":np.random.beta(0.1,0.2,n)
})

df["y"] = 2 + 3*df["x1"] + (-2)*df["x2"] + df["e"]

X = sm.add_constant(df[["x1","x2"]], prepend=False)
mod = sm.OLS(df["y"], X)
res_beta = mod.fit()
print(res_beta.summary())
sns.kdeplot(res_beta.resid)

# %% [markdown]
"""
#### QQ Plots

A QQ plot sorts your sample in ascending order (here, the sample are the residuals) 
against the quantiles of a distribution (default is normal distribution). 

The number of quantiles is selected to match the size of the sample data.

In each of the two plots below, the red line is a 45-degree line. If the distribution 
is normal, the values should "hug" the line.
"""

# %%
g = sm.graphics.qqplot(res.resid, line="45")
g_beta = sm.graphics.qqplot(res_beta.resid, line="45")

# %% [markdown]
"""
***
## Multicollinearity

Condition number can be used (high numbers indicate multicollinearity), but usually just look at 
correlation table and plots and VIF
"""

# %%
df = pd.DataFrame({
    "x1":np.random.normal(10,1,n),
    "x2":np.random.normal(2,1,n),
    "e":np.random.beta(0.1,0.2,n)
})

df["x3"] = df["x1"] * 1.3 + np.random.normal(0,2,n)
df["x4"] = df["x2"] * 1.3 + np.random.normal(0,1,n)
df["x5"] = df["x1"] * 1.3 + np.random.normal(0,0.5,n)

# %% [markdown]
"""
#### Condition Number
"""

# %%
# Condition Number
np.linalg.cond(res.model.exog)

# %% [markdown]
"""
#### Correlation Matrix
"""

# %%
features = ["x1","x2","x3","x4","x5"]
corr_mat = df[features].corr()
corr_mat

# %%
sns.heatmap(corr_mat)

# %%
g = sns.PairGrid(df[features])
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)

# %% [markdown]
"""
#### VIF

The variance inflation factor is a measure for the increase of the variance of the 
parameter estimates if an additional variable, given by exog_idx is added to the linear regression. 
It is a measure for multicollinearity of the design matrix.

To get VIF, pick each feature and regress it against all other features. For each regression,
the factor is calculated as $VIF = \frac{1}{1-R^2}$

Generally, VIF > 5 indicates high multicollinearity
"""
# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df[features]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]
vif_data

# %% [markdown]
"""
***
## Heteroskedasticity tests

Also see: Breush-Pagan test and Goldfeld-Quandt test
"""
# %%
n = 500
df = pd.DataFrame({
    "x1":np.random.normal(10,1,n),
    "x2":np.random.normal(2,1,n),
    "e":np.random.normal(0,1,n)
})

df["y"] = 2 + 3*df["x1"] + (-2)*df["x2"] + df["e"]

X = sm.add_constant(df[["x1","x2"]], prepend=False)
mod = sm.OLS(df["y"], X)
res = mod.fit()
print(res.summary())

# %% [markdown]
"""
Plot Sqrt(Standarized Residual) vs Fitted values 

Horizontal line suggests homoscedasticity
"""
# %%
model_norm_residuals = res.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
sns.scatterplot(res.predict(), model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(res.predict(), model_norm_residuals_abs_sqrt,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

# %% [markdown]
"""
Heteroscedasticity may occur if misspecified model:
"""
# %%
n = 500
df = pd.DataFrame({
    "x1":np.random.normal(10,1,n),
    "x2":np.random.normal(2,1,n),
    "e":np.random.normal(0,1,n)
})

df["y"] = 2 + 2*df["x1"] + (3)*df["x2"]**2 + df["e"]

X = sm.add_constant(df[["x1","x2"]], prepend=False)
mod = sm.OLS(df["y"], X)
res = mod.fit()
# %%
sns.scatterplot(df["x1"], df["x2"], df["y"])
# %%
model_norm_residuals = res.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
sns.scatterplot(res.predict(), model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(res.predict(), model_norm_residuals_abs_sqrt,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

# %% [markdown]
"""
Heteroscedasticity may occur if autocorrelation:
"""
# %%
n = 500
df = pd.DataFrame({
    "x1":np.random.normal(0,1,n),
    "time":np.arange(0,n),
    "e":np.random.normal(0,1,n)
})

y_list = [0]
for i in range(n):
    y = df.loc[i,"x1"] + 0.95*y_list[i] + df.loc[i,"e"]
    y_list.append(y)

df["y"] = y_list[1:]

X = sm.add_constant(df[["x1","time"]], prepend=False)
mod = sm.OLS(df["y"], X)
res = mod.fit()
# %%
sns.scatterplot(df["time"], df["y"])
# %%
model_norm_residuals = res.get_influence().resid_studentized_internal
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
sns.scatterplot(df["time"], model_norm_residuals_abs_sqrt, alpha=0.5)
sns.regplot(df["time"], model_norm_residuals_abs_sqrt,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})



# %% [markdown]
"""
***
## Linearity tests

Also see Harvey-Collier multiplier test for Null hypothesis that the linear specification is correct.
"""

# %% [markdown]
"""
Plot residuals aggainst fitted values

(or plot observed vs predicted values)
"""

# %%
sns.residplot(
    res.predict(), 
    res.resid, 
    lowess=True, 
    scatter_kws={'alpha': 0.5},
    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
)

# %%

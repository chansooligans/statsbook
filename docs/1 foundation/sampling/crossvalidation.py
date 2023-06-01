# %% [markdown]
"""
# Cross Validation
"""
# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(12, 10)})

# %% [markdown]
"""
## Create Fake Data
"""

# %%
np.random.seed(1234)
df = pd.DataFrame({
    "x0":np.random.normal(0,4,700),
    "x1":np.random.normal(1,4,700),
})

radius = np.sqrt(df["x0"]**2 + df["x1"]**2)

circles = [
    [0,2],
    [2,4],
    [4,8],
    [8,np.inf],
]

df["y"] = 0
for i,circle in enumerate(circles):
    df.loc[radius.between(circle[0],circle[1]), "y"] = i

X = df[["x0","x1"]]
y = df["y"]

sns.scatterplot(X["x0"], X["x1"], hue=df["y"], palette="tab10")

df = df.sample(len(df), replace=False)

# %% [markdown]
"""
## Cross Validation Using sklearn

Create the classifier object, then use sklearn.cross_val_score to cross validate.
The `cv` parameter is used to determine splitting strategy. The default is `cv=5`, which 
uses 5-fold cross validation.  
"""

# %%
clf = RandomForestClassifier(max_depth=2, random_state=0)
scores = cross_val_score(clf, X, y, cv=5, n_jobs=15)
scores

# %% [markdown]
"""
## Cross Validation Manually

Would never actually do this, but for instructional purposes, manually conduct 5-fold cross validation.

Algorithm:  

1. Divide observations into $k$ folds  
2. Use first fold as test set and fit model on remaining $k-1$ folds
3. Repeat $k$ times
"""

# %%
folds = 5
df["folds"] = np.repeat([x for x in range(folds)], len(df) // 5)[:700]

# %%
score = []
for fold in range(folds):
    test = df.loc[df["folds"]==fold]
    train = df.loc[df["folds"]!=fold]
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X=train[["x0","x1"]], y=train["y"])
    score.append(np.mean(test["y"] == clf.predict(test[["x0","x1"]])))

# %%
score
# %%

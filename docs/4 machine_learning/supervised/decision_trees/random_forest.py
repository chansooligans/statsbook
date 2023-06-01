# %% [markdown]
"""
# Random Forest

A problem with growing a deep tree is that it overfits (low bias, but high variance).  
Solutions to this problem include cross validation and bagging. 

Bagging is equivalent to bootstrapping. You randomly select a subset from your data and 
train a deep ("overfit") tree. Then you repeat this step N times. Then you average your 
results. To make predictions, you can average the predicted values if a regression tree. 
Or if classification, you may use the majority of the predicted classes.

Random Forest adds a step to the Bagging method. Again, we train many decision trees but 
each time a split is considered, you draw a random sample of $m$ predictors and the split 
must occur on one of these sampled predictors. Typically, we choose $m = \sqrt(p)$. 

The purpose is to "decorrelate" the trees. Since we determine each split based on the 
optimal decrease in loss function (e.g. RSS, Gini Index), even if we grow many trees using 
different subsamples, it could be that they all make the initial cut on the same predictor. 
By decorrelating the trees, we force some trees to use a different predictor for the 
initial cut, potentially realizing more optimal solutions. 

## Variable Importance Measures  

Using bagging and random forest approaches, we're training many different trees and one drawback 
of this is that you can't print out a single tree to show what the model is doing. 

But you may still be able to estimate the average importance of each predictor. If bagging 
regression trees, you might record the total amount that RSS is reduced due to splits over a given 
predictor, averaged over all trees. Or if classification trees, record the total amount that the 
Gini index decreased by splits over a given predictor, averaged over all trees. 
"""

# %%
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
## Using sklearn 

Create the classifier object, then use sklearn.cross_val_score to cross validate.
The `cv` parameter is used to determine splitting strategy. The default is `cv=5`, which 
uses 5-fold cross validation.  
"""

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
y_preds = clf.predict(X_test)
np.mean(y_preds == y_test)


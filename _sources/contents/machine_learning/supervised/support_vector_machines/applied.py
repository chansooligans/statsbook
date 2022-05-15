# %% [markdown]
"""
# Applied
"""

# %%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

centers = [(-3, -3), (2, 2)]
cluster_std = [1, 1]
X, y = make_blobs(
    n_samples=500, 
    cluster_std = cluster_std, 
    centers = centers, 
    n_features = 2, 
    random_state=2
)

# %%
pipe = Pipeline(
    [
        ('scaler', StandardScaler()), 
        ('clf', SVC())
    ]
)

param_grid = {
    'clf__C': np.logspace(0, 4, 10)
}

search = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid, 
    cv=5, 
    n_jobs=10,
    scoring='accuracy',
    refit="roc_auc",
)

search.fit(X,y)


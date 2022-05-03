# %% [markdown]
"""
# Pipeline Code
"""

# %%
# fake data
from sklearn.datasets import make_classification
X, y = make_classification(random_state=0) 

# %%
import numpy as np
from typing import Mapping, List, Any, Optional
from dataclasses import dataclass
from functools import cached_property

# models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier

# pipeline
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


class DummyEstimator(BaseEstimator):
    def fit(self): pass
    def score(self): pass

@dataclass
class Pipe:
    model: str = "LogReg"
    
    def __post_init__(self):
        self.scoring = [
            'accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc'
        ]

    @property
    def pipe(self):
        return Pipeline(
            [
                ('scaler', StandardScaler()), 
                ('clf', DummyEstimator())
            ]
        )

    @cached_property
    def search_space(self):
        return [
            {
                'clf': [LogisticRegression()], 
                'clf__penalty': ['l1','l2'],
                'clf__C': np.logspace(0, 4, 10)
            },
            {
                'clf': [SVC()],  # Actual Estimator
                'clf__C': np.logspace(0, 4, 10)
            }
        ]

    @cached_property
    def search(self): 
        return GridSearchCV(
            estimator=self.pipe, 
            param_grid=self.search_space, 
            cv=5, 
            scoring='accuracy',
            refit="roc_auc",
        )

    def fit(self, X, y):
        return self.search.fit(X, y)

# %%
pipe = Pipe()

pipe.fit(X,y)
# %%
pipe.search.best_estimator_
# %%

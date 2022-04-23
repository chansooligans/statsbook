# %% [markdown]
"""
# Regression Trees

Algorithm:  

1. Use recursive binary splitting to grow a large tree on the training data, stopping only when 
each terminal node has fewer than some minimum number of observations.  
2. Apply cost complexity pruning to the large tree in order to obtain a sequence 
of best subtrees, as a function of $\alpha$  
3. Use K-fold cross-validation tto choose $\alpha$. That is, divide the training observations 
into K folds. For each k = 1,...,K:  
    - Repeat steps 1 and 2 on all but the $k$th fold of the training data.  
    - Evaluate the mean squared prediction error on the data in the left-out $k$th fold, as a function of $\alpha$
    Average the results for each value of $\alpha$, and pick $\alpha$ to minimize the average error.  
4. Return the subtree from Step 2 that corresponds to the chosen value of $\alpha$  

**Recursive Binary Splitting**: 

A top-down, greedy approach that begins at the top of the tree (i.e. all points belong
to one single region), then successively splits the predictor space into two new branches. It is greedy because 
at each step of the tree-building process, the best split is made at that particular step, instead of looking 
ahead and picking a plit that will lead to a better tree in some future step. 

At each step, select the predictor $X_j$ and cutpoint s such that splitting the predictor space leads to the 
greatest possible reduction in RSS. 

**Cost Complexity Tree Pruning**:

One approach is to build the tree only until the decrease in RSS at a step is less than some threshold. 
BUT a large decrease may follow a cut that yields a small decrease. So this appraoch is short-sighted.

An alternative it to build a large tree then prune it back to find the optimal subtree. So we look for the subtree 
that yields the best fit. But since more complex trees (trees with more cuts) will overfit data, we want to 
cross validate. Cross validating every subtree is expensive, so instead with include the # of terminal nodes into the 
loss function with tuning parameter \alpha:

e.g. for m nodes, for each $\alpha$, identify subtree $T \in T_{0}$ that minimizes:

$$\sum_{m=1}^{|T|} \sum_{x_i \in R_m} (y_i - \hat{y}_{R_m})^2 + \alpha |T|$$
"""

# %%
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import GridSearchCV
import seaborn as sns
sns.set(rc={'figure.figsize':(12, 10)})

fp = "/Users/chansoosong/Desktop/statsbook/data"

# %% [markdown]
"""
Create Fake Data
"""

# %%
np.random.seed(1234)
df = pd.DataFrame({
    "x0":np.random.normal(20,5,100),
    "e":np.random.normal(0,5,100),
})

df["y"] = df.sum(axis=1)

X_test = np.arange(df["x0"].min()-2, df["x0"].max()+2, 0.01)[:, np.newaxis]

# %%
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

# %% [markdown]
"""
## Decesion Tree Regressor Class
"""

# %%
@dataclass
class DecisionTree:
    X: pd.DataFrame
    y: pd.Series
    X_test: np.array
    mod: str
    depth: Optional[int] = ""

    @property
    def regressor(self):
        reg = DecisionTreeRegressor(random_state=0, max_depth = self.depth)
        reg.fit(self.X, self.y)
        return reg

    @cached_property
    def preds(self):
        return self.regressor.predict(X_test)

    @classmethod
    def get_model(cls,X, y, X_test, depth, mod):
        return cls(X=X,y=y,X_test=X_test,depth=depth, mod=mod)

    def get_by_depth(self, depth):
        return self.get_model(
            X=self.X,
            y=self.y,
            X_test=self.X_test,
            depth=depth,
            mod=self.mod
        )

    @cached_property
    def res(self):
        return pd.DataFrame({
            "x0":self.X_test.reshape(-1),
            "preds":self.preds
        }).assign(mod = self.mod+str(self.depth))

# %% [markdown]
"""
## Cross Validation Class
"""

# %%
# cross validation child class
class GridSearchCrossValidate(DecisionTree):

    def __init__(self, parameters, X, y, X_test, mod):
        self.parameters = parameters
        self.X = X
        self.y = y
        self.X_test = X_test
        self.mod = mod

    @property
    def fit(self):
        reg = DecisionTreeRegressor()
        
        mod = GridSearchCV(
            reg,
            param_grid=self.parameters,
            scoring='neg_mean_squared_error',
            cv=3,
            verbose=1,
            n_jobs = 15
        )

        mod.fit(X=self.X, y=self.y)

        return mod

    @cached_property
    def preds(self):
        return self.fit.predict(X_test)


# %% [markdown]
"""
## Set Parameters for Grid Search
"""

# %%
parameters={
    "splitter":["best","random"],
    "max_depth" : [1,3,5,8,10,15],
    "min_samples_leaf":[1,2,3],
    "max_features":["auto","log2","sqrt",None],
    "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] 
}

# %% [markdown]
"""
## Get Results and Plot
"""

# %% tags=["no-execute"]
dt = DecisionTree(
    X=df[["x0"]],
    y=df["y"],
    X_test=X_test, 
    mod="depth"
)
cv = GridSearchCrossValidate(
    parameters=parameters, 
    X=df[["x0"]],
    y=df["y"],
    X_test=X_test, 
    mod="cv"
)

res = pd.concat(
    [
        dt.get_by_depth(depth=2).res,
        dt.get_by_depth(depth=15).res,
        cv.res
    ]
).reset_index()


pickle.dump(res, open(f"{fp}/regression_tree_output.p","wb"))

# %% [markdown]
"""
Plot demonstrates underfit tree with max_depth = 2, overfit tree with max_depth = 15, 
and cross validation best parameters result.
"""

# %%
res = pickle.load(open(f"{fp}/regression_tree_output.p","rb"))

sns.lineplot(
            res["x0"], 
            res["preds"], 
            hue=res["mod"],
            style=res["mod"]
        )

sns.scatterplot(df["x0"], df["y"], color="red")

# %%

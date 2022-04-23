# %% [markdown]
"""
# Classification Trees

See "Regression Tree" page first. The classification tree is similar to the regression tree, except it's used where the output is a categorical variable. 
Since the output is categorical, we can no longer use RSS as our loss function. Instead, we use the **Gini index** or **entropy**. Another option would 
be simple classification error rate. 

Similar to regression trees, recursive binary splitting is used to grow a large tree, picking splits that maximize the 
"purity" of nodes. 

```
Purity:

"Purity" can be measured by Gini Index, Entropy, classification error rate, etc.
Purity is highest when all the nodes in a leaf are of the same class.
```

Let $\hat{p}_{mk}$ be the proportion of observations in the $m$th region that are from the $k$th class

**Gini Index**

$$G = \sum_{k=1}^{K} \hat{p}_{mk} ( 1 - \hat{p}_{mk} ) $$


**Entropy**

$$D = - \sum_{k=1}^{K} \hat{p}_{mk}\ \log(\hat{p}_{mk}) $$

**Classification Error Rate**

$$ E = 1 - max_{k}(\hat{p}_{mk})$$
"""


# %%
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from sklearn.model_selection import GridSearchCV
import seaborn as sns
sns.set(rc={'figure.figsize':(12, 10)})
import matplotlib.pyplot as plt

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

from six import StringIO  
from IPython.display import Image  
import pydotplus

fp = f"../../../../../data"


# %% [markdown]
"""
Create Fake Data
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

sns.scatterplot(df["x0"],df["x1"],hue=df["y"], palette="tab10")

# %% [markdown]
"""
## Decesion Tree Classification Class
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
        reg = DecisionTreeClassifier(random_state=0, max_depth = self.depth)
        reg.fit(self.X, self.y)
        return reg

    @cached_property
    def preds(self):
        return self.regressor.predict(self.X_test)

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
            "x0":self.X_test["x0"],
            "x1":self.X_test["x1"],
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
        reg = DecisionTreeClassifier()
        
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
        return self.fit.predict(self.X_test)

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
    X=df[["x0","x1"]],
    y=df["y"],
    X_test=df[["x0","x1"]], 
    mod="depth"
)
cv = GridSearchCrossValidate(
    parameters=parameters, 
    X=df[["x0","x1"]],
    y=df["y"],
    X_test=df[["x0","x1"]], 
    mod="cv"
)

res = pd.concat(
    [
        dt.get_by_depth(depth=5).res,
        dt.get_by_depth(depth=15).res,
        cv.res
    ]
).reset_index()

pickle.dump(res, open(f"{fp}/classification_tree_output.p","wb"))

# %% [markdown]
"""
Plot demonstrates underfit tree with max_depth = 5, overfit tree with max_depth = 15, 
and cross validation best parameters result.
"""

# %%
res = pickle.load(open(f"{fp}/classification_tree_output.p","rb"))

# %%
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

sns.scatterplot(df["x0"],df["x1"], hue=df["y"], ax=ax, palette="tab10")
sns.scatterplot("x0", "x1", hue="preds", data=res.loc[res["mod"]=="depth5"], ax=ax2, palette="tab10")
sns.scatterplot("x0", "x1", hue="preds", data=res.loc[res["mod"]=="depth15"], ax=ax3, palette="tab10")
sns.scatterplot("x0", "x1", hue="preds", data=res.loc[res["mod"]=="cv"], ax=ax4, palette="tab10")

ax.set_title("train")
ax2.set_title("depth 5")
ax3.set_title("depth 15")
ax4.set_title("cv")
plt.show()


# %% [markdown]
"""
## Visualize Decision Tree (max_depth = 5 version)
"""
# %%
dt = DecisionTree(
    X=df[["x0","x1"]],
    y=df["y"],
    X_test=df[["x0","x1"]], 
    mod="depth"
)
dot_data = StringIO()
export_graphviz(dt.get_by_depth(depth=5).regressor, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = ["x0","x1"],class_names=['0','1','2','3'])
# %%
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
# %%

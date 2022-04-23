# %% [markdown]
"""
# K-Nearest Neighbors

KNN is a supervised learning method. 

Let k = 3. Given a test data point, identify the 3 ($k$) training data points 
that are closest, using some distance metric (e.g. Euclidean distance). Use 
a majority-wins rule to label the test data point. If using KNN for regression 
instead of classification, you might use the average of the k-nearest neighbors.

"""

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

centers = [(-3, -3), (4, 4), (4, -4)]
cluster_std = [2, 3, 2]
X, y = make_blobs(
    n_samples=500, 
    cluster_std = cluster_std, 
    centers = centers, 
    n_features = 2, 
    random_state=2
)

sns.scatterplot(X[:,0], X[:,1], hue=y, palette="tab10")

# %% [markdown]
"""
## Using Sklearn
"""

# %%
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
print(neigh.predict([[1, 1]]))
print(neigh.predict_proba([[1, 1]]))

# %% [markdown]
"""
## Things to Consider

1. Parameter Selection: what "k" should you use? You can cross validate to optimize 
this hyperparameter
2. Balance: In the classification setting, majority-rules solution can be problematic 
if the distribution of classes is skewed. One way to address this would be to weight 
each training data point by the inverse of class proportion. Or you can construct a 
balanced training set
3. Distance: Euclidean distance is default, but other popular ones include: Minkowski, 
Manhattan Distance, Cosine Distance, Jaccard Distance, Hamming Distance
4. Weighting Neighbors: Given k nearest neighbors, some neighbors are closer than others. 
So you might let a vote = inverse of distance rather than letting all votes = 1. In general, 
we can let a vote = some function k(x, c) where x = obs and c = class albel. We call k(.) 
a kernel. 
5. Local Sensitivity
6. Curse of High Dimensionality: key assumption of KNN is that similar inputs means same output.
But as dimensions increase, inputs may never bee similar! One solution is to use a
 dimension-reduction method such as PCA to preprocess data.
"""


# %% [markdown]
"""
## Manually
"""

# %% [markdown]
"""
For a test point [1,1], identify 3 nearest neighbors.
"""
# %%
test_point = np.array([1,1])
# euclidean distance between test point and all training points
dists = np.linalg.norm(X - test_point, axis=1)
# argpartition identifies kth element in array s.t. all smaller elements are 
# moved before it and all larger elements are moved after it
ind = np.argpartition(dists, kth=3)[:3]
sns.scatterplot(X[:,0], X[:,1], hue=y, palette="tab10")
plt.scatter(test_point[0], test_point[1], c="red")
plt.scatter(X[ind,0], X[ind,1], c="purple")

# %% [markdown]
"""
their classes are all 1 so we would classify 1.
"""
# %%
y[ind]

# %% [markdown]
"""
For a test point [5,0], identify 3 nearest neighbors.
"""

# %%
test_point = np.array([5.5,0])
dists = np.linalg.norm(X - test_point, axis=1)
ind = np.argpartition(dists, 3)[:3]
sns.scatterplot(X[:,0], X[:,1], hue=y, palette="tab10")
plt.scatter(test_point[0], test_point[1], c="red")
plt.scatter(X[ind,0], X[ind,1], c="purple")

# %% [markdown]
"""
their classes are [2, 1, 1]; majority wins
"""

# %%
y[ind]

# %%

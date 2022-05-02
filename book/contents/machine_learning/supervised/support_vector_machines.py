# %% [markdown]
"""
# Support Vector Machines

Support Vector Machines are often used for binary classfication problems. The algorithm looks for the optimal boundary 
to separate the data and determines the boundary based on the "support vectors", the points that lie 
closest to the decision bounadry (hyperplanes in multiple dimensions). 

See: https://web.mit.edu/6.034/wwwbob/svm.pdf
"""

# %% tags=['hide-cell']
from markdown import markdown
from sklearn.datasets import make_blobs
import pandas as pd
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

# %% [markdown]
"""
Start with a classification scenario with perfect and linear separation, as in the example below. 
I draw three different possible candidates to use as the decision boundary. 
"""

# %%
scatter = sns.scatterplot(X[:,0], X[:,1], hue=y, palette="tab10")
scatter.plot([-6,4], [5,-4], '-', color="red", linewidth = 2)
scatter.plot([-6,3], [6,-4], '-', color="red", linewidth = 2)
scatter.plot([-6,5], [4,-4], '-', color="red", linewidth = 2)

# %% [markdown]
"""
Picking just one line as an example, we can find the points that lie closest to this
arbitrary line.
"""

# %%
p1 = np.array([-6,3.5])
p2 = np.array([6,-5.2])
dists = []
for p3 in X:
    dists.append(np.linalg.norm(np.cross(p1-p3,p2-p1))/np.linalg.norm(p2-p1))

# %%
k=4
scatter = sns.scatterplot(X[:,0], X[:,1], hue=y, palette="tab10")
scatter.plot([-6,6], [3.5,-5.2], '-', color="red", linewidth = 2)
supports = X[np.argpartition(dists, kth=k)[:k]]
sns.scatterplot(supports[:,0], supports[:,1], color="purple", s=100)

def p4(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dx, dy = x2-x1, y2-y1
    det = dx*dx + dy*dy
    a = (dy*(y3-y1)+dx*(x3-x1))/det
    return x1+a*dx, y1+a*dy

for i in range(k):
    p4line = p4(p1=p1,p2=p2,p3=supports[i,:])
    scatter.plot([supports[i,0],p4line[0]], [supports[i,1],p4line[1]], '-', color="red", linewidth = 2)

# %% [markdown]
"""
The algorithm estimates the optimal decision boundary by maximizing the gap between the decision boundary and the
support vectors. The image below (https://blog-c7ff.kxcdn.com/blog/wp-content/uploads/2017/02/constraints.png) 
illustrates the maximal margin, the optimal separating hyperplane, as well as the support vectors.

![svm](../constraints.png)
"""


# %% [markdown]
"""
## Beyond Linear Separation

Real-world data is not neat like this and there's no decision boundary that perfectly separates two clusters. 

Couple solutions:

1. Soft margins: allow SVM to make a few "mistakes" where a training data point is on the wrong side of the
decision boundary. 
2. Kernel trick: map the data to a higher dimensional space, e.g. project 2D data $(x,y)$ to 3d by transforming to 
$(x^2, y^2, \sqrt(2)xy)$.
"""


# %%
"""
## Coding From Scratch

To-Do...

see: https://towardsdatascience.com/svm-implementation-from-scratch-python-2db2fc52e5c2
"""

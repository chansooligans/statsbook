# %% [markdown]
"""
# K-Means

K-means just requires $k$, the number of clusters, as an input. There are few variations of the algorithm 
and different distance measures can be used. The following is one example:

The algorithm uses the following steps:

1. Randomly initialize all points to one of $k$ clusters.  
2. Compute the centroids of each cluster. 
3. Assign all points to the cluster whose centroid is closest 
4. Repeat steps 2 and 3 until convergence, i.e. fewer than X% of points are re-assigned clusters at step 3. 
"""

# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# sudo apt update
# sudo apt install ffmpeg

# %% [markdown]
"""
## Create Fake Data

With clustering problems, you don't have a labelled training set. 
But for this demo, we generate y for fake data to be able to evaluate results. 
"""

# %%
centers = [(-3, -3), (4, 4), (4, -4)]
cluster_std = [2, 3, 2]
X, y = make_blobs(
    n_samples=500, 
    cluster_std = cluster_std, 
    centers = centers, 
    n_features = 2, 
    random_state=0
)

plt.scatter(X[:,0], X[:,1], c=y, cmap="Set1")


# %% [markdown]
"""
## Using sklearn
"""
# %%
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
kmeans.labels_

# %%
sns.scatterplot(X[:,0], X[:,1], hue=y, style=kmeans.predict(X), s=50)
sns.scatterplot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], hue=[0,2,1], s=200)

# %% [markdown]
"""
## Run Manually for Demonstration
"""

# %%
class KmeansDemo:

    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.n = len(self.X)
        self.distances = []
        self.clusters = {0:np.random.choice(self.K, self.n)}
        self.centroids = {}

    def get_centroids(self, clusters):
        return np.array([
            X[clusters==k].mean(axis=0)
            for k in range(self.K)
        ])

    def get_clusters(self, centroids):
        distances = np.linalg.norm(
            self.X[:,None,:] - centroids[None,:,:], 
            axis=-1
        )
        return np.argmin(distances, axis=1)

    def __call__(self):
        self.centroids[0] = self.get_centroids(self.clusters[0])

        # save each iter for plotting
        for iter in range(1, 15):
            self.clusters[iter] = self.get_clusters(self.centroids[iter-1])
            self.centroids[iter] = self.get_centroids(self.clusters[iter])
            update = np.mean(mod.clusters[iter-1]!=mod.clusters[iter])
            print(iter, update)
            if update < 0.002:
                break

# %% [markdown]
"""
**Run k-means:**

Prints iteration and percentage of points updated.
Stops when no new points are updated.
"""

# %%
mod = KmeansDemo(X=X, K=3)
mod()

# %% [markdown]
"""
## Graph Each Iteration
"""

# %%
color_dict =  {0:"red", 1:"green", 2:"purple"}

sns.set(rc={'figure.figsize':(8,5)})

def plot_iter(i):
    
    fig = plt.figure(i)
    ax = sns.scatterplot(X[:,0], X[:,1], hue=mod.clusters[i])
    sns.scatterplot(mod.centroids[i][:,0], mod.centroids[i][:,1], color="red", s=100)
    ax.set_title(f"Iter {i}")

for _ in range(len(mod.centroids)):
    plot_iter(_)


# %%
# animated
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=1, bitrate=1800)
# fig,ax = plt.subplots()
# scat = ax.scatter(X[:,0], X[:,1], c=mod.clusters[0], cmap="Set1")
# def animate(i):
#     scat.set_array(mod.clusters[i])
# ani = matplotlib.animation.FuncAnimation(fig, animate, frames=4, repeat=True)
# ani.save('animate.mp4')
# %%

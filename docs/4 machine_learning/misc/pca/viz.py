# %% [markdown]
"""
# Principal Components Analysis

For theory, see: foundation/linear_algebra_review.md
"""

# %% [markdown]
"""
## 2d Components with Seaborn
"""

# %%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

np.random.seed(0)

df = pd.DataFrame(
    np.random.multivariate_normal([0,0],[[1,.7],[.7,1]],200),
    columns = ["x0","x1"]
)

# %%
sns.scatterplot(x="x0",y="x1",data=df)

# %%
pca = PCA(n_components=2)
pca.fit(df[["x0","x1"]])

# %%
def draw_vector(v0, v1, ax=None):
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    color="black",
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

fig, ax = plt.subplots()
ax.set(ylim=(-7, 7), xlim=(-3, 3))
sns.scatterplot(x="x0",y="x1",data=df)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * np.sqrt(length)
    print(pca.mean_ + v)
    draw_vector(pca.mean_, pca.mean_ + v, ax=ax)


# %%
X_transform = pca.transform(df[["x0","x1"]])
fig, ax2 = plt.subplots()
ax2.set(ylim=(-5, 5))
sns.scatterplot(X_transform[:,0],X_transform[:,1])
draw_vector([0,0], [pca.explained_variance_[0],0], ax=ax2)
draw_vector([0,0], [0,pca.explained_variance_[1]], ax=ax2)

# %% [markdown]
"""
## Plotly
"""

# %%
df = pd.DataFrame(
    np.random.multivariate_normal([0,0],[[1,.7],[.7,1]],200),
    columns = ["x0","x1"]
)
df["x2"] = df["x1"] * 2

pca_3d = PCA(n_components=3)
pca_3d.fit(df[["x0","x1","x2"]])
X_transform_3d = pca_3d.transform(df[["x0","x1","x2"]])

# %% [markdown]
"""
#### 3 dimensions with perfectly correlated 3rd feature
"""

# %%
import plotly.express as px
fig = px.scatter_3d(
    X_transform_3d, x=0, y=1, z=2,
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.update_traces(marker_size = 5)
fig.show()


# %% [markdown]
"""
#### 3 dimensions with 3 uncorrelated features
"""
# %%
df = pd.DataFrame(
    np.random.multivariate_normal([0,0,0],[[1,0,0],[0,1,0],[0,0,1]],200),
    columns = ["x0","x1","x2"]
)

pca_3d2 = PCA(n_components=3)
pca_3d2.fit(df[["x0","x1","x2"]])
X_transform_3d2 = pca_3d2.transform(df[["x0","x1","x2"]])

# %%
import plotly.express as px
fig = px.scatter_3d(
    X_transform_3d2, x=0, y=1, z=2,
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.update_traces(marker_size = 5)
fig.show()

# %%

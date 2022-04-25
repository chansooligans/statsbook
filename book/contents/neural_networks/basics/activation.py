# %% [markdown]
"""
# Activation Functions

Most common functions:

1. Sigmoid
2. Tanh
3. ReLu
4. Leaky ReLu
"""

# %%
import numpy as np
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
x = np.linspace(-10, 10, 1000)

# %% [markdown]
"""
## Sigmoid
"""

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sns.lineplot(x, sigmoid(x))


# %% [markdown]
"""
## Tanh
"""

# %%
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

sns.lineplot(x, tanh(x))


# %% [markdown]
"""
## ReLu
"""

# %%
def relu(x):
    return [max(0, _) for _ in x]

sns.lineplot(x, relu(x))


# %% [markdown]
"""
## Leaky ReLu
"""

# %%
def leakyRelu(x, e=0.05):
    return [max(e*z, z) for z in x]

sns.lineplot(x, leakyRelu(x))

# %%

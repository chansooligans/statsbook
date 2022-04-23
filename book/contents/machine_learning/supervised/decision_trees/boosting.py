# %% [markdown]
"""
# Boosting

Similar to bagging, boosting grows a large number of decision trees. Unlike bagging, however, each tree 
depends strongly on the trees that have already been grown. 

Parameters:

1. B: the number of trees
2. $\lambda$: the shrinkage parameter (typically 0.01 or 0.001)
3. d: the number ofsplits (interaction depth)

Algorithm:  

1. Initialize predicted outputs, e.g. $\hat{f}(x) = 0$ and $r_i = y_i$
2. For each tree (out of B trees):  
    - fit a tree $\hat{f}^b(x)$ with d splits to the training data (X, r), note that output is r not y
    - update initialized predicted outputs: $\hat{f}(x) = \hat{f}(x) + \lambda\hat{f}^b(x)$
    - update the residuals: $r_i = r_i - \lambda\hat{f}^b(x_i)$  
3. Output boosted model  
    - $\hat{f}(x) = \sum_{b=1}^{B}\lambda\hat{f}^b(x)$
"""

# %%

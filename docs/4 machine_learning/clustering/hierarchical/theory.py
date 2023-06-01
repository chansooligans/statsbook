# %% [markdown]
"""
# Theory
"""

# %% [markdown]
"""
Agglomerative Algorithm (Bottom-Up):

1. Define a similarity measure, e.g. Euclidean distance  
2. Start at bottom of dendrogram (assign each node to its own cluster = n clusters)  
3. The two clusters that are most similar to each other are fused (the similarity measure
will be the height in the dendrogram where fuse occurs)
4. Repeat with the n-1 remaining clusters, until all observations are assigned to a single cluster  
"""

# %% [markdown]
"""
Dendrogram Interpretation  
- Leaves that are fused at bottom of tree are more similar to each other compared to leaves
that fuse higher atop.
- Use vertical axis to assess similarity, not the horizontal axis
- The height of the cut serves same purpose as "K" in k-means clustering: it controls the # of 
clusters obtained
"""

# %% [markdown]
"""
Types of Linkage (measures to compare two clusters' similarity):  
- complete: compute all pairwise similarity between observations in cluster A and cluster B, 
and record the LARGEST of these similarities
- single: compute all pairwise similarity between observations in cluster A and cluster B,
and record the smallest 
- average: compute all pairwise similarity between observations in cluster A and cluster B, 
and record the average 
- centroid: dissimilarity between centroid for cluster A and cluster B (inversions may occur, 
two clusters are fused BELOW either of the two clusters -- makes dendrogram visualization 
difficult)
- correlation-based: focuses on shapes of observation profiles rather than magnitudes (e.g. 
clustering users where features are movies watched)
- ward: minimizes variance of clusters being merged
"""
# %%

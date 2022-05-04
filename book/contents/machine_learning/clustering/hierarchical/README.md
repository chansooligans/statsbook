# Hierarchical

Pros:
- Unlike K-means, does not require "K" - the # of clusters
- Yields a tree-based representation of observations, called a dendrogram

Cons:
- "Hierachical" means that clusters obtained by cutting the dendrogram at a lower height are 
necessarily nested within clusters obtained by cutting the dendogram at a higher height. So 
if you make a cut that generates 4 clusters. And also make a cut that generates 2 clusters, 
these 4 clusters are necessarily nested within the 2 clusters. As an example, if the data 
contain 50-50 split of male vs female and contain Americans, Japanese, and French. A cut that generates 3 clusters may split data by ethnicity. But the split by 2 clusters would generate clusters split by gender. 
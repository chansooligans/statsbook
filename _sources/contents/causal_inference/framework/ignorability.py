# %% [markdown]
"""
# Ignorability

Ignorability is often expressed using this simple but confusing equation:

$$ (Y_1, Y_0) \perp\!\!\!\perp T|X $$

This says that the distribution of potential outcomes is independent of the treatment assignment mechanism, controlling for 
covariates $X$. Importantly, note that this is NOT saying $ Y \perp\!\!\!\perp T|X$. It is NOT saying that 
the distribution of the observed outcome is independent of treatment assignment. If treatment is effective, of course 
we would expect that the outcome and treatment are related. 

Rather, ignorability is saying that the potential outcome is not correlated with treatment assignment... that the 
**potential** outcome $Y_1$ (observed for treated group and not observed for control group) is not correlated with treatment assignment. 
And that the **potential** outcome $Y_0$ is not correlated with treatment assignment.

It's helpful to imagine a violation of this assumption: Suppose that we want to evaluate the effectiveness of a summer learning program 
for middle schoolers on academic outcomes in the following year. 

First, suppose this treatment (the learning program) is randomly assigned. Then $Y_1$ is independent of treatment assignment. That is, 
the academic outcome that each student would obtain after the summer learning program IS independent of treatment assignment. Why? Because 
the treatment assignment is RANDOM. Similarly, $Y_0$, the potential outcome each student would obtain if they did NOT attend the 
summer learning program is also independent of treatment assignment. Note that if the program is effective, the observed $Y$ is NOT independent of $T$, 
even if treatment is random.

Now, suppose treatment is NOT random and higher performing students are more likely to attend the program. 
Then neither $Y_1$ nor $Y_0$ is no longer independent of treatment assignment. Students that would have higher $Y_1$ and $Y_0$, 
whether they received treatment or not, are more likely to receive treatment. So treatment is positively correlated with $Y_1$ and also 
positively correlated with $Y_0$. 

Let X = previous year's academic outcome. In this observational data setting, if the treatment assignment mechanism IS random, 
conditioned on previous year's academic outcome, then ignorability would be satisfied, conditioned on X. In other words, given a cohort 
of students who received an A, treatment is not correlated with $Y_1$ or $Y_0$. 
"""

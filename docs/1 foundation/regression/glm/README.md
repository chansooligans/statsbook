# Generalized Linear Models

A couple restrictions of OLS are that it is a linear model and requires a continuous dependent variable. 
We often come across other response types: binary, probabilities, counts, rates, and categorical. 
We generalize the linear model to relate to thsee different response types by (1) assuming the dependent variable 
follows a distribution from the exponential family, including Normal (as in OLS), Binomial, Gamma, 
Negative Binomial, Poisson, etc.; and (2) applying a non-linear transformation to the linear model. 
We call this non-linear transformation a "link function". 

$$E(Y|X)  = \mu = g^{-1}(\eta)$$

The equation above shows the generalized linear model. $E(Y|X)$ is the expected value, the mean. 
$g^{-1}()$ is the link function. $\eta=X\beta$ is the linear component.

The variance is modeld as a function of the mean, $V(Y|X) = V(\mu)$

A linear regression simply has an identity link function, $g^{-1}(X\beta)=X\beta$.

For a table of link functions, see the "Common distributions with typical uses and canonical link functions" table 
on the Generalized_linear_model wikipedia page, [here](Common distributions with typical uses and canonical link functions):


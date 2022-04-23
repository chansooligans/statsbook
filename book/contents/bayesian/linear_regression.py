# %% [markdown]
"""
# Linear Regression (Bayesian)

First, we use grid approximation using fake data to demnonstrate Bayesian model approach to a 
simple linear model.  

Then, I show how this is done using Stan. This is the standard tool for Bayesians in research and industry.
(Gelman created it!).

Linear regression has an analytical solution so no one would estimate a linear model in this way, but this is 
just for demonstration. The Stan process can easily be adapted to much more sophisticated models.

Stan Reference Manual: https://mc-stan.org/docs/2_19/reference-manual/index.html#overview  
e.g. page on hamiltonian monte carlo: https://mc-stan.org/docs/2_28/reference-manual/hamiltonian-monte-carlo.html
"""

# %% tags=['hide-cell']
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')


# %%
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% [markdown]
"""
## Linear Regression Using Bayesian Model

Bayesian estimation of linear regression works in a similar way:
    
- The model:
    - $y = \beta X + \epsilon$
- Aka:
    - $y_i \sim Normal(\mu,\sigma)$
- The frequentist approach tries to maximize the likelihood: $p(y |X, \mu, \sigma)$
- As bayesians, we're interested in $p(\mu, \sigma | X, y)$

### No Parameter Example

True Model:

$y \sim Normal(100, 10)$

Generate Fake Data:
"""
# %%
y = np.random.normal(100,10,1000)

# %% [markdown]
"""
Bayesian model:

$y_i \sim Normal(\mu,\sigma)$  
$\mu \sim Normal(90,20)$  
$\sigma \sim Uniform(0, 50)$
"""

# %%
# Grid approximation of posterior:
import itertools
import scipy
mu_list = np.linspace(98,102,200)
sigma_list = np.linspace(9.5,10.5,200)
grid = list(itertools.product(mu_list,sigma_list))
df_grid = pd.DataFrame(grid, columns = ['mu', 'sigma'])
print(len(df_grid))
df_grid

# %% [markdown]
"""
We're trying to solve:

$p(\mu, \sigma | X, y) \propto \prod (Normal(y_i | \mu, \sigma) * Normal(\mu| 90, 20) * Uniform(\sigma| 0,50) ) $

(This is just Bayes Theorem. It just says the posterior is proportional to the numerator: prior*likelihood. We can ignore the denominator since its function is just for scaling.)

Normal distribution likelihood:

$L(\mu, \sigma | x_1,x_2,...,x_n) = \prod_i{\cfrac{1}{\sigma\sqrt{2\pi}}\epsilon^{-\frac{(x-u)^2}{2\sigma^2}}}$

Uniform likelihood:

$L(a,b) = \cfrac{1}{(b-a)}$

To simplify the calculation, we use log likelihoods.  
For each mu/sigma in our grid, we compute the posterior probabilities.
"""
# %% tags=['hide-output']
log_likelihood = []
post = []
for mu,sigma in grid:
    ll = sum(scipy.stats.norm.logpdf(y, mu, sigma))
    log_likelihood.append(ll)
    
    # Normal log-likelihood + normal log-likelihood + uniform log-lik
    post.append(ll + scipy.stats.norm.logpdf(mu, 90, 20) + scipy.stats.uniform.logpdf(sigma, 0, 50))

# %%
# exp of difference to get probability
# scale using the maximum in order to avoid getting all zeros (if you just do np.exp(post), results are all zero)
# this is due to rounding error
df_grid['prob'] = np.exp(post-np.max(post))

# %%
# import plotly.express as px
# fig = px.scatter_3d(df_grid.sample(5000), x='mu', y='sigma', z='prob',color='prob',
#                     width=1500,height=700, opacity = 0.8)

# fig.show()
# fig.write_image("lin_reg_grid_search.png")

# %% [markdown]
"""
Since interactive plotly 3d plot doesn't work in markdown.
See image:
![lin_reg_grid_search.png](lin_reg_grid_search.png)

Using grid search, we estimated the joint distribution of mu and sigma.  
"""

# %% [markdown]
"""
## PyStan

#### Suppose we add a parameter to the linear model.

Grid approximation is super inefficient and not feasible with more complicated models.  
A big part of Bayesian modeling to estimate the joint posterior distribution as in the graph above  
in the most efficient way possible.  
Common algorithms include Monte Carlo Markov Chain, Gibbs Sampler, Hamiltonian MCMC.  

Here's a visualized example of how Hamiltonian MCMC estimates a posterior: https://chi-feng.github.io/mcmc-demo/app.html

(How's this different from gradient descent in ML?)

True model:

$y = ax + b + \epsilon$


Bayesian model:

$y_i \sim Normal(\mu,\sigma)$  
$\mu = \alpha x + \beta$  
    
"""
# %% 
import nest_asyncio
nest_asyncio.apply()
import stan
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
"""
The chunk below is "stan" code, it's C++.  
The first chunk "data" initialize the variables and "parameters" initialize parameters.   
n is the number of samples.  

"Transformed parameters" just defines mu as a linear function of x and intercept. 

In the "model" section, we set the prior distribution for sigma.   
And we define the linear model: normal distribution with mean mu and variance sigma.   

Then, we generate fake data to test.  

The plots at bottom are "trace" plots. They show the outcome of many simulations.  
"""

# %% tags=['hide-output']
lin_reg_code = """
data {
    int<lower=0> n;
    real x[n];
    real y[n];
}
transformed data {}
parameters {
    real a;
    real b;
    real sigma;
}
transformed parameters {
    real mu[n];
    for (i in 1:n) {
        mu[i] <- a*x[i] + b;
        }
}
model {
    sigma ~ uniform(0, 20);
    y ~ normal(mu, sigma);
}
generated quantities {}
"""

n = 11
_a = 6
_b = 2
x = np.linspace(0, 1, n)
y = _a*x + _b + np.random.randn(n)

lin_reg_dat = {
             'n': n,
             'x': x,
             'y': y
            }

posterior = stan.build(lin_reg_code, data=lin_reg_dat)
fit = posterior.sample(num_chains=4, num_samples=1000)

# %%
fit
df = fit.to_frame()  # pandas `DataFrame, requires pandas
# %%
#pip install arviz
import arviz
arviz.plot_trace(fit)

# %%

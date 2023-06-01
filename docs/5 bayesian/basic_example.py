# %% [markdown]
"""
# Basic Example

**Question:**

Suppose you are spot-checking 10,000 documents for errors. Assume errors are independently distributed.  

You check 5 items and none have errors. So the sample mean is 0. What's your estimate of the # of errors in the total population of 10,000?
i.e. what's the probability of error in the total population?

How would a non-Bayesian answer this?

We use Bayes theorem:

$$P(\text{Error|No Errors in First 5 Checks}) \propto P(\text{No Errors in First 5 Checks|Error})P(\text{Error})$$

Note that we don't need the denominator since it's just a scaling factor.  
Just that posterior is proportional to the numerator (prior*likelihood).

Below, I define a function that runs simulations. Then, I set a prior then estimate the posterior.  
I repeat with different scenarios to demonstrate how it changes when the data changes (e.g. more samples than 5).  
And how the results may or may not be sensitive to the prior.  
"""

# %% tags=['hide-cell']
from IPython import get_ipython
if get_ipython() is not None:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
"""
Let's use a Beta Distribution for our prior, $P(\text{Error})$  
And a Binomial Distribution for our likelihood, $P(\text{No Errors in First 5 Checks|Error})$
"""

# %%
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy.stats import chi2
from scipy.stats import binom
from scipy.stats import beta
sns.set(rc={'figure.figsize':(15,3)})

def binom_pmf(p, n, k): 
    return binom.pmf(n, k, p)

def beta_pdf(x,a,b):
    return beta.pdf(x,a,b)

# Use grid search to estimate posterior using prior and likelihood

def run_sim(
    # a and b are hyperparameters for the beta prior (determines shape of the prior)
    a,
    b,
    # n: # of checks
    n,
    # k: # of errors
    k,
    nsim = 1000
):
    
    # Prior
    x = [x/nsim for x in range(0,nsim)]
    prior = [beta_pdf(x/nsim, a,b) for x in range(0,nsim)]
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=False)
    sns.lineplot(x=x,y=prior,ax=ax1).set_title('Prior')

    # Likelihood
    ll = [binom_pmf(x/nsim, k, n) for x in range(0,nsim)]
    g2 = sns.lineplot(x=x,y=ll, color='orange',ax=ax2).set_title('Likelihood')

    # Posterior
    posterior = [x*y for x,y in zip(prior,ll)]
    sns.lineplot(x=x, y=posterior, color='green', ax=ax3).set_title('Posterior')


    f, ax4 = plt.subplots(1,1)
    sns.lineplot(x=x, y=posterior, ax=ax4).set_title('95% Interval')

    posterior = pd.Series(posterior)
    posterior = posterior[posterior.notnull()]
    idx = np.where(posterior==np.max(posterior))[0][0]
    for i in range(0,1000):
        normalized = (posterior/sum(posterior))
        left = sum(normalized[max(idx-i,0):idx])
        right = sum(normalized[idx:min(idx+i,1000)]) 
        if left + right  > 0.95:
            low = max(idx-i,0)
            high = min(idx+i,999)
            break

    print("95% interval: ({},{})".format(low*.1,high*.1))
    ax4.axvline(x[low], color='red')
    ax4.axvline(x[high], color='red')

# %% [markdown]
"""
Below, I use Beta(a=1,b=1), which is equivalent to a uniform distribution for my prior. This suggests that I have no information about the true probability and the true probability is equally likely to be 0 as it is to be 1.

The "95% credible interval" is the posterior distribution with vertical lines indicating the credible interval.  
The credible interval shows that I am 95% certain that the true probability is at least 60%.  
My best guess would be to select the peak of the posterior distribution.
"""

# %%
results = run_sim(a=1,b=1,n=5,k=5)

# %% [markdown]
"""
With more samples (20 out of 20), the 95% interval tightens
"""

# %%
results = run_sim(a=1,b=1,n=20,k=20)

# %% [markdown]
"""
We can adjust our prior if we have a strong belief that the true probability is larger than 0.7 and close to 1:
"""

# %%
results = run_sim(a=15,b=1,n=20,k=20)

# %% [markdown]
"""
For a less skewed example:
"""

# %%
results = run_sim(a=1,b=1,n=10,k=5)

# %% [markdown]
"""
With so few data (n=10), the prior can have a dramatic effect:
"""

# %%
results = run_sim(a=1,b=10,n=10,k=5)

# %% [markdown]
"""
**But with lots of data (n=1000), the prior is hardly influential:**
"""

# %%
results = run_sim(a=1,b=1,n=10000,k=5000)

# %% [markdown]
"""
With this much data, even with a strong prior, the posterior isn't too different from when we used the weak, uniform prior.
"""

# %%
results = run_sim(a=20,b=1,n=10000,k=5000)


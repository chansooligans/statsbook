import markdown


# %% [markdown]
"""
# Metrics

1. Classification
    - Confusion Matrix
    - Accuracy / Precision / Recall
    - ROC / AUC
    - Common Loss Functions:
        - Log loss / cross entropy
2. Regression
    - MSE
    - r-squared; adjusted-r-squared
    - Log Likelihood
    - AIC
    - BIC
"""

# %% [markdown]
"""
## Classification

#### Confusion Matrix

![confuse](confusion_matrix.png)
"""

# %%
import numpy as np
np.random.seed(0)
from sklearn.metrics import confusion_matrix
acc = 0.9
y_true = np.random.binomial(1,0.7,100)
y_pred = [x if np.random.binomial(1,acc,1) == 1 else abs(x-1) for x in y_true]
confusion_matrix(y_true, y_pred)

# %% [markdown]
"""
In the example above, there are: 
- 30 predicted zeros and 23 actual zeros
- 70 predicted ones and 77 actual ones
- 22 zeros where predicted values are zeros  
- 1 zeros where predicted values are ones  
- 69 ones where predicted values are ones  
- 8 ones where predicted values are zeros  

Metrics using confusion matrix:
- Accuracy: the true positives and negatives out of the total population  
- Precision: the true positives out of total predicted positives  
- Recall / Sensitivity / True Positive Rate: the true positives out of the total actual positives  
- Specificity /True Negative Rate: the true negatives out of the total actual negatives
- True Positive Rate: True positives / Actual Positives
- False Positive Rate: False positives / Actual Positives
- True Negative Rate: True negatives / Actual negatives
- False Negative Rate: False negative / Actual positives
"""

# %% [markdown]
"""
#### ROC (Receiver Operating Characteristic)

A ROC is a plot of the TPR against the FPR. It shows the relationship between 
sensitivity and specificity and accuracy of the model. 

You need the true labels + either predicted values or predicted probabilities 
to generate a ROC curve. When using predicted probabilities, you can use different
threshold to classify label = 1. By generating predictions for different thresholds, 
you get different tpr and fpr for each threshold. Plot these tpr against the fpr 
to generate a ROC curve.

Slightly confusing, but note that in the example below, 0 = True.

When the threshold is extremely high, e.g. 1, then all predictions are 0 so 
both TPR and FPR = 1. We get all the 0s right, but we're also predicting 0 
for all the 1s, so there are many false 0s.

When the threshold is ~ 0.5, the tpr is high (~0.97) and the fpr is low (~0.13). 
"""
# %%
from sklearn.metrics import roc_curve, RocCurveDisplay

y_true = np.random.binomial(1,0.7,100)
probs = [
    np.random.beta(3,1,1)[0]
    if x == 1 
    else np.random.beta(1,5,1)[0]
    for x in y_true
]

# %%
RocCurveDisplay.from_predictions(y_true, probs)

# %%
for threshold in np.linspace(0,1,20):
    preds = [1 if x > threshold else 0 for x in probs ]
    [[tp, fn],[fp, tn]] = confusion_matrix(y_true, preds)
    print(f"thresh: {round(threshold,1)}, tpr: {round(tp / (tp + fn),2)} fpr: {round(fp / (fp + tn),2)}")

# %% [markdown]
"""
#### AUC

AUC summarizes the performance of the classification model across all the possible thresholds. 
It can be calculated as the integral (hence, area under the curve) of the ROC curve. 
High values close to 1 means that the model performs well under any threshold. 
"""

# %% [markdown]
"""
#### Log Loss / Cross Entropy

The likelihood function for a sequence of Bernoulli trials is:

$$\prod_i^n{p^{y_i}(1-p_i)^{1-y_i}}$$

Taking the log gives us the log likelihood:

$$\sum_i^n{y_i log p_i + (1-y_i) log (1-p_i)}$$

To get a loss function, we often multiply by $-\frac{1}{N}$. Reversing the sign let's us minimize 
and divide by N to normalize. 

Note intuitively how this differs from accuracy: when a classification is correct, we add $p$ to the likelihood 
(or equivalently, subtract $p$ from the loss). So the model is rewarded for probabalistic certainty. 
"""

# %% [markdown]
"""
## Regression

#### MSE
"""
# %%
y = np.random.normal(10,2,100)
y_pred = y + np.random.normal(0,1,100)
mse = np.mean(np.sqrt((y - y_pred)**2))

# %% [markdown]
"""
#### r-squared

R^2 is a goodness of fit measure that tells you the amount of variation in the output that 
can be explained by the covariates. It is computed as $R^2 = 1 - \frac{RSS}{TSS}$ where RSS 
is the residual sum of squares and TSS is the total sum of squares, $TSS = \sum{(y_i-\bar{y})^2}$

The adjusted r-squared is defined as:

$$R^2 = 1 - \frac{RSS/df_{res}}{TSS/df_{tot}}$$

where $df_{res}$ is $n-p$ and $df_{tot}$ is $n-1$. Importantly, the adjustment penalizes complex models 
(i.e. as the # of parameters increases)
"""

# %% [markdown]
"""
#### Likelihood

The likelihood function describes the probability of some parameter values, given some data are observed: $L(\theta|x)$.  
One method of estimating a model is to maximize the likelihood function (maximum likelihood estimation). 
The log likelihood function is often used out of convenience.

"""

# %% [markdown]
"""
#### AIC / BIC

Like MSE and R^2, AIC and BIC are also used to compare different models. 
In both cases, we want to select the model with lowest AIC/BIC. Both metrics 
penalize models for complexity. BIC penalizes the model MORE for its complexity compared 
to AIC.

$$AIC = 2K - 2ln(L)$$

where K is equal to the number of parameters. (For multiple regression, 
 include intercept and constant variance parameters). L is the model likelihood.

$$BIC = -2ln(L) + K*log(n)$$

where k is the # of parameters and N is the number of observations
"""

# %%

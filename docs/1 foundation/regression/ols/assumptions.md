# Assumptions

In order of importance:

1. Validity  
    - As obvious as it may sound, make sure the data map to the research question
    - Does the outcome variable accurately capture what you are trying to answer?  
    - Are you using the right data for this question?  
2. Representativeness  
    - Is your sample representative of the population?
    - To be more precise, need to make sure that the data are representative of the distribution of the outcome given the predictors. 
    - As an example, in a regerssion of earnings on height and sex, it would be acceptable for women and tall people to be overrepresented 
    in the sample, compared to general population, but not if rich people are overrepresented. Selection on y is a problem, not selection on x.
3. Additivity + Linearity
    - Outcome needs to be a linear function of the separate predictors
    - Often can transform variables (e.g. x^2, 1/x, log(x))
4. Independence of Errors   
    - This assumption is violated if:   
        - data are clustered and clusters are correlated with outcome  
        - time series ($y$ depends on $y_{t-1}$)
        - spatial (counties are correlated within state)
5. Homoscedasticity (Equal Variance of Errors)
    - usually a minor issue and can be remedied with weighted least squares or robust standard errors
6. Normality of Errors
    - only relevant when making predictions and often not a problem
    - if normal errors, least squares solution is also the maximum likelihood solution

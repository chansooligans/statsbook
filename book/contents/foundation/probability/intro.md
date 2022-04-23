# Probability Introduction

**sample space**:

A sample space associated with a data generating process is the set consisting of all possible sample points.

If the data generating process is a roll of a dice, the sample space is ${1,2,3,4,5,6}$.  

**event**:

An event in a discrete sample space is a collection of sample points (any subset of S).

Rolling a dice, examples of an "event":
- 6
- 1 or 2
- An even number

**kolmogorov axioms**:

Suppose S is a sample space. For every event A in S:

> 1. $P(A) >= 0$
> 2. $P(S) == 1$
> 3. If $A_1, A_2, ...$ form a sequence of pairwise mutually exclusive events in $S$, then:  
> $P(A_1 \bigcup A_2 \bigcup A_3 \bigcup ...) = \sum_{i=1}^{\infty}P(A_i)$

**conditional probability**: 

probability of event A given event B:

> $P(A|B) = \frac{P(A\bigcap B)}{p(B)}$

Example:   
Suppose you toss a die once. Given that an odd number was obtained (B), what is the probability that it was a 1 (A)?

> $P(A|B) = \frac{P(A\bigcap B)}{P(B)} = \frac{1/6}{1/2} = 1/3$

Note that since $A \subset B$, $P(A\bigcap B) = P(A)$

**independence**:

Two events A and B are independent if any of the following holds:

> $P(A|B) = P(A)$  
> $P(B|A) = P(A)$  
> $P(A\bigcap B) = P(A)P(B)$

Otherwise, A and B are dependent.

Example:  
Given a single die toss, consider:

A. Observe an odd number  
B. Observe an even number  
C. Observe a 1 or 2  

Are A and B independent events? Are A and C independent events?

**multiplicative law of probability**:

> $P(A\bigcap B) = P(A) P(B|A) = P(B) P(A|B)$

If A and B are independent, then:

> $P(A\bigcap B) = P(A)P(B)$

**additive law of probability**:

> $P(A \bigcup B) = P(A) + P(B) - P(A\bigcap B)$

If mutually exclusive:

> $P(A \bigcup B) = P(A) + P(B)$

**random variable**:

A random variable is a function for which the domain is a sample space.

**probability distribution**:

The probability distribution for a discrete random variable Y is represented as P(Y = y): probability that the random variable takes on the value y. 

A pmf is a probability mass function and it is used for discrete distributions.  
A pdf is a probability density function and is used for continuous distributions.  
A cdf is a cumulative distribution function and expresses the cumulative probability for $Y<=y$.
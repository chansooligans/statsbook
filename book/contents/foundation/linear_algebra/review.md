# Lin Alg Review

## Linear Equations / Matrix Operations:

A system of linear equations is:  
- consistent: one solution or infinitely many solutions  
- inconsistent: no solution  

Linear combination:  
- Given vectors $v_1, v_2, .., v_p$ in $\mathbb{R}^n$ and given scalars $c_1, c_2, ... c_p$, 
the vector y defined by $y = c_1v_1 + c_2v_2 + ... + c_pv_p$ is called a linear combination of 
$v_1, v_2, .., v_p$ with weights $c_1, c_2, ... c_p$.
- Given an $m$ x $n$ matrix, with columns $a_1, ... a_n$, and if $x$ is in $\mathbb{R}^n$, then $Ax$ 
is the lienar combination of the columns of A using the corresponding entries in $x$ as weights.

Spans:  
- "the columns of A span $\mathbb{R}^n$ means that every b in $\mathbb{R}^n$ is a linear combination of 
the columns of A. 

Trivial Solutions:  
- if $Ax = 0$, the zero vector is always a solution and is called the trivial solution  
- a nonzero vector x that satisfies $Ax = 0$ is called a nontrivial solution  
- $Ax = 0$ has a nontrivial solution iff the equation has at least one free variable  

Linear Independence:  
- the columns of a matrix A are linearly independent iff the equation $Ax=0$ has ONLY the trivial solution.  
- a set of vectors is linearly dependent if:
    - at least one of the vectors is a linear combination of the others  
    - the # of vectors ,p, is greater than the # of entries, n: (p > n)  
    - it contains the zero vector  

Invertible Matrices: 
- An $n$ x $n$ matrix $A$ is invertible (non-singular) if:
    - $A^{-1}A = I$
    - for each $b$ in $\mathbb{R}^n$, $Ax=b$ has the unique solution $x=A^{-1}b$
    - the equation AX=0 has only the trivial solution
    - the columns of A are linearly independent
    - the linear transformation $x \rightarrow Ax$ is one-to-one
    - $A^T$ is an invertible matrix
    - the columns of A form a basis of $\mathbb{R}^n$
    - rank $A = n$
    - Nul $A = {0}$
    - the number 0 is not an eigenvalue of A
    - det(A) is not zero
- $A$ is a matrix representing some linear transformation. $A^{-1}$, if it exists, is a linear transformation that "undos" $A$.

LU Factorization:
- An $m$ x $n$ matrix A can be expressed in the form $A = LU$, where $L$ is an $m$ x $n$ lower triangular matrix and 
U is an $m$ x $n$ echelon form of A.
- Suppose A can be reduced to echelon form U using only row replacements that add a multiple of one row to another row below it. In this case, there exist unit lower triangular elementary matrices $E_1, E_2, ... E_p$ s.t.:
    - $E_p ... E_1A = U$
    - then, $A = (E_p...E_1)^{-1}U = LU$
    - so, $L = (E_p...E_1)^{-1}$
- Algorithm: 
    - reduce A to an echelon form U by a sequence of row replacement operations, if possible
    - place entries in L s.t. the same sequence of row operations reduces L to I


Rank and Dimension:
- definitions:
    - the rank of a matrix A is the dimension of the column space of A
    - the dimension of a nonzero subspace H is the number of vectors in any basis for H
    - a basis for a subspace H is a linearly independent set in H that spans H
- if the rank of a linear transformation is 1, the output is a line. if the rank is 2, the output vectors land on a 2d plane

***

## Determinant:  

A determinant may be understood as a scaling factor by which a linear transformation changes any area. If 
a determinant squishes an area to a smaller dimension, determinant is 0. Negative determinants invert the 
orientation of a matrix.

Theorems:
- A square matrix A is invertible if and only if det $A$ != 0.
- If A is an n x n matrix, then det $A^T$ = det $A$

Cramer's Rule:   
- Let A be an invertible n x n matrix. For any $b$ in $\mathbb{R}^n$, the unique solution x of $Ax = b$ has entries 
given by:

$$x_i = \frac{detA_i(b)}{detA}$$

where $A_i(b)$ is the matrix obtained from A by replacing column i by the vector b.

***

## Vector Spaces

Vector Space:
- set of vectors on which axioms hold (addition, multiplication, transitive, etc.)

Null Space:
- the null space of an m x n matrix A is the set of all solutions of the homogenous equation Ax = 0.

Column Space:
- the column space of an m x n matrix A is the set of all linear combinations of the columns of A. (The span of the columns of A)
- set of all possible outputs

Linear Transformation:
- a linear transformation T from a vector space V into a vector space W is a rule that assigns to each vector x in V a unique vector T(x).  
- a kernel (null space) of such a T is the set of all u in V such that T(u) = 0. 

***

## Eigenvectors and Eigenvalues

Eigenvector / Eigenvalues: 
- an eigenvector of an n x n matrix A is a nonzero vector x such that $Ax = \lambda x$ for some scalar $\lambda$
- a scalar $\lambda$ is called an eigenvalue if there is a nontrivial solution of $Ax = \lambda x$ 
- such an x is called an eigenvector corresponding to $\lambda$

An eigenvector is a vector that stays on its own span, after a linear transformation. An eigenvalue is a factor 
by which eigenvector is streteched / squished during a linear transformation. 

Characteristic Equation:
- $(A - \lambda I)x = 0$
- a scalar $\lambda$ is an eigenvalue of n x n matrix A iff $\lambda$ satisfies characteristic equation: $det (A - \lambda I) = 0$

If diagonal matrix, each column is an eigenvector.
If symmetric matrix, eigenvectors are always orthogonal.

Diagonalization:
- An n x n matrix is diagonalizable iff A has n linearly independent eigenvectors
- $A = PDP^{-1}$ with D a diagonal matrix, iff the columns of P are n linearly independent eigenvectors of A. 
Then, the diagonal entries of D are eigenvalues of A that correspond, respectively, to the eigenvectors in P
- so an n x n matrix with n distinct eigenvalues is diagonalizable
- n x n matrix with fewer than n distinct eigenvalues may be diagonalizable:
    - iff sum of dimensions of the eigenspaces equals n

***

## Ordinary Least Squares

inner product, orthogonality, orthogonal projections, gram-schmidt process, QR factorization

***

## Symmetric Matrices / Singular Value Decomposition

Symmetric Matrices
- An n x n matrix A is orthogonally diagonalizable iff A is symmetric
- And P (in $A = PDP^{-1}$) is orthonormal eigenvectors. 

Constrained Optimization
- Let $m = min({x^TAx: ||x|| = 1})$ and $M = max({x^TAx: ||x|| = 1})$
- Let A be a symmetric matrix, then $M$ is greatest eigenvalue $\lambda_1$ of A and m is the least eigenvalue of A. 
The value of $x^TAx$ is M when x is a unit eigenvector $u_1$ corresponding to M. The value of $x^TAx$ is m when 
x is a unit eigenvector corresponding to m. 

Singular Value Decomposition
- Any m x n matrix can be factored into $A = QDP^{-1}$
- SVD is based on the following property of ordinary diagonalization: the absolute values of the eigenvalues of a symmetric matrix A measure the 
amounts that A stretches or shrinks certain vectors (the eigenvectors). 
- Singular Values
    - Let A be an m x n matrix
    - Then $A^TA$ is symmetric and diagonlizable
    - ${v_1, v_2, ..., v_n}$ is the orthonormal basis consisting of the n eigenvectors of $A^TA$
    - ${\lambda_1, \lambda_2, ..., \lambda_n}$ are the corresponding eigenvalues
    - Then:
        - $||Av_i||^2 = (Av_i)^T(Av_i) = v_i^TA^TAv_i$ 
        - $ = v_i^T\lambda_iv_i$ since $v_i$ is eigenvector of $A^TA$
        - $ = \lambda_i$ since $v_i$ is a unit vector
    - Then ordering $\lambda_i$ by magnitude for all i from 1 to n, $\lambda_1 = ||Av_1||$ is the maximum.
    - $\lambda_2$ is the maximum of all unit vectors orthogonal to $\lambda_2$ = $||A_v2||$
    - Singular values are the square roots of the eigenvalues of $A^TA$
- Suppose ${v_1, v_2, ..., v_n}$ is an orthonormal basis consisting of eigenvectors of $A^TA$, arranged so that corresponding eigenvalues of $A^TA$ 
satisfy $\lambda_1 > \lambda_2 > ... > \lambda_n$, and suppose A has r nonzero singular valuse. Then ${Av_1, ..., Av_r}$ is an orthogonal basis for Col A, 
and rank A = r.
- SVD
    - Let A be an m x n matrix with rank r
    - There exists m x n matrix $\Sigma$  for which diagonals are the first r singular values of A, 
    $\sigma_1 \geq \sigma_2 \geq ... \geq \sigma_r > 0 $
    - $$\Sigma = \begin{bmatrix} D & 0 \\ 0 & 0\end{bmatrix}$$
    - D is a r x r diagonal matrix for some r not exceeding n or m
    - And there exists m x m U (left singular vectors) and n x n V (right singular vectors), such that:
    - $$A = U\Sigma V^T$$
    - The right singular vectors are the unit eigenvectors of $A^TA$: $v_1,v_2,...,v_n$ where square root of their 
    eigenvalues are the singular values
    - The left singular vectors are made up of the normalized vectors $\frac{Av_1}{\sigma_1}, \frac{Av_2}{\sigma_2}, ..., \frac{Av_r}{\sigma_r}$ (where r is the rank of A). Then if r < m, the remaining columns are 0.

***

## Stats Application / PCA

Let $X$ be a mean-centered p x N matrix of observations. 
And let $S$ be its covariance matrix (e.g. $S = \frac{1}{N-1}XX^T$)

The goal of Principal Components Analysis is to find an orthogonal p x p matrix that determines a change of variable, $X = PY$, with the property that the new variables in Y are uncorrelated and are arranged in order of decreasing variance. 

Substituting in $PY$ for $X$ in $S = \frac{1}{N-1}XX^T$, we can get the covariance matrix of $Y$: $P^TSP$. 
So we want to find P that makes $P^TSP$ diagonal, since vectors in Y are uncorrelated so off-diagonals of its 
covariance matrix should be zero. 

Let D be the diagonal matrix with eigenvalues $\lambda_1,\lambda_2, ..., \lambda_p$ of S on the diagonal, arranged 
in decreasing order. And let P be an orthogonal matrix whose columns are corresponding unit vectors $u$. Then 
$S = PDP^T$ (diagonalization of a symmetric matrix) and rearranging terms -> $P^TSP = D$

















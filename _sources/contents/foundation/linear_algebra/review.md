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
    - the equation AX=b has only the trivial solution
    - the columns of A are linearly independent
    - the linear transformation $x \rightarrow Ax$ is one-to-one
    - $A_T$ is an invertible matrix
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
- $A = PDP^{-1}$ with D a diagonal matrix, iff the columns of P are n linearly independent eigenvectors of A. Then, the diagonal entries of D are eigenvalues of A that correspond, respectively, to the eigenvectors in P
- so an n x n matrix with n distinct eigenvalues is diagonalizable
- n x n matrix with fewer than n distinct eigenvalues may be diagonalizable:
    - iff sum of dimensions of the eigenspaces equals n

Eigenvectors and Linear Transformations:


## Ordinary Least Squares

## Symmetric Matrices / Singular Value Decomposition

## Stats Application / PCA





















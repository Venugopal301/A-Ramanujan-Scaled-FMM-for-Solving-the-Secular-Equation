# A Ramanujan-Scaled Fast Multipole Method for Solving the Secular Equation

This repository contains the Python implementation accompanying the research work:

**A Ramanujan-Scaled Fast Multipole Method for Solving the Secular Equation**

The code implements a one-dimensional Fast Multipole Method (FMM) for the efficient evaluation of the secular equation arising from a rank-one modification of a symmetric eigenvalue problem.

## 1. Problem Formulation

For

\[
A = D + \sigma qq^T,
\]

where \(D=\operatorname{diag}(x_1,\ldots,x_n)\), the eigenvalues are determined from the roots of the secular equation

\[
f(\lambda)
=
1+\sigma\sum\_{j=1}^{n}
\frac{b_j^2}{x_j-y}.
\]

Its derivative is

\[
f'(\lambda)
=
\sigma\sum\_{j=1}^{n}
\frac{b_j^2}{(x_j-y)^2}.
\]

Direct evaluation of these quantities requires \(O(n^2)\) operations. The present implementation uses the one-dimensional FMM for the Cauchy kernel

\[
K(x,y)=\frac{1}{x-y}
\]

to accelerate these evaluations.

## 2. Ramanujan Scaling

The implementation incorporates Ramanujan-based scaling into the FMM translation operators to control the growth of high-order expansion coefficients.

The Ramanujan scaling factor is based on

\[
R_r =
\sqrt{\pi}
\left(
8r^3+4r^2+r+\frac{1}{30}
\right)^{1/6},
\]

and the scaling coefficient is

\[
\eta_j =
\left[
\frac{j}{e}
R_r^{1/r}
\frac{1}{\delta}
\right]^j,
\qquad j\geq1,
\]

with \(\eta_0=1\).

A Stirling-based implementation is also provided for comparison.

## 3. Main Features

The code provides:

- Ramanujan-scaled FMM evaluation.
- Stirling-scaled FMM evaluation.
- M2L translation.
- Hierarchical tree traversal.
- Secular function and derivative evaluation.
- Newton iteration for solving the secular equation.
- Direct \(O(n^2)\) evaluation for reference.
- Timing and speedup comparisons.
- Numerical error analysis for different expansion orders and problem sizes.

## Please run the following:

fmm_Rmnj_scaling.py
max_diff_norm_plot.py
norm_compare.py

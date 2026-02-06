"""
Author: Noah van der Meer
Description: Implementation of the log-likelihood of the proportional
    hazards model, along with its gradient. These are based on Efron's
    formulation.


License: MIT License

Copyright (c) 2025, Noah van der Meer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

"""

# MPyC
from mpyc.runtime import mpc

# Numpy
import numpy as np

# This repository
from secure_survival_analysis.aggregation import group_propagate_right, group_sum
from secure_survival_analysis import np_logarithm
from secure_survival_analysis import np_pow

def negative_log_likelihood(beta, X, delta, grouping, ld):
    """Compute the negative log-likelihood function for the Proportional Hazards Model

    It is assumed that the samples are already sorted on survival times.

    Parameters
    ----------
    beta : array
        model parameters
    X : array
        covariates
    delta : array
        censoring indicator
    grouping : array
        grouping indexing array of bits
    ld : array
        for each non-censored subject, the subject index divided by the total size of the group for
        that survival time

    Returns
    ------
    Negative log-likelihood (fixed point)

    """

    # Total number of records
    N = len(X)
    assert (len(delta) == N) and (len(grouping) == N) and (len(ld) == N)
    assert len(X[0]) == len(beta)

    # Compute the inner products (beta, x_j), and the powers e^(beta, x_j) for all j
    #
    # All can be done in parallel
    #
    w = mpc.np_matmul(X, beta)
    e = np_pow.np_exp(w, -7)

    # Compute the at-risk sums for each subject. This is a local operation
    r = mpc.np_flip(mpc.np_cumsum(mpc.np_flip(e)))

    # Compute the at-risk sums for each subject taking into account the groups
    r_hat = group_propagate_right(r, grouping)

    # Compute the exponent sums for each group
    u = group_sum(delta * e, grouping)

    # Compute the logarithms
    s = np_logarithm.np_log(r_hat - ld * u)

    # Result: one inner product
    q = mpc.np_matmul(delta, w - s)
    return -q


def negative_log_likelihood_gradient(beta, X, delta, grouping, ld):
    """Compute the gradient of the negative log-likelihood function for the Proportional Hazards Model

    It is assumed that the samples are already sorted on survival times.

    Parameters
    ----------
    beta : array
        model parameters
    X : array
        covariates
    delta : array
        censoring indicator
    grouping : array
        grouping indexing array of bits
    ld : array
        for each non-censored subject, the subject index divided by the total size of the group for
        that survival time

    Returns
    ------
    Gradient of negative log-likelihood (fixed point)

    """

    # Total number of records
    N = len(X)
    assert (len(delta) == N) and (len(grouping) == N) and (len(ld) == N)
    assert len(X[0]) == len(beta)

    # Compute the inner products (beta, x_j), the powers e^(beta, x_j)
    # as well as x_j * e^(beta, x_j) for all j
    #
    # All can be done in parallel
    #
    w = mpc.np_matmul(X, beta)
    e = np_pow.np_exp(w, -7)
    c = e[:, np.newaxis] * X  # row-wise product

    # Compute the at-risk sums for each subject. This is a local operation
    r = mpc.np_flip(mpc.np_cumsum(mpc.np_flip(e)))
    v = mpc.np_flip(mpc.np_cumsum(mpc.np_flip(c, axis=0), axis=0), axis=0)

    # Compute the at-risk sums for each subject taking into account the groups
    r_hat = group_propagate_right(r, grouping)
    # Compute the exponent sums for each group
    u = group_sum(delta * e, grouping)

    # Compute at-risk sums and exponent sums for each group, column by column
    v_hat = group_propagate_right(v, grouping)
    h = group_sum(delta[:, np.newaxis] * c, grouping)

    # Perform the divisions
    rec = 1 / (r_hat - ld * u)
    s = mpc.np_multiply(v_hat - ld[:, np.newaxis] * h, rec[:, np.newaxis])

    # Result: matrix product
    q = mpc.np_matmul(delta, X - s)
    return -q

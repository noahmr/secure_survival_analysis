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

import numpy as np
from mpyc.runtime import mpc
from secure_survival_analysis.aggregation import group_propagate_right, group_sum
from secure_survival_analysis.np_logarithm import np_log
from secure_survival_analysis.np_pow import np_exp


def negative_log_likelihood(beta, X, delta, grouping, ld):
    """Compute the negative log-likelihood function for the Proportional Hazards Model

    It is assumed that the samples are already sorted on survival times.

    beta : model parameters
    X : covariates
    delta: censoring indicator
    grouping: grouping indexing array of bits
    ld : for each non-censored subject, the subject index divided by the total size of the group for
        that survival time
    """
    w = X @ beta                        # inner products <beta, x_j> for all j
    e = np_exp(w)                   # e^<beta, x_j>
    r = np.flip(np.cumsum(np.flip(e)))  # at-risk sums for each subject

    # Compute the at-risk sums for each subject taking into account the groups
    r_hat = group_propagate_right(r, grouping)
    u = group_sum(delta * e, grouping)  # exponent sums for each group

    s = np_log(r_hat - ld * u)
    return delta @ (s - w)


def negative_log_likelihood_gradient(beta, X, delta, grouping, ld):
    """Compute gradient of the negative log-likelihood function for the Proportional Hazards Model

    It is assumed that the samples are already sorted on survival times.

    beta : model parameters
    X : covariates
    delta: censoring indicator
    grouping: grouping indexing array of bits
    ld : for each non-censored subject, the subject index divided by the total size of the group for
        that survival time
    """
    # shapes: X (n,d) beta (d,) delta, grouping, ld (n,)
    w = X @ beta              # inner products <beta, x_j> for all j
    e = np_exp(w)         # e^<beta, x_j>
    c = e[:, np.newaxis] * X  # row-wise product

    # Compute the at-risk sums for each subject. This is a local operation
    r = np.flip(np.cumsum(np.flip(e)))  # at-risk sums for each subject
    v = np.flip(np.cumsum(np.flip(c, axis=0), axis=0), axis=0)

    # Compute the at-risk sums for each subject taking into account the groups
    r_hat = group_propagate_right(r, grouping)
    u = group_sum(delta * e, grouping)  # exponent sums for each group

    # Compute at-risk sums and exponent sums for each group, column by column
    v_hat = group_propagate_right(v, grouping)
    h = group_sum(delta[:, np.newaxis] * c, grouping)

    # Perform the divisions
    s = (v_hat - ld[:, np.newaxis] * h) / (r_hat - ld * u)[:, np.newaxis]
    return delta @ (s - X)

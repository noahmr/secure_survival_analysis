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
from secure_survival_analysis.aggregation import group_sum
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
        that survival time, including factor delta
    """
    w = X @ beta                        # inner products <beta, x_j> for all j
    e = np_exp(w)                       # e^<beta, x_j>
    r = np.flip(np.cumsum(np.flip(e)))  # at-risk sums for each subject

    # Compute the at-risk / exponent sums for each subject taking into account the groups
    u = grouping * r - ld * e
    s = group_sum(u, grouping)

    s = np_log(s)
    return delta @ (s - w)


def negative_log_likelihood_gradient(beta, X, delta, grouping, ld):
    """Compute gradient of the negative log-likelihood function for the Proportional Hazards Model

    It is assumed that the samples are already sorted on survival times.

    beta : model parameters
    X : covariates
    delta: censoring indicator
    grouping: grouping indexing array of bits
    ld : for each non-censored subject, the subject index divided by the total size of the group for
        that survival time, including factor delta
    """
    # shapes: X (n,d) beta (d,) delta, grouping, ld (n,)
    if not isinstance(beta, type(X)):
        assert np.all(beta == 0)
        e = type(X)(np.ones((len(X), 1), dtype='O'))
        c = X
    else:        
        w = X @ beta              # inner products <beta, x_j> for all j
        e = np_exp(w)             # e^<beta, x_j>
        e = e[:, np.newaxis]
        c = e * X  # row-wise product
    e_c = np.hstack((e, c))

    # Compute the at-risk sums for each subject. This is a local operation
    r_v = np.flip(np.cumsum(np.flip(e_c, axis=0), axis=0), axis=0)

    # Compute at-risk / exponent sums for each subject taking into account the groups, col by col
    u = (grouping * r_v.T).T - ld[:, np.newaxis] * e_c
    s = group_sum(u, grouping)

    s = s[:, 1:] / s[:, :1]  # perform the divisions
    return delta @ (s - X)
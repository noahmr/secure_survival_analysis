"""
Author: Noah van der Meer
Description: Implementation of fitting procedure for the proportional
    hazards model, based on gradient descent, BFGS or L-BFGS


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

# Python
import time
import logging

# Numpy
import numpy as np

# This repository
from secure_survival_analysis import ph_log_likelihood
from secure_survival_analysis.aggregation import group_values, _group_sum
from secure_survival_analysis.optimize import gradient_descent, bfgs, lbfgs

async def fit_proportional_hazards_model(table, method = 'l-bfgs', alpha = 1, num_iterations = 10, tolerance=0.005):
    """Fit the proportional hazards model

    Parameters
    ----------
    table : array
        table containing the times, censoring status, and then the covariates (in this order)
    method : str
        optimization method
    alpha :
        step size
    num_iterations :
        maximum number of iterations to perform
    tolerance :
        tolerance for stopping criterion

    Returns
    ------
    Minimizer beta, log-likelihood at last iteration
    """

    ttype = type(table)

    logging.info("###### Fitting proportional hazards model ######")

    # Sort and group on survival time
    logging.info("fit_proportional_hazards_model(): grouping values")
    X_sorted, grouping = group_values(table, group_column=0, sort_column=0)

    # Separate the status
    delta_sorted = X_sorted[:,1]

    # Drop the times and status; this isolates the covariates
    X_sorted = X_sorted[:,2:]

    await mpc.barrier(f"sorting & grouping on survival time")
    logging.info("finished sorting & grouping on survival time")

    # Compute group sizes, and indices within the groups
    d, l = _group_sum(delta_sorted, grouping)
    ld = delta_sorted * ((l - 1) / (d + (1 - delta_sorted)))

    await mpc.barrier(f"computing group sizes & indices")
    logging.info("finished computing group sizes & indices")

    # Define log-likelihood function
    def f(b):
        return ph_log_likelihood.negative_log_likelihood(b, X_sorted, delta_sorted, grouping, ld)

    # Define gradient of log-likelihood function
    def f_grad(b):
         return ph_log_likelihood.negative_log_likelihood_gradient(b, X_sorted, delta_sorted, grouping, ld)

    num_features = len(X_sorted[0])
    beta0 = ttype(np.array([np.float64(0)] * num_features))

    start = time.time()

    H = None
    if method == 'gd':
        beta, likelihoods_ = await gradient_descent(f, f_grad, beta0, alpha, num_iterations, tolerance=tolerance)
    elif method == 'bfgs':
        beta, likelihoods_, H = await bfgs(f, f_grad, beta0, alpha, num_iterations, tolerance=tolerance)
    elif method == 'l-bfgs':
        beta, likelihoods_ = await lbfgs(f, f_grad, beta0, alpha, num_iterations, m = 7, tolerance=tolerance)
    else:
        raise ValueError("invalid method specified")

    await mpc.barrier(f"final barrier")
    logging.info("finished fitting the model through optimization process")

    dt = time.time() - start
    logging.info(f"fitting model took {dt:.3f} seconds")

    likelihoods = mpc.np_fromlist(likelihoods_)
    return beta, likelihoods, H

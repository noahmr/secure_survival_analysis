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

import time
import logging
import numpy as np
from mpyc.runtime import mpc
from secure_survival_analysis import ph_log_likelihood
from secure_survival_analysis.aggregation import group_values, _group_sum
from secure_survival_analysis.optimize import gradient_descent, bfgs, lbfgs

def initial_point_heuristic_death_alive(X_sorted, delta_sorted):
    """Heuristic: average of covariates of deaths (higher risk), minus average
        of covariates of all subjects (average risk)
    """

    num_events = mpc.np_sum(delta_sorted)
    avg_events = mpc.np_matmul(delta_sorted, X_sorted) / num_events
    avg_all = mpc.np_sum(X_sorted, axis=0) / len(X_sorted)

    beta0 = (avg_events - avg_all)
    return beta0

async def initial_point_heuristic_parts(X_sorted, delta_sorted, part_size):
    """Heuristic: average of covariates of subjects that died earlier (higher risk),
        minus average of covariates of subjects that died later (lower risk)

        Size of each set is controlled by the 'part_size' parameter.
    """

    X_low = X_sorted[:part_size, :]
    delta_low = delta_sorted[:part_size]
    
    events_low = mpc.np_sum(delta_low)
    avg_low = mpc.np_matmul(delta_low, X_low) / events_low

    X_high = X_sorted[-part_size:, :]
    delta_high = delta_sorted[-part_size:]

    events_high = mpc.np_sum(delta_high)
    avg_high = mpc.np_matmul(delta_high, X_high) / events_high

    print("average of early deaths: ", await mpc.output(avg_low))
    print("average of late deaths: ", await mpc.output(avg_high))

    return avg_low - avg_high

async def fit_proportional_hazards_model(table, method='l-bfgs', alpha=1, num_iterations=10, tolerance=0.005, sort_column=0):
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

    logging.info("###### Fitting proportional hazards model ######")

    # Sort and group on survival time
    logging.info("fit_proportional_hazards_model(): grouping values")
    X_sorted, grouping = group_values(table, sort_column=sort_column, group_column=0)

    # Separate the status
    delta_sorted = X_sorted[:, 1]

    # Drop the times and status; this isolates the covariates
    X_sorted = X_sorted[:, 2:]

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

    #ttype = type(table)
    #num_features = len(X_sorted[0])
    #beta0 = ttype(np.array([np.float64(0)] * num_features))

    #beta0 = initial_point_heuristic_death_alive(X_sorted, delta_sorted)
    beta0 = await initial_point_heuristic_parts(X_sorted, delta_sorted, len(X_sorted) // 4)
    print("beta0: ", await mpc.output(beta0))

    # another heuristic: first fit on 1/10th of the dataset. Then use this as initial
    # estimate for complete procedure.

    start = time.time()

    H = None
    if method == 'gd':
        beta, likelihoods_ = await gradient_descent(f, f_grad, beta0, alpha, num_iterations, tolerance=tolerance)
    elif method == 'bfgs':
        beta, likelihoods_, H = await bfgs(f, f_grad, beta0, alpha, num_iterations, tolerance=tolerance)
    elif method == 'l-bfgs':
        beta, likelihoods_ = await lbfgs(f, f_grad, beta0, alpha, num_iterations, m=7, tolerance=tolerance)
    else:
        raise ValueError("invalid method specified")

    await mpc.barrier(f"final barrier")
    logging.info("finished fitting the model through optimization process")

    dt = time.time() - start
    logging.info(f"fitting model took {dt:.3f} seconds")

    likelihoods = mpc.np_fromlist(likelihoods_)
    return beta, likelihoods, H

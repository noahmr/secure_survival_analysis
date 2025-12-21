"""
Author: Noah van der Meer
Description: Implementation of statistical tests for the parameters of
    the proportional hazards model. This is based on SciPy.

    
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

## Imports
from mpyc.runtime import mpc
import numpy as np
import scipy

def compute_z_scores(beta, se):
    """Compute z scores (e.g. square root of the Wald test statistic)
        
    Parameters
    ----------
    beta : np.array
        model parameters (public)
    se : np.array
        standard error estimates for each of the model parameters (public)

    Returns
    ------
    array containing z scores

    """

    # Replace zero by NaN, to avoid division by zero
    se_ = se.copy()
    se_[se_ == 0] = np.nan

    # testing against null hypothesis of beta=0; therefore
    # computing (beta-0) / se = beta / se
    return beta / se_


def compute_p_values(z_scores):
    """Compute p values for the model parameters
        
    Parameters
    ----------
    z_scores : np.array
        parameter z scores

    Returns
    ------
    array containing p values for each model parameter

    """

    # Compute Wald test statistics as z^2
    ts = (z_scores ** 2)

    # Test statistics now follow a Chi2 distribution with 1 degree of freedom
    return scipy.stats.chi2.sf(ts, 1)


def compute_coefficient_confidence_intervals(beta, se, alpha):
    """Compute 100(1 - alpha) confidence intervals for the model parameters

    e.g. for alpha=0.05, these would be the 95% confidence intervals

    Parameters
    ----------
    beta : np.array
        estimated MLE model parameters (public)
    se : np.array
        standard error estimates for each of the model parameters (public)
    alpha : float
        confidence level
        
    Returns
    ------
    k * 2 matrix containing confidence intervals, where k is
    the number of model parameters
    
    """

    # Value 'c' such that P(-c <= T <= c) = alpha, where T is standard normal
    c = scipy.stats.norm.ppf(1 - alpha / 2)

    return np.column_stack((beta - c * se, beta + c * se))


async def model_overview_table(table, ll):
    """Format the model overview table

    Parameters
    ----------
    table : secfxp.array
        original secret shared data on which the model was fitted
    ll : float
        partial log-likelihood value of the model (assumed to be open)
    
    Returns
    ------
    Tuple with row names, and array containing the relevant statistical values
    
    """

    num_observations = len(table)

    # censoring variable; count the number of observed events
    delta = table[:,1]
    observed_events = await mpc.output(mpc.np_sum(delta))

    row_names = np.array(['number of observations', 'number of events observed', 'partial log-likelihood'])
    values = np.array([num_observations, observed_events, ll])
    
    return row_names, values


def summary_table(beta, se):
    """Obtain a summary table of the Proportional Hazards model, including z scores, confidence
    intervals etc.

    Parameters
    ----------
    beta : np.array
        model parameters (public)
    se : np.array
        standard error estimates (public)
    
    Returns
    ------
    Tuple with column names, and table containing the relevant statistical values
    
    """

    z_scores = compute_z_scores(beta, se)
    p_values = compute_p_values(z_scores)

    ci_coeff = compute_coefficient_confidence_intervals(beta, se, 0.05)
    ci_exp_coeff = np.exp(ci_coeff)

    columns = ['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p']

    # Gather all values into the summary table
    values = np.column_stack((beta, np.exp(beta), se, ci_coeff[:,0], ci_coeff[:,1], ci_exp_coeff[:,0], ci_exp_coeff[:,1], z_scores, p_values))

    return columns, values

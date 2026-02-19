"""Generate a synthetic dataset corresponding to a Proportional Hazards model
and Weibull baseline hazard

This code is taken directly from Appendix A of:
Van der Meer, Noah. "Privacy-Preserving Survival Analysis" Master Thesis, Eindhoven
University of Technology (2025).

Refer to Appendix A for further context, explanation and motivation for the code.
"""

import numpy as np
import pandas as pd
import scipy
import sys

def generate_survival_times(l, k, beta, X, rounding=False):
    """Generate survival times according to the Proportional Hazards model, following a specific beta
        
    Parameters
    ----------
    l : float
        weibull scale parameter
    k : float
        weibull shape parameter
    beta : array
        proportional hazards parameters
    X : array
        covariates
    rounding : bool
        whether to apply rounding or not

    Returns
    ------
    Array of survival times for the survival subjects. Of the same size as X
    """
    
    n = len(X)
    print("Generating", n, "random survival times")
    
    # Inverse of cumulative hazard function of Weibull distribution
    def h0_inv(y):
        return l * np.pow(y, 1/k)
    
    # Generate uniform random values over [0, 1]
    u = np.random.rand(n)
    
    # Compute e^{(beta, x)} for all covariate vectors x
    ip = np.matmul(X, beta)
    exps = np.exp(ip)
        
    T = h0_inv(-np.log(u) / exps)
    if rounding:
        T = np.round(T)
    
    unique = np.unique(T)
    print("Generated", len(unique), "unique event times")
    
    return T

def generate_covariates(n, c):
    """Generate matrix of random covariates, containing n subjects with c covariates each

    Uniform distribution over [-1, 1] is used.

    Parameters
    ----------
    n : int
        number of subjects
    c: int
        number of covariates

    Returns
    ------
    Matrix of n x c, containing the covariates
    """
    
    return 1 - 2 * np.random.rand(n, c)

def generate_random_censoring(s, l, n):
    """Generate n independent random censoring times following an
    exponential distribution
    
    Parameters
    ----------
    s : float
        exponential distribution scale parameter
    l : float
        exponential distribution location parameter
    n : int
        total number of subjects

    Returns
    ------
    Array of length 'n' containing the censoring times
    """
    
    return scipy.stats.expon.rvs(loc=l, scale=s, size=n)

def generate_survival_sample(weibull_scale, weibull_shape, censoring_scale, censoring_location, n, beta):
    """Generate a survival sample corresponding to a proportional hazards
    model, including covariates, survival times and random independent censoring
    
    Parameters
    ----------
    weibull_scale : float
        weibull scale parameter (for survival times)
    weibull_shape : float
        weibull shape parameters (for survival times)
    censoring_scale : float
        exponential distribution scale parameter (for censoring pattern)
    censoring_location : float
        exponential distribution location parameter (for censoring pattern)
    n : int
        total number of subjects
    beta : array
        proportional hazards parameters

    Returns
    ------
    Tuple (X, t, status), with X a matrix containing the covariates, t an array
    of the survival times, and status an array containing the censoring status
    """
    
    X = generate_covariates(n, len(beta))
    t = generate_survival_times(weibull_scale, weibull_shape, beta, X, True)
    
    c = generate_random_censoring(censoring_scale, censoring_location, n)
    
    # Censoring status: event occurred before the censoring time
    status = (t <= c).astype(int)
    
    # Apply censoring pattern: r = min(c, t)
    z_ = status * t + (1 - status) * c
    z = np.round(z_)
    
    num_events = np.sum(status)
    num_censored = n - num_events
    print("Generated sample contains", num_events, "events and", num_censored, "censored observations")

    unique = np.unique(z)
    print("Generated sample contains", len(unique), "unique times overall")
    
    return X, z, status

def main():
    # Handle arguments
    args = sys.argv
    if len(args) <= 2:
        print("Usage: generate_data <num_values> <filename>")
        return 1
    num_values = int(args[1])
    filename = args[2]

    # Proportional hazards model parameters
    beta = [0.02, 0.9, 0.6, 0.03, 0.8]

    # The values 30, 2, 50 and 10 are the parameters associated with the Weibull and
    # exponential distribution. These were chosen as to generate datasets with a
    # significant number of censored subjects, along with introducing many tied event
    # times in order to demonstrate the accuracy of the methods in handling these.
    #
    # Different values can be chosen to simulate different settings.
    X, t, status = generate_survival_sample(30, 2, 50, 10, num_values, beta)

    # Save to CSV
    df = pd.DataFrame(X)
    df['status'] = status
    df['time'] = t

    print(f"Saving to file: {filename}")
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    main()

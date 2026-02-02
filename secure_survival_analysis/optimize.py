"""
Author: Noah van der Meer
Description: Implementation of optimization methods, including gradient
    descent, and the BFGS and L-BFGS quasi-newton methods

    
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
import logging

async def gradient_descent(f, f_grad, beta0, alpha, num_iterations, tolerance=0.005):
    """Use gradient descent to minimize the objective function
        
    Parameters
    ----------
    f : 
        objective function
    f_grad :
        gradient of objective function
    beta0 :
        starting value, should be numpy array
    alpha :
        step size
    num_iterations :
        maximum number of iterations to perform
    tolerance : 
        tolerance for stopping criterion
        
    Returns
    ------
    Minimizer beta, list of function values of each iteration

    """
    
    secfxp = type(beta0[0])

    logging.info("gradient_descent(): starting gradient descent algorithm")
    beta = beta0

    # norm of gradient; for stopping criterion
    grad_norm = secfxp(1)  # initialize to 1 with the correct type

    # List of function values for each iteration, for analysis purposes
    #likelihoods = [f(beta)]
    
    for i in range(0, num_iterations):

        # Evaluate stopping criterion
        stopping_criterion = await mpc.output(grad_norm < tolerance)
        if i >= 2 and stopping_criterion:
            logging.info("gradient_descent(): stopping criterion triggered")
            break

        logging.info(f"gradient_descent(): starting iteration {i}")

        # Evaluate gradient
        grad = f_grad(beta)
        grad_norm = mpc.statistics._fsqrt(mpc.np_matmul(grad, grad))

        # Update estimate through gradient descent step
        beta = beta - alpha * grad
        
        #ll = f(beta)
        #likelihoods.append(ll)

        await mpc.barrier(f"iteration {i}")
        

    logging.info("gradient_descent() finished")

    logging.info("gradient_descent(): evaluating objective function")
    likelihoods = [f(beta)]

    await mpc.barrier(f"objective function")
    logging.info("finished computing objective function")

    return beta, likelihoods


def update_inverse_hessian(H, s, y):
    rho = 1 / y @ s
    r = rho * s
    A = np.eye(len(s)) - np.outer(r, y)  # I - rho_k s_k y_k^T
    C = np.outer(r, s)  # (rho_k s_k s_k^T)
    return A @ H @ A.T + C


async def bfgs(f, f_grad, beta0, alpha, num_iterations, tolerance=0.005):
    """Minimize the objective function f (with gradient f_grad) through the BFGS method

    Parameters
    ----------
    f : 
        objective function
    f_grad :
        gradient of objective function
    beta0 : array
        starting value, should be numpy array
    alpha :
        step size
    num_iterations :
        maximum number of iterations to perform
    tolerance : 
        tolerance for stopping criterion
        
    Returns
    ----------
    BFGS estimate for a minimizer of f, list of function values

    """
    
    secarray = type(beta0)
    secfxp = type(beta0[0])

    logging.info("bfgs(): starting bfgs algorithm")
    beta = beta0
    beta_prev = None

    # keep track of gradient 
    grad_prev = None

    # norm of step direction; for stopping criterion
    w_norm = secfxp(1)  # initialize to 1 with the correct type

    # estimate inverse Hessian matrix
    H = None

    # List of function values for each iteration, for analysis purposes
    #likelihoods = [f(beta)]
    
    for i in range(0, num_iterations):

        # Evaluate stopping criterion
        stopping_criterion = await mpc.output(w_norm < tolerance)
        if i >= 2 and stopping_criterion:
            logging.info("bfgs(): stopping criterion triggered")
            break

        logging.info(f"bfgs(): starting iteration {i}")
        grad = f_grad(beta)

        # Bookkeeping
        w = None
        if i > 0:
            s_i = beta - beta_prev
            y_i = grad - grad_prev

        # First iteration is simply gradient descent; We only have the gradient
        # at one point, no info about curvature yet.
        if i == 0:
            logging.info("--- gradient descent step")

            # Explicitly Dampen the first step, since the gradient is typically very large.
            # 
            # The l-bfgs steps are damped in a different manner.
            l2 = mpc.statistics._fsqrt(mpc.np_matmul(grad, grad))
            gamma = 1 / l2
            w = gamma * grad

        else:
            logging.info("--- bfgs step")

            # Update estimate of inverse Hessian matrix
            if H is None:
                gamma = mpc.np_matmul(y_i, s_i) / mpc.np_matmul(y_i, y_i)
                H = gamma * secarray(np.eye(len(beta)))
            H = update_inverse_hessian(H, s_i, y_i)

            w = mpc.np_matmul(H, grad)


        w_norm = mpc.statistics._fsqrt(mpc.np_matmul(w, w))

        # Bookkeeping
        beta_prev = beta
        grad_prev = grad

        # Update beta with l-bfgs step
        beta = beta - alpha * w

        #ll = f(beta)
        #likelihoods.append(ll)

        await mpc.barrier(f"iteration {i}")

        
    logging.info("bfgs() finished")

    logging.info("bfgs(): evaluating objective function")
    likelihoods = [f(beta)]

    await mpc.barrier(f"objective function")
    logging.info("finished computing objective function")

    return beta, likelihoods, H


def two_loop_recursion(s, y, rho, grad):
    """Use two-loop recursion to determine the l-bfgs optimization direction

    This is based on Algorithm 2.1 of:
    Erway, J., and R. F. Marcia. "Solving limited-memory bfgs systems with
    generalized diagonal updates." World Congress on Engineering. 2012.

    as well as Section 4 of:
    Liu, Dong C., and Jorge Nocedal. "On the limited memory BFGS method for large
    scale optimization." Mathematical programming 45.1 (1989): 503-528.

    which specifically suggests using the last y[-1], s[-1] for scaling H_0.

    Parameters
    ----------
    s : list(np.array)
        history of point differences
    y : list(np.array)
        history of gradient differences
    rho : list(secfxp)
        history of rho values
    grad :
        gradient at current point

    Returns
    ------
    Estimate of H^{-1} * grad, e.g. the l-bfgs direction

    """

    l = len(y)
    a = [None] * l
    
    w = grad
    for j in range(l - 1, -1, -1):
        # note: w is changing every iteration. Therefore inner products are
        # computed sequentially
        a[j] = rho[j] * mpc.np_matmul(s[j], w)
        w = w - a[j] * y[j]

    # Note: scale using the last y, s vectors
    gamma = mpc.np_matmul(y[-1], s[-1]) / mpc.np_matmul(y[-1], y[-1])
    w = gamma * w
    
    for j in range(0, l):
        b = rho[j] * mpc.np_matmul(y[j], w)
        w = w + (a[j] - b) * s[j]

    return w

async def lbfgs(f, f_grad, beta0, alpha, num_iterations, m, tolerance=0.005):
    """Minimize the objective function f (with gradient f_grad) through the l-bfgs method

    Parameters
    ----------
    f : 
        objective function
    f_grad :
        gradient of objective function
    beta0 : array
        starting value, should be numpy array
    alpha :
        step size
    num_iterations :
        maximum number of iterations to perform
    m :
        l-bfgs memory size (e.g. 5)
    tolerance : 
        tolerance for stopping criterion
        
    Returns
    ----------
    L-BFGS estimate for a minimizer of f, list of function values

    """
    
    secfxp = type(beta0[0])

    logging.info("lbfgs(): starting lbfgs algorithm")
    beta = beta0
    beta_prev = None

    # memory of previous point/gradient differences
    s = []
    y = []
    rho = []
    
    # keep track of gradient 
    grad_prev = None

    # norm of step direction; for stopping criterion
    w_norm = secfxp(1)  # initialize to 1 with the correct type

    # List of function values for each iteration, for analysis purposes
    #likelihoods = [f(beta)]
    
    for i in range(0, num_iterations):

        # Evaluate stopping criterion
        stopping_criterion = await mpc.output(w_norm < tolerance)
        if i >= 2 and stopping_criterion:
            logging.info("lbfgs(): stopping criterion triggered")
            break

        logging.info(f"lbfgs(): starting iteration {i}")
        grad = f_grad(beta)
        logging.info(f"lbfgs(): end grad starting iteration {i}")

        # Bookkeeping
        w = None
        if i > 0:
            s_i = beta - beta_prev
            s.append(s_i)

            y_i = grad - grad_prev
            y.append(y_i)

            rho_i = 1 / mpc.np_matmul(y_i, s_i)
            rho.append(rho_i)

        if len(s) > m:
            s.pop(0)
            y.pop(0)
            rho.pop(0)

        # First iteration is simply gradient descent; We only have the gradient
        # at one point, no info about curvature yet.
        if i == 0:
            logging.info("--- gradient descent step")

            # Explicitly dampen the first step, since the first gradient is typically very large.
            # 
            # The l-bfgs steps are damped in a different manner.
            l2 = mpc.statistics._fsqrt(mpc.np_matmul(grad, grad))
            gamma = 1 / l2
            w = gamma * grad

        else:
            logging.info("--- l-bfgs step")
            w = two_loop_recursion(s, y, rho, grad)

        w_norm = mpc.statistics._fsqrt(mpc.np_matmul(w, w))

        # Bookkeeping
        beta_prev = beta
        grad_prev = grad

        # Update beta with l-bfgs step
        beta = beta - alpha * w

        #ll = f(beta)
        #likelihoods.append(ll)

        await mpc.barrier(f"iteration {i}")

        
    logging.info("lbfgs() finished")

    logging.info("lbfgs(): evaluating objective function")
    likelihoods = [f(beta)]

    await mpc.barrier(f"objective function")
    logging.info("finished computing objective function")

    return beta, likelihoods

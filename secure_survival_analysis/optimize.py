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

import logging
import numpy as np
from mpyc.runtime import mpc


def norm(a):
    return mpc.statistics._fsqrt(a @ a)


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
    beta = beta0
    for i in range(num_iterations):
        logging.info(f"gradient_descent(): iteration {i}")
        grad = f_grad(beta)
        beta = beta - alpha * grad  # gradient descent step
        if await mpc.output(norm(grad) < tolerance):
            break

    logging.info("gradient_descent(): evaluating objective function")
    likelihoods = [f(beta)]
    await mpc.barrier(f"objective function")
    logging.info("finished computing objective function")
    return beta, likelihoods


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
    logging.info("bfgs(): starting bfgs algorithm")
    beta = beta0
    grad = f_grad(beta)

    # First iteration is simply gradient descent; We only have the gradient
    # at one point, no info about curvature yet.
    logging.info("--- gradient descent step")

    # Explicitly dampen the first step, since the gradient is typically very large.
    # The bfgs steps are damped in a different manner.
    w = grad / norm(grad)
    beta_prev = beta
    beta = beta - alpha * w  # bfgs step
    await mpc.barrier(f"gradient descent step")

    for i in range(1, num_iterations):
        logging.info(f"bfgs(): {i}")
        grad_prev = grad
        grad = f_grad(beta)
        s_i = beta - beta_prev
        y_i = grad - grad_prev

        logging.info("--- bfgs step")
        rho_i = 1 / (y_i @ s_i)
        if i == 1:
            gamma = 1 / (rho_i * (y_i @ y_i))
            H = np.diag(mpc.np_fromlist([gamma] * len(beta)))
        r = rho_i * s_i
        A = np.eye(len(beta)) - np.outer(r, y_i)  # I - rho_i s_i y_i^T
        C = np.outer(r, s_i)  # rho_i s_i s_i^T
        H = A @ H @ A.T + C  # update Hessian
        w = H @ grad
        beta_prev = beta
        beta = beta - alpha * w  # bfgs step
        if await mpc.output(norm(w) < tolerance):
            break

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
    w = grad
    l = len(y)
    a = [None] * l
    for j in range(l-1, -1, -1):
        a[j] = rho[j] * (s[j] @ w)
        w -= a[j] * y[j]
    w *= (y[-1] @ s[-1]) / (y[-1] @ y[-1])  # NB: scale w using last y, s vectors
    for j in range(l):
        b = rho[j] * (y[j] @ w)
        w += (a[j] - b) * s[j]
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
    logging.info("lbfgs(): starting lbfgs algorithm")
    beta = beta0
    grad = f_grad(beta)

    # memory of previous point/gradient differences
    s = []
    y = []
    rho = []

    # First iteration is simply gradient descent; We only have the gradient
    # at one point, no info about curvature yet.
    logging.info("--- gradient descent step")

    # Explicitly dampen the first step, since the first gradient is typically very large.
    # The l-bfgs steps are damped in a different manner.
    w = grad / norm(grad)  # NB: norm = 1
    beta_prev = beta
    beta = beta - alpha * w  # l-bfgs step
    await mpc.barrier(f"gradient descent step")

    for i in range(1, num_iterations):
        logging.info(f"lbfgs(): iteration {i}")
        grad_prev = grad
        grad = f_grad(beta)
        s_i = beta - beta_prev
        y_i = grad - grad_prev
        rho_i = 1 / (y_i @ s_i)
        s.append(s_i)
        y.append(y_i)
        rho.append(rho_i)
        if len(s) > m:
            s.pop(0)
            y.pop(0)
            rho.pop(0)

        logging.info("--- l-bfgs step")
        w = two_loop_recursion(s, y, rho, grad)
        beta_prev = beta
        beta = beta - alpha * w  # l-bfgs step
        if await mpc.output(norm(w) < tolerance):
            break

    logging.info("lbfgs() finished")
    likelihoods = [f(beta)]
    await mpc.barrier(f"objective function")
    logging.info("finished computing objective function")
    return beta, likelihoods

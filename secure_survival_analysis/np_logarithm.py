"""
Author: Noah van der Meer
Description: Implementation of a protocol for computing the logarithm of
    fixed point numbers, using MPyC numpy arrays


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

import math
import functools
import logging
import numpy as np
from mpyc.runtime import mpc
import mpyc.mpctools


# Cache the result to avoid re-computing it every time, since the fixed point
# bitlength typically stays the same in any case.
@functools.cache
def log_taylor_degree(f):
    """Determine the degree of the Taylor polynomial necessary to (theoretically)
    achieve the full accuracy of the logarithm approximation

    Parameters
    ----------
    f : int
        number of fractional bits

    Returns
    -------
    minimum degree necessary
    """

    assert isinstance(f, int)

    # Determine the degree of the Taylor polynomial by trying k = 1, 2, 3, ...
    #
    # This is not very efficient, but due to the use of Python caching, only
    # needs to happen once for a given base and precision.
    k = 1
    while math.log2(k + 1) + k + 1 < f:
        k = k + 1
    return k


def np_log_taylor(c):
    """Approximate the logarithm of secret fixed point numbers (within [0.5, 1))

    Parameters
    ----------
    c : secfxp.array
        array of secret shared fixed point numbers, within the interval [0.5, 1)

    Returns
    -------
    logarithms of the inputs
    """

    f = c[0].frac_length

    # Determine degree of Taylor polynomial
    theta = math.ceil(log_taylor_degree(f))

    # Taylor polynomial centered around a=0.75
    a = 0.75
    y = c - a

    # Compute the powers y^0, y^1, y^2, ..., y^theta
    y_powers = mpc.np_vander(y, theta + 1, increasing=True)[:, 1:]

    # Taylor polynomial coefficients (public)
    rng = np.arange(1, theta + 1)
    coeff = (1 / (rng * (a**rng))) * (-1)**(rng + 1)
    constant_term = math.log(a)  # constant term

    # Inner product between powers of y and the coefficients to evaluate the polynomial.
    z = constant_term + mpc.np_matmul(y_powers, coeff)
    return z


def np_log2(x):
    """Approximate the logarithm base 2 of an array of secret fixed point numbers

    Parameters
    ----------
    x : secfxp.array
        array of secret shared fixed point numbers

    Returns
    -------
    logarithms (base 2) of the inputs
    """

    ### Step 1: normalize the inputs; keep the normalization bits for later
    k = x[0].bit_length
    f = x[0].frac_length

    # Bit-decomposition of the inputs; least significant bit is placed at the front
    b = mpc.np_to_bits(x)
    assert (b.shape == (len(x), k))

    # Prefix-OR of the bits
    or_op = lambda v1, v2: v1 + v2 - v1 * v2
    y_ = mpc.np_vstack(list(mpyc.mpctools.accumulate(mpc.np_rot90(b, k=1), or_op)))
    y = mpc.np_rot90(y_, k=3)  # rotate back; and reverse order for each input

    # Detect the first transition; which represents the bit length for each input number
    z_ = y[:, :-1] - y[:, 1:]
    z = mpc.np_concatenate((z_, y[:, -1:]), axis=1)  # append the last element of y

    # Normalization factors
    indices = k - 1 - np.arange(k)
    factors = 2**indices
    v = mpc.np_matmul(z, factors) / (2**f)
    c = x * v

    ### Step 2: taylor approximation
    log_c = np_log_taylor(c)
    log2_c = log_c / math.log(2)

    ### Step 3: extract result
    l = mpc.np_matmul(z, indices)

    log2_x = log2_c - l + f
    return log2_x


def np_log(x):
    """Approximate the natural logarithm of an array of secret fixed point numbers

    Parameters
    ----------
    x : secfxp.array
        array of secret shared fixed point numbers

    Returns
    -------
    logarithms of the inputs
    """

    logs = np_log2(x)
    return (logs / math.log2(math.e))

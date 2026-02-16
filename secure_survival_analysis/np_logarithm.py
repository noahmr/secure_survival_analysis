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


@functools.cache
def _log_taylor_degree(f):
    """Required degree of Taylor polynomial for log x as a function of f."""
    # Degree k s.t. maximum error of 1/(k+1) 2^-(k+1) <= 2^-f, that is log2(k+1) + k+1 >= f.
    k = f  # invariant: math.log2(k+1) + k+1 >= f
    while math.log2(k) + k >= f:
        k -= 1
    return k


def np_log2(a):
    """Secure elementwise logarithm base 2 of a."""
    f = a.frac_length
    l = a.sectype.bit_length
    x = mpc.np_to_bits(a, l=l-1)  # low to high bits, ignore sign bit
    x = np.flip(x, axis=-1)
    k, v = mpc.np_find(x, 1, cs_f=lambda b, i: (i+1+b, (b+1) * 2**i))
    v *= 2**(f - (l-1))  # NB: f <= l
    b = a * v  # 1/2 <= b < 1

    # Evaluate Taylor polynomial around 0.75 at b:
    alpha = 0.75
    theta = _log_taylor_degree(f)
    y = b - alpha
    y_powers = np.vander(y, theta + 1, increasing=True)[:, 1:]  # [y^1 y^2 ... y^theta]
    i = np.arange(1, theta + 1)  # [1 2 ... theta]
    coefficients = -1/math.log(2) / (i * (-alpha)**i)
    log2_b = math.log2(alpha) + y_powers @ coefficients
    return log2_b - k + f


def np_log(a):
    """Secure elementwise natural logarithm of a."""
    return np_log2(a) * math.log(2)

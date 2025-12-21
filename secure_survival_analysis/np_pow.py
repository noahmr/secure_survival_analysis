"""
Author: Noah van der Meer
Description: Implementation of a secure exponentiation with fixed point
    exponents, using MPyC numpy arrays

    
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


# MPyC imports
from mpyc.runtime import mpc
from mpyc.asyncoro import mpc_coro
import gmpy2

# Numpy
import numpy as np

# Python imports
import math
import secrets
import functools

# Cache the result to avoid re-computing it every time, since the fixed point
# bitlength typically stays the same in any case.
@functools.cache
def pow_taylor_degree(b, f):
    """Determine the degree of the Taylor polynomial necessary to (theoretically)
    achieve the full accuracy of the exponentiation approximation
        
    Parameters
    ----------
    b : int
        public integer base
    f : int
        number of fractional bits

    Returns
    -------
    minimum degree necessary
    """

    assert isinstance(b, int)
    assert isinstance(f, int)

    log2b = math.log2(b)
    log2logb = math.log(log2b)

    # Determine the degree of the Taylor polynomial by trying k = 1, 2, 3, ...
    # 
    # This is not very efficient, but due to the use of Python caching, only
    # needs to happen once for a given base and precision.
    k = 1
    factorial = k
    while math.log2(factorial) - log2b - (k+1) * log2logb < f:
        factorial = (k + 1) * factorial
        k = k + 1

    return k

def np_pow_taylor(b, e):
    """Approximate b^e using a Taylor approximation, for public integer base b
    and secret-shared fixed point exponents e which are close to zero

    Parameters
    ----------
    b : int
        public integer base
    e : secfxp.array
        secret-shared fixed point exponents, with absolute value <1

    Returns
    -------
    secret-shared array containing b^e
    """

    assert isinstance(b, int)
    f = e.sectype.frac_length

    # Determine degree of Taylor polynomial necessary to utilize full fixed point precision
    theta = pow_taylor_degree(b, f)

    # Transform computation as b^e = exp(e * log(b))
    y = e * math.log(b)

    # Compute the powers y^1, y^2, ..., y^theta
    y_powers = mpc.np_vander(y, theta + 1, increasing=True)[:, 1:]

    # Taylor polynomial coefficients (public): 1 / k! for k = 1, ..., theta
    rng = np.arange(1, theta + 1)
    coeff = 1 / np.cumprod(rng)

    # Inner product between powers of y and the coefficients to evaluate the polynomial.
    z = 1 + mpc.np_matmul(y_powers, coeff)
    return z

@mpc_coro
async def np_pow_integer_exponent(b, e):
    """Compute b^e, for public integer base b and secret-shared integer exponents e

    This is based on Protocol 7 of:
    Schoenmakers, Berry, and Toon Segers. "Secure Groups for Threshold Cryptography
    and Number-Theoretic Multiparty Computation." Cryptography 7.4 (2023): 56.

    along with Section 7.2 of:
    Damgård, Ivan, et al. "Unconditionally secure constant-rounds multi-party computation
    for equality, comparison, bits and exponentiation." Theory of Cryptography Conference.
    Berlin, Heidelberg: Springer Berlin Heidelberg, 2006.

    Note: this function takes as input and also outputs a secint array specifically,
    not a fixed point array.

    Parameters
    ----------
    b : int
        public integer base
    e : secint.array
        secret-shared integer exponents

    Returns
    -------
    secret-shared secint array containing b^e
    """

    assert isinstance(b, int)
    assert isinstance(e, mpc.SecureIntegerArray)

    ttype = type(e)
    await mpc.returnType((ttype, e.shape))

    modulus = e.sectype.field.modulus
    num_parties = len(mpc.parties)
    sec_param = mpc.options.sec_param

    k = e.sectype.bit_length
    assert (modulus // num_parties) > (2**(k + sec_param))


    ### Step 1: generate the random pair [r], [b^{-r}]

    # Let each party locally generate a random number r_i
    upper = math.ceil(2 ** (k + sec_param) / num_parties)
    r_i = np.array([secrets.randbelow(upper) for _ in range(0, len(e))], dtype=object)

    # Secret-share with the other parties, and sum to obtain r
    r_summands = mpc.np_stack(mpc.input(ttype(r_i)), axis=0)
    r = mpc.np_sum(r_summands, axis=0)

    # Locally compute b^{-r_i} at each party
    b_pow_r_i_inverse_ = gmpy2.powmod_exp_list(b, -r_i, modulus)
    b_pow_r_i_inverse = np.vectorize(int, otypes='O')(b_pow_r_i_inverse_)
    
    # Secret-share b^{-r_i} with the other parties, and multiply to obtain b^{-r}
    b_pow_r_inverse_factors = mpc.np_stack(mpc.input(ttype(b_pow_r_i_inverse)), axis=0)
    b_pow_r_inverse = mpc.np_prod(b_pow_r_inverse_factors, axis=0)


    ### Step 2: open e + r to all parties, and compute b^{e + r} locally
    e_plus_r_ = await mpc.output(e + r)
    e_plus_r = (e_plus_r_ % modulus) # ensure (e_plus_r_ > 0)

    b_pow_e_r_ = gmpy2.powmod_exp_list(b, e_plus_r, modulus)
    b_pow_e_r = np.vectorize(int, otypes='O')(b_pow_e_r_)


    ### Step 3: extract the result b^e by multiplying with [b^{-r}]
    b_pow_e = b_pow_e_r * b_pow_r_inverse

    return b_pow_e

@mpc_coro
async def np_pow_integer_base(b, e, e_lower_bound):#
    """Compute b^e, for public integer base b and secret-shared fixed point exponents e

    Parameters
    ----------
    b : int
        public integer base
    e : secfxp.array
        secret-shared fixed point exponents
    e_lower_bound : float
        public lower bound on the exponents
        
    Returns
    -------
    secret-shared array containing b^e
    """

    assert isinstance(b, int)
    assert (isinstance(e_lower_bound, int) or isinstance(e_lower_bound, float))

    ttype = type(e)
    await mpc.returnType((type(e), e.integral, e.shape))

    l = e.sectype.bit_length
    f = e.sectype.frac_length
    secint = mpc.SecInt(l=l - f, p=e.sectype.field.modulus)

    # For it to be possible for b^e to be represented as a fixed point:
    #
    # (b^e) < 2^{l - f}  <=>  e < log_b(2^{l - f})  <=>  e < (l - f)log_b(2)
    # 
    # Therefore log_2(e) < log_2((l - f)log_b(2)), which gives an upper bound on the
    # bitlength of e.
    e_max_bitlength = f + math.ceil(((l - f) * math.log(2, b))).bit_length() + 1

    ### Step 1: bring negative exponents into positive range
    e_prime = e
    if e_lower_bound < 0:
        ltz = mpc.np_sgn(e, l=e_max_bitlength, LT=True)
        e_prime = e + ltz * abs(e_lower_bound)

    ### Step 2: split the inputs into an integral and fractional part
    e_int_ = mpc.np_trunc(e_prime, l=e_max_bitlength, f=f)
    e_frac = e_prime - e_int_ * (2**f)

    ### Step 3: approximate b^{e_frac} using taylor approximation
    e_pow_frac = np_pow_taylor(b, e_frac)

    ### Step 4: compute b^{e_int} exactly
    e_int_ = await mpc.gather(e_int_) 
    e_int = secint.array(e_int_.value)  # convert from secfxp to secint

    e_pow_int_ = np_pow_integer_exponent(b, e_int)
    e_pow_int_ = await mpc.gather(e_pow_int_)
    e_pow_int = ttype(e_pow_int_.value)  # convert result from secint to secfxp

    ### Step 5: result is b^e = b^{e_frac + e_int} = b^{e_frac} * b^{e_int}
    e_pow = e_pow_frac * e_pow_int

    if e_lower_bound < 0:
        # for the negative exponents, divide out the b^{|e_lower_bound|} again
        d = ltz * (b**(e_lower_bound)) + (1 - ltz)
        e_pow = e_pow * d
    
    return e_pow

@mpc_coro
async def np_pow(b, e, e_lower_bound):
    """Compute b^e, for public base b and secret-shared fixed point exponents e

    Parameters
    ----------
    b : int/float
        public base, either an integer of floating-point number
    e : secfxp.array
        secret-shared fixed point exponents
    e_lower_bound : int/float
        public lower bound on the exponents

    Returns
    -------
    secret-shared array containing b^e
    """

    await mpc.returnType((type(e), (isinstance(b, int) and e.integral), e.shape))

    lower = e_lower_bound

    if isinstance(b, int):
        # Integer base. Can use protocol directly
        return np_pow_integer_base(b, e, lower)

    elif isinstance(b, float):
        # Floating-point base. Convert to base 2, by using
        #
        # b^e = (2^(log_2(b)))^e = 2^(log_2(b) * e)
        log2b = math.log2(b)
        return np_pow_integer_base(2, log2b * e, log2b * lower)

    else:
        raise ValueError("Unsupported base", b)


def np_exp(x, x_lower_bound):
    """Evaluate the exponential function on an array of secret fixed point numbers

    Parameters
    ----------
    x : secfxp.array
        secret-shared fixed point exponents
    x_lower_bound : int/float
        public lower bound on the exponents

    Returns
    -------
    secret-shared array containing exp(x)
    """

    base = math.e
    exp = np_pow(base, x, x_lower_bound)
    return exp

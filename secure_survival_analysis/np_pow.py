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

import math
import secrets
import functools
import gmpy2
from mpyc.numpy import np
from mpyc.runtime import mpc


@functools.cache
def _exp2_taylor_degree(f):
    """Required degree of Taylor polynomial for 2^x as a function of f."""
    log2ln2 = math.log2(math.log(2))
    k = 1
    log2factorial = 1  # log2factorial = log2 (k+1)!
    while log2factorial - (k+1) * log2ln2 < f+1:
        k += 1
        log2factorial += math.log2(k + 1)
    return k


def np_exp2_taylor(a):
    """Approximate 2^a using Taylor approximation, |a|<1."""
    f = a.frac_length
    theta = _exp2_taylor_degree(f)
    y = a * math.log(2)  # 2^a = exp(a log 2)
    y_powers = np.vander(y, theta + 1, increasing=True)  # [y^0 y^1 ... y^theta]
    i = np.arange(1, theta + 1)
    coefficients = 1 / np.cumulative_prod(i, include_initial=True)  # [1/0! 1/1! ... 1/theta!]
    return y_powers @ coefficients


@mpc.coroutine
async def np_pow_integer_exponent(b, a):
    """Compute b^a, for public integer base b and secret-shared integer exponents a

    This is based on Protocol 7 of:
    Schoenmakers, Berry, and Toon Segers. "Secure Groups for Threshold Cryptography
    and Number-Theoretic Multiparty Computation." Cryptography 7.4 (2023): 56.

    along with Section 7.2 of:
    Damgård, Ivan, et al. "Unconditionally secure constant-rounds multi-party computation
    for equality, comparison, bits and exponentiation." Theory of Cryptography Conference.
    Berlin, Heidelberg: Springer Berlin Heidelberg, 2006.
    """
    ttype = type(a)
    await mpc.returnType((ttype, True, a.shape))

    modulus = a.sectype.field.modulus
    m = len(mpc.parties)
    k = mpc.options.sec_param
    l = a.sectype.bit_length
    f = a.sectype.frac_length

    # Let each party locally generate a random number r_i
    bound = 1<<(l + k) // m
    r = np.array([secrets.randbelow(bound) for _ in range(len(a))], dtype='O')
    # Locally compute b^{-r_i} at each party
    b_pow_r = gmpy2.powmod_exp_list(b, -r, modulus)
    b_pow_r = np.vectorize(int, otypes='O')(b_pow_r)

    r_1 = np.stack(mpc.input(ttype(np.vstack((r, b_pow_r)))))
    r = np.sum(r_1[:, 0], axis=0)
    b_pow_r = np.prod(r_1[:, 1], axis=0)  # b^-r
    del r_1

    c = await mpc.output(a + r, raw=True)
    c = c.value >> f
    b_pow_a_r = gmpy2.powmod_exp_list(b, c, modulus)  # b^(a+r)
    b_pow_a_r = np.vectorize(int, otypes='O')(b_pow_a_r)
    b_pow_a_r *= 2**f
    return b_pow_a_r * b_pow_r


def np_exp2(a):
    """Compute 2^a, secret-shared fixed point exponents a."""
    l = a.sectype.bit_length
    f = a.sectype.frac_length
    # For it to be possible for 2^a to be represented as a signed fixed-point number:
    # 2^a < 2^(l-1-f)  <=>  a < l-1-f
    # Therefore log_2 a < log_2(l-1 - f), which gives an upper bound on the bit length of a.
    a_max_bitlength = f + (l-1 - f).bit_length() + 1
    a_int = mpc.np_trunc(a, l=a_max_bitlength, f=f)  # integral part of a
    a_int *= 1<<f
    a_frac = a - a_int
    a_pow = np_exp2_taylor(a_frac)  # 2^a_frac
    a_int += 2**(l-1-f)  # ensure nonnegative a_int
    a_int.integral = True
    a_pow *= np_pow_integer_exponent(2, a_int)  # 2^a_int
    a_pow /= a.sectype.field(2)**(1<<(l-1-f))
    return a_pow  # 2^a


def np_pow(b, a):
    """Secure elementwise exponentiation of array a with public base b (scalar)."""
    if b != 2:
        # Convert to base 2, using b^a = 2^(a log_2 b):
        a *= math.log2(b)
    return np_exp2(a)

def batch(ufunc, batch_size=4096):

    @mpc.coroutine
    async def batched_ufunc(a):  # TODO: generalize
        await mpc.returnType((type(a), a.shape))
        b = []
        for i in range(0, a.size, batch_size):
            b.append(ufunc(a[i: i+batch_size]))
            await b[-1].share
            mpc.peek(b[-1][0], f'await batch {i=} in {ufunc.__name__}')
        return np.concatenate(b)

    return batched_ufunc


@batch
def np_exp(a):
    """Secure elementwise (natural) exponential function of a."""
    return np_pow(math.e, a)

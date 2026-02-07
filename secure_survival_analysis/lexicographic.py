
"""
Author: Noah van der Meer
Description: Tools for lexicographic sorting using MPyC


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

from mpyc.runtime import mpc


class ordered_row:
    """
    Class with which to instantiate objects, to define a lexicographic order
    on rows (which are assumed to be numpy arrays)
    """

    def __init__(self, row):
        # only select relevant indices
        self.row = row[self.indices]

    def __lt__(self, rhs):
        # compute the signs
        #   -1 if self.row > rhs.row
        #   0 if equal
        #   1 if self.row < rhs.row
        sgns = mpc.np_sgn(rhs.row - self.row, l=self.bitlength)
        squares = (sgns**2)

        # by construction guaranteed to be either 0 or 2, thus divisible by 2
        z = (sgns + squares) / 2

        n = len(z)
        r = z[-1]
        for j in range(n - 2, -1, -1):
            r = z[j] + (1 - squares[j]) * r
        return r


def create_ordering(indices, bitlength):
    """
    Create lexicographic ordering class, based on the given indices

    Parameters
    ----------
    indices : list
        list containing the indices which should be used in the ordering, in the given order
    bitlength : int
        upper bound on the bitlengths for all columns

    Returns
    ------
    ordering type
    """

    assert type(indices) is list

    tp = type(f"ordered_row()", (ordered_row,), dict())
    tp.indices = indices
    tp.bitlength = bitlength

    return tp

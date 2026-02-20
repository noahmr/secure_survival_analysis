"""
Author: Noah van der Meer
Description: This script implements the grouping of identical keys, along
    with group aggregates such as the group sum and group sizes


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

import operator
import numpy as np
from mpyc.runtime import mpc
from mpyc import mpctools


def mark_differences(values):
    """
    Mark the places in which values change

    Example:
    [5, 6, 6, 6, 7, 7, 8, 9, 10] -> [1, 1, 0, 0, 1, 0, 1, 1, 1]

    Note: the first value is always marked 1, since it is considered to be
    the start of a new group.

    Parameters
    ----------
    table : array
        table of secret-shared values

    Returns
    ------
    Array of bits, indicating where values change
    """

    ttype = type(values)

    # Create group indexing array by determining when new groups start, i.e.
    # when the value changes
    diff = values[1:] != values[:-1]

    # The first value always marks the start of a group
    one = ttype(np.array([1]))
    b = np.concatenate((one, diff))
    return b


def group_values(table, sort_column=0, group_column=0):
    """
    Sort and create the grouping indexing array for a list of secret shared values

    Parameters
    ----------
    table : array
        table of secret-shared values
    sort_column : int
        index of column to use for sorting. If -1, then no sorting is performed.
    group_column : int
        index of column to use for grouping.

    Returns
    ------
    Table sorted according to the specified sorting column, group indexing array
    """

    # Sort according to desired column
    w = table
    if sort_column >= 0:
        w = mpc.np_sort(w, axis=0, key=lambda v: v[sort_column])

    b = mark_differences(w[:, group_column])
    return w, b


def selective_operator(op):
    """
    Create a selective operator for the associative binary operator 'op'

    It is assumed that f takes two numpy arrays as inputs, with the indexing
    bits placed at the end of these arrays.

    Parameters
    ----------
    op : function/lambda
        associative binary operator

    Returns
    ------
    associative binary operator f((x1, x2, ..., b_x), (y1, y2, ..., b_y)) on tuples
    """

    def f(t1, t2):
        # ((1 - b_y)(x op y) + b_y y, b_x or b_y)
        o = op(t1, t2)

        # Note: this operation performs the selection for the first n-1 columns,
        #  and then the OR for the last column of bits
        return t2[-1] * (t2 - o) + o

    return f


def selective_sum(values, grouping):
    """
    Sum the values within 'values', with a reset occurring at every positive bit in the grouping

    Note: it is assumed that the first bit in 'grouping' actually marks the
    start of a group, i.e. the first elements cannot be "without a group".

    Parameters
    ----------
    values : array
        array of secret shared numbers (can be 2-dimensional)
    grouping : array
        1 dimensional array of indicator bits

    Returns
    ------
    array, containing the selective sum of the values; i.e. the sum of all of the values
    will be stored at the end of each group.
    """

    # TODO: handle arbitrary axis to sum over, in the same way numpy functions do

    assert len(values) == len(grouping)

    # Selective sum operator
#    ssum_op = selective_operator(operator.add)
    ssum_op = lambda t1, t2: (1 - t2[-1])*t1 + t2

    # Combine into a single matrix values|grouping
    combined = np.concatenate((values.reshape((len(values), -1)), grouping[:, np.newaxis]), axis=1)

    # Prefix-OP; Note: the result is an iterator of secure arrays (of length 2 each)
    prefix_result = np.vstack(tuple(mpctools.accumulate(combined, ssum_op, method='Brent-Kung')))
    # TODO: use numpy function for this? E.g. np_accumulate

    # Select all except for the last entry (which has the indicator bits)
    return np.reshape(prefix_result[:, :-1], values.shape)


def group_propagate(values, grouping):
    """
    Propagate the last value for each group to the rest of the group

    Parameters
    ----------
    values : array
        array of secret shared numbers (can be 2-dimensional)
    grouping : array
        one dimensional array of indicator bits

    Returns
    ------
    array of same length as inputs, containing the combined value within each group
    """

    assert len(values) == len(grouping)
    assert values.ndim <= 2
    assert grouping.ndim == 1

    ttype = type(grouping)

    # Shift to the left once and insert a 1 at the end
    grouping_shifted = np.roll(grouping, -1)

    # Ensure grouping array can be multiplied row-wise with values
    if values.ndim == 2:
        grouping_reshaped = grouping_shifted[:, np.newaxis]
    else:  # values.ndim == 1
        grouping_reshaped = grouping_shifted
    # TODO: can this be done in a cleaner way?

    # Apply the grouping as a mask to the values
    u = grouping_reshaped * values

    # Selective sum to propagate last value of each group to the rest of the group
    r = np.flip(selective_sum(np.flip(u, axis=0), np.flip(grouping_shifted, axis=0)), axis=0)
    return r


def group_propagate_right(values, grouping):
    """
    Propagate the first value for each group to the rest of the group
    (direct implementation)

    Parameters
    ----------
    values : array
        array of secret shared numbers (can be 2-dimensional)
    grouping : array
        one dimensional array of indicator bits

    Returns
    ------
    array of same length as inputs, containing the combined value within each group
    """

    assert len(values) == len(grouping)
    assert values.ndim <= 2
    assert grouping.ndim == 1

    # Ensure grouping array can be multiplied row-wise with values
    if values.ndim == 2:
        grouping_reshaped = grouping[:, np.newaxis]
    else:  # values.ndim == 1
        grouping_reshaped = grouping
    # TODO: can this be done in a cleaner way?

    # Apply the grouping as a mask to the values
    u = grouping_reshaped * values

    # Selective sum to propagate first value in each group to the rest of the group
    r = selective_sum(u, grouping)
    return r


def _group_sum(values, grouping):
    assert len(values) == len(grouping)

    # Selective-sum within each group; The sum per group is then stored at the end of each group
    u = selective_sum(values, grouping)

    # Propagate the "last value" to the rest of the group
    r = group_propagate(u, grouping)

    return r, u


def group_sum(values, grouping):
    """
    Compute the sum of the values within each group; Each element within the group will be
    assigned that sum

    Parameters
    ----------
    values : array
        secret shared numbers
    grouping : array
        array of indicator bits

    Returns
    ------
    array of same length as inputs, containing the combined value within each group
    """
    return _group_sum(values, grouping)[0]


class comparable_bit:
    """
    Class which defines an efficient comparison on secret-shared bits
    """

    def __init__(self, bit):
        self.bit = bit

    def __lt__(self, rhs):
        return self.bit - self.bit * rhs.bit


def extract_aggregates(values, grouping):
    """
    Extract the aggregate values and position these at the front

    In principle, it is assumed that the aggregate values are stored at the
    position of the first member of each group.

    Parameters
    ----------
    values : array
        array of secret shared numbers (can be 2-dimensional)
    grouping : array
        one dimensional array of indicator bits

    Returns
    ------
    array of same length as inputs, containing the aggregate values placed at the front,
    with the remaining values as zeros.
    """

    assert len(values) == len(grouping)
    assert values.ndim <= 2
    assert grouping.ndim == 1

    # Ensure grouping array can be multiplied row-wise with values
    if values.ndim == 2:
        grouping_reshaped = grouping[:, np.newaxis]
    else:  # values.ndim == 1
        grouping_reshaped = grouping
    # TODO: can this be done in a cleaner way?

    # Apply the grouping as a mask to the values
    u = grouping_reshaped * values

    u_reshaped = np.reshape(u, (len(u), -1))
    grouping_reshaped = np.reshape(grouping_reshaped, (len(grouping_reshaped), -1))

    # Sort based on the group indicator bits
    tab = np.concatenate((grouping_reshaped, u_reshaped), axis=1)
    w = mpc.np_sort(tab, axis=0, key=lambda v: comparable_bit(v[0]))

    # Return just the values
    return w[:, 1]


def group_count(grouping):
    """
    Determine the sizes of groups

    Parameters
    ----------
    grouping : array
        one dimensional array of indicator bits

    Returns
    ------
    array of same length as inputs, containing the group sizes within each group
    """

    assert grouping.ndim == 1

    n = len(grouping)
    array_type = type(grouping)

    # Assign indices, and append n+1 at the end
    w = grouping * np.arange(1, n+1)
    w = np.concatenate((w, array_type(np.array([n+1]))))

    # Propagate indices to group boundaries
    g_prime = np.concatenate((grouping, array_type(np.array([1]))))
    u = np.flip(selective_sum(np.flip(w), np.flip(g_prime)))

    # Compute group sizes through successive differences
    t = u[1:] - u[:-1]

    # Selective sum to propagate first value in each group to the rest of the group
    r = selective_sum(t, grouping)
    return r


def group_propagate_right_group_sum(values1, values2, grouping):
    """Combine group_propagate_right(values1) and group_sum(values2)."""
    u = (grouping * values1.T).T

    u_values2 = np.column_stack((u, values2))
    R_u = selective_sum(u_values2, grouping)
    if u.ndim == 1:
        R, u = R_u[:, 0], R_u[:, 1]
    else:
        D = R_u.shape[1]
        R, u = R_u[:, :D//2], R_u[:, D//2:]

    # Shift to the left once and insert a 1 at the end
    grouping_shifted = np.roll(grouping, -1)

    # Selective sum to propagate last value of each group to the rest of the group
    r = np.flip(selective_sum(np.flip(u, axis=0), np.flip(grouping_shifted, axis=0)), axis=0)

    return R, r

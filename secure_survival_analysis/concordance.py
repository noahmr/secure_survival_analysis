"""Implementation of concordance index for survival analysis
using MPyC"""

# MPyC imports
from mpyc.runtime import mpc

# Numpy
import numpy as np

# Python imports
import logging

# My code
from secure_survival_analysis.aggregation import mark_differences, group_values, selective_sum, group_sum
from secure_survival_analysis.lexicographic import create_ordering

def harrell_sort_times(times, hazards, status):
    """
    Sort the provided survival data lexicographically on survival time, status (such
    that non-censored subjects come first) and finally hazard score.

    Parameters
    ----------
    times : array
        array containing the actual survival times (secret shared)
    hazards : array
        array containing the predicted hazard corresponding to each subject (secret shared)
    status : array
        censoring status of the subjects (secret shared)

    Returns
    ------
    tuple (t, h, s) containing the sorted times, hazard scores and censoring information
    """

    assert len(times) == len(status)
    assert len(hazards) == len(status)

    # Since the status is a bit, it can be combined into a single variable with
    # the survival times, to save comparisons
    #
    # Invert status such that non-censored subjects (with 1) are placed
    # before censored subjects (with 0)
    #
    sorting_column = 2 * times - status

    table = mpc.np_column_stack((sorting_column, hazards, status))

    # Sort lexicographically on survival time, status and then hazards
    ordering = create_ordering([0, 1], times[0].bit_length)
    sorted_table = mpc.np_sort(table, axis=0, key = lambda v: ordering(v))

    # Extract the sorted times from the first column
    sorted_status = sorted_table[:,2]
    sorted_times = (sorted_table[:,0] + sorted_status) / 2

    return sorted_times, sorted_table[:,1], sorted_status


def harrell_count_comparable_pairs(grouping, status):
    """
    Count the number of comparable pairs

    It is assumed that the values are sorted and grouped by (actual) survival time.

    Parameters
    ----------
    grouping : array
        grouping indexing array of bits (for the survival times)
    censoring : array
        censoring status of the subjects

    Returns
    ------
    number of comparable pairs
    """

    # Example (part 1)
    #
    # Suppose the groups (sorted and grouped on survival time) look like this:
    # [1, 1, 1, 2, 2, 3, 4, 4]  (times)
    # [1, 0, 0, 1, 0, 1, 1, 0]  (grouping)
    #
    # Using a selective sum on an array of ones, and then propagate to get the group sizes
    # [1, 2, 3, 1, 2, 1, 2, 3]  (intermediate result)
    # [3, 3, 3, 2, 2, 3, 3, 3]  (group sizes)
    #
    # By shifting the group indexing once to the left, and then multiplying it in:
    # [0, 0, 3, 0, 2, 0, 0, 3]
    #
    # Next, a summation from the right:
    # [8, 8, 8, 5, 5, 3, 3, 3]
    #
    # Then, subtract the group sizes from this, which gives:
    # [5, 5, 5, 3, 3, 0, 0, 0]
    #
    # For each element, this now stores the number of elements that come afterwards (in different groups). For
    # everything which is not censored, everything afterwards is comparable.
    #
    # Thus multiply by the 'status', and then compute the sum.
    #
    #
    #
    # Example (part 2)
    #
    # What remains are the comparable elements within groups of equivalent survival times. Only elements with
    # unequal censoring are comparable.
    #
    # Consider one single group:
    # [1, 0, 1, 1, 0]
    #
    # which can be seen to have:
    # size: 5
    # non-censored: 3
    # censored: 2
    #
    # The number of comparable pairs is (non-censored) * (censored) = 3 * 2 = 6
    # while the total number of pairs is 1+2+3+4 = 10, therefore, the number of incomparable
    # pairs is 10-6 = 4.
    #

    n = len(grouping)
    array_type = type(grouping)

    # Compute group sizes 'd', with 'u_' as intermediate result
    d = group_sum(array_type(np.ones(n, dtype=int)), grouping)

    # Apply grouping as a mask (shifted to the left by 1 position), in order to set all of
    # the elements (except the last element) within each group to zero
    group_mask = mpc.np_concatenate((grouping[1:], array_type(np.array([1]))))
    u = d * group_mask


    # For each element, compute the total number of elements outside of its own group that
    # still come afterwards.
    remainders_ = mpc.np_flip(mpc.np_cumsum(mpc.np_flip(u)))
    remainders = remainders_ - d

    # For elements that are not censored, everything afterwards is comparable
    comparable_outside_groups = mpc.np_matmul(remainders, status)


    # For each group, get the number of non-censored subjects in the group; Placed at the
    # last value of that group
    noncensored = selective_sum(status, grouping) * group_mask
    censored = d - noncensored

    comparable_within_groups = mpc.np_matmul(censored, noncensored)

    return comparable_outside_groups + comparable_within_groups

def np_xor(a, b):
    """
    Simple helper function to compute the xor of two bit arrays a, b

    Parameters
    ----------
    a : array
        first input array
    b : array
        second input array

    Returns
    ------
    a xor b
    """

    z = (a + b) - 2 * (a * b)
    return z

async def harrell_count_concordant_and_tied(times, times_grouping, hazards, status):
    """
    Count the number of concordant and tied pairs

    It is assumed that the values are sorted lexicographically by (actual) survival
    time, status (inverted) and finally the hazard rate.

    Parameters
    ----------
    times : array
        array containing the actual survival times (secret shared)
    times_grouping : array
        array containing indexing for groups (of equal survival times)
    hazards : array
        array containing the predicted hazard corresponding to each subject (secret shared)
    status : array
        censoring status of the subjects (secret shared)

    Returns
    ------
    tuple (concordant, tied), with both numbers secret shared
    """

    assert len(times) == len(status)
    assert len(times_grouping) == len(status)
    assert len(hazards) == len(status)

    n = len(times)
    array_type = type(times)

    ######## Step 1: sort lexicographically on (hazard rate, survival time, status inverted)
    ########

    # Since the input is already sorted on survival time, status inverted and then hazard
    # rate, the desired ordering can be achieved by sorting on hazard rate, and then the original
    # ordering as the second column.
    #
    # By also sorting on (the inverse of) status, such that non-censored subjects come
    # first, we handle cases where the survival time and hazard rate are equivalent but the
    # censoring is different from the concordant count. Such tuples should in fact be
    # counted as tied values.
    #
    # Finally, a case which causes issues is when the survival time, hazard rate and status
    # are all equivalent. These cases are not comparable, and should be excluded altogether.
    #
    # To do so, also the original order is included in the sorting as a final tie breaker. In
    # that manner, subjects which are completely equivalent will also be ranked with an
    # increasing hazard rank, and thus repeatedly excluded from the count. In essence, this
    # turns the sorting procedure into a stable sort.
    #

    increasing_range = array_type(np.arange(n))  # interpret as secfxp.array()!

    # Insert original indices to remember permutation
    table = mpc.np_column_stack((hazards, status, increasing_range))

    # Sort using lexicographic ordering. By using the original order as the
    # second sorting column, this essentially becomes a stable sort.
    ordering = create_ordering([0, 2], hazards[0].bit_length)
    sorted_table_ = mpc.np_sort(table, axis=0, key = lambda v: ordering(v))

    # Insert increasing indices again; These are the hazard ranks
    sorted_table = mpc.np_hstack((sorted_table_, increasing_range[:, np.newaxis]))
    hazards_sorted = sorted_table[:, 0]
    status_sorted = sorted_table[:, 1]

    # Revert the sorting using the original indices, in order to obtain
    # the desired permutation
    #
    # Include only the last 2 columns in this sorting operation as a
    # slight optimization
    #
    result_table = mpc.np_sort(sorted_table[:, [2, 3]], axis=0, key = lambda v: v[0])

    hazard_ranks = result_table[:, 1]

    await mpc.barrier("lexicographic sorting on hazard rate")
    logging.info("finished lexicographic sorting on hazard rate")


    ######## Step 2: compute the unit vector corresponding to each element. These
    ######## indicate where in the sorted hazards array they are positioned
    ########

    unit_vectors_ = []
    for i in range(0, n):    # note: can be done in parallel
        unit_vectors_.append(mpc.np_unit_vector(hazard_ranks[i], n))
        await mpc.throttler(load_percentage=0.01)
    # TODO: can this be done through numpy directly, instead of a for-loop?

    #unit_vectors_.append(array_type(np.zeros(n)))
    unit_vectors = mpc.np_vstack(unit_vectors_)  # convert to Numpy array

    # Next, sum them up in reverse order
    masks = mpc.np_flip(mpc.np_cumsum(mpc.np_flip(unit_vectors, axis=0), axis=0), axis=0)

    await mpc.barrier("generating unit vectors and masks")
    logging.info("finished generating unit vectors and masks")


    ######## Step 3: count the number of concordant pairs
    ########

    # Example
    #
    # Looking at the mask corresponding to a value:
    # [0, 1, 1, 1, 0, 0, 0, 1]  (mask)
    # This mask marks all of the other subjects that died at a later time. The mask has
    # a 0 at the position of the subject of interest, since it shouldn't be counted
    # against itself.
    #
    # The unit vector corresponding to this value is:
    # [0, 0, 0, 0, 1, 0, 0, 0]
    #
    # Everything to the left of this value has a smaller risk score, and is thus concordant. By
    # computing the prefix sum and then inverting it the array, we obtain:
    # [1, 1, 1, 1, 0, 0, 0, 0]
    # which marks all of these subjects.
    #
    # By computing the inner product of this array with the mask, we obtain the contribution of
    # concordant pairs for this subject.
    #
    #
    # Note: ties are deliberately excluded in the counting, due to the lexicographic sorting
    # first on hazard rate, then on survival time, which determines the hazard ranks.
    #
    # Specifically, for the hazard ranks, if multiple subjects have the same risk score, the
    # subject with the lowest survival time will be placed first, excluding the others from the
    # count. For the second lowest survival time, it will also have a higher hazard rank, and thus
    # again is not included.
    #

    # For each subject, mark all subjects with a lower risk score
    U = (1 - mpc.np_cumsum(unit_vectors, axis=1))

    concordant_results = []
    for i in range(0, n):    # note: can be done in parallel
        # Inner product to obtain concordant pairs
        concordant_results.append(mpc.np_matmul(masks[i], U[i]))
    # TODO: can this be done using numpy, instead of for-loop?

    concordant_results = mpc.np_fromlist(concordant_results)

    # note: each entry only has a contribution if not censored; thus,
    # multiply in the status bits
    num_concordant = mpc.np_matmul(concordant_results, status)

    await mpc.barrier("counting concordant pairs")
    logging.info("finished counting concordant pairs")


    ######## Step 4: count the number of duplicate pairs (equal survival time, hazard rate and censoring)
    ########

    # Note: the inputs are already sorted on survival time, then status, and then the hazards
    #
    # The transitions in survival time are given by the 'times_grouping'. Similarly, the
    # transitions for hazards and status can also be computed through the same protocol.
    #
    # If all of these are 0, this marks a duplicate.
    #
    hazard_duplicates = 1 - mark_differences(hazards)
    status_duplicates_ = 1 - np_xor(status[1:], status[:-1])
    status_duplicates =  mpc.np_concatenate((array_type(np.array([0])), status_duplicates_))
    times_duplicates = 1 - times_grouping

    # Array marking all of the duplicates (with a 1)
    duplicates = times_duplicates * hazard_duplicates * status_duplicates


    ######## Step 5: count the number of tied pairs
    ########

    _, hazard_grouping = group_values(hazards_sorted[:, np.newaxis], sort_column=-1, group_column=0)

    one = array_type(np.array([1]))
    ones = array_type(np.ones(n, dtype=int))

    # Count group sizes; Shift grouping vector to the left once
    hazard_grouping_flipped = mpc.np_concatenate((one, mpc.np_flip(hazard_grouping[1:])))
    hazard_group_sizes = mpc.np_flip(selective_sum(ones, hazard_grouping_flipped))
    tied_pairs = hazard_group_sizes - 1

    # Only non-censored subjects have a contribution, thus multiply with the status variable
    #
    # note: tied_pairs has the subjects sorted on hazard rate; thus, the status array
    #   with the same order is used here as well.
    num_ties_and_duplicates = mpc.np_matmul(status_sorted, tied_pairs)

    # One limitation of the above manner for computing the ties, is that duplicate
    # subjects (with equal survival time, hazard rate, status) are also added (repeatedly),
    # even though these are incomparable.
    #
    # Previously, an array marking the duplicates was computed. By inverting this array, and
    # then counting the group sizes (considering mutual duplicates as one group), we can
    # compute the total (erroneous) contribution to the ties.
    #
    # This value can then be subtracted from the tie count, giving the correct number of ties.
    #
    duplicate_grouping = 1 - (status * duplicates)
    duplicate_counts = selective_sum(ones, duplicate_grouping)
    duplicates_counted = mpc.np_sum(duplicate_counts - 1)

    num_ties = num_ties_and_duplicates - duplicates_counted

    await mpc.barrier("counting tied pairs")
    logging.info("finished counting tied pairs")

    ######## Result
    return (num_concordant, num_ties)


async def harrell_count_pairs(times, hazards, status):
    """
    Count the number of concordant, tied and comparable pairs according
    to Harrell's concordance score.

    Parameters
    ----------
    times : array
        array containing the actual survival times (secret shared)
    hazards : array
        array containing the predicted hazard corresponding to each subject (secret shared)
    status : array
        censoring status of the subjects (secret shared)

    Returns
    ------
    tuple (concordant, tied, comparable), all secret shared
    """

    # Sort according to lexicographic ordering
    times_sorted, hazards_sorted, status_sorted = harrell_sort_times(times, hazards, status)

    # Compute grouping (times are already sorted)
    _, times_grouping = group_values(times_sorted[:,np.newaxis], sort_column=-1, group_column=0)

    await mpc.barrier("lexicographic sorting & grouping on survival time")
    logging.info("finished lexicographic sorting & grouping on survival time")


    # Count comparable
    comparable_pairs = harrell_count_comparable_pairs(times_grouping, status_sorted)

    await mpc.barrier("counting comparable pairs")
    logging.info("finished counting comparable pairs")

    # Compute concordant and tied pairs
    concordant_pairs, tied_pairs = await harrell_count_concordant_and_tied(times_sorted, times_grouping, hazards_sorted, status_sorted)

    await mpc.barrier("counting concordant and tied pairs")
    logging.info("finished counting concordant and tied pairs")

    return (concordant_pairs, tied_pairs, comparable_pairs)


def harrell_concordance_index(concordant, tied, comparable):
    """
    Compute Harrell's concordance index based on the number of concordant,
    tied and comparable pairs.

    These counts can be determined through the function harrell_count_pairs().

    Note: the inputs of this function can be either public, secret fixed-point
    numbers, or a mix of both.

    Parameters
    ----------
    concordant :
        number of concordant pairs
    tied :
        number of tied pairs
    comparable :
        number of comparable pairs

    Returns
    ------
    concordance index within the interval [0, 1]
    """

    c = (concordant + 0.5 * tied) / comparable
    return c
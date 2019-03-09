"""
Improved implementation of the concepts of embargo and purge in chapter 7 of the book
"Advances in Financial Machine Learning" by Marcos de Prado.

In purge(), "for loop" is removed.
"""
import os
import pandas as pd


def get_embargo_table(event_idx, embargo_pct=.01):
    """
    Get embargo times for every bar(event).
    Embargo should be done before purging.

    Args:
        event_idx(DatetimeIndex): t0(start time) of every event
        embargo_pct(float): percentage of bars to embargo

    Returns:
        embargo_table(Series):
            index: t0(start time) of the final test observation
            values: t0(start time) of the final observation to embargo
    """
    step = int(len(event_idx) * embargo_pct)

    if step == 0:  # No embargo needed
        embargo_table = pd.Series(event_idx, index=event_idx)
    else:
        embargo_table = pd.Series(event_idx[step:], index=event_idx[:-step])
        # For those with no training event later than 'step', just embargo to the final start time
        embargo_table = embargo_table.append(
            pd.Series(event_idx[-1], index=event_idx[-step:]))

    return embargo_table


def embargo(cand_times, test_times, embargo_table):
    """
    "Embargo" observations from the training set.

    Args:
        cand_times(Series): times of candidates to be the "embargoed set"
            index: t0(start time)
            value: t1(end time)
        test_times(Series): times of the test set
            index: t0(start time)
            value: t1(end time)
        embargo_table(Series): embargo times table returned by get_embargo_table()

    Returns:
        embargoed_times(Series): times of embargoed training set
            index: t0(start time)
            value: t1(end time)
    """
    first_test_start = test_times.index[0]
    final_test_start = test_times.index[-1]

    final_embargo_start = embargo_table[final_test_start]  # end time of the embargo

    to_embargo_idx = cand_times.loc[first_test_start:final_embargo_start].index
    embargoed_times = cand_times.drop(to_embargo_idx)

    return embargoed_times


def purge(cand_times, test_times):
    """
    "Purge" observations that overlap with the test set,
    which meet any one of these three conditions:
        Case 1. t_j0 <= t_i0 <= t_j1  # training starts within test
        Case 2. t_j0 <= t_i1 <= t_j1  # training ends within test
        Case 3. t_i0 <= t_j0 <= t_j1 <= t_i1  # training envelops test
    where [t_i0, t_i1] are training times and [t_j0, t_j1] are test times.

    Args:
        cand_times(Series): times of candidates to be the training set.
            You could enter either all events or embargoed events
            index: t0(start time)
            value: t1(end time)
        test_times(Series): times of the test set
            index: t0(start time)
            value: t1(end time)

    Returns:
        purged_train_times(Series): times of the purged training set
            index: t0(start time)
            value: t1(end time)
    """
    # Remove "for loop" by forming "test period"
    test_period0 = test_times.index[0]  # test period start
    test_period1 = test_times.max()     # test period end

    # training starts within test
    case_1_idx = cand_times[
        (test_period0 <= cand_times.index) & (cand_times.index <= test_period1)].index
    # training ends within test
    case_2_idx = cand_times[
        (test_period0 <= cand_times) & (cand_times <= test_period1)].index
    # training envelops test
    case_3_idx = cand_times[
        (cand_times.index <= test_period0) & (test_period1 <= cand_times)].index

    purged_train_times = cand_times.drop(
        case_1_idx.union(case_2_idx).union(case_3_idx))

    return purged_train_times

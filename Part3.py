import random


def distribute_values(c_est, d):
    n = len(c_est)
    S = sum(x for x in c_est if x is not None)
    diff = n * d - S

    # Find indices of None values
    none_indices = [i for i, x in enumerate(c_est) if x is None]
    none_count = len(none_indices)

    if none_count == 0:
        return c_est  # No None values, return directly

    # Initial even distribution
    avg_add = diff // none_count
    remainder = diff % none_count  # Remaining part

    # Fill None values first
    base_value = d + avg_add
    values = [base_value] * none_count

    # Randomly distribute the remainder
    for i in random.sample(range(none_count), remainder):
        values[i] += 1

    # Assign values back to c_est
    for i, index in enumerate(none_indices):
        c_est[index] = values[i]

    return c_est


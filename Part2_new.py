import math


def solve_integer_planning_problem_v3(c_est, C, B, k, d, S_unknown):
    """
    Assigns None entries in c_est based on the described non-integer-programming procedure.

    - U3 indices are those with c_est[i] is None and B[i] > 0.
    - U4 indices are the remaining None entries.
    - For U3: x_i ∝ B[i], scaled so that sum(x_i) >= |U3|*d - S3_max,
      where S3_max = S_unknown - |U4|*d.
    - Each x_i = int(round(l * (B[i]/B_min))), where l is chosen minimally.
    - c_est[i] = d - x_i for i in U3.
    - The remaining sum for U4 is split equally: each U4 entry = round((S_unknown - sum_U3) / |U4|).
    """
    # Copy and initialize
    result = list(c_est)
    n = len(result)

    # Identify U3 and U4
    U3 = [i for i in range(n) if result[i] is None and B[i] > 0]
    U4 = [i for i in range(n) if result[i] is None and B[i] <= 0]

    nU3, nU4 = len(U3), len(U4)

    # Compute S3_max and missing sum needed for x_i
    S3_max = S_unknown - nU4 * d
    S3_missing = nU3 * d - S3_max

    # Proportions for U3
    B_vals = [B[i] for i in U3]
    B_min = min(B_vals)
    ratios = [b_i / B_min for b_i in B_vals]
    sum_ratios = sum(ratios)

    # Find minimal scaling l so that sum(x_i) >= S3_missing
    l = math.ceil(S3_missing / sum_ratios)
    while True:
        x_vals = [int(round(l * r)) for r in ratios]
        if sum(x_vals) >= S3_missing:
            break
        l += 1

    # Assign U3 c_est values and check bounds
    for idx, x in zip(U3, x_vals):
        lower, upper = C[idx], d - k[idx]
        if not (lower <= x <= upper):
            raise ValueError(f"x[{idx}] = {x} out of bounds [{lower}, {upper}]")
        result[idx] = d - x

    # Allocate remaining sum to U4
    sum_U3 = sum(result[i] for i in U3)
    remaining = S_unknown - sum_U3
    if nU4 > 0:
        avg_u4 = int(round(remaining / nU4))
        for idx in U4:
            result[idx] = avg_u4

    return result
def adversary_accurate(n, d, A, B, k, R):
    c_est = [None] * n
    S_known = 0  # Sum of c_i we have estimated so far

    for i in range(n):
        # If A[i] < k[i]*R, that suggests c_i > d (the invitation was randomly dropped sometimes)
        if k[i] > 0 and A[i] < k[i] * R:
            # Probability that an adversary-controlled invitation is retained is d / c_i
            # So A[i] / (k[i]*R) ~ d / c_i  => c_i ~ d*k[i]*R / A[i]
            # Protect against division by zero:
            if A[i] == 0:
                # If A[i] is 0, we can't do the ratio directly; it implies c_i is definitely > d
                # We could choose a large integer or handle it differently
                c_est[i] = d + 1
            else:
                ratio = (A[i] / (k[i] * R))
                c_est[i] = round(d / ratio)
            S_known += c_est[i]

    # ------------------------------------------------------------
    # STEP 3: Among those participants for which c_est[i] is still None,
    # we have c_i <= d. We distinguish between c_i = d and c_i < d using B[i].
    # ------------------------------------------------------------
    U1 = []  # Those we suspect have c_i < d
    U2 = []  # Those we confirm have c_i = d

    for i in range(n):
        if c_est[i] is None:
            # Means A[i] == k[i]*R (or k[i] == 0) => c_i <= d
            if B[i] == 0:
                # No replacements needed, so c_i = d
                c_est[i] = d
                S_known += d
                U2.append(i)
            else:
                # We suspect c_i < d
                U1.append(i)

    # ------------------------------------------------------------
    # STEP 4: Use the global constraint sum_{i=1 to n} c_i = n*d to allocate
    # the missing counts among U1 participants proportionally to B[i].
    # ------------------------------------------------------------
    S_total = n * d
    S_unknown = S_total - S_known  # The sum of c_i for i in U1 must be exactly this

    # If each participant i in U1 has a gap x_i = d - c_i, we assume x_i is
    # proportional to B[i]:  x_i = k_ratio * B[i].
    # Then sum_{i in U1} x_i = k_ratio * sum_{i in U1} B[i].
    # Meanwhile, sum_{i in U1} c_i = |U1|*d - sum_{i in U1} x_i = S_unknown.
    # => sum_{i in U1} x_i = |U1|*d - S_unknown
    # => k_ratio * sum_{i in U1} B[i] = |U1|*d - S_unknown
    # => k_ratio = (|U1|*d - S_unknown) / sum_{i in U1} B[i]
    sum_B_U1 = sum(B[i] for i in U1)
    if sum_B_U1 > 0:
        k_ratio = (len(U1) * d - S_unknown) / sum_B_U1
    else:
        # If sum_B_U1 == 0, it means all B[i] in U1 are 0, so presumably c_i = d for them,
        # but that should've been caught above. We can set k_ratio = 0 as a fallback.
        k_ratio = 0

    for i in U1:
        x_i = k_ratio * B[i]  # The gap
        c_est[i] = max(round(d - x_i),0)  # c_i = d - x_i
        S_known += c_est[i]

    c_honest = [c_est[i] - k[i] for i in range(len(c_est))]

    return c_est, c_honest
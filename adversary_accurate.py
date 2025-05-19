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



import numpy as np

def adversary_accurate_numpy(d, A, B, k, R):
    """
    Vectorized adversary_accurate, all inputs are numpy arrays:
      d   -- 阈值
      A   -- 大小为 n 的 ndarray，记录 adversary 收到的邀请数累积
      B   -- 大小为 n 的 ndarray，记录替换邀请的次数累积
      k   -- 大小为 n 的 ndarray，记录 adversary 控制的邀请数
      R   -- 轮数（标量）
    返回：
      c_est    -- ndarray，size n，估计的每个参与者的真实邀请数
      c_honest -- ndarray，size n，估计的 honest 邀请数（c_est - k）
    """
    # 确保 numpy 数组
    print("accurate")
    A = np.asarray(A)
    B = np.asarray(B)
    k = np.asarray(k)
    n = A.shape[0]

    # 用 -1 标记还未估计的 c_est
    c_est = np.full(n, -1, dtype=int)

    # STEP 1: k>0 且 A < k*R 的位置
    mask1 = (k > 0) & (A < k * R)
    # A==0 的子集，直接设为 d+1
    zero_idx = mask1 & (A == 0)
    c_est[zero_idx] = d + 1
    # A>0 的子集，按比例估计
    nz_idx = mask1 & (A > 0)
    ratio = A[nz_idx] / (k[nz_idx] * R)
    c_est[nz_idx] = np.round(d / ratio).astype(int)

    # 已知部分总和
    S_known = c_est[c_est != -1].sum()

    # STEP 2: 剩余位置 (c_est==-1)
    unknown = (c_est == -1)
    # B==0 的位置，说明 c_i = d
    eq_idx = unknown & (B == 0)
    c_est[eq_idx] = d
    S_known += eq_idx.sum() * d
    # B!=0 的位置，暂存到 U1
    lt_idx = unknown & (B != 0)
    U1 = np.where(lt_idx)[0]

    # STEP 3: 全局约束 sum c_i = n*d
    S_total   = n * d
    S_unknown = S_total - S_known
    sum_B_U1  = B[U1].sum()
    if sum_B_U1 > 0:
        k_ratio = (len(U1) * d - S_unknown) / sum_B_U1
    else:
        k_ratio = 0.0

    # 按 B[i] 成比例分配剩余部分
    x_U1 = k_ratio * B[U1]
    c_est[U1] = np.clip(np.round(d - x_U1), 0, None).astype(int)

    # STEP 4: honest 部分
    c_honest = c_est - k

    return c_est, c_honest


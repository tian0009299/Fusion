import random
import Part2, Part3
import numpy as np
import cupy as cp


def adversary_estimate(n, d, A, B, C, k, R):
    # Initial estimation of c values
    c_est = [None] * n
    S_known = 0  # Total sum of the known part

    # Phase 1: For invitations controlled by the adversary,
    # if A[i] < k[i] * R, it indicates that the participant received more than d invitations.

    for i in range(n):
        if k[i] > 0 and A[i] < k[i] * R:
            if A[i] == 0:
                # Prevent division by zero, directly set to d+1
                c_est[i] = d + 1
            else:
                # Using the proportion: d / c_i ≈ A[i] / (k[i] * R)
                c_est[i] = round(d * k[i] * R / A[i])
            S_known += c_est[i]

    # Phase 2: Handling the case where A[i] == k[i] * R
    # If B[i] == 0 and A[i] is not 0, directly set c_i = d; otherwise, mark it as unknown.
    for i in range(n):
        if k[i] > 0 and A[i] == k[i] * R:
            if B[i] == 0 and A[i] != 0:
                c_est[i] = d
                S_known += d

    # Phase 3: Adjust unknown values using global constraints
    S_total = n * d
    S_unknown = S_total - S_known  # Total sum that the unknown part must satisfy

    # Divide the unknown set U into two parts:

    c_est = Part2.solve_integer_planning_problem_v4(c_est, C, B, k, d, S_unknown)


    c_honest = [c_est[i] - k[i] for i in range(len(c_est))]

    return c_est, c_honest


from Part2 import solve_integer_planning_problem_v4, solve_integer_planning_problem_v4_numpy, solve_integer_planning_problem_v4_cupy


def adversary_estimate_numpy(n,d, A, B, C, k, R):
    """
    Vectorized version of adversary_estimate.

    Parameters
    ----------
    d : int
        The drop threshold.
    A, B, C, k : array-like of shape (n,)
        A: adversary-controlled invitations retained count
        B: replacements count
        C: upper bounds / coefficient list for integer programming
        k: adversary-controlled invitations total count
    R : int
        Number of rounds.

    Returns
    -------
    c_est : ndarray of shape (n,)
        Estimated true invitation counts.
    c_honest : ndarray of shape (n,)
        Estimated honest invitation counts (c_est – k).
    """
    # ensure numpy arrays
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    k = np.asarray(k)
    n = A.shape[0]

    # initialize c_est with None (object dtype)
    c_est = np.full(n, None, dtype=object)

    # ----- Phase 1: adversary-controlled invitations dropped sometimes -----
    mask1 = (k > 0) & (A < k * R)
    # if A==0, set to d+1
    zero_idx = mask1 & (A == 0)
    c_est[zero_idx] = d + 1
    # else estimate by ratio: d * k[i] * R / A[i]
    nz_idx = mask1 & (A > 0)
    est_vals = np.round(d * k[nz_idx] * R / A[nz_idx]).astype(int)
    c_est[nz_idx] = est_vals

    # compute S_known so far
    known_mask = np.array([ci is not None for ci in c_est])
    S_known = sum(int(ci) for ci in c_est[known_mask])

    # ----- Phase 2: A[i] == k[i]*R and no replacements => c_i = d -----
    mask2 = (k > 0) & (A == k * R) & (B == 0) & (A != 0)
    c_est[mask2] = d
    S_known += mask2.sum() * d

    # ----- Phase 3: solve for the remaining unknowns via integer programming -----
    S_total = n * d
    S_unknown = S_total - S_known

    # Part2.solve_integer_planning_problem_v4 will fill in the None entries
    c_est_solved = solve_integer_planning_problem_v4_numpy(c_est,
                                                     C,
                                                     B,
                                                     k,
                                                     d,
                                                     S_unknown)
    # back to numpy
    c_est_arr = np.array(c_est_solved, dtype=int)
    c_honest = c_est_arr - k

    return c_est_arr, c_honest



def adversary_estimate_cupy(n, d, A, B, C, k, R):
    """
    全部在 GPU（CuPy）上完成，未知项用 NaN 标记，不再用 dtype=object。
    """
    # 1) 转 float GPU 数组
    A = cp.asarray(A, dtype=float)
    B = cp.asarray(B, dtype=float)
    C = cp.asarray(C, dtype=float)
    k = cp.asarray(k, dtype=float)

    # 2) 用 NaN 初始化 c_est
    c_est = cp.full(n, cp.nan, dtype=float)

    # ---- Phase 1 ----
    m1 = (k > 0) & (A < k * R)
    c_est[m1 & (A == 0)] = d + 1
    idx1 = m1 & (A > 0)
    c_est[idx1] = cp.round(d * k[idx1] * R / A[idx1])

    # ---- Phase 2 ----
    m2 = (k > 0) & (A == k * R) & (B == 0) & (A != 0)
    c_est[m2] = d

    # 3) 已知和 & 剩余总和
    S_known   = float(cp.nansum(c_est))
    S_total   = n * d
    S_unknown = int(S_total - S_known)

    # ---- Phase 3: 调用你的 CuPy solver ----
    #    这里假设 solve_integer_planning_problem_v4_cupy
    #    能接收一个含 NaN 的 float64 CuPy 数组，
    #    并返回一个完整填好的同形 int32/64 CuPy 数组。
    c_filled = solve_integer_planning_problem_v4_cupy(
        c_est, C, B, k, d, S_unknown
    )

    # ---- 结果转换 ----
    c_est_gpu    = c_filled.astype(int)
    c_honest_gpu = c_est_gpu - k.astype(int)
    return c_est_gpu, c_honest_gpu

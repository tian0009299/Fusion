import pulp
import random
import numpy as np
import cupy as cp


def solve_integer_planning_problem_v2(c_est, C, B, k, d, U1, U2, S_unknown, epsilon=0.1, solver=None):
    """
    Use integer programming methods to solve and update c_est.
    """
    print("integer, c_est: ", c_est)
    print("C: ", C)
    print("B: ", B)
    print("k: ", k)
    print("S_unknown: ", S_unknown)
    # 1. Prepare parameters required for integer programming
    p = len(U1)  # Number of variables
    lower_bounds = [C[i] for i in U1]  # Minimum value for each variable
    upper_bounds = [d - k[i] for i in U1]  # Maximum value for each variable
    # Total lower bound: S = len(U1) * d - (S_unknown - len(U2) * d)
    # Equivalent form: S = d * (len(U1) + len(U2)) - S_unknown
    S = p * d - (S_unknown - len(U2) * d)
    # Ratio array: ratio[i] = B[U1[i]]
    ratios = [B[i] for i in U1]

    # 2. Create an integer programming model
    prob = pulp.LpProblem("Minimize_Sum", pulp.LpMinimize)

    # 3. Define decision variables (integer type)
    x_vars = [
        pulp.LpVariable(
            f"x_{i}",
            lowBound=lower_bounds[i],
            upBound=upper_bounds[i],
            cat=pulp.LpInteger
        )
        for i in range(p)
    ]
    slack = pulp.LpVariable("slack", lowBound=0, cat=pulp.LpContinuous)

    # 4. Objective function: minimize the sum of all variables
    prob += pulp.lpSum(x_vars), "Objective_Minimize_Sum"

    # 5. Total sum constraint: sum(x_i) >= S
    prob += pulp.lpSum(x_vars) + slack>= S, "Sum_Constraint"

    # 6. Ratio constraints (assuming ratios[0] != 0)
    #    For i = 1..p-1, ensure x_i is within [ (ratios[i]/ratios[0])*(1-epsilon)*x_0,  (ratios[i]/ratios[0])*(1+epsilon)*x_0 ]
    for i in range(1, p):
        prob += x_vars[i] >= (ratios[i] / ratios[0]) * (1 - epsilon) * x_vars[0], f"Ratio_Lower_{i}"
        prob += x_vars[i] <= (ratios[i] / ratios[0]) * (1 + epsilon) * x_vars[0], f"Ratio_Upper_{i}"

    # 7. Solve the problem
    prob.solve()

    # 8. Collect results
    status = pulp.LpStatus[prob.status]
    objective_value = pulp.value(prob.objective)
    solution = [var.varValue for var in x_vars]

    # 9. Write the solution back to c_est, storing d - x
    #    If the solution is [a, b, c, ...] and U1 = [i1, i2, i3, ...],
    #    then store d - a in c_est[i1], d - b in c_est[i2], ...
    for idx_in_sol, idx_in_U1 in enumerate(U1):
        c_est[idx_in_U1] = max(d - int(solution[idx_in_sol]), 0)

    return c_est


def solve_integer_planning_problem_v3(c_est, C, B, k, d, S_unknown, epsilon=0.4, solver=None):
    """
    Solve for unknown entries in c_est via integer programming using PuLP,
    maximizing c_est values for U3 and then distributing remainder to U4.

    Parameters:
    - c_est: list of ints or None, length n. Unknowns marked as None.
    - C, B, k: lists of length n, providing bounds and ratio info.
    - d: integer constant.
    - S_unknown: integer, total sum of all None entries in c_est.
    - epsilon: relative tolerance for ratio constraints (default 0.1).
    - solver: optional PuLP solver instance.

    Returns:
    - Updated c_est with all None replaced by integer values.
    """
    # Identify U3 (to maximize) and U4 (even split) among unknowns
    U3 = [i for i, val in enumerate(c_est) if val is None and B[i] > 0]
    U4 = [i for i, val in enumerate(c_est) if val is None and B[i] <= 0]
    m4 = len(U4)
    print("c_est: ", c_est)
    print("C: ", C)
    print("B: ", B)
    print("k: ", k)
    print("S_unknown: ", S_unknown)



    # U3 total of x = d - c_est values must leave room for U4 at least d each
    S3_max = S_unknown - m4 * d

    # Create integer program: minimize sum x_i = sum(d - c_est[i])
    prob = pulp.LpProblem("Solve_U3", pulp.LpMinimize)
    x = {
        i: pulp.LpVariable(
            f"x_{i}", lowBound=C[i], upBound=d - k[i], cat=pulp.LpInteger
        )
        for i in U3
    }

    # Objective: minimize sum of x_i so that c_est[i] = d - x_i is maximized
    prob += pulp.lpSum(x[i] for i in U3), "Minimize_d_minus_c_est"
    print("len(U3)", len(U3))
    print("S3_max: ", S3_max)
    print("len(U3) * d - S3_max", len(U3) * d - S3_max)
    # Constraint: total of U3 <= S3_max
    if U3:
        prob += pulp.lpSum(x[i] for i in U3) >= len(U3) * d - S3_max, "Sum_U3_Constraint"

    # Ratio constraints relative to the first U3 element
    if U3:
        # choose baseline index with smallest B[i]
        i0 = max(U3, key=lambda i: B[i])
        for i in U3:
            if i == i0:
                continue
            # x[i] represents d - c_est[i]
            prob += x[i] >= (B[i] / B[i0]) * (1 - epsilon) * x[i0], f"Ratio_Lower_{i}"
            prob += x[i] <= (B[i] / B[i0]) * (1 + epsilon) * x[i0], f"Ratio_Upper_{i}"

    for i in U3:
        prob += x[i] <= (d - k[i]), f"Enforce_Upper_{i}"

    # Solve the model
    if solver:
        prob.solve(solver)
    else:
        prob.solve()
    x_sum = 0
    # Update c_est for U3: c_est[i] = d - x_i
    for i in U3:
        xi = x[i].varValue if i in x else None
        print(xi)
        x_sum += xi

        c_est[i] = d - int(xi) if xi is not None else None
    print(x_sum)
    status = pulp.LpStatus[prob.status]
    print("Solver status:", status)

    # …前面都保持不变…

    # 1) 解完 U3，更新 c_est 后，先计算真正的 sum_c3
    sum_c3 = sum(c_est[i] for i in U3) if U3 else 0

    # 2) 正确的剩余量
    remainder = S_unknown - sum_c3

    # 3) 均匀分配，并 clamp 到 ≥ d
    if m4 > 0:
        avg_val = round(remainder / m4)
        avg_val = max(avg_val, d)
        for i in U4:
            c_est[i] = avg_val
    print("c_est: ", c_est)

    return c_est

def l1_accurate_test(c_true, c_hat):
    if len(c_true) != len(c_hat):
        raise ValueError("c_true and c_hat must have the same length")

    # 只保留 c_hat 不为 None 的位置
    paired = [(t, h) for t, h in zip(c_true, c_hat) if h is not None]
    if not paired:
        raise ValueError("There are no valid entries in c_hat to compare")

    # 计算这些位置上的绝对差之和
    total_diff = sum(abs(t - h) for t, h in paired)

    # 归一化时，只累加对应的 c_true
    norm = sum(t for t, _ in paired)
    if norm == 0:
        raise ValueError("Sum of selected c_true entries is zero, cannot normalize")

    # 计算 L1 准确率
    error = 1 - (total_diff / norm)
    return error



def solve_integer_planning_problem_v4(c_est, C, B, k, d, S_unknown):
    import math
    # Identify U3 and U4 indices
    U3 = [i for i in range(len(c_est)) if c_est[i] is None and B[i] > 0]
    U4 = [i for i in range(len(c_est)) if c_est[i] is None and B[i] <= 0]
    nU3, nU4 = len(U3), len(U4)

    # Compute required sums
    S3_max = S_unknown - nU4 * d
    total_needed = nU3 * d - S3_max

    if nU3 > 0:
        # Prepare B values for U3
        B_vals = [B[i] for i in U3]
        B_max = max(B_vals)

        l = d
        if l < 1:
            l = 1  # 保证 l 至少为 1

        # Decremental search for minimal l
        while l > 0:
            raw_list = []
            for idx, b in zip(U3, B_vals):
                raw = l * b / B_max
                if raw < 1:
                    raw = 1
                raw = max(C[idx], raw)
                raw_list.append(int(round(raw)))
            if sum(raw_list) < total_needed:
                break
            l -= 1
        l += 1  # 回退到第一个满足 sum(raw_list) >= total_needed 的 l

        count = 0
        # Final assignment for U3
        for idx, b in zip(U3, B_vals):
            raw = l * b / B_max
            if raw < 1:
                raw = 1
            raw = max(C[idx], raw)
            x_i = int(round(raw))

            # 限制 x_i 在 [C[idx], d - k[idx]] 范围内
            lower_bound = C[idx]
            upper_bound = d - k[idx]
            if x_i < lower_bound:
                x_i = lower_bound
            elif x_i > upper_bound:
                x_i = upper_bound


            # 然后再更新 c_est
            c_est[idx] = d - x_i
            count += 1



    # Distribute remainder to U4
    sum_U3 = sum(c_est[i] for i in U3)
    remaining = S_unknown - sum_U3
    if nU4 > 0:
        base = remaining // nU4
        r = remaining % nU4

        # 构造一个长度为 nU4 的列表，其中 r 个元素是 base+1，其余是 base
        shares = [base + 1] * r + [base] * (nU4 - r)
        # 随机打乱，使得哪几个 idx 得到 +1 是随机的
        random.shuffle(shares)

        # 将分好的 shares 赋值回 c_est
        for idx, share in zip(U4, shares):
            c_est[idx] = share

    return c_est


import numpy as np


def solve_integer_planning_problem_v4_numpy(c_est, C, B, k, d, S_unknown):
    """
    NumPy 化的整数规划分配函数，直接接受并返回 numpy 数组：

    参数
    ----
    c_est : ndarray of float
        初始估计的 c_est 数组，其中未知项用 np.nan 表示。
    C, B, k : ndarray of float
        长度均为 n 的数组，分别对应原来的 C, B, k。
    d : int
        阈值 d。
    S_unknown : int
        剩余总和约束（所有 np.nan 项之和应为该值）。

    返回
    ----
    c_est_filled : ndarray of int
        填充完成后的 c_est，所有 np.nan 项被替换为整数估值。
    """
    # 把输入转成浮点型，方便用 np.nan 标记未知
    c = c_est.astype(float)
    C = C.astype(float)
    B = B.astype(float)
    k = k.astype(float)

    # 标记 U3（B>0 且未知）和 U4（B<=0 且未知）
    unknown = np.isnan(c)
    U3_mask = unknown & (B > 0)
    U4_mask = unknown & (B <= 0)

    B_U3 = B[U3_mask]
    C_U3 = C[U3_mask]
    k_U3 = k[U3_mask]
    nU3 = B_U3.size
    nU4 = U4_mask.sum()

    # 计算 total_needed
    S3_max = S_unknown - nU4 * d
    total_need = nU3 * d - S3_max

    # 如果 U3 非空，搜索合适的 l 并分配
    if nU3 > 0:
        B_max = B_U3.max()
        l = max(d, 1)

        # 向下搜索 l，直到 sum(round(raw)) < total_need
        while l > 0:
            raw = (l * B_U3) / B_max
            raw = np.maximum(raw, 1)
            raw = np.maximum(raw, C_U3)
            if np.round(raw).sum() < total_need:
                break
            l -= 1
        l += 1

        # 最终计算 U3 的分配 x_i
        raw = (l * B_U3) / B_max
        raw = np.maximum(raw, 1)
        raw = np.maximum(raw, C_U3)
        lower = C_U3
        upper = d - k_U3
        x_U3 = np.clip(np.round(raw), lower, upper).astype(int)

        # c_i = d - x_i
        c[U3_mask] = d - x_U3

    # U4 平均分配剩余
    sum_U3 = np.nansum(c[U3_mask])
    remaining = S_unknown - sum_U3
    if nU4 > 0:
        base = int(remaining // nU4)
        r = int(remaining % nU4)
        shares = np.concatenate([
            np.full(r, base + 1, dtype=int),
            np.full(nU4 - r, base, dtype=int)
        ])
        np.random.shuffle(shares)
        c[U4_mask] = shares

    return c.astype(int)



def solve_integer_planning_problem_v4_cupy(c_est, C, B, k, d, S_unknown):
    """
    GPU 上用 CuPy 完成的整数规划分配函数，直接接受并返回 CuPy 数组：

    参数
    ----
    c_est : cupy.ndarray, float
        初始估计的 c_est 数组，其中未知项用 NaN 表示。
    C, B, k : cupy.ndarray, float
        长度均为 n 的数组，分别对应原来的 C, B, k。
    d : int
        单参与者最大邀请数阈值。
    S_unknown : int
        剩余总和约束（所有 NaN 项的总和应为该值）。

    返回
    ----
    c_est_filled : cupy.ndarray, int
        填充完成后的 c_est，所有 NaN 项被替换为整数估值。
    """
    # 确保为 float 型，以便用 NaN 标记未知
    c = c_est.astype(float)
    C = C.astype(float)
    B = B.astype(float)
    k = k.astype(float)

    # U3: B>0 且未知；U4: B<=0 且未知
    unknown  = cp.isnan(c)
    U3_mask  = unknown & (B > 0)
    U4_mask  = unknown & (B <= 0)

    # 提取子数组
    B_U3 = B[U3_mask]
    C_U3 = C[U3_mask]
    k_U3 = k[U3_mask]
    nU3   = B_U3.size
    nU4   = int(U4_mask.sum())

    # 计算 U3 部分所需的总和
    S3_max     = S_unknown - nU4 * d
    total_need = nU3 * d - S3_max

    # 如果 U3 非空，寻找最小 l 并分配
    if nU3 > 0:
        B_max = B_U3.max()
        l     = max(d, 1)

        # 递减搜索 l，直到 sum(round(raw)) < total_need
        while l > 0:
            raw = (l * B_U3) / B_max
            raw = cp.maximum(raw, 1)
            raw = cp.maximum(raw, C_U3)
            if cp.round(raw).sum() < total_need:
                break
            l -= 1
        l += 1  # 回退到第一个满足条件的 l

        # 按比例计算最终 x_i，并裁剪到 [C_i, d - k_i]
        raw   = (l * B_U3) / B_max
        raw   = cp.maximum(raw, 1)
        raw   = cp.maximum(raw, C_U3)
        lower = C_U3
        upper = d - k_U3
        x_U3  = cp.clip(cp.round(raw), lower, upper).astype(int)

        # 填回 c_est：c_i = d - x_i
        c[U3_mask] = d - x_U3

    # 计算 U4 应平均分配的剩余量
    sum_U3    = cp.nansum(c[U3_mask])
    remaining = S_unknown - sum_U3

    if nU4 > 0:
        base   = int(remaining // nU4)
        r      = int(remaining % nU4)
        # 构造 r 个 (base+1) 和 (nU4-r) 个 base
        shares = cp.concatenate([
            cp.full((r,   ), base + 1, dtype=float),
            cp.full((int(nU4) - r,), base,     dtype=float),
        ])
        # 打乱索引
        perm   = cp.random.permutation(int(nU4))
        shares = shares[perm]
        # 填回 c_est
        c[U4_mask] = shares

    # 最终转成整型返回
    return c.astype(int)


if __name__ == "__main__":
    # c_est = [None,None,None,None,None,None,None,3,4,5]
    # C = [1,1,1,2,3,0,0,0,0,0]
    # B = [105,97,100,197,310,0,0,0,0,0]
    # k = [0,1,0,1,0,2,3,1,2,1]
    # d = 3
    # S_unknown = 18
    c_est= [None, 20, 23, None, 20, None, None, None, None, 28, None, 22, 20, 20, None, None, None, None, 27, None,
            None, None, 26, 24, None, 20, None, None, 34, 23, 20, None, None, 25, None, None, None, None, None, None,
            21, None, None, 28, 20, None, 20, 20, None, 28, 27, None, None, None, 22, 25, 23, 26, None, 25, 22, None,
            None, 24, 23, 25, 23, 24, 26, 21, 21, None, 27, None, 28, None, None, None, None, 26, None, 21, None, 21,
            23, 28, None, None, None, None, None, 21, None, None, None, None, 21, 27, 24, None]
    C= [3, 0, 0, 1, 0, 1, 2, 2, 1, 0, 1, 0, 0, 0, 2, 4, 2, 2, 0, 3, 1, 2, 0, 0, 1, 0, 1, 3, 0, 0, 0, 3, 2, 0, 2, 2, 3,
        1, 2, 3, 0, 1, 3, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 2, 0, 0, 0, 0, 2, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1,
        0, 0, 2, 1, 2, 0, 1, 0, 0, 0, 0, 0, 4, 1, 2, 2, 0, 0, 3, 3, 3, 1, 0, 0, 0, 2]
    B= [32, 0, 0, 11, 0, 9, 7, 19, 15, 0, 7, 0, 0, 0, 19, 38, 15, 10, 0, 41, 11, 14, 0, 0, 6, 0, 6, 17, 0, 0, 0, 33, 14,
        0, 17, 11, 35, 4, 20, 30, 0, 14, 20, 0, 0, 12, 0, 0, 16, 0, 0, 0, 1, 19, 0, 0, 0, 0, 6, 0, 0, 44, 8, 0, 0, 0, 0,
        0, 0, 0, 0, 25, 0, 1, 0, 0, 35, 5, 9, 0, 5, 0, 0, 0, 0, 0, 54, 11, 10, 17, 0, 0, 42, 32, 51, 6, 0, 0, 0, 25]
    k= [1, 2, 3, 2, 2, 1, 2, 4, 0, 2, 3, 4, 1, 1, 2, 1, 4, 2, 2, 1, 1, 1, 3, 2, 0, 3, 2, 1, 1, 2, 1, 1, 0, 5, 4, 1, 2,
        1, 0, 2, 1, 0, 3, 3, 3, 1, 3, 3, 2, 4, 2, 0, 0, 2, 1, 2, 3, 4, 2, 2, 1, 2, 3, 2, 5, 2, 4, 1, 2, 3, 3, 2, 4, 1,
        4, 0, 1, 4, 0, 3, 2, 2, 0, 3, 1, 3, 3, 1, 2, 3, 0, 2, 1, 4, 0, 0, 3, 4, 4, 1]
    S_unknown= 887
    d=20
    epsilon = 0.1

    result = solve_integer_planning_problem_v4(c_est, C, B, k, d, S_unknown)


    # result = solve_integer_planning_problem_v2(
    #     c_est=c_est,
    #     C=C,
    #     B=B,
    #     k=k,
    #     d=d,
    #     U1=U1,
    #     U2=U2,
    #     S_unknown=S_unknown,
    #     epsilon=epsilon,
    #     solver=None
    # )
    print(result)


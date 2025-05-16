import pulp


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

        # Decremental search for minimal l
        l = d
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
        l += 1  # first l that ensures sum >= total_needed

        # Final assignment for U3
        for idx, b in zip(U3, B_vals):
            raw = l * b / B_max
            if raw < 1:
                raw = 1
            raw = max(C[idx], raw)
            x_i = int(round(raw))
            c_est[idx] = d - x_i

    # Distribute remainder to U4
    sum_U3 = sum(c_est[i] for i in U3)
    remaining = S_unknown - sum_U3
    if nU4 > 0:
        avg_u4 = int(round(remaining / nU4))
        for idx in U4:
            c_est[idx] = avg_u4

    return c_est




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


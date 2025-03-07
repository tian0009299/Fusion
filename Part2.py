import pulp


def solve_integer_planning_problem_v2(c_est, C, B, k, d, U1, U2, S_unknown, epsilon=0.1, solver=None):
    """
    Use integer programming methods to solve and update c_est.
    """

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

    # 4. Objective function: minimize the sum of all variables
    prob += pulp.lpSum(x_vars), "Objective_Minimize_Sum"

    # 5. Total sum constraint: sum(x_i) >= S
    prob += pulp.lpSum(x_vars) >= S, "Sum_Constraint"

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


if __name__ == "__main__":
    c_est = [5, 0, 4, 0, 0]
    C = [0, 2, 0, 2, 0]
    B = [0, 72, 0, 73, 0]
    k = [3, 1, 1, 1, 0]
    d = 3
    U1 = [1,3]
    U2 = [4]
    S_unknown = 6
    epsilon = 0.1

    result = solve_integer_planning_problem_v2(
        c_est=c_est,
        C=C,
        B=B,
        k=k,
        d=d,
        U1=U1,
        U2=U2,
        S_unknown=S_unknown,
        epsilon=epsilon,
        solver=None
    )


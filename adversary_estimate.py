import random
import Part2, Part3


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
    U = []  # Set of indices for unknown participants
    for i in range(n):
        if k[i] > 0 and A[i] == k[i] * R:
            if B[i] == 0 and A[i] != 0:
                c_est[i] = d
                S_known += d
            else:
                U.append(i)
        elif k[i] == 0 and B[i] >= 0:
            U.append(i)

    # Phase 3: Adjust unknown values using global constraints
    S_total = n * d
    S_unknown = S_total - S_known  # Total sum that the unknown part must satisfy

    # Divide the unknown set U into two parts:
    # U1 contains elements where B[i] > 0, and U2 contains elements where B[i] == 0
    U1 = [i for i in U if B[i] > 0]
    U2 = [i for i in U if B[i] == 0]

    c_est = Part2.solve_integer_planning_problem_v2(c_est=c_est, C=C, B=B, k=k, d=d, U1=U1, U2=U2, S_unknown=S_unknown,
                                                    epsilon=0.1, solver=None)
    c_est = Part3.distribute_values(c_est, d)

    c_honest = [c_est[i] - k[i] for i in range(len(c_est))]

    return c_est, c_honest

import adversary_accurate, adversary_estimate
import dialing
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import csv



def generate_data(n, d, h):
    cn = round((1-h)*n)
    I = [[random.randint(1, n) for _ in range(d)] for _ in range(n)]
    corrupted = random.sample(range(1, n + 1), cn)

    # Compute c_true, where c_true[i] represents the number of times the value i+1 appears in I
    c_true = [sum(row.count(i + 1) for row in I) for i in range(n)]
    # Compute the contribution of adversary-controlled inputs, k
    # k[i] represents the total number of invitations targeting P_{i+1} in adversary-controlled rows
    k = [0] * n
    for r in corrupted:
        for j in range(d):
            target = I[r - 1][j]  # Target number (1-based)
            k[target - 1] += 1
    c_honest_true = [c_true[i] - k[i] for i in range(len(c_true))]


    return I, corrupted, c_true, k, c_honest_true

def adversary_estimation(n, d, A, B, C, k, R):
    A_ = np.array(A)
    B_ = np.array(B)

    if np.any((A_ == 0) & (B_ == 0)):
        return adversary_estimate.adversary_estimate(n=n, d=d, A=A, B=B, C=C, k=k, R=R)
    else:
        return adversary_accurate.adversary_accurate(n=n, d=d, A=A, B=B, k=k, R=R)


def l1_accurate(c_true, c_hat):
    if len(c_true) != len(c_hat):
        raise ValueError("c_true and c_hat must have the same length")

    # Compute the sum of absolute differences between corresponding elements
    total_diff = sum(abs(a - b) for a, b in zip(c_true, c_hat))

    # Compute the sum of c_true
    norm = sum(c_true)

    # Calculate L1 error without worrying about division by zero
    error = 1 - (total_diff / norm)
    return error


def adversary_view(I, adversary_controls, times):
    n = len(I)  # Number of participants
    d = len(I[0])  # Number of invitations sent by each participant

    # Initialize statistical tables A, B, and C, all as lists of length n
    A = [0] * n
    B = [0] * n
    C = [0] * n

    # Run the protocol for multiple rounds and collect statistics for A, B, and C
    for _ in range(times):
        O = dialing.assign_invitations(I)  # O is an n*d matrix of Invitation objects
        # Temporary statistics for this round: temp[i] records the number of times
        # the invitation target was replaced with P_{i+1} in adversary-controlled rows
        temp = [0] * n
        for r in adversary_controls:
            for j in range(d):
                inv = O[r - 1][j]
                if inv.final_receiver == inv.intended_receiver:
                    A[inv.final_receiver - 1] += 1
                else:
                    B[inv.final_receiver - 1] += 1
                    temp[inv.final_receiver - 1] += 1
        # Update table C: each participant takes the maximum replacement count across all rounds
        for i in range(n):
            C[i] = max(C[i], temp[i])

    return A, B, C


def experiment_for_R(R, n, d, cn, I, adversary_controls, c_true, k, c_honest_true,dp=False, p=0):
    """
    Runs the experiment with the given parameters and returns the computed l1_accurate value.

    Parameters:
        R (int): Number of runs for the adversary_view and adversary_estimation functions.
        n (int): Experiment parameter.
        d (int): Experiment parameter.
        cn (int): Experiment parameter.

    Returns:
        float: The computed l1_accurate value.
    """
    if dp:
        I = dialing.DP_modify_input(I,cn=adversary_controls, p=p)
    # Compute the adversary's view using the provided R
    A, B, C = adversary_view(I=I, adversary_controls=adversary_controls, R=R)

    # Estimate the adversary's values
    c_est, c_honest = adversary_estimation(n=n, d=d, A=A, B=B, C=C, k=k, R=R)

    # Calculate and return the l1_accurate error
    return l1_accurate(c_honest_true, c_honest)


def experiment_for_single_input(I, adversary_controls, k, c_honest_true, R_list):
    n = len(I)
    d = len(I[0])

    record = []
    prev_R = 0

    A = [0] * n
    B = [0] * n
    C = [0] * n

    for R in R_list:
        times = R - prev_R
        A_new, B_new, C_new = adversary_view(I=I, adversary_controls=adversary_controls, times=times)

        A = [a + an for a, an in zip(A, A_new)]
        B = [b + bn for b, bn in zip(B, B_new)]
        C = [max(c_old, c_new) for c_old, c_new in zip(C, C_new)]

        # 调用估计算法
        c_est, c_honest = adversary_estimation(n=n, d=d, A=A, B=B, C=C, k=k, R=R)



        acc = l1_accurate(c_honest_true, c_honest)
        print("Round: ", R, "acc: ", acc)
        record.append(acc)
        prev_R = R
    return record


def experiment_for_multiple_inputs(n,d,h, R_list, runs=50):
    all_records = []

    for run in range(runs):
        print("run: ", run)
        print("n: ", n, "d: ", d, "h: ", h)
        I, corrupted, c_true, k, c_honest_true = generate_data(n, d, h)
        record = experiment_for_single_input(I, adversary_controls=corrupted, k=k, c_honest_true=c_honest_true,
                                             R_list=R_list)
        all_records.append(record)

    avg_record = []
    for idx in range(len(R_list)):
        mean_val = sum(rec[idx] for rec in all_records) / runs
        avg_record.append(mean_val)

    os.makedirs('result', exist_ok=True)
    filename = f"result/result_n{n}_d{d}_h{h}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['R', 'accuracy'])
        for R, acc in zip(R_list, avg_record):
            writer.writerow([R, acc])

    return avg_record


def run_experiments(n, R_list, runs=30):
    """
    先按 d 从小到大依次运行，再对每个 d 里所有支持该 d 的 h 运行实验。
    不打印、不存储，只依赖 experiment_for_multiple_inputs 自身的 CSV 输出。
    """
    param_grid = {
        0.7: [1, 2, 5, 10, 20, 40],
        0.9:  [2, 5, 10, 20, 40],
        0.97: [1, 2, 5, 10, 20, 40],
    }

    all_ds = sorted({d for d_list in param_grid.values() for d in d_list})

    for d in all_ds:
        for h, d_list in param_grid.items():
            if d in d_list:
                experiment_for_multiple_inputs(
                    n=n,
                    d=d,
                    h=h,
                    R_list=R_list,
                    runs=runs
                )





if __name__ == "__main__":
    n = 10000
    d = 20
    h = 0.97
    R_list = [1,5,10,20,50,100,150,200]
    # print(experiment_for_multiple_inputs(n, d, h, R_list, runs=50))
    run_experiments(n, R_list, runs=50)






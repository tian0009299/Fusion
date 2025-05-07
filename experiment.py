import adversary_accurate, adversary_estimate
import dialing
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import csv


def adversary_estimation(n, d, A, B, C, k, R):
    A_ = np.array(A)
    B_ = np.array(B)

    if np.any((A_ == 0) & (B_ == 0)):
        return adversary_estimate.adversary_estimate(n=n, d=d, A=A, B=B, C=C, k=k, R=R)
    else:
        return adversary_accurate.adversary_accurate(n=n, d=d, A=A, B=B, k=k, R=R)


def adversary_view(I, adversary_controls, R):
    n = len(I)  # Number of participants
    d = len(I[0])  # Number of invitations sent by each participant

    # Initialize statistical tables A, B, and C, all as lists of length n
    A = [0] * n
    B = [0] * n
    C = [0] * n

    # Run the protocol for multiple rounds and collect statistics for A, B, and C
    for _ in range(R):
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


def generate_data(n, d, cn):
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


def calculate_accuracy(list1, list2):
    if len(list1) != len(list2):
        return 0.0  # If lengths are different, return 0 accuracy

    correct_count = sum(1 for x, y in zip(list1, list2) if x == y)
    return correct_count / len(list1)


def calculate_similarity(list1, list2, threshold=1, percentage=0.2):
    if len(list1) != len(list2):
        return 0.0  # If lengths are different, return 0 similarity accuracy

    similar_count = sum(
        1 for x, y in zip(list1, list2) if abs(x - y) <= threshold or abs(x - y) / max(y, 1) <= percentage)
    return similar_count / len(list1)


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


def experiment_without_noise(R_list, n, d, cn, runs_per_R=10):
    """
    Run the experiment for each R in R_list, compute the average L1 error,
    and append the results to a CSV file unique to this (n, d, cn) combination.
    Ensures each row is flushed to disk immediately after writing.

    Parameters:
        R_list (list of int): A list of R values to test.
        n (int): Experiment parameter n.
        d (int): Experiment parameter d.
        cn (int): Experiment parameter cn (number of corrupted nodes).
        runs_per_R (int): Number of times to run the experiment per R (default 10).

    Returns:
        list of float: Averaged L1 error values for each R.
    """
    # Construct filename based on parameters
    file_name = f"result\\results_n{n}_d{d}_cn{cn}.csv"

    # Open the CSV file in append+ mode (creates file if it doesn't exist)
    with open(file_name, 'a+', newline='') as f_csv:
        writer = csv.writer(f_csv)

        # Move to the start and check if file is empty to write header
        f_csv.seek(0)
        if not f_csv.read(1):
            writer.writerow(['R', 'n', 'd', 'cn', 'avg_l1'])
            # flush header to disk
            f_csv.flush()
            os.fsync(f_csv.fileno())
            print(f"Created and initialized results file: {file_name}")

        # Move back to end for appending data rows
        f_csv.seek(0, os.SEEK_END)

        # Generate a single dataset for all R runs
        I, adversary_controls, c_true, k, c_honest_true = generate_data(n=n, d=d, cn=cn)
        l1_accurate_averages = []
        print(f"Running experiments for n={n}, d={d}, cn={cn}")

        # Iterate over each R, compute average L1, and append to CSV
        for R in R_list:
            # Collect L1 values and compute mean
            l1_values = [
                experiment_for_R(R, n, d, cn, I, adversary_controls, c_true, k, c_honest_true)
                for _ in range(runs_per_R)
            ]
            avg_l1 = sum(l1_values) / runs_per_R
            l1_accurate_averages.append(avg_l1)

            # Append a new row and immediately flush to disk
            writer.writerow([R, n, d, cn, avg_l1])
            f_csv.flush()
            os.fsync(f_csv.fileno())

    return l1_accurate_averages



def experiment_section4(R_list, n, d_list, cn_percent, runs_per_R=10):
    """
    For each d in d_list, runs experiment_without_noise over R_list,
    then plots all average L1 results on one figure, labeling each line by d.
    Annotates the shared n and computed cn in the title.

    Parameters:
        R_list (list of int): different R values to test.
        n (int): experiment parameter.
        d_list (list of int): different d values to test.
        cn_percent (float): fraction to compute cn = round(cn_percent * n).
        runs_per_R (int): how many repeats per R (default 10).

    Returns:
        dict: mapping each d to its list of averaged l1_accurate values.
    """
    # compute cn
    cn = round(cn_percent * n)

    # collect results
    results = {}
    for d in d_list:
        l1_averages = experiment_without_noise(
            R_list=R_list,
            n=n,
            d=d,
            cn=cn,
            runs_per_R=runs_per_R
        )
        results[d] = l1_averages

    # plot
    plt.figure(figsize=(8, 6))
    for d, averages in results.items():
        plt.plot(R_list, averages, marker='o', label=f'd = {d}')
    plt.xlabel("R (Number of Runs)")
    plt.ylabel("Average L1 Accurate")
    plt.title(f"n = {n}, cn = {cn} (cn_percent = {cn_percent})")
    plt.legend(title="d values")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results



def experiment_with_noise(R_list, n, d, cn, dp_p=0, runs_per_R=10):
    """
    For each value in R_list, runs the experiment multiple times in two modes:
    1. Regular version (no differential privacy noise).
    2. Differential privacy version (dp=True with probability parameter dp_p).

    It then computes the average l1_accurate value over runs_per_R runs for each mode,
    and plots the relationship between R (x-axis) and the averaged l1_accurate (y-axis)
    for both versions on the same graph.

    Parameters:
        R_list (list of int): A list of different R values to test.
        n (int): Experiment parameter (the number of participants).
        d (int): Experiment parameter (the number of the invitations per party).
        cn (int): Experiment parameter (the amount of the corrupted participants).
        runs_per_R (int): Number of times to run the experiment for each R (default is 10).
        dp_p (float): The probability parameter to be passed when dp=True.

    Returns:
        tuple: Two lists, (l1_accurate_normal, l1_accurate_dp), containing the average l1_accurate
               values for the regular version and the differential privacy version, respectively.
    """
    I, adversary_controls, c_true, k, c_honest_true = generate_data(n=n, d=d, cn=cn)
    l1_accurate_normal = []
    l1_accurate_dp = []

    # For each R, run experiments for both modes and compute averages
    for R in R_list:
        l1_values_normal = []
        l1_values_dp = []

        for _ in range(runs_per_R):
            # Regular version: dp=False
            l1_value_normal = experiment_for_R(R, n, d, cn, I, adversary_controls, c_true, k, c_honest_true,dp=False, p=0)
            l1_values_normal.append(l1_value_normal)

            # Differential privacy version: dp=True, with probability parameter dp_p
            l1_value_dp = experiment_for_R(R, n, d, cn, I, adversary_controls, c_true, k, c_honest_true,dp=True, p=dp_p)
            l1_values_dp.append(l1_value_dp)

        avg_normal = sum(l1_values_normal) / runs_per_R
        avg_dp = sum(l1_values_dp) / runs_per_R

        l1_accurate_normal.append(avg_normal)
        l1_accurate_dp.append(avg_dp)

    # Plot the results for both versions
    plt.figure(figsize=(8, 6))
    plt.plot(R_list, l1_accurate_normal, marker='o', label="Without Noise")
    plt.plot(R_list, l1_accurate_dp, marker='s', label=f"With Differential Privacy (p={dp_p})")
    plt.xlabel("R (Number of Runs)")
    plt.ylabel("Average L1 Accurate")
    plt.title("Comparison: Regular vs. Differential Privacy Versions")
    plt.grid(True)
    plt.legend()
    plt.show()

    return l1_accurate_normal, l1_accurate_dp


if __name__ == "__main__":

    R_list = [256]
    n = 100
    d_list = [1,2,4,8,16,32]
    cn_percent = 0.03


    results = experiment_section4(R_list, n, d_list, cn_percent, runs_per_R=15)


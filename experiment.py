import adversary_accurate, adversary_estimate
import dialing
import numpy as np
import random
import matplotlib.pyplot as plt


def adversary_estimation(n, d, A, B, C, k, R):
    A_ = np.array(A)
    B_ = np.array(B)

    if np.any((A_ == 0) & (B_ == 0)):
        return adversary_estimate.adversary_estimate(n=n, d=d, A=A, B=B, C=C, k=k, R=R)
    else:
        return adversary_accurate.adversary_accurate(n=n, d=d, A=A, B=B, k=k, R=R)


def mean_absolute_error(A, B):
    A, B = np.array(A), np.array(B)
    return np.mean(np.abs(A - B))


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
    """
    Compute the L1 error based on two lists of integers, c_true and c_hat.

    The function computes the absolute difference for each corresponding element,
    sums all the differences, normalizes by the sum of c_true, and finally subtracts
    the normalized value from 1 to produce the error.

    Parameters:
        c_true (list of int): The list of true values.
        c_hat (list of int): The list of predicted values.

    Returns:
        float: The L1 error.

    Example:
        c_true = [1, 2, 4, 6, 3]
        c_hat  = [1, 2, 7, 7, 1]
        Absolute differences: 0, 0, 3, 1, 2 (sum = 6)
        Sum of c_true: 16
        L1 error = 1 - (6 / 16) = 0.625
    """
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
        For each value in R_list, runs the experiment multiple times and computes the average l1_accurate value.
        Then plots the relationship between R (x-axis) and the averaged l1_accurate (y-axis).

        Parameters:
            R_list (list of int): A list of different R values to test.
            n (int): Experiment parameter.
            d (int): Experiment parameter.
            cn (int): Experiment parameter.
            runs_per_R (int): Number of times to run the experiment for each R (default is 10).
        """
    I, adversary_controls, c_true, k, c_honest_true = generate_data(n=n, d=d, cn=cn)
    l1_accurate_averages = []

    # Run the experiment for each R in R_list and average the l1_accurate value over multiple runs.
    for R in R_list:
        l1_values = []
        for _ in range(runs_per_R):
            l1_value = experiment_for_R(R, n, d, cn, I, adversary_controls, c_true, k, c_honest_true)
            l1_values.append(l1_value)
        avg_l1 = sum(l1_values) / runs_per_R
        l1_accurate_averages.append(avg_l1)

    # Plot the results.
    plt.figure(figsize=(8, 6))
    plt.plot(R_list, l1_accurate_averages, marker='o')
    plt.xlabel("R (Number of Runs)")
    plt.ylabel("Average L1 Accurate")
    plt.title("Relationship between R and Average L1 Accurate")
    plt.grid(True)
    plt.show()

    return l1_accurate_averages


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
        n (int): Experiment parameter.
        d (int): Experiment parameter.
        cn (int): Experiment parameter.
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
    # print(experiment_for_R(R=100, n = 10, d = 5, cn=2))
    R = [1, 10,50,200,500,1000,5000,10000]
    #print(experiment_without_noise(R_list=R, n=50, d=5, cn=2))
    print(experiment_with_noise(R_list=R, n=50, d=5,cn=2,dp_p=0.2))

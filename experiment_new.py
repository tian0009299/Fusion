import adversary_accurate, adversary_estimate
import dialing
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import csv
import cupy as cp



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

def generate_data_fast(n, d, h):
    """
    Input:
        n: number of participants
        d: number of invitations per participant
        h: fraction of honest participants
    Returns:
        I             : n×d list of lists, each entry in [0..n-1]
        corrupted     : list of corrupted participant indices (0-based)
        c_true        : list of length n, counts of how many invitations each participant received
        k             : list of length n, counts of invitations to each participant coming from corrupted rows
        c_honest_true : list of length n, c_true[i] - k[i]
    """
    # number of corrupted participants
    cn = int(round((1 - h) * n))

    # 1) generate I as an n×d NumPy array of ints in [0..n-1]
    I_arr = np.random.randint(0, n, size=(n, d))

    # 2) pick 'cn' distinct corrupted rows (0-based)
    corrupted_arr = np.random.choice(n, size=cn, replace=False)

    # 3) total counts per participant
    flat = I_arr.ravel()
    counts = np.bincount(flat, minlength=n)
    c_true = counts.tolist()

    # 4) counts coming only from corrupted rows
    corrupted_flat = I_arr[corrupted_arr].ravel()
    k_counts = np.bincount(corrupted_flat, minlength=n)
    k = k_counts.tolist()

    # 5) honest counts
    c_honest_true = (counts - k_counts).tolist()

    # 6) convert back to Python lists
    I = I_arr.tolist()
    corrupted = corrupted_arr.tolist()

    return I, corrupted, c_true, k, c_honest_true


def generate_data_numpy(n, d, h):
    """
    Generate synthetic data for the dialing protocol model.

    Parameters:
    - n (int): number of participants (also number of rows in I)
    - d (int): number of invitations per participant (number of columns in I)
    - h (float): honesty fraction; cn = round((1-h)*n) corrupted rows

    Returns:
    - I (ndarray): shape (n, d), random integers in [0, n)
    - corrupted (ndarray): 1D array of length cn, unique indices in [0, n)
    - c_true (ndarray): length n, counts of each value in I
    - k (ndarray): length n, counts of each value in I restricted to corrupted rows
    - c_honest_true (ndarray): length n, c_true - k
    """
    rng = np.random.default_rng()
    # number of corrupted participants
    cn = int(round((1 - h) * n))

    # 1) Invitation matrix: shape (n, d)
    I = rng.integers(low=0, high=n, size=(n, d))

    # 2) Randomly choose corrupted participant indices
    corrupted = rng.choice(n, size=cn, replace=False)

    # 3) True counts across all invitations
    c_true = np.bincount(I.ravel(), minlength=n)

    # 4) Counts for corrupted rows only
    k = np.bincount(I[corrupted].ravel(), minlength=n)

    # 5) Honest counts = total minus corrupted
    c_honest_true = c_true - k

    return I, corrupted, c_true, k, c_honest_true


def generate_data_cupy(n, d, h):
    """
    Generate synthetic data for the dialing protocol model.

    Parameters:
    - n (int): number of participants (also number of rows in I)
    - d (int): number of invitations per participant (number of columns in I)
    - h (float): honesty fraction; cn = round((1-h)*n) corrupted rows

    Returns:
    - I (ndarray): shape (n, d), random integers in [0, n)
    - corrupted (ndarray): 1D array of length cn, unique indices in [0, n)
    - c_true (ndarray): length n, counts of each value in I
    - k (ndarray): length n, counts of each value in I restricted to corrupted rows
    - c_honest_true (ndarray): length n, c_true - k
    """
    rng = cp.random.default_rng()
    # number of corrupted participants
    cn = int(round((1 - h) * n))

    # 1) Invitation matrix: shape (n, d)
    I = rng.integers(low=0, high=n, size=(n, d))

    # 2) Randomly choose corrupted participant indices
    corrupted = cp.random.choice(n, size=cn, replace=False)

    # 3) True counts across all invitations
    c_true = cp.bincount(I.ravel(), minlength=n)

    # 4) Counts for corrupted rows only
    k = cp.bincount(I[corrupted].ravel(), minlength=n)

    # 5) Honest counts = total minus corrupted
    c_honest_true = c_true - k

    return I, corrupted, c_true, k, c_honest_true



def adversary_estimation(n, d, A, B, C, k, R):
    A_ = np.array(A)
    B_ = np.array(B)

    if np.any((A_ == 0) & (B_ == 0)):
        return adversary_estimate.adversary_estimate(n=n, d=d, A=A, B=B, C=C, k=k, R=R)
    else:
        return adversary_accurate.adversary_accurate(n=n, d=d, A=A, B=B, k=k, R=R)



def adversary_estimation_numpy(n, d, A, B, C, k, R):
    A_ = np.array(A)
    B_ = np.array(B)

    if np.any((A_ == 0) & (B_ == 0)):
        return adversary_estimate.adversary_estimate_numpy(n=n, d=d, A=A, B=B, C=C, k=k, R=R)
    else:
        return adversary_accurate.adversary_accurate_numpy(n=n, d=d, A=A, B=B, k=k, R=R)

def adversary_estimation_cupy(n, d, A, B, C, k, R):
    A_ = cp.array(A)
    B_ = cp.array(B)

    if np.any((A_ == 0) & (B_ == 0)):
        return adversary_estimate.adversary_estimate_cupy(n=n, d=d, A=A, B=B, C=C, k=k, R=R)
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

def l1_accurate_numpy(c_true: np.ndarray, c_hat: np.ndarray) -> float:
    """
    Compute the L1‐based accuracy metric:
        1 − (‖c_true − c_hat‖₁ / ‖c_true‖₁)

    Parameters
    ----------
    c_true : np.ndarray, shape (n,)
        True counts vector.
    c_hat : np.ndarray, shape (n,)
        Estimated counts vector.

    Returns
    -------
    error : float
        The L1 accuracy in [−∞, 1], higher is better.

    Raises
    ------
    ValueError
        If the two inputs have mismatched shapes or if sum(c_true) == 0.
    """
    c_true = np.asarray(c_true, dtype=float)
    c_hat  = np.asarray(c_hat,  dtype=float)

    if c_true.shape != c_hat.shape:
        raise ValueError("c_true and c_hat must have the same shape")

    total_diff = np.abs(c_true - c_hat).sum()
    norm       = c_true.sum()

    if norm == 0:
        raise ValueError("Sum of c_true is zero; cannot normalize.")

    return 1.0 - (total_diff / norm)


def l1_accurate_cupy(c_true: np.ndarray, c_hat: np.ndarray) -> float:
    """
    Compute the L1‐based accuracy metric:
        1 − (‖c_true − c_hat‖₁ / ‖c_true‖₁)

    Parameters
    ----------
    c_true : np.ndarray, shape (n,)
        True counts vector.
    c_hat : np.ndarray, shape (n,)
        Estimated counts vector.

    Returns
    -------
    error : float
        The L1 accuracy in [−∞, 1], higher is better.

    Raises
    ------
    ValueError
        If the two inputs have mismatched shapes or if sum(c_true) == 0.
    """
    c_true = cp.asarray(c_true, dtype=float)
    c_hat  = cp.asarray(c_hat,  dtype=float)

    if c_true.shape != c_hat.shape:
        raise ValueError("c_true and c_hat must have the same shape")

    total_diff = cp.abs(c_true - c_hat).sum()
    norm       = c_true.sum()

    if norm == 0:
        raise ValueError("Sum of c_true is zero; cannot normalize.")

    return 1.0 - (total_diff / norm)



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


def adversary_view_fast(I, adversary_controls, times):
    n = len(I)  # Number of participants
    d = len(I[0])  # Number of invitations sent by each participant

    # Initialize statistical tables A, B, and C, all as lists of length n
    A = [0] * n
    B = [0] * n
    C = [0] * n

    # Run the protocol for multiple rounds and collect statistics for A, B, and C
    for _ in range(times):
        O = dialing.assign_invitations_fast(I)  # O is an n*d matrix of Invitation objects
        # Temporary statistics for this round: temp[i] records the number of times
        # the invitation target was replaced with P_{i+1} in adversary-controlled rows
        temp = [0] * n
        for r in adversary_controls:
            for j in range(d):
                receiver = O[r][j]
                if O[r][j] == I[r][j]:
                    A[receiver] += 1
                else:
                    B[receiver] += 1
                    temp[receiver] += 1
        # Update table C: each participant takes the maximum replacement count across all rounds
        for i in range(n):
            C[i] = max(C[i], temp[i])

    return A, B, C

def adversary_view_fast_numpy(I: np.ndarray,
                              adversary_controls: np.ndarray,
                              times: int,
                              seed: int = None):
    """
    使用 cap_and_fill_matrix 生成 O，并统计 A, B, C。

    Parameters
    ----------
    I : np.ndarray, shape (n, d)
        输入的邀请矩阵，每个元素是接收者的索引（0..n-1）。
    adversary_controls : np.ndarray, shape (cn,)
        被破坏方的行索引数组。
    times : int
        实验轮数。
    seed : int, optional
        随机种子，同传给 cap_and_fill_matrix。

    Returns
    -------
    A : np.ndarray, shape (n,)
        统计在破坏方行上，“正确保留”的接收次数。
    B : np.ndarray, shape (n,)
        统计在破坏方行上，“被替换”的接收次数。
    C : np.ndarray, shape (n,)
        每轮内“被替换”次数的最大值（跨轮最大）。
    """
    rng = np.random.default_rng(seed)
    n, d = I.shape
    A = np.zeros(n, dtype=int)
    B = np.zeros(n, dtype=int)
    C = np.zeros(n, dtype=int)

    for _ in range(times):
        # 生成本轮输出
        O = dialing.assign_invitations_numpy(I, seed=rng.integers(0, 1<<32))

        # 取出被控行
        O_sub = O[adversary_controls]
        I_sub = I[adversary_controls]

        # 区分 match/mismatch
        match_mask = (O_sub == I_sub)
        mismatch_mask = ~match_mask

        # 累加 A
        matched = O_sub[match_mask]
        if matched.size:
            A += np.bincount(matched, minlength=n)

        # 累加 B 和临时替换计数 temp
        mismatched = O_sub[mismatch_mask]
        if mismatched.size:
            counts_mis = np.bincount(mismatched, minlength=n)
            B += counts_mis
            temp = counts_mis
        else:
            temp = np.zeros(n, dtype=int)

        # 更新 C
        C = np.maximum(C, temp)

    return A, B, C


def adversary_view_fast_cupy(I: cp.ndarray,
                              adversary_controls: cp.ndarray,
                              times: int,
                              seed: int = None):
    """
    使用 cap_and_fill_matrix 生成 O，并统计 A, B, C。

    Parameters
    ----------
    I : np.ndarray, shape (n, d)
        输入的邀请矩阵，每个元素是接收者的索引（0..n-1）。
    adversary_controls : np.ndarray, shape (cn,)
        被破坏方的行索引数组。
    times : int
        实验轮数。
    seed : int, optional
        随机种子，同传给 cap_and_fill_matrix。

    Returns
    -------
    A : np.ndarray, shape (n,)
        统计在破坏方行上，“正确保留”的接收次数。
    B : np.ndarray, shape (n,)
        统计在破坏方行上，“被替换”的接收次数。
    C : np.ndarray, shape (n,)
        每轮内“被替换”次数的最大值（跨轮最大）。
    """
    rng = cp.random.default_rng()
    n, d = I.shape
    A = cp.zeros(n, dtype=int)
    B = cp.zeros(n, dtype=int)
    C = cp.zeros(n, dtype=int)

    for _ in range(times):
        # 生成本轮输出
        O = dialing.assign_invitations_cupy(I, seed=rng.integers(0, 1<<32))

        # 取出被控行
        O_sub = O[adversary_controls]
        I_sub = I[adversary_controls]

        # 区分 match/mismatch
        match_mask = (O_sub == I_sub)
        mismatch_mask = ~match_mask

        # 累加 A
        matched = O_sub[match_mask]
        if matched.size:
            A += cp.bincount(matched, minlength=n)

        # 累加 B 和临时替换计数 temp
        mismatched = O_sub[mismatch_mask]
        if mismatched.size:
            counts_mis = cp.bincount(mismatched, minlength=n)
            B += counts_mis
            temp = counts_mis
        else:
            temp = cp.zeros(n, dtype=int)

        # 更新 C
        C = cp.maximum(C, temp)

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
        A_new, B_new, C_new = adversary_view_fast(I=I, adversary_controls=adversary_controls, times=times)

        A = [a + an for a, an in zip(A, A_new)]
        B = [b + bn for b, bn in zip(B, B_new)]
        C = [max(c_old, c_new) for c_old, c_new in zip(C, C_new)]

        # 调用估计算法
        c_est, c_honest = adversary_estimation_numpy(n=n, d=d, A=A, B=B, C=C, k=k, R=R)



        acc = l1_accurate(c_honest_true, c_honest)
        print("Round: ", R, "acc: ", acc)
        record.append(acc)
        prev_R = R
    return record


def experiment_for_single_input_numpy(I: np.ndarray,
                                      adversary_controls: np.ndarray,
                                      k: np.ndarray,
                                      c_honest_true: np.ndarray,
                                      R_list: list,
                                      seed: int = None) -> np.ndarray:
    """
    Run the adversary experiment for a single input scenario using NumPy.

    Parameters
    ----------
    I : np.ndarray, shape (n, d)
        Invitation matrix.
    adversary_controls : np.ndarray, shape (cn,)
        Indices of corrupted rows.
    k : np.ndarray, shape (n,)
        True counts from corrupted rows.
    c_honest_true : np.ndarray, shape (n,)
        True honest counts.
    R_list : list of int
        Round thresholds at which to estimate.
    seed : int, optional
        RNG seed.

    Returns
    -------
    record : np.ndarray
        L1 accuracy at each R in R_list.
    """
    n, d = I.shape
    record = []
    prev_R = 0

    # Initialize statistics arrays
    A = np.zeros(n, dtype=int)
    B = np.zeros(n, dtype=int)
    C = np.zeros(n, dtype=int)

    rng = np.random.default_rng(seed)

    for R in R_list:
        times = R - prev_R

        # Get new stats for this batch of rounds
        A_new, B_new, C_new = adversary_view_fast_numpy(
            I=I,
            adversary_controls=adversary_controls,
            times=times,
            seed=rng.integers(0, 1 << 32)
        )

        # Update cumulative stats
        A += A_new
        B += B_new
        C = np.maximum(C, C_new)

        # Perform estimation (expects numpy arrays)
        c_est, c_honest = adversary_estimation_numpy(
            n=n, d=d, A=A, B=B, C=C, k=k, R=R
        )

        # Compute L1 accuracy
        acc = l1_accurate_numpy(c_honest_true, c_honest)
        print(f"Round: {R}, acc: {acc}")

        record.append(acc)
        prev_R = R

    return np.array(record)



def experiment_for_single_input_cupy(I: cp.ndarray,
                                      adversary_controls: cp.ndarray,
                                      k: cp.ndarray,
                                      c_honest_true: cp.ndarray,
                                      R_list: list,
                                      seed: int = None) -> cp.ndarray:
    """
    Run the adversary experiment for a single input scenario using NumPy.

    Parameters
    ----------
    I : np.ndarray, shape (n, d)
        Invitation matrix.
    adversary_controls : np.ndarray, shape (cn,)
        Indices of corrupted rows.
    k : np.ndarray, shape (n,)
        True counts from corrupted rows.
    c_honest_true : np.ndarray, shape (n,)
        True honest counts.
    R_list : list of int
        Round thresholds at which to estimate.
    seed : int, optional
        RNG seed.

    Returns
    -------
    record : np.ndarray
        L1 accuracy at each R in R_list.
    """
    n, d = I.shape
    record = []
    prev_R = 0

    # Initialize statistics arrays
    A = cp.zeros(n, dtype=int)
    B = cp.zeros(n, dtype=int)
    C = cp.zeros(n, dtype=int)

    rng = cp.random.default_rng(seed)

    for R in R_list:
        times = R - prev_R

        # Get new stats for this batch of rounds
        A_new, B_new, C_new = adversary_view_fast_cupy(
            I=I,
            adversary_controls=adversary_controls,
            times=times,
            seed=rng.integers(0, 1 << 32)
        )

        # Update cumulative stats
        A += A_new
        B += B_new
        C = cp.maximum(C, C_new)

        # Perform estimation (expects numpy arrays)
        c_est, c_honest = adversary_estimation_cupy(
            n=n, d=d, A=A, B=B, C=C, k=k, R=R
        )

        # Compute L1 accuracy
        acc = l1_accurate_cupy(c_honest_true, c_honest)
        print(f"Round: {R}, acc: {acc}")

        record.append(acc)
        prev_R = R

    return np.array(record)

def experiment_for_multiple_inputs(n,d,h, R_list, runs=50):
    all_records = []

    for run in range(runs):
        print("run: ", run)
        print("n: ", n, "d: ", d, "h: ", h)
        I, corrupted, c_true, k, c_honest_true = generate_data_fast(n, d, h)
        record = experiment_for_single_input(I, adversary_controls=corrupted, k=k, c_honest_true=c_honest_true,
                                             R_list=R_list)
        all_records.append(record)

    avg_record = []
    for idx in range(len(R_list)):
        mean_val = sum(rec[idx] for rec in all_records) / runs
        avg_record.append(mean_val)

    os.makedirs('result', exist_ok=True)
    filename = f"result/dresult_n{n}_d{d}_h{h}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['R', 'accuracy'])
        for R, acc in zip(R_list, avg_record):
            writer.writerow([R, acc])

    return avg_record


def experiment_for_multiple_inputs_numpy(n,d,h, R_list, runs=50):
    all_records = []

    for run in range(runs):
        print("run: ", run)
        print("n: ", n, "d: ", d, "h: ", h)
        I, corrupted, c_true, k, c_honest_true = generate_data_numpy(n, d, h)
        record = experiment_for_single_input_numpy(I, adversary_controls=corrupted, k=k, c_honest_true=c_honest_true,
                                             R_list=R_list)
        all_records.append(record)

    avg_record = []
    for idx in range(len(R_list)):
        mean_val = sum(rec[idx] for rec in all_records) / runs
        avg_record.append(mean_val)

    os.makedirs('result', exist_ok=True)
    filename = f"result/dialingresult_n{n}_d{d}_h{h}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['R', 'accuracy'])
        for R, acc in zip(R_list, avg_record):
            writer.writerow([R, acc])

    return avg_record



def experiment_for_multiple_inputs_cupy(n,d,h, R_list, runs=50):
    all_records = []

    for run in range(runs):
        print("run: ", run)
        print("n: ", n, "d: ", d, "h: ", h)
        I, corrupted, c_true, k, c_honest_true = generate_data_cupy(n, d, h)
        record = experiment_for_single_input_cupy(I, adversary_controls=corrupted, k=k, c_honest_true=c_honest_true,
                                             R_list=R_list)
        all_records.append(record)

    avg_record = []
    for idx in range(len(R_list)):
        mean_val = sum(rec[idx] for rec in all_records) / runs
        avg_record.append(mean_val)

    # os.makedirs('result', exist_ok=True)
    # filename = f"result/dresult_n{n}_d{d}_h{h}.csv"
    # with open(filename, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(['R', 'accuracy'])
    #     for R, acc in zip(R_list, avg_record):
    #         writer.writerow([R, acc])

    return avg_record


def run_experiments(n, R_list, runs=30):
    """
    先按 d 从小到大依次运行，再对每个 d 里所有支持该 d 的 h 运行实验。
    不打印、不存储，只依赖 experiment_for_multiple_inputs 自身的 CSV 输出。
    """
    param_grid = {
        0.7: [1,2,5,10,20],
        0.9:  [1,2,5,10,20],
        0.97: [1,2,5,10,20],
    }

    all_ds = sorted({d for d_list in param_grid.values() for d in d_list})

    for d in all_ds:
        for h, d_list in param_grid.items():
            if d in d_list:
                experiment_for_multiple_inputs_numpy(
                    n=n,
                    d=d,
                    h=h,
                    R_list=R_list,
                    runs=runs
                )





if __name__ == "__main__":
    n = 50000
    R_list = [1,5,10,20,50,75,100]
    # print(experiment_for_multiple_inputs_numpy(n, d, h, R_list, runs=10))
    run_experiments(n, R_list, runs=10)






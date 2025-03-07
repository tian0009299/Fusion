import adversary_accurate, adversary_estimate
import dialing
import numpy as np
import random


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


def experiment():
    R = 10000
    n = 50
    d = 100
    cn = 2
    I, adversary_controls, c_true, k, c_honest_true = generate_data(n=n, d=d, cn=cn)

    A, B, C = adversary_view(I=I, adversary_controls=adversary_controls, R=R)

    c_est, c_honest = adversary_estimation(n=n, d=d, A=A, B=B, C=C, k=k, R=R)

    print("c_true: ", c_honest_true)
    print("c_est: ", c_honest)
    print(mean_absolute_error(c_honest, c_honest_true))
    print(calculate_accuracy(c_honest, c_honest_true))
    print(calculate_similarity(c_honest, c_honest_true, threshold=1, percentage=0.2))


if __name__ == "__main__":
    experiment()

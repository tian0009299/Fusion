import random
import numpy as np
import cupy as cp


class Invitation:
    def __init__(self, sender, orig_row, orig_col, intended_receiver):
        self.sender = sender  # The user who sends the invitation (numbering starts from 1)
        self.orig_row = orig_row  # The row index in the input matrix I (0-indexed, for tracking)
        self.orig_col = orig_col  # The column index in the input matrix I (0-indexed, for tracking)
        self.intended_receiver = intended_receiver  # The intended receiver of the invitation (numbering starts from 1)
        self.final_receiver = None  # The final receiver of the invitation (to be determined)

    def __repr__(self):
        # Display format: Inv(Px->Py) indicates an invitation from P_x to P_y
        return f"Inv(P{self.sender}->P{self.final_receiver})"


def assign_invitations(I):
    """
    Input:
        I: An n*d matrix (list of lists), where each element is a number representing that
           the invitation is sent to P_i (participant numbering starts from 1)
    Output:
        O: An n*d matrix, where O[i][j] is an Invitation object representing the invitation
           that P_(i+1) ultimately receives.
    """
    n = len(I)
    if n == 0:
        return []
    d = len(I[0])

    # Create all Invitation objects and record their original positions in the input matrix.
    # Note: sender numbering starts from 1.
    invitations = []
    for i in range(n):
        for j in range(d):
            inv = Invitation(sender=i + 1, orig_row=i, orig_col=j, intended_receiver=I[i][j])
            invitations.append(inv)

    # Group invitations by intended_receiver; construct a dictionary for participants numbered 1 to n.
    received_by = {i: [] for i in range(1, n + 1)}

    for inv in invitations:
        received_by[inv.intended_receiver].append(inv)

    # For each participant, if the number of received invitations is not more than d, accept all.
    # Otherwise, randomly choose d invitations to accept; extra invitations are stored for redistribution.
    accepted_by = {i: [] for i in range(1, n + 1)}
    extra_invitations = []
    for user in range(1, n + 1):
        inv_list = received_by[user]
        if len(inv_list) <= d:
            for inv in inv_list:
                inv.final_receiver = user
            accepted_by[user] = inv_list
        else:
            accepted = random.sample(inv_list, d)
            for inv in accepted:
                inv.final_receiver = user
            accepted_by[user] = accepted
            for inv in inv_list:
                if inv not in accepted:
                    extra_invitations.append(inv)

    # Identify participants who received fewer than d invitations and record how many they are missing.
    deficiency = {user: d - len(accepted_by[user]) for user in range(1, n + 1) if len(accepted_by[user]) < d}

    # Randomly redistribute extra invitations to participants with a deficiency.
    while extra_invitations and deficiency:
        inv = extra_invitations.pop()
        user = random.choice(list(deficiency.keys()))
        accepted_by[user].append(inv)
        inv.final_receiver = user
        deficiency[user] -= 1
        if deficiency[user] == 0:
            del deficiency[user]

    # Construct the output matrix O using each invitation's orig_row and orig_col.
    O = [[None for _ in range(d)] for _ in range(n)]
    for inv in invitations:
        O[inv.orig_row][inv.orig_col] = inv

    return O


def assign_invitations_fast(I):
    """
    Input:
        I: An n×d list of lists, where each element is an int in [0..n-1]
           indicating the intended receiver (0-based).
    Output:
        O: An n×d list of lists, where each element is the final receiver (0-based).
    """
    n = len(I)
    if n == 0:
        return []
    d = len(I[0])

    # 1) Group all invitation positions (i,j) by their intended receiver
    received_by = [[] for _ in range(n)]
    for i in range(n):
        for j in range(d):
            user = I[i][j]
            received_by[user].append((i, j))

    # 2) Prepare output matrix and helper structures
    O = [[None] * d for _ in range(n)]
    accepted_by = [[] for _ in range(n)]
    extra_invitations = []

    # 3) For each user, accept up to d invitations; surplus go to extra_invitations
    for user in range(n):
        inv_list = received_by[user]
        if len(inv_list) <= d:
            # Accept all if count ≤ d
            for (i, j) in inv_list:
                O[i][j] = user
                accepted_by[user].append((i, j))
        else:
            # Otherwise randomly pick d to accept
            accepted = random.sample(inv_list, d)
            for (i, j) in accepted:
                O[i][j] = user
                accepted_by[user].append((i, j))
            # The rest become extra
            for (i, j) in inv_list:
                if (i, j) not in accepted:
                    extra_invitations.append((i, j))

    # 4) Compute how many more each user needs
    deficiency = [d - len(accepted_by[user]) for user in range(n)]

    # 5) Randomly redistribute extras to users with remaining deficiency
    while extra_invitations:
        candidates = [u for u in range(n) if deficiency[u] > 0]
        if not candidates:
            # Should not happen if total extras == total deficiency
            break
        i, j = extra_invitations.pop()
        user = random.choice(candidates)
        O[i][j] = user
        accepted_by[user].append((i, j))
        deficiency[user] -= 1

    # 6) Verify every slot got assigned
    for i in range(n):
        for j in range(d):
            if O[i][j] is None:
                raise ValueError(f"Position {(i, j)} was never assigned")

    return O

def assign_invitations_numpy(I: np.ndarray, seed: int = None) -> np.ndarray:
    """
    对输入的 n×d 矩阵 I，按以下规则生成同形状的 O：
      1. 初始化 O 全为 -1。
      2. 统计每个 k∈{0,…,n-1} 在 I 中出现的次数 count[k]。
      3. 对于 count[k] <= d，保留 I 中所有值为 k 的位置；对于 count[k] > d，随机挑 d 个位置保留。
      4. 对于 count[k] < d，最后再从剩余的 “-1” 位置里随机补齐到恰好 d 个 k。
    :param I: 整数矩阵，shape=(n,d)，元素 ∈ [0, n-1]
    :param seed: 随机种子（可选）
    :return: 整数矩阵 O，shape=(n,d)，元素 ∈ {−1, 0,…,n−1}
    """
    rng = np.random.default_rng(seed)
    n, d = I.shape

    # 1) 初始化
    O = -np.ones_like(I, dtype=int)

    # 2) 统计出现次数
    counts = np.bincount(I.ravel(), minlength=n)

    # 3) 按上限 d 先赋值
    for k in range(n):
        # 找到所有 I == k 的坐标
        idx = np.argwhere(I == k)  # shape=(counts[k], 2)
        c = counts[k]
        if c <= d:
            # 全部保留
            O[I == k] = k
        else:
            # 随机挑 d 个
            chosen = idx[rng.choice(c, size=d, replace=False)]
            O[chosen[:,0], chosen[:,1]] = k

    # 4) 对 count[k] < d 的 k，从剩余空位中补齐
    deficits = np.maximum(d - counts, 0)       # 每个 k 还需要补的数量
    free_slots = np.argwhere(O == -1)          # 所有还未赋值的位置
    perm = rng.permutation(len(free_slots))

    offset = 0
    for k in range(n):
        need = deficits[k]
        if need > 0:
            sel = free_slots[perm[offset : offset + need]]
            O[sel[:,0], sel[:,1]] = k
            offset += need

    return O

def assign_invitations_cupy(I: cp.ndarray, seed: int = None) -> np.ndarray:
    """
    对输入的 n×d 矩阵 I，按以下规则生成同形状的 O：
      1. 初始化 O 全为 -1。
      2. 统计每个 k∈{0,…,n-1} 在 I 中出现的次数 count[k]。
      3. 对于 count[k] <= d，保留 I 中所有值为 k 的位置；对于 count[k] > d，随机挑 d 个位置保留。
      4. 对于 count[k] < d，最后再从剩余的 “-1” 位置里随机补齐到恰好 d 个 k。
    :param I: 整数矩阵，shape=(n,d)，元素 ∈ [0, n-1]
    :param seed: 随机种子（可选）
    :return: 整数矩阵 O，shape=(n,d)，元素 ∈ {−1, 0,…,n−1}
    """

    n, d = I.shape

    # 1) 初始化
    O = -cp.ones_like(I, dtype=int)

    # 2) 统计出现次数
    counts = cp.bincount(I.ravel(), minlength=n)

    # 3) 按上限 d 先赋值
    for k in range(n):
        idx = cp.argwhere(I == k)  # shape=(counts[k], 2)
        c = counts[k]
        if c <= d:
            O[I == k] = k
        else:
            c_py = int(c)
            perm = cp.random.permutation(c_py)
            sel = perm[:d]
            chosen = idx[sel]
            O[chosen[:, 0], chosen[:, 1]] = k

    # 4) 对 count[k] < d 的 k，从剩余空位中补齐
    deficits = cp.maximum(d - counts, 0)  # cupy array
    free_slots = cp.argwhere(O == -1)  # shape=(num_free, 2)

    # 改这里：不要用 rng.permutation，转为 Python int 后用全局 permutation
    num_free = int(free_slots.shape[0])
    perm = cp.random.permutation(num_free)

    offset = 0
    for k in range(n):
        need = int(deficits[k])  # 把 cupy scalar 转成 Python int
        if need > 0:
            sel = free_slots[perm[offset: offset + need]]
            O[sel[:, 0], sel[:, 1]] = k
            offset += need

    return O


def convert_output_matrix(O):
    """
    Convert the output matrix O (a matrix of Invitation objects) into a numeric matrix.
    Each element in the resulting matrix is the final receiver's number.
    """
    numeric_matrix = []
    for row in O:
        numeric_row = [inv.final_receiver for inv in row]
        numeric_matrix.append(numeric_row)
    return numeric_matrix


def DP_modify_input(I, cn, p):
    """
    For each row in I whose 1-indexed index is not in cn,
    each of its d elements is replaced with a random integer
    between 1 and n (inclusive) with probability p.

    Parameters:
        I (list of list of int): An n*d list, where n is the number of rows and each row has d integers.
        cn (list of int): A list of 1-indexed integers (can be empty) indicating rows to be excluded.
        p (float): The probability with which each element is replaced.

    Returns:
        list of list of int: The modified list with replacements.
    """
    n = len(I)  # n is the number of rows in I
    modified_I = []

    # Iterate over each row in I using 0-indexing
    for idx, row in enumerate(I):
        # Check if the row's 1-indexed index is in cn
        if (idx + 1) in cn:
            # If yes, do not modify this row
            modified_I.append(row.copy())
        else:
            # Otherwise, process each element in the row
            new_row = []
            for element in row:
                # With probability p, replace the element with a random integer in [1, n]
                if random.random() < p:
                    new_row.append(random.randint(1, n))
                else:
                    new_row.append(element)
            modified_I.append(new_row)

    return modified_I


if __name__ == "__main__":
    # Construct an example 4x3 input matrix,
    # where each number represents the invitation target.
    # For example, 2 means the invitation is sent to P2.
    I = [
        [1, 2, 3],
        [4, 5, 1],
        [1, 1, 1],
        [3, 3, 3],
        [5,5,2]
    ]

    for i in range(5):
        for j in range(3):
            I[i][j] -= 1
    print(I)
    print(assign_invitations_fast(I))



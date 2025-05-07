def compute_nn(I, corrupted):
    """
    Compute the nn values based on the invitation matrix I and the list of corrupted parties.

    The steps are as follows:
    1. Exclude the rows corresponding to corrupted parties from I.
    2. For the remaining (honest) rows, count the number of times each party number (1-indexed) appears.
       Let c_i be the count for party i.
    3. For each party i (from 1 to n):
         - If c_i < d, then set nn[i] = d - c_i.
         - If c_i >= d, then set nn[i] = 0.

    Parameters:
      I         : A matrix (list of lists) where I[i][j] is the party number that party (i+1) invites.
      corrupted : A list of corrupted party numbers (1-indexed).

    Returns:
      nn        : A list where nn[i] is the computed value for party (i+1).
    """
    n = len(I)
    if n == 0:
        return []
    d = len(I[0])  # number of invitations per party

    # Initialize counts for each party (using 0-indexed list for parties 1 to n)
    counts = [0] * n

    # Process only honest parties (i.e. parties not in the corrupted list)
    for i in range(n):
        if (i + 1) in corrupted:
            continue  # skip corrupted parties
        for invitation in I[i]:
            # invitation is a 1-indexed party number, convert to 0-indexed for counts list
            counts[invitation - 1] += 1

    # Compute nn values based on counts
    nn = []
    for i in range(n):
        if counts[i] < d:
            nn.append(d - counts[i])
        else:
            nn.append(0)

    return nn

I = [[1, 1, 1], [4, 4, 5], [3, 3, 4], [2, 4, 3], [1, 1, 1]]
corrupted = [5,1]

print(compute_nn(I, corrupted))

import experiment, dialing
import copy


def find_a_party(I, corrupted):
    # Call generate_data to obtain the data matrix I and the list of corrupted parties (plus other variables which we don't need)
    n = len(I)
    d = len(I[0])

    # Iterate candidate party numbers from 1 to n
    while True:
        for candidate in range(1, n + 1):
            # For each corrupted party (given as 1-indexed), adjust their invitations to the current candidate
            for party in corrupted:
                index = party - 1  # convert 1-indexed party number to 0-indexed row index
                I[index] = [candidate] * d

            # Process the modified matrix I with dialing.assign_invitations to obtain the output matrix O
            O = dialing.assign_invitations(I)

            # Check the corrupted parties' portion; if any invitation changes, return the first changed O[i][j]
            for party in corrupted:
                index = party - 1
                for j in range(d):
                    if O[index][j].final_receiver != I[index][j]:
                        return O[index][j].final_receiver


def estimate_NO(I, corrupted, t, R):
    """
    Estimate the number of times each party appears in the NO set.

    Parameters:
      I         : the original invitation matrix (size n x d)
      corrupted : list of corrupted party numbers (1-indexed)
      t         : a value returned by find_a_party
      R         : a parameter used in the equations

    Returns:
      nn: a list where nn[i] represents the number of times party (i+1) appears in the NO set.
    """
    # Determine d (number of invitations per party) and n (total number of parties)
    n = len(I)
    d = len(I[0])
    cn = len(corrupted)
    print("corrupted: ",corrupted)
    print("I: ", I)
    print("t: ", t)

    # -----------------------------
    # Step 1: Create I1 by setting all corrupted parties' inputs to t.
    # -----------------------------
    I1 = copy.deepcopy(I)
    for party in corrupted:
        idx = party - 1  # convert party number (1-indexed) to row index (0-indexed)
        I1[idx] = [t] * d
    print("I1: ", I1)
    # Run adversary_view on I1; it returns three lists A, B, and C (C is not used)
    A1, B1, _ = experiment.adversary_view(I1, adversary_controls=corrupted, R=R)
    print("A1: ", A1)
    print("B1: ", B1)

    # -----------------------------
    # Step 2: Select a candidate a (different from t) such that B1[candidate-1] != 0.
    #         Here we iterate candidate party numbers from 1 to n.
    # -----------------------------
    a_candidate = None
    for candidate in range(1, n + 1):
        if candidate != t and B1[candidate - 1] != 0:
            a_candidate = candidate
            break
    if a_candidate is None:
        raise ValueError("No valid candidate found in B1 for 'a' (candidate != t with nonzero value).")

    print("a: ", a_candidate)
    # -----------------------------
    # Step 3: Create I2 by setting all corrupted parties' inputs to a_candidate.
    # -----------------------------
    I2 = copy.deepcopy(I)
    for party in corrupted:
        idx = party - 1
        I2[idx] = [a_candidate] * d
    print("I2: ", I2)

    # Run adversary_view on I2; we get new lists A2, B2, and C2 (C2 is ignored)
    A2, B2, _ = experiment.adversary_view(I2, adversary_controls=corrupted, R=R)

    print("A2: ", A2)
    print("B2: ", B2)

    # -----------------------------
    # Step 4: For every index (except for those corresponding to party t and party a_candidate),
    #         ensure that both B1 and B2 have nonzero values by re-running adversary_view as needed.
    #         We accumulate (add) the returned values until both are nonzero.
    # -----------------------------
    for b in range(n):
        # Skip the indices corresponding to party t and candidate a_candidate
        if b == (t - 1) or b == (a_candidate - 1):
            continue
        # Continue to accumulate until both B1[b] and B2[b] are nonzero.
        while (B1[b] == 0 and B2[b] != 0) or (B1[b] != 0 and B2[b] == 0):
            # If B1[b] is zero but B2[b] is nonzero, run adversary_view on I1 and add the new value.
            if B1[b] == 0 and B2[b] != 0:
                _, new_B, _ = experiment.adversary_view(I1, adversary_controls=corrupted, R=R)
                B1[b] += new_B[b]
            # If B2[b] is zero but B1[b] is nonzero, run adversary_view on I2 and add the new value.
            if B2[b] == 0 and B1[b] != 0:
                _, new_B2, _ = experiment.adversary_view(I2, adversary_controls=corrupted, R=R)
                B2[b] += new_B2[b]


    # -----------------------------
    # Step 5: Compute nn[t-1] and nn[a_candidate-1] based on the equations:
    #         A[t-1] = (R * d * d) / (2*d - x)  ==>  x = 2*d - (R*d*d) / A[t-1]
    #         and similarly for A[a_candidate-1].
    # -----------------------------
    nn = [0] * n  # initialize nn for all parties

    if A1[t - 1] == 0:
        raise ValueError("A1[t-1] is zero, cannot compute nn[t-1].")
    nn[t - 1] = round((cn + 1) * d - (R * d * d * cn) / A1[t - 1])

    if A2[a_candidate - 1] == 0:
        raise ValueError("A2[a_candidate-1] is zero, cannot compute nn[a_candidate-1].")
    nn[a_candidate - 1] = round((cn+1) * d - (R * d * d * cn) / A2[a_candidate - 1])

    # -----------------------------
    # Step 6: For all other indices (call these indices U), compute nn[i] based on proportionality.
    #         From adversary_view on I1, we have:
    #             nn[i] : nn[t-1] = B1[i] : B1[t-1]   =>   nn[i] = (B1[i] / B1[t-1]) * nn[t-1]
    #         And from adversary_view on I2:
    #             nn[i] : nn[a_candidate-1] = B2[i] : B2[a_candidate-1]   =>   nn[i] = (B2[i] / B2[a_candidate-1]) * nn[a_candidate-1]
    #         We combine these two estimates (here, by averaging).
    #         If both B1[i] and B2[i] are zero, then we set nn[i] = 0.
    # -----------------------------
    for i in range(n):
        if i == (t - 1) or i == (a_candidate - 1):
            continue
        # If both B1[i] and B2[i] are zero, then nn[i] remains 0.
        if B1[i] == 0 and B2[i] == 0:
            nn[i] = 0
        else:
            estimate1 = (B1[i] / B1[a_candidate - 1]) * nn[a_candidate - 1]
            estimate2 = (B2[i] / B2[t - 1]) * nn[t - 1]
            # Here we take the average of the two estimates.
            nn[i] = round((estimate1 + estimate2) / 2)

    return nn

def experiment_active(n, d, cn, R):
    I, corrupted, _, _, _ = experiment.generate_data(n, d, cn)
    nn_real = compute_nn(I=I,corrupted=corrupted)
    t = find_a_party(I=I, corrupted=corrupted)
    nn_estimate = estimate_NO(I, corrupted, t, R)
    acc = experiment.l1_accurate(nn_real,nn_estimate)
    print(nn_real)
    print(nn_estimate)
    print(acc)
    return acc

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


if __name__ == "__main__":
    experiment_active(n=5,d=100,cn=2,R=1000)
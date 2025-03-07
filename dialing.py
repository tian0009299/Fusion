import random


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


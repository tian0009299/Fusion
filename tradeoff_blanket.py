import numpy as np
import matplotlib.pyplot as plt

# Fixed parameters
N = 10
delta = 0.0001
log_term = np.log(2 / delta)

# Range of p values
p_vals = np.linspace(0.01, 1.0, 300)

# Different hd values to compare
hd_vals = [100,200, 500]

# Create plot
plt.figure(figsize=(10, 6))

for hd in hd_vals:
    hNd_minus_1 = hd * N - 1
    C = p_vals * hNd_minus_1 / N
    eps1 = 27 / C
    eps2 = np.sqrt(14 * log_term / C)
    epsilon = np.maximum(eps1, eps2)

    # Only keep epsilon < 1
    mask = epsilon < 1
    if np.any(mask):  # Avoid plotting empty lines
        plt.plot(p_vals[mask], epsilon[mask], label=f'hd = {hd}')

# Labels and title
plt.xlabel("p")
plt.ylabel("ε (epsilon)")
plt.title("Lower bound on ε vs. p (ε < 1) for different hd values (N=10000, δ=0.0001)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


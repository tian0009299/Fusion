import numpy as np
import matplotlib.pyplot as plt

def compute_epsilon_min(p, N, h, d, delta):
    """
    Compute the minimum epsilon required for (epsilon, delta)-DP
    given noise probability p.
    """
    A = 14 * N * np.log(2 / delta) / (h * N * d - 1)
    B = 27 * N / (h * N * d - 1)
    return np.maximum(np.sqrt(A / p), B / p)

def compute_alpha_min(p, d, beta):
    """
    Compute the minimum alpha required for correctness (confidence beta)
    given noise probability p.
    """
    return np.sqrt(2 * np.log(1 / (1 - beta)) / ((1 - p) * d))

# Example usage
if __name__ == "__main__":
    N = 1000
    h = 0.95
    d = 200
    delta = 0.01
    beta = 0.2

    p_values = np.linspace(0.4, 0.9, 200)
    epsilon_vals = compute_epsilon_min(p_values, N, h, d, delta)
    alpha_vals   = compute_alpha_min(p_values, d, beta)

    # Plotting trade-offs
    plt.figure()
    plt.plot(p_values, epsilon_vals, label='ε_min(p)')
    plt.xlabel('p')
    plt.ylabel('Minimum ε')
    plt.title('Minimum ε vs. p')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(p_values, alpha_vals, label='α_min(p)')
    plt.xlabel('p')
    plt.ylabel('Minimum α')
    plt.title(f'Minimum α vs. p (β={beta})')
    plt.grid(True)
    plt.legend()

    plt.show()
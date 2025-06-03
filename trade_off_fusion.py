import numpy as np
import matplotlib.pyplot as plt

# 固定参数
N = 10
delta = 0.0001
log_term = np.log(2 / delta)
p_vals = np.linspace(0.001, 1.0, 1000)
hd_vals = [100, 150,200, 300,500]

# 创建图像
plt.figure(figsize=(10, 6))

for hd in hd_vals:
    denom = p_vals * (hd * N - 1)

    # 第一项
    term1 = 2 * np.log(1 + N / denom + np.sqrt(3 * N * log_term / denom))

    # 第二项的合法判断和计算
    valid_mask = (2 * N * log_term / denom) < 1
    safe_sqrt = np.zeros_like(p_vals)
    safe_sqrt[valid_mask] = np.sqrt(2 * N * log_term / denom[valid_mask])
    term2 = np.full_like(p_vals, np.inf)
    term2[valid_mask] = -2 * np.log(1 - safe_sqrt[valid_mask])

    # 两项取最大值
    epsilon = np.maximum(term1, term2)

    # 改为画 e^ε
    exp_epsilon = np.exp(epsilon)

    # 只画出 epsilon < 1 的部分（即 e^epsilon < e^1 ≈ 2.718）
    mask = epsilon < np.log(5)
    if np.any(mask):
        plt.plot(p_vals[mask], exp_epsilon[mask], label=f'hd = {hd}')

# 图像标签与格式
plt.xlabel("p")
plt.ylabel("e^ε")
plt.title("Fusion Lower bound on e^ε vs. p (ε < 1) for different hd values (N=10, δ=0.0001)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 显示图像
plt.show()

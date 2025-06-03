import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_accuracy_for_params(params, d_list, result_dir='result'):
    """
    params: list of tuples (n, h)
    d_list: list of d values
    """
    num = len(params)
    fig, axs = plt.subplots(1, num, figsize=(5 * num, 4))

    for ax, (n, h) in zip(axs, params):
        for d in d_list:
            filename = os.path.join(result_dir, f"dialingresult_n{n}_d{d}_h{h}.csv")
            if not os.path.isfile(filename):
                print(f"Warning: file not found for n={n}, d={d}, h={h}: {filename}")
                continue
            df = pd.read_csv(filename)
            ax.plot(df['R'], df['accuracy'], marker='o', label=f"d = {d}")

        ax.set_xlabel("Number of rounds (R)")
        ax.set_ylabel("accuracy")
        ax.set_title(f"n={n}, h={h}")
        ax.legend(title="d values")
        ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.show()


# 示例调用
if __name__ == '__main__':
    # 三组 (N, h) 参数
    param_list = [(10000, 0.9), (1000, 0.9), (100, 0.9)]
    d_values = [1, 2, 5, 10, 20]
    plot_accuracy_for_params(param_list, d_values)

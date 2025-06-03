import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# 参数设置
ps = np.linspace(0.01, 1.0, 100)      # p 的取值范围
hd_values = [100,200,400,1000,2000,5000]                    # hd 值列表

plt.figure(figsize=(8, 5))

for hd in hd_values:
    if hd <= 1:
        continue

    mu = ps * (hd - 1)
    # 令 delta = 2 * exp( - (hd-1)*p / 2 )
    delta_expr = 2 * np.exp(- (hd - 1) * ps / 2) *1.1+0.0001  # +0.001 避免 0

    with np.errstate(divide='ignore', invalid='ignore'):
        eps2 = 2 * np.log(1 + 1/mu + np.sqrt(3 * np.log(2/delta_expr) / mu))
        eps3 = -2 * np.log(1 - np.sqrt(2 * np.log(2/delta_expr) / mu))
        eps = np.nanmax(np.vstack([eps2, eps3]), axis=0)

    # 截断到 [0,2]
    eps = np.clip(eps, 0, 2)

    plt.plot(ps, eps, label=f'hd={hd}')

plt.xlabel('p')
plt.ylabel(r'$\epsilon$')
plt.title(r'$\epsilon$ clipped to [0,2]$')
plt.ylim(0, 2)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

ps = np.linspace(0, 1.0, 1000)
hd_values = [100, 200, 400, 1000, 2000, 5000]

plt.figure(figsize=(8, 5))

for hd in hd_values:
    delta_expr = 2 * np.exp(- (hd - 1) * ps / 2) * 1.1 + 0.0001
    mask = delta_expr <= 0.01
    plt.plot(ps[mask], delta_expr[mask], label=f'hd={hd}')

plt.xlabel('p')
plt.ylabel('delta')
plt.title('Segment where delta <= 0.01')
plt.legend()
plt.ylim(0, 0.01)
plt.grid(True)
plt.xlim(0, 1)      # 强制横轴从0到1
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 1000
delta = 0.05
beta = delta

# p values
p = np.linspace(0, 1, 500)

# d values to plot
ds = [100]

# Plot
plt.figure()
for d in ds:
    alpha = p + np.sqrt(2 * (1 - p) / (N * d) * np.log(1 / beta))
    alpha = np.minimum(1, alpha)
    plt.plot(p, alpha, label=f'd={d}')

plt.xlabel('p')
plt.ylabel(r'$\alpha$')
plt.title(r'$\alpha = \min(1,\,p + \sqrt{\frac{2(1-p)\ln(1/\beta)}{N\,d}})$')
plt.legend()
plt.show()




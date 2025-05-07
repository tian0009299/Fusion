import glob, os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data_dir = 'result'   # ← adjust to your folder

pattern = os.path.join(data_dir, 'results_n100_d*_cn3.csv')
files = sorted(glob.glob(pattern),
               key=lambda f: int(os.path.basename(f).split('_')[2][1:]))

plt.figure(figsize=(8, 6))
legend_handles = []
labels = []

for d_val, fpath in [(int(os.path.basename(f).split('_')[2][1:]), f) for f in files]:
    df = pd.read_csv(fpath)
    df['R'] = pd.to_numeric(df['R'], errors='coerce')
    df['avg_l1'] = pd.to_numeric(df['avg_l1'], errors='coerce')
    df = df[df['R'] <= 256].sort_values('R')
    if df.empty:
        continue

    # moving average smoothing
    df['smoothed'] = df['avg_l1'].rolling(window=5, min_periods=1, center=True).mean()

    # plot with markers on the curve
    line, = plt.plot(df['R'], df['smoothed'],
                     marker='o', linewidth=1.5, label=f'd = {d_val}')

    # build a legend handle that has no marker
    lh = Line2D([], [],
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                marker='')
    legend_handles.append(lh)
    labels.append(f'd = {d_val}')

# draw legend: using our handles, and set markerscale=0 to hide markers
plt.legend(legend_handles,
           labels,
           title="d values",
           loc='best',
           markerscale=0,
           handlelength=2)

plt.xlabel("R (Number of Runs)")
plt.ylabel("Average L1 Error")
plt.title("Average L1 Error vs R for n=100, cn=3 (R ≤ 256)")
plt.grid(True)
plt.tight_layout()
plt.show()






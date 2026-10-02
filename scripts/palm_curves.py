import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
figure_dir = f"{output_root}/figures"
mlruns = os.environ.get("PALM_MLRUNS", os.path.expanduser("~/PALM/mlruns"))
run_id = os.environ.get("PALM_RUN", "")

os.makedirs(figure_dir, exist_ok = True)

# PALM reloads the epoch with the lowest validation loss before testing, so the test
# number is only as trustworthy as that choice. A minimum in the first few epochs means
# the model never learned, one at the last epoch means it was still improving when
# patience ran out, and a flat noisy valley means the chosen epoch was close to
# arbitrary. This reads the curves mlflow wrote and says which of those happened.


def read_metric(run_dir, name):
    """mlflow's file store writes one line per point: timestamp, value, step."""
    path = f"{run_dir}/metrics/{name}"
    if not os.path.exists(path):
        return None
    steps, values = [], []
    for line in open(path):
        parts = line.split()
        if len(parts) < 3:
            continue
        values.append(float(parts[1]))
        steps.append(int(float(parts[2])))
    if not values:
        return None
    frame = pd.DataFrame({'step': steps, name: values})
    # lightning can log a metric more than once per epoch; the last write wins
    return frame.groupby('step', as_index = False).last()


runs = sorted(glob.glob(f"{mlruns}/*/*/metrics"), key = os.path.getmtime)
runs = [os.path.dirname(r) for r in runs]
if run_id:
    runs = [r for r in runs if os.path.basename(r) == run_id]
if not runs:
    raise SystemExit(f"no mlflow runs with metrics under {mlruns}\n"
                     f"  set PALM_MLRUNS if the PALM checkout is elsewhere")

run_dir = runs[-1]
train = read_metric(run_dir, 'train.loss')
val = read_metric(run_dir, 'val.loss')
if val is None:
    raise SystemExit(f"{run_dir} has no val.loss: training never reached validation")

curves = val if train is None else train.merge(val, on = 'step', how = 'outer').sort_values('step')
curves = curves.reset_index(drop = True)
n_epochs = len(curves)
best_row = curves['val.loss'].idxmin()
best_step = int(curves.loc[best_row, 'step'])
best_val = float(curves.loc[best_row, 'val.loss'])

print(f"run {os.path.basename(run_dir)}")
print(f"epochs recorded      {n_epochs}")
print(f"best validation loss {best_val:.5f} at epoch {best_step}")

# is the chosen epoch a real minimum, or did it land somewhere arbitrary?
drop = float(curves['val.loss'].iloc[0] - best_val)
near = curves.loc[max(0, best_row - 10):best_row + 10, 'val.loss']
noise = float(np.abs(np.diff(curves['val.loss'].values)).mean()) if n_epochs > 1 else np.nan
print(f"fall from first epoch {drop:+.5f}")
print(f"spread within 10 epochs of the best {float(near.max() - near.min()):.5f}")
print(f"mean change between epochs {noise:.5f}"
      f"  ({noise / drop:.1%} of the total improvement)" if drop > 0 else "")

verdicts = []
if n_epochs < 10:
    verdicts.append("fewer than ten epochs ran, so there was no curve to choose from")
if best_row <= 2 and n_epochs > 10:
    verdicts.append("the best epoch is at the very start: validation loss never improved, "
                    "so the model did not learn")
if best_row >= n_epochs - 2 and n_epochs > 10:
    verdicts.append("the best epoch is the last one: it was still improving when training "
                    "ended, so raise max_epochs or patience")
first_val = float(curves['val.loss'].iloc[0])
if drop <= 0:
    verdicts.append("validation loss never fell below its starting value")
elif first_val > 0 and drop < 0.05 * first_val:
    # a clean curve can still be a curve that went nowhere. the selection being sound
    # says nothing about whether there was anything worth selecting between
    verdicts.append(f"validation loss fell by only {drop / first_val:.1%} of where it "
                    f"started ({first_val:.4f} to {best_val:.4f}), so the model barely "
                    f"trained whatever the curve looks like: check the spread of the "
                    f"predictions, and raise the number of optimiser steps per epoch "
                    f"by lowering the batch size before reading anything into the test score")
elif noise > 0.05 * drop:
    verdicts.append(f"epoch-to-epoch noise is {noise / drop:.0%} of the total improvement, so "
                    f"which epoch wins is close to arbitrary: look at the figure, and consider "
                    f"a lower learning rate or a larger batch before trusting the test number")
if not verdicts:
    verdicts.append("the minimum sits inside a settled valley, so the selected epoch is sound")
print()
for line in verdicts:
    print(f"  - {line}")

slide = os.environ.get("SPF_SLIDE", "") == "1"
scale = 1.4 if slide else 1.0
blue, vermillion = "#0072B2", "#D55E00"
ink, muted = "#1a1a1a", "#6b6b6b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11 * scale,
    "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": muted, "ytick.color": muted,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "#e4e4e2", "grid.linewidth": 0.8,
})

fig, axes = plt.subplots(1, 2, figsize = (12 * scale, 4.6 * scale))
for ax, window in zip(axes, [None, 'tail']):
    frame = curves if window is None else curves.iloc[max(0, best_row - 40):best_row + 40]
    ax.grid(True, zorder = 0)
    ax.set_axisbelow(True)
    if 'train.loss' in frame:
        ax.plot(frame['step'], frame['train.loss'], color = blue, linewidth = 1.8,
                label = 'train', zorder = 3)
    ax.plot(frame['step'], frame['val.loss'], color = vermillion, linewidth = 1.8,
            label = 'validation', zorder = 3)
    ax.axvline(best_step, color = ink, linestyle = '--', linewidth = 1.4, zorder = 4)
    ax.set_xlabel('epoch')
    ax.set_title('whole run' if window is None else 'around the selected epoch',
                 loc = 'left', fontsize = 10.5 * scale)
axes[0].set_ylabel('loss')
axes[0].legend(loc = 'upper right', fontsize = 10 * scale)
fig.suptitle(f"PALM training: lowest validation loss {best_val:.4f} at epoch {best_step} "
             f"of {n_epochs}\ndashed line is the epoch reloaded for testing",
             fontsize = 12.5 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.88])
path = f"{figure_dir}/palm_loss_curves.png"
plt.savefig(path)
plt.close()
curves.to_csv(f"{output_root}/results/palm_loss_curves.csv", index = False)
print(f"\nsaved {path}")

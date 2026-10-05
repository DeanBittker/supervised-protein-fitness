import os
import glob
import time
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
# binary cross entropy against a soft target cannot reach zero: its floor is the mean
# binary entropy of the targets themselves. On targets squashed into [0, 1] that floor
# sits near 0.65, so measuring progress against the starting loss makes a model that
# captured half of everything available look like one that did nothing. Point this at
# the exported csv and the improvement is reported against what was reachable
targets_path = os.environ.get("PALM_TARGETS", "")

os.makedirs(figure_dir, exist_ok = True)

# PALM reloads the epoch with the lowest validation loss before testing, so the test
# number is only as trustworthy as that choice. A minimum in the first few epochs means
# the model never learned, one at the last epoch means it was still improving when
# patience ran out, and a flat noisy valley means the chosen epoch was close to
# arbitrary. This reads the curves mlflow wrote and says which of those happened.


def read_metric(run_dir, name):
    """Read a metric as one value per epoch.

    PALM logs each epoch twice, once against the epoch number and once against the
    global optimiser step, so the file holds two interleaved series and is twice as
    long as the run. Walking it in order and keeping the line whose step is the epoch
    being waited for picks the epoch series out, whichever of the pair was written
    first. Sorting on step instead splices the two together and invents a restart.
    """
    path = f"{run_dir}/metrics/{name}"
    if not os.path.exists(path):
        return None
    epoch, values = 0, []
    for line in open(path):
        parts = line.split()
        if len(parts) < 3:
            continue
        if int(float(parts[2])) == epoch:
            values.append(float(parts[1]))
            epoch += 1
    if not values:
        return None
    return pd.DataFrame({'epoch': range(len(values)), name: values})


runs = sorted(glob.glob(f"{mlruns}/*/*/metrics"), key = os.path.getmtime)
runs = [os.path.dirname(r) for r in runs]
if run_id:
    runs = [r for r in runs if os.path.basename(r) == run_id]
if not runs:
    raise SystemExit(f"no mlflow runs with metrics under {mlruns}\n"
                     f"  set PALM_MLRUNS if the PALM checkout is elsewhere")

run_dir = runs[-1]
label = os.environ.get("PALM_LABEL", "")
suffix = f"_{label}" if label else ""
print(f"run {os.path.basename(run_dir)}"
      f"{'  (chosen as the most recent, PALM_RUN pins one)' if not run_id else ''}")
# picking the newest directory is wrong whenever a second job is still writing, and
# the numbers then belong to a run that has not finished. say so rather than let the
# curves be read against the wrong job
age = time.time() - max(os.path.getmtime(p) for p in glob.glob(f"{run_dir}/metrics/*"))
if age < 180:
    others = [os.path.basename(r) for r in runs[-4:-1]]
    print(f"  WARNING: written to {age:.0f}s ago, so this run is probably still going and")
    print(f"  these numbers are not final. check squeue before reading them.")
    if not run_id:
        print(f"  it was also chosen only for being newest; pin one with PALM_RUN. "
              f"others: {', '.join(others)}")
train = read_metric(run_dir, 'train.loss')
val = read_metric(run_dir, 'val.loss')
if val is None:
    raise SystemExit(f"{run_dir} has no val.loss: training never reached validation")

curves = val if train is None else train.merge(val, on = 'epoch', how = 'outer').sort_values('epoch')
curves = curves.reset_index(drop = True)
# train and validation are written at different points in the run, so the merge leaves
# a gap in whichever column was not written at that step. carrying the last value
# forward keeps both lines continuous instead of drawing them through the holes
for column in ['train.loss', 'val.loss']:
    if column in curves:
        curves[column] = curves[column].ffill()
curves = curves[curves['val.loss'].notna()].reset_index(drop = True)
n_epochs = len(curves)
best_row = curves['val.loss'].idxmin()
# the step mlflow records is the global optimiser step, which is the epoch number
# multiplied by the batches in an epoch. the position in the record is the epoch
best_step = int(curves.loc[best_row, 'epoch'])
best_val = float(curves.loc[best_row, 'val.loss'])

print(f"epochs recorded      {n_epochs}")
print(f"best validation loss {best_val:.5f} at record {best_row} of {n_epochs}"
      f" (optimiser step {best_step})")

# is the chosen epoch a real minimum, or did it land somewhere arbitrary?
drop = float(curves['val.loss'].iloc[0] - best_val)
near = curves.loc[max(0, best_row - 10):best_row + 10, 'val.loss']
noise = float(np.abs(np.diff(curves['val.loss'].values)).mean()) if n_epochs > 1 else np.nan
print(f"fall from first epoch {drop:+.5f}")
print(f"spread within 10 epochs of the best {float(near.max() - near.min()):.5f}")
print(f"mean change between epochs {noise:.5f}"
      f"  ({noise / drop:.1%} of the total improvement)" if drop > 0 else "")

first_val = float(curves['val.loss'].iloc[0])
floor = None
if targets_path and os.path.exists(targets_path):
    target_values = pd.read_csv(targets_path)['value_real'].astype(float).values
    target_values = (target_values - target_values.min()) / \
                    (target_values.max() - target_values.min())
    target_values = np.clip(target_values, 1e-7, 1 - 1e-7)
    floor = float(np.mean(-(target_values * np.log(target_values) +
                            (1 - target_values) * np.log(1 - target_values))))
    headroom = first_val - floor
    print(f"irreducible loss for these targets {floor:.5f}")
    print(f"headroom that ever existed         {headroom:+.5f}")
    if headroom > 0:
        print(f"share of it captured               {drop / headroom:.0%}")

verdicts = []
# the clearest thing a pair of curves can say: training loss going down while
# validation goes up means the model is fitting what it was given and losing ground
# on what it was not. with whole domains held out, that is a statement about
# generalising across domains rather than about the optimiser
if 'train.loss' in curves and n_epochs > 10:
    train_fall = float(curves['train.loss'].iloc[0] - curves['train.loss'].iloc[-1])
    val_rise = float(curves['val.loss'].iloc[-1] - curves['val.loss'].iloc[0])
    first_train = float(curves['train.loss'].iloc[0])
    if train_fall > 0.01 * first_train and val_rise > 0:
        verdicts.append(f"training loss fell {train_fall:.4f} while validation rose "
                        f"{val_rise:.4f}: the model is fitting the domains it was trained "
                        f"on and getting worse on the held-out ones, so this is about "
                        f"generalising across domains rather than about the optimiser")
if n_epochs < 10:
    verdicts.append("fewer than ten epochs ran, so there was no curve to choose from")
if best_row <= 2 and n_epochs > 10:
    verdicts.append("the best epoch is at the very start: validation loss never improved, "
                    "so the model did not learn")
if best_row >= n_epochs - 2 and n_epochs > 10:
    verdicts.append("the best epoch is the last one: it was still improving when training "
                    "ended, so raise max_epochs or patience")
if drop <= 0:
    verdicts.append("validation loss never fell below its starting value")
if floor is not None and first_val - floor > 0:
    captured = drop / (first_val - floor)
    if captured < 0.15:
        verdicts.append(f"validation loss moved {captured:.0%} of the way from where it "
                        f"started to the floor these targets allow ({floor:.4f}), so the "
                        f"model barely trained: check the spread of the predictions and "
                        f"raise the optimiser steps per epoch by lowering the batch size")
elif drop > 0 and first_val > 0 and drop < 0.05 * first_val:
    # without the targets there is no floor to measure against, so this falls back to
    # the starting loss, which understates progress on a loss that cannot reach zero
    verdicts.append(f"validation loss fell by only {drop / first_val:.1%} of where it "
                    f"started ({first_val:.4f} to {best_val:.4f}). if the loss is cross "
                    f"entropy against a soft target it cannot reach zero, so set "
                    f"PALM_TARGETS to the exported csv and read this against the floor "
                    f"rather than against the start")
if drop > 0 and noise > 0.05 * drop:
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
    frame = curves if window is None else curves.iloc[max(0, best_row - 20):best_row + 60]
    ax.grid(True, zorder = 0)
    ax.set_axisbelow(True)
    if 'train.loss' in frame:
        ax.plot(frame['epoch'], frame['train.loss'], color = blue, linewidth = 1.8,
                label = 'train', zorder = 3)
    ax.plot(frame['epoch'], frame['val.loss'], color = vermillion, linewidth = 1.8,
            label = 'validation', zorder = 3)
    ax.axvline(best_step, color = ink, linestyle = '--', linewidth = 1.4, zorder = 4)
    ax.set_xlabel('optimiser step')
    ax.set_title('whole run' if window is None else 'around the selected epoch',
                 loc = 'left', fontsize = 10.5 * scale)
axes[0].set_ylabel('loss')
axes[0].legend(loc = 'upper right', fontsize = 10 * scale)
fig.suptitle(f"PALM training: lowest validation loss {best_val:.4f} at optimiser step "
             f"{best_step}, record {best_row} of {n_epochs}"
             f"\ndashed line is the checkpoint reloaded for testing",
             fontsize = 12.5 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.88])
path = f"{figure_dir}/palm_loss_curves{suffix}.png"
plt.savefig(path)
plt.close()
curves.to_csv(f"{output_root}/results/palm_loss_curves{suffix}.csv", index = False)
print(f"\nsaved {path}")

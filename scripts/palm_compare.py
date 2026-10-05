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
results_dir = f"{output_root}/results"
figure_dir = f"{output_root}/figures"
mlruns = os.environ.get("PALM_MLRUNS", os.path.expanduser("~/PALM/mlruns"))
# "<run id>:<label>,<run id>:<label>", left to right. A ";" starts another row, so
# "a:held out,b:control;c:held out,d:control" is a two by two. Each row shares a y axis
# with itself and nothing else, which is the only honest way to put two losses on one
# figure: cross entropy sits near 0.65 and squared error near 0.01.
runs_spec = os.environ.get("PALM_RUNS", "")
targets_path = os.environ.get("PALM_TARGETS", "")
# one name per row, used as that row's y label
row_labels_spec = os.environ.get("PALM_ROW_LABELS", "")
# 1-based rows the entropy floor belongs on. It is a cross entropy floor, so it is wrong
# on a squared error row; default is every row, which is what a single row run wants.
floor_rows_spec = os.environ.get("PALM_FLOOR_ROWS", "")
# suffix on the output names, so a second comparison does not overwrite the first
label_suffix = os.environ.get("PALM_LABEL", "")
title_override = os.environ.get("PALM_TITLE", "")

os.makedirs(figure_dir, exist_ok = True)

# The two PALM runs differ in one thing: whether whole domains were held out or whether
# variants were split inside each protein. Put their curves side by side and the
# difference is the whole argument. The control is what says the model and the wiring
# work, so that the divergence on held-out domains is a property of the task.


def read_metric(run_dir, name):
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
    return pd.DataFrame({'step': steps, name: values}).groupby('step', as_index = False).last()


def load(run_id):
    found = [os.path.dirname(p) for p in glob.glob(f"{mlruns}/*/{run_id}*/metrics")]
    if not found:
        raise SystemExit(f"no run under {mlruns} matching {run_id}")
    train, val = read_metric(found[0], 'train.loss'), read_metric(found[0], 'val.loss')
    if val is None:
        raise SystemExit(f"{found[0]} has no val.loss")
    frame = val if train is None else train.merge(val, on = 'step', how = 'outer').sort_values('step')
    frame = frame.reset_index(drop = True)
    for column in ['train.loss', 'val.loss']:
        if column in frame:
            frame[column] = frame[column].ffill()
    return frame[frame['val.loss'].notna()].reset_index(drop = True)


if not runs_spec:
    raise SystemExit('set PALM_RUNS="<run id>:<label>,<run id>:<label>"'
                     ' and separate rows with ";"')
rows = []
for row_spec in runs_spec.split(';'):
    if not row_spec.strip():
        continue
    wanted = []
    for item in row_spec.split(','):
        run_id, _, label = item.partition(':')
        if not run_id.strip():
            continue
        wanted.append((run_id.strip(), label.strip() or run_id.strip()[:8]))
    if wanted:
        rows.append(wanted)
if not rows:
    raise SystemExit(f"no run ids in PALM_RUNS={runs_spec!r}")
widths = {len(row) for row in rows}
if len(widths) > 1:
    raise SystemExit(f"every row needs the same number of panels, got {sorted(widths)}")

row_labels = [part.strip() for part in row_labels_spec.split(';')] if row_labels_spec else []
row_labels += ['loss'] * (len(rows) - len(row_labels))
if floor_rows_spec:
    floor_rows = {int(part) for part in floor_rows_spec.replace(',', ' ').split()}
else:
    floor_rows = set(range(1, len(rows) + 1))

floor = None
if targets_path and os.path.exists(targets_path):
    y = pd.read_csv(targets_path)['value_real'].astype(float).values
    y = np.clip((y - y.min()) / (y.max() - y.min()), 1e-7, 1 - 1e-7)
    floor = float(np.mean(-(y * np.log(y) + (1 - y) * np.log(1 - y))))
    print(f"irreducible loss for these targets {floor:.4f}")

slide = os.environ.get("SPF_SLIDE", "") == "1"
scale = 1.4 if slide else 1.0
blue, vermillion, muted_line = "#0072B2", "#D55E00", "#009E73"
ink, muted = "#1a1a1a", "#6b6b6b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11 * scale,
    "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": muted, "ytick.color": muted,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "#e4e4e2", "grid.linewidth": 0.8,
})

grid = [[(label, load(run_id)) for run_id, label in row] for row in rows]
ncol = len(grid[0])
fig, axes = plt.subplots(len(grid), ncol, squeeze = False,
                         figsize = (6.2 * ncol * scale, 4.6 * len(grid) * scale),
                         sharey = 'row')
summary = []
for row_index, row in enumerate(grid):
    row_floor = floor if (row_index + 1) in floor_rows else None
    for col_index, (label, frame) in enumerate(row):
        ax = axes[row_index][col_index]
        best = frame['val.loss'].idxmin()
        first = float(frame['val.loss'].iloc[0])
        drop = first - float(frame.loc[best, 'val.loss'])
        ax.grid(True, zorder = 0)
        ax.set_axisbelow(True)
        if 'train.loss' in frame:
            ax.plot(frame.index, frame['train.loss'], color = blue, linewidth = 1.8,
                    label = 'train', zorder = 3)
        ax.plot(frame.index, frame['val.loss'], color = vermillion, linewidth = 1.8,
                label = 'validation', zorder = 3)
        ax.axvline(best, color = ink, linestyle = '--', linewidth = 1.3, zorder = 4)
        if row_floor is not None:
            ax.axhline(row_floor, color = muted_line, linestyle = ':', linewidth = 1.6,
                       zorder = 2, label = 'floor these targets allow')
        captured = (drop / (first - row_floor)
                    if row_floor is not None and first > row_floor else np.nan)
        ax.set_title(f"{label}\n{captured:.0%} of the available range"
                     if np.isfinite(captured) else label,
                     loc = 'left', fontsize = 11 * scale)
        if row_index == len(grid) - 1:
            ax.set_xlabel('epoch')
        summary.append({'row': row_labels[row_index], 'run': label, 'epochs': len(frame),
                        'first_val': round(first, 5),
                        'best_val': round(float(frame.loc[best, 'val.loss']), 5),
                        'best_epoch': int(best),
                        'share_of_range': None if not np.isfinite(captured)
                        else round(captured, 3)})
    axes[row_index][0].set_ylabel(row_labels[row_index])
    # each row carries its own legend because only a floored row has a floor entry
    axes[row_index][0].legend(loc = 'center right', fontsize = 9.5 * scale)
fig.suptitle(title_override or
             "PALM: holding whole domains out against splitting inside each protein\n"
             "same model and variants, only the split differs",
             fontsize = 12.5 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.88 if len(grid) == 1 else 0.94])
suffix = f"_{label_suffix}" if label_suffix else ""
path = f"{figure_dir}/palm_loss_comparison{suffix}.png"
plt.savefig(path)
plt.close()

report = pd.DataFrame(summary)
report.to_csv(f"{results_dir}/palm_loss_comparison{suffix}.csv", index = False)
print(report.to_string(index = False))
print(f"\nsaved {path}")

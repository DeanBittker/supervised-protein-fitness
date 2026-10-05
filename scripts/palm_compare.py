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
# "<run id>:<label>,<run id>:<label>", left to right
runs_spec = os.environ.get("PALM_RUNS", "")
targets_path = os.environ.get("PALM_TARGETS", "")

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
    raise SystemExit('set PALM_RUNS="<run id>:<label>,<run id>:<label>"')
wanted = []
for item in runs_spec.split(','):
    run_id, _, label = item.partition(':')
    wanted.append((run_id.strip(), label.strip() or run_id.strip()[:8]))

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

frames = [(label, load(run_id)) for run_id, label in wanted]
fig, axes = plt.subplots(1, len(frames), figsize = (6.2 * len(frames) * scale, 4.6 * scale),
                         sharey = True)
axes = np.atleast_1d(axes)
summary = []
for ax, (label, frame) in zip(axes, frames):
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
    if floor is not None:
        ax.axhline(floor, color = muted_line, linestyle = ':', linewidth = 1.6, zorder = 2,
                   label = 'floor these targets allow')
    captured = drop / (first - floor) if floor is not None and first > floor else np.nan
    ax.set_title(f"{label}\n{captured:.0%} of the available range" if np.isfinite(captured)
                 else label, loc = 'left', fontsize = 11 * scale)
    ax.set_xlabel('epoch')
    summary.append({'run': label, 'epochs': len(frame), 'first_val': round(first, 5),
                    'best_val': round(float(frame.loc[best, 'val.loss']), 5),
                    'best_epoch': int(best),
                    'share_of_range': None if not np.isfinite(captured) else round(captured, 3)})

axes[0].set_ylabel('loss')
axes[0].legend(loc = 'center right', fontsize = 9.5 * scale)
fig.suptitle("PALM: holding whole domains out against splitting inside each protein\n"
             "same model and variants, only the split differs",
             fontsize = 12.5 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.88])
path = f"{figure_dir}/palm_loss_comparison.png"
plt.savefig(path)
plt.close()

report = pd.DataFrame(summary)
report.to_csv(f"{results_dir}/palm_loss_comparison.csv", index = False)
print(report.to_string(index = False))
print(f"\nsaved {path}")

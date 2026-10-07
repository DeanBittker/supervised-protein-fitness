import os
import glob
import numpy as np
import pandas as pd

# PALM keeps the epoch with the lowest validation loss. On held-out domains that loss is
# dominated by calibration: the model puts an unseen domain's mean in the wrong place and
# pays for it on every one of that domain's variants, while still ordering them. So the
# rule selects for calibration, and the metric everything else in this project is judged
# by is a rank correlation. This asks what that costs, by reading the validation Spearman
# PALM already logs at the epoch it chose and at the epoch it should have chosen.

mlruns = os.environ.get("PALM_MLRUNS", os.path.expanduser("~/PALM/mlruns"))
output_root = os.environ.get("SPF_OUTPUT",
                             os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
results_dir = f"{output_root}/results"
# only runs whose data_name contains this, so a sweep can be read on its own
wanted = os.environ.get("PALM_MATCH", "")


def series(path):
    """One value per epoch.

    PALM logs each epoch twice, once against the epoch and once against the global
    optimiser step, so the epoch series is picked out by waiting for each epoch in turn.
    """
    epoch, values = 0, []
    for line in open(path):
        parts = line.split()
        if len(parts) >= 3 and int(float(parts[2])) == epoch:
            values.append(float(parts[1]))
            epoch += 1
    return values


rows = []
for run_dir in sorted(glob.glob(f"{mlruns}/*/*/")):
    loss_path = f"{run_dir}metrics/val.loss"
    rho_path = f"{run_dir}metrics/val.spearman_r"
    name_path = f"{run_dir}params/dataset.data_name"
    if not (os.path.exists(loss_path) and os.path.exists(rho_path)):
        continue
    name = open(name_path).read().strip() if os.path.exists(name_path) else "?"
    if wanted and wanted not in name:
        continue
    loss, rho = series(loss_path), series(rho_path)
    width = min(len(loss), len(rho))
    if width < 5:
        continue
    loss, rho = np.array(loss[:width]), np.array(rho[:width])
    chosen, best = int(loss.argmin()), int(rho.argmax())
    rows.append({'run': os.path.basename(run_dir.rstrip('/'))[:8], 'data': name,
                 'epochs': width, 'chosen_epoch': chosen, 'rho_at_chosen': rho[chosen],
                 'best_epoch': best, 'rho_at_best': rho[best],
                 'left_on_table': rho[best] - rho[chosen]})

if not rows:
    raise SystemExit(f"no run under {mlruns} has both val.loss and val.spearman_r"
                     + (f" matching {wanted!r}" if wanted else ""))

frame = pd.DataFrame(rows).sort_values('left_on_table', ascending = False)
print(frame.to_string(index = False,
                      formatters = {'rho_at_chosen': '{:+.3f}'.format,
                                    'rho_at_best': '{:+.3f}'.format,
                                    'left_on_table': '{:+.3f}'.format}))
print()
print(f"{len(frame)} runs, median left on the table {frame['left_on_table'].median():+.3f}, "
      f"worst {frame['left_on_table'].max():+.3f}")
held = frame[frame['data'].str.contains('replicate|loco', na = False)]
if not held.empty:
    print(f"  held-out runs only: median {held['left_on_table'].median():+.3f} "
          f"over {len(held)} runs")

os.makedirs(results_dir, exist_ok = True)
path = f"{results_dir}/palm_checkpoint_check.csv"
frame.to_csv(path, index = False)
print(f"\nsaved {path}")

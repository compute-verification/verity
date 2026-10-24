"""Survival curves P(round trip later than x) for the network-only runs: the concurrent ping series (left) and the
receive-CPU pinning runs (right), from analyze.py's JSON files.

    python plot_net.py OUT.png 'JSON|LABEL|KEY;...' 'JSON|LABEL|KEY;...'

Each panel's curves: analyze.py's JSON, a legend label, and the cell's key in it."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = sys.argv[1]
panels = [("concurrent 5-minute ping series, 1 ms apart", sys.argv[2]),
          ("challenger on node 1, responder on cpu-1: challenger on / off its flow's receive-interrupt CPU",
           sys.argv[3])]
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)
for ax, (title, spec) in zip(axes, panels):
    for item in spec.split(";"):
        path, label, key = item.split("|")
        c = json.load(open(path))["cells"][key]
        s = c["survival"]
        pts = [(x, y) for x, y in zip(s["x_us"], s["p_late"]) if y > 0]
        ax.loglog([x for x, _ in pts], [y for _, y in pts], label=f"{label} (n={c['n']:,})", lw=1.4)
    for t in (50, 100, 150, 300, 840):
        ax.axvline(t, color="grey", lw=0.6, ls=":")
    ax.set_title(title)
    ax.set_xlabel("x (µs)")
    ax.set_xlim(20, 2e4)
    ax.set_ylim(1e-6, 1.5)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
axes[0].set_ylabel("P(answer later than x)")
fig.suptitle("Nebius VPC round trips, host-memory answers, no GPU (dotted: τ = 50, 100, 150, 300, 840 µs)")
fig.tight_layout()
fig.savefig(out, dpi=110)

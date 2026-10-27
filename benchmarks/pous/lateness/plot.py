"""Survival curves P(answer later than x) from analyze.py's JSON: on-node, cpu-1 to node 2, node 1 to node 2.

    python plot.py ANALYZE_JSON OUT.png LABEL=RUN/CONDITION/OP..."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rep = json.load(open(sys.argv[1]))
out = sys.argv[2]
panels = [("on-node challenger (loopback): the lower bound", "local"), ("challenger on a CPU node", "cpu1"),
          ("challenger on a second GPU node", "node1")]
want = [a.split("=") for a in sys.argv[3:]]  # label=run/cond/OPNAME, drawn in every panel
fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
for ax, (title, who) in zip(axes, panels):
    for label, spec in want:
        run, cond, op = spec.split("/")
        key = f"{run}/{cond}/{who}/{op}"
        c = rep["cells"].get(key)
        if not c:
            continue
        s = c["survival"]
        pts = [(x, y) for x, y in zip(s["x_us"], s["p_late"]) if y > 0]
        ax.loglog([x for x, _ in pts], [y for _, y in pts], label=f"{label} (n={c['n']:,})", lw=1.4)
    for t in rep["taus_us"]:
        ax.axvline(t, color="grey", lw=0.6, ls=":")
    ax.set_title(title)
    ax.set_xlabel("x (µs)")
    ax.set_xlim(5, 1e5)
    ax.set_ylim(1e-6, 1.5)
    ax.grid(True, which="major", alpha=0.3)
axes[0].set_ylabel("P(answer later than x)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, [lab.split(" (n=")[0] for lab in labels], loc="lower center", ncol=5, fontsize=9)
fig.suptitle("PoUS honest lateness, 4 KiB challenge answers from GPU memory, node 2 (dotted: τ = 50, 100, 150, 300, 840 µs)")
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(out, dpi=110)

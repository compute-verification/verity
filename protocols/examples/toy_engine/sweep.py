"""PoUS's certified fraction as a function of the engine's yield latency.

    uv run python protocols/examples/toy_engine/sweep.py [--plot out.png]

Part 1 evaluates `DesignC.certified` (the timing term only, 1 - tau) for a few hardware profiles.  Part 2 runs the toy
engine at three yield latencies under a stress audit rate, so that audits land inside steps, and reports what the
challenger observed and certified (sampling term included).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toy_engine import demo  # noqa: E402
from verity_pous import DesignC  # noqa: E402

LATENCIES = (1e-6, 1.92e-6, 16.36e-6, 1e-4, 3.4e-4, 1e-3, 2.5e-3)
PROFILES = {
    # H100 figures measured in the PoUS research (challenger2): 3.1 TB/s random sectors, 55 GB/s over PCIe tiers.
    "H100, X = 4 GB": DesignC(),
    "H100, X = 1.6 GB": DesignC(audit_bytes=1.6e9),
    # an NVLink peer (450 GB/s published) that is not challenged at the same instant
    "H100, X = 4 GB, unchallenged NVLink peer": DesignC(b_alt=55e9 + 450e9),
    # published, unmeasured: B200 HBM3e 8 TB/s, NVLink-C2C 450-550 GB/s (500 used), X = 5% of 186 GB
    "GB200, C2C, X = 9.3 GB": DesignC(b_hbm=8e12, b_alt=500e9, audit_bytes=9.3e9),
}
REBUILD = {"H100, X = 4 GB, weights rebuilt at 100 GB/s": DesignC(b_rec={"weights": 100e9})}


def zero_at(cfg: DesignC, region: str = "filler") -> float:
    """The yield latency at which nothing is certified."""
    return cfg.audit_bytes / (cfg.b_alt + cfg.b_rec.get(region, 0.0)) - cfg.audit_bytes / cfg.b_hbm - cfg.jitter


def table() -> list[str]:
    head = "| profile | " + " | ".join(f"{l * 1e6:g} us" for l in LATENCIES) + " | zero at |"
    rows = [head, "| --- " * (len(LATENCIES) + 2) + "|"]
    for name, cfg in {**PROFILES, **REBUILD}.items():
        region = "weights" if cfg.b_rec else "filler"
        cells = " | ".join(f"{cfg.certified(region, l):.3f}" for l in LATENCIES)
        rows.append(f"| {name} | {cells} | {zero_at(cfg, region) * 1e3:.1f} ms |")
    return rows


def stress() -> list[str]:
    rows = ["| engine | yield latency | audits (in steps) | slowest wait | planned (timing) | certified filler |",
            "| --- | --- | --- | --- | --- | --- |"]
    base = demo.Config(epoch=0.05, audit_rate=400.0, mean_gap=5e-5)
    for label, cfg in (("preemptible per request", base), ("non-preemptible step", replace(base, preemptible=False)),
                       ("1 ms kernel per step", replace(base, launch=1e-3))):
        world = demo.setup(cfg)
        demo.serve(world)
        spaces = world.challenger.fold([(a[0], a[2]) for a in world.audits])
        L = world.engine.cfg.yield_latency
        in_steps = sum(a[1] > 0 for a in world.audits)
        rows.append(f"| {label} | {L * 1e6:.2f} us | {len(world.audits)} ({in_steps}) | "
                    f"{max(a[1] for a in world.audits) * 1e6:.2f} us | {world.pous.certified('filler', L):.4f} | "
                    f"{1 - spaces['filler'].deficit:.4f} |")
    return rows


def plot(path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ls = np.logspace(-6, -2, 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, cfg in {**PROFILES, **REBUILD}.items():
        region = "weights" if cfg.b_rec else "filler"
        ax.plot(ls * 1e6, [cfg.certified(region, l) for l in ls], label=name)
    for l, text in ((1.92e-6, "toy, preemptible"), (16.36e-6, "toy step loop"), (1e-3, "1 ms")):
        ax.axvline(l * 1e6, color="grey", lw=0.6, ls=":")
        ax.text(l * 1e6 * 1.08, 0.45, text, rotation=90, fontsize=7, color="grey")
    ax.set(xscale="log", xlabel="engine yield latency (us)", ylabel="certified fraction (timing term)",
           ylim=(0, 1.02), title="PoUS design C: certified fraction vs yield latency")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot")
    a = ap.parse_args(argv)
    print("\n".join(table()))
    print()
    print("\n".join(stress()))
    if a.plot:
        plot(a.plot)


if __name__ == "__main__":
    main()

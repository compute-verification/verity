"""The tensor pipe's use in each kernel of one arm, from Nsight Compute (run.sh's NCU=1: `ncu --csv --page raw` over a
short period of the arm, kernels serialized, clocks as the node sets them): per kernel name, its launches, time, INT8
tensor ops and the tensor pipe's active share of its elapsed cycles (time-weighted), and its SM clock. With `--card`,
each kernel's MACs (the counter counts a MAC as two ops) against θ at its own clock: for the GEMM, the share the pipe
fills when nothing runs beside it (the counted side of terms.py's GEMM slack). On the RTX the two agree: cuBLASLt's s8
GEMM at 87.4% of θ and the pipe active 87.4% (r20261027-072824-8af6).

    pipes.py NCU_CSV [--card SPEC]      prints one JSON object
"""
from __future__ import annotations

import argparse
import csv
import json

TIME = "gpu__time_duration.sum"
OPS = "sm__ops_path_tensor_op_imma_src_int8.sum"
ACTIVE = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
CLOCK = "sm__cycles_elapsed.avg.per_second"
METRICS = (TIME, OPS, ACTIVE, CLOCK)
MACS_PER_OP = 0.5
SCALE = {"": 1.0, "%": 1.0, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0, "nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0, "hz": 1.0, "khz": 1e3,
         "mhz": 1e6, "ghz": 1e9, "cycle/second": 1.0, "cycle/usecond": 1e6, "cycle/nsecond": 1e9, "inst": 1.0}


def _num(x: str) -> float:
    return float(x.replace(",", "")) if x.strip() not in ("", "n/a") else 0.0


def launches(text: str) -> list[dict]:
    """Each profiled launch: its kernel name and the four metrics in base units (seconds, ops, a share, Hz)."""
    rows = list(csv.reader(line for line in text.splitlines() if line.startswith('"')))
    head = rows[0]
    units = next((r for r in rows[1:] if not r[head.index("ID")].strip().isdigit()), [""] * len(head))
    col = {m: head.index(m) for m in METRICS}
    out = []
    for r in rows[1:]:
        if not r[head.index("ID")].strip().isdigit():
            continue
        v = {m: _num(r[c]) * SCALE.get(units[c].strip().lower(), 1.0) for m, c in col.items()}
        out.append({"name": r[head.index("Kernel Name")], "s": v[TIME], "ops": v[OPS], "active": v[ACTIVE] / 100,
                    "hz": v[CLOCK]})
    return out


def pipes(text: str, card: dict | None = None) -> dict:
    by: dict[str, dict] = {}
    for x in launches(text):
        k = by.setdefault(x["name"], {"name": x["name"], "launches": 0, "s": 0.0, "ops": 0.0, "_a": 0.0, "_hz": 0.0})
        k["launches"] += 1
        k["s"] += x["s"]
        k["ops"] += x["ops"]
        k["_a"] += x["active"] * x["s"]
        k["_hz"] += x["hz"] * x["s"]
    ops = sum(k["ops"] for k in by.values())
    kernels = []
    for k in sorted(by.values(), key=lambda k: -k["s"]):
        a, hz = k.pop("_a"), k.pop("_hz")
        k["tensor_active"] = a / k["s"] if k["s"] else 0.0
        k["mhz"] = hz / k["s"] / 1e6 if k["s"] else 0.0
        k["macs"] = k["ops"] * MACS_PER_OP
        if card is not None and k["s"]:
            from period import theta
            k["share_of_theta"] = k["macs"] / (k["s"] * theta(card, k["mhz"]))
        kernels.append(k)
    total = sum(k["s"] for k in kernels)
    return {"kernels": kernels, "s": total, "int8_ops": ops, "macs": ops * MACS_PER_OP,
            "tensor_active": sum(k["tensor_active"] * k["s"] for k in kernels) / total if total else 0.0}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv")
    p.add_argument("--card")
    a = p.parse_args(argv)
    card = None
    if a.card:
        from period import load_card
        card = load_card(a.card)
    with open(a.csv, encoding="utf-8", errors="replace") as f:
        print(json.dumps(pipes(f.read(), card), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The machine model's sm_89 census (`benchmarks/pouw/machine_model/data/sm89/`) from a run of `run.sh` (scalar.cu) and a
run of tensor_kinds with TK_ARCH=89, in the machine model's schema with one epoch (no `since` or `until`).

  python census.py SCALAR_OUT TENSOR_KINDS_OUT [--write DIR]   # prints rows.csv and census.csv, or writes them in DIR

Every kind is one warp instruction; MMA kinds are tensor_kinds' (one a SASS form, `KIND`). The rows:

- `issue`: one warp instruction a SMSP-cycle, every kind.
- `M`: the tensor pipe, each MMA kind at its fastest form's alone rate.
- `JI89`: the joint issue row, every MMA kind's tensor_kinds coefficient and every other kind at one SMSP-cycle (every
  SASS instruction of a program counted), both divided by tensor_kinds' SCHED, as `refit` does.
- one pipe row a scalar kind, at the largest pipe rate its alone probes reach. A probe's pipe rate counts every
  instruction of its loop on the counted one's pipe (the ALU pipe's are `ALU_PIPE`; the loops' IADD3 and ISETP run there),
  so it is the pipe's own rate, not the probe's. Kinds share a row only where that pipe accounting shows it (`A`).
- the `E*` facets of each MMA + X family (eps1.py `facets`), with every non-MMA instruction of a probe's loops counted
  as X: an overhead instruction could be an X in an attacker's program, so the facets stay sound, and loose at the X
  axis, where the pipe row binds.

FFMA has no pipe row: it issues 1.000 a SMSP-cycle alone (tensor_kinds' lane ops), so `issue` binds first.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("opcode_probe_eps1", HERE.parent / "eps1.py")
TK = _load("tensor_kinds_fit", HERE.parents[1] / "tensor_kinds" / "tensor_kinds.py")

ALU_PIPE = {"IADD3", "LOP3", "SHF", "PRMT", "ISETP", "IMNMX", "LEA", "SEL"}
#: scalar kind: (row, alone probes, what the row holds)
PIPES = {
    "ALU": ("A", ("alone_iadd3", "alone_lop3", "alone_shf", "alone_prmt"),
            "ALU pipe: IADD3, LOP3, SHF, PRMT, IMNMX, LEA.HI"),
    "VIMNMX": ("A", ("alone_imnmx",), ""),
    # duo's fold (LOP3 + IMAD + LEA.HI a word): its LOP3, LEA.HI and loop counter together run at the pipe's rate
    "LEA": ("A", ("fold_only",), ""),
    "IMAD": ("I", ("alone_imad",), "IMAD"),
    "IDP4A": ("I4A", ("alone_idp4a",), "IDP.4A"),
    "HFMA2": ("H", ("alone_hfma2",), "HFMA2"),
    "SHFL": ("SHFL", ("alone_shfl",), "SHFL.BFLY"),
    "XU": ("XU", ("alone_popc",), "POPC"),
    "FLO": ("FLO", ("alone_flo",), "FLO"),
    "LDSM_X1": ("LDSM_X1", ("ldsm_x1",), "LDSM.16.M88"),
    "LDSM_X2": ("LDSM_X2", ("ldsm_x2",), "LDSM.16.M88.2"),
    "LDSM_X4": ("LDSM_X4", ("ldsm_x4",), "LDSM.16.M88.4"),
    "LDSM_X1T": ("LDSM_X1T", ("ldsm_x1t",), "LDSM.16.MT88"),
    "LDSM_X2T": ("LDSM_X2T", ("ldsm_x2t",), "LDSM.16.MT88.2"),
    "LDSM_X4T": ("LDSM_X4T", ("ldsm_x4t",), "LDSM.16.MT88.4"),
    "LDS32": ("LDSi", ("lds32",), "LDS.32"),
    "LDS128": ("R6r", ("lds128",), "LDS.128"),
    "STS128": ("R6w", ("sts128",), "STS.128"),
    "FFMA": (None, (), ""),
}
MARGIN = 0.02


def pipe_class(op: str) -> str:
    return "ALU" if op.split(".")[0] in ALU_PIPE else op


def loop_rates(row: dict, verdict: dict) -> tuple[float, float]:
    """(IMMA.16832 rate, every non-MMA instruction's rate) over the probe's loops, warp instructions/SM/clk."""
    m = x = 0.0
    seen = set()
    for key, r in row["rates"].items():
        pat = key.split(":", 1)[1]
        i = next(j for j, lp in enumerate(verdict["loops"]) if lp["counts"].get(pat, 0) > 0)
        if i in seen:
            continue
        seen.add(i)
        lp = verdict["loops"][i]
        per_iter = r / lp["counts"][pat]
        mma = sum(n for op, n in lp["top"].items() if TK.MMA.match(op))
        m += per_iter * sum(n for op, n in lp["top"].items() if op.startswith(E.IMMA))
        x += per_iter * (sum(lp["top"].values()) - mma)
    return m, x


def pipe_rate(row: dict, verdict: dict) -> float:
    """The probe's pipe rate: its loop's rate times the loop's instructions on the counted ones' pipe (every ALU-pipe
    instruction when one counted pattern is on it)."""
    (lp,) = verdict["loops"]
    pats = [p for p, n in lp["counts"].items() if n]
    per_iter = next(r for k, r in row["rates"].items() if k.split(":", 1)[1] == pats[0]) / lp["counts"][pats[0]]
    ops = [op for op in lp["top"] for p in pats if op.startswith(p.rstrip("*"))]
    if any(pipe_class(op) == "ALU" for op in ops):
        return per_iter * sum(n for op, n in lp["top"].items() if pipe_class(op) == "ALU")
    return per_iter * sum(lp["top"][op] for op in ops)


def build(scalar: Path, tk: Path) -> tuple[list[list[str]], list[list[str]]]:
    rows = {r["probe"]: r for r in E.load(scalar / "rows.jsonl")}
    gate = {v["probe"]: v["verdict"] for v in json.loads((scalar / "gate.json").read_text())["verdicts"]}
    s_run = next(iter(rows.values()))["run"]
    t_run = json.loads((tk / "meta.json").read_text())["run_id"]
    summary, fails = TK.analyze(tk)
    if fails:
        raise SystemExit(f"{tk}: tensor_kinds gate failures: {fails}")
    rws: list[list[str]] = []
    cen: list[list[str]] = []

    kinds: dict[str, dict] = {}
    for form, s in summary.items():
        if form == "lane_op_alone":
            continue
        k = kinds.setdefault(TK.KIND[form], {"forms": [], "alone": float("inf"), "a": float("inf"), "slots": s["slots"]})
        k["forms"].append(form)
        k["alone"] = min(k["alone"], s["alone_smsp_cycles_per_mma"])
        k["a"] = min(k["a"], *((1 - TK.NOISE) * p["cycles_per_mma"] - p["ops_per_mma"] for p in s["points"]))
    scalar_kinds = list(PIPES)

    lanes = summary["lane_op_alone"]
    rws.append(["issue", "measured", t_run, "1 warp instruction a SMSP-cycle, every kind: FFMA alone "
                f"{lanes['ffma']['issue_per_smsp_cycle']:.3f}, IADD3 and FFMA alternating "
                f"{lanes['iadd3_ffma']['issue_per_smsp_cycle']:.3f} a SMSP-cycle (tensor_kinds' lane ops)"])
    cen += [[k, "issue", "", "", "1", "", "", "", t_run] for k in [*kinds, *scalar_kinds]]

    rws.append(["M", "measured", t_run, "tensor pipe: each MMA kind at its fastest form's alone rate + 2%; "
                + ", ".join(f"{k} {4 * v['slots'] / v['alone']:.2f} words/SM/clk" for k, v in kinds.items())])
    cen += [[k, "M", f"{4 / v['alone']:.6g}", f"{MARGIN}", "", "", "", "", t_run] for k, v in kinds.items()]

    rws.append(["JI89", "joint", t_run, f"each MMA kind a SMSP-cycles + every other warp instruction 1, / {TK.SCHED} "
                "(tensor_kinds, every SASS instruction counted, 2% noise margin): "
                + ", ".join(f"{k} {v['a']:.3f}" for k, v in kinds.items())])
    cen += [[k, "JI89", "", "", f"{v['a'] / TK.SCHED:.9g}", "", "", "", t_run] for k, v in kinds.items()]
    cen += [[k, "JI89", "", "", f"{1 / TK.SCHED:.9g}", "", "", "", t_run] for k in scalar_kinds]

    done_rows = set()
    for kind, (row, probes, what) in PIPES.items():
        if row is None:
            continue
        rates = {p: pipe_rate(rows[p], gate[p]) for p in probes}
        best = max(rates, key=rates.get)
        cen.append([kind, row, f"{rates[best]:.6g}", f"{MARGIN}", "", "", "", "", s_run])
        if row in done_rows:
            continue
        done_rows.add(row)
        shared = [k for k, v in PIPES.items() if v[0] == row]
        allr = {p: pipe_rate(rows[p], gate[p]) for k in shared for p in PIPES[k][1]}
        rws.append([row, "measured", s_run, f"{what} {max(allr.values()):.5f} warp instructions/SM/clk + 2%, the loop's "
                    "instructions on the pipe counted (" + ", ".join(f"{p} {v:.5f}" for p, v in allr.items()) + ")"])

    pts_rows = []
    for name, r in rows.items():
        m, x = loop_rates(r, gate[name])
        pts_rows.append({"probe": name, "run": r["run"], "rates": {f"0:{E.IMMA}": m, "0:*": x}})
    for x, _ in E.OPS.items():
        kind, _, new = E.FAMILY[x]
        fs = E.facets(E.frontier_points(pts_rows, x, "*"), MARGIN)
        for i, f in enumerate(fs):
            tight = " and ".join(t.split("@")[0] for t in f["tight"])
            rws.append([f"{new}{i}", "joint", s_run, f"IMMA16832 {f['MMA']:.3f} + {kind} {f['X']:.3f} SMSP-cycles (sm89/"
                        f"scalar.cu's {x} frontier, every non-MMA instruction of the loops counted as {kind}, 2% noise "
                        f"margin; tight at {tight})"])
            for k, c in (("IMMA16832", f["MMA"]), (kind, f["X"])):
                if c > 0:
                    cen.append([k, f"{new}{i}", "", "", f"{c:.9g}", "", "", "", s_run])
    return rws, cen


def csv_text(header: list[str], lines: list[list[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(lines)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scalar", type=Path)
    ap.add_argument("tensor_kinds", type=Path)
    ap.add_argument("--write", type=Path, metavar="DIR")
    a = ap.parse_args(argv)
    rws, cen = build(a.scalar, a.tensor_kinds)
    out = {"rows.csv": csv_text(["row", "status", "runs", "description"], rws),
           "census.csv": csv_text(["kind", "row", "rate", "noise", "cycles", "u", "since", "until", "runs"], cen)}
    for name, text in out.items():
        if a.write:
            a.write.mkdir(parents=True, exist_ok=True)
            (a.write / name).write_text(text)
        else:
            print(f"# {name}\n{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

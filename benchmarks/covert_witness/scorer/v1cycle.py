"""Night 2's formalism-v1 refresh (note:training-20261102T1817Z-report-training-formalism-v1), every 90 minutes.

    python v1cycle.py [--out DIR] [--no-publish] [--no-put] [--no-post]

One refresh:

1. Rebuilds the merged ledger (every ingested checkpoint's MACs log) from STATE/ingest/<TE_CURRENT_COMMIT[:12]>.
2. Runs the re-score, benchmarks/covert_witness/analysis/rescore_v1.py at the newest commit that touches it on
   cursor/te-rescore-v1-4207 or cursor/training-efficiency-4207: per checkpoint of every valid v2.3 record Delta (= u(1),
   the best-known value, an upper bound), Delta_grad, phi, Fit (running and training) and the Slowdown theorem's rho
   (target) on the (c, k) grid, under both readings of the u profile (`spec`: evenly spread, the frozen-tail form for
   one-layer runs; `witness`: the frozen-tail form for every run with a frozen tail, the tied head included).
3. Scores it under the coordinator's 19:50Z rule: H(l*) and W*(l*; c, k) = min |C| rho both read every checkpoint of
   every valid record, so S = W*/H >= 1.
4. Builds the console's five panels (note:dashboard-20261102T1934Z-draft-training-plots-contract, columns per
   note:training-20261102T1938Z-draft-training-plots-contract-reply), draws each as a PNG from the same rows, writes them to
   /opt/cursor/artifacts/te-panels/, publishes each plot (PNG + panel JSON) and the tables to the evidence store, PUTs the
   panels, and posts one DIGEST line with the art ids.
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import scorer as S

HERE = Path(__file__).resolve().parent
RESCORE_WT = Path(os.environ.get("TE_RESCORE_WT", "/tmp/scorer/rescore"))
RESCORE_PATH = "benchmarks/covert_witness/analysis/rescore_v1.py"
RESCORE_REFS = ("cursor/te-rescore-v1-4207", "cursor/training-efficiency-4207")
HARNESS_WT = Path(os.environ.get("TE_HARNESS_WT", "/tmp/scorer/harness-head"))
PANELS_DIR = Path(os.environ.get("TE_PANELS_DIR", "/opt/cursor/artifacts/te-panels"))
PANELS_URL = "https://website-docs-sage.vercel.app/api/panels"
READINGS = {"spec": "evenly spread", "witness": "frozen tail"}
REF_TARGET = 0.01
NIGHT2 = "2026-11-02T19:15Z"
BINS = (("<=10kB", 8e4), ("<=100kB", 8e5), ("<=1MB", 8e6), ("<=10MB", 8e7), ("<=100MB", 8e8), ("full", math.inf))
FAMILY_COLORS = {"full": "black", "data": "dimgray", "partial": "tab:blue", "adapter": "tab:green",
                 "update-code": "tab:purple", "state-compressed": "tab:brown", "activation-code": "tab:cyan",
                 "seed": "magenta"}


def fmt(x) -> str:
    if x is None:
        return "–"
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-2:
        return f"{x:.2e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str, cwd: Path = RESCORE_WT) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True).stdout.strip()


def rescore_head() -> str:
    """Check out, detached, the newest commit touching the re-score on either branch; returns it."""
    for ref in RESCORE_REFS:
        subprocess.run(["git", "-C", str(RESCORE_WT), "fetch", "-q", "origin", ref], check=False, capture_output=True)
    commit = git("log", "-1", "--format=%H", *(f"origin/{r}" for r in RESCORE_REFS), "--", RESCORE_PATH)
    git("checkout", "-q", "--detach", commit)
    return commit


def ledger(out: Path) -> int:
    """STATE's ingested checkpoints as rescore_v1's ledger rows (run_id, macs, macs_log)."""
    commit = os.environ.get("TE_CURRENT_COMMIT", "df4acd427d3151ec8d1c955f9b3e93ff38de40f4")
    rows = 0
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "index.jsonl.tmp"
    with tmp.open("w") as f:
        for p in sorted((S.STATE / "ingest" / commit[:12]).glob("*/records/scorer-ingest/*.json")):
            r = json.loads(p.read_text())
            if r.get("run_id") and r.get("macs") is not None and r.get("macs_log"):
                f.write(json.dumps({k: r.get(k) for k in ("run_id", "macs", "macs_log", "fraction", "checkpoint_step")})
                        + "\n")
                rows += 1
    tmp.rename(out / "index.jsonl")
    return rows


def load_rescore():
    spec = importlib.util.spec_from_file_location("rescore_v1", RESCORE_WT / RESCORE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


POLICY = """
import json
from te_core import policy as P
out = {}
for name, shape in P.starts().items():
    c0, how = P.c0(shape)
    out[name] = {"parameters": shape["parameters"], "tokens_per_param": shape.get("tokens_per_param"),
                 "c0_gates": c0, "c0_counted": not how.startswith("stub"), "c0_how": how,
                 "marks": {m: {"c": v["c"], "k": v["k"]} for m, v in P.shapes(shape).items()}}
print(json.dumps(out))
"""


def policy_marks() -> dict:
    """Each start's shape (parameters, tokens per parameter, C0) and te_core.policy's marked blocks, in gates and bits,
    from the harness head's te_core (a process of its own: scorer.py has imported the pinned one)."""
    subprocess.run(["git", "-C", str(HARNESS_WT), "fetch", "-q", "origin", "cursor/covert-witness-4207"], check=False,
                   capture_output=True)
    git("checkout", "-q", "--detach", "origin/cursor/covert-witness-4207", cwd=HARNESS_WT)
    r = subprocess.run([sys.executable, "-c", POLICY], cwd=HARNESS_WT / "benchmarks" / "covert_witness",
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def family(r: dict) -> str:
    mech, fmt, opt, d = r["mechanism"] or "", r["E_format"] or "", r["optimizer"], r["direction"]
    if opt == "seed" or fmt == "seedlog":
        return "seed"
    if "act" in mech.split("-")[0]:
        return "activation-code"
    if fmt == "quant" or "qat" in mech or "4bit" in mech or "signrows" in mech:
        return "update-code"
    if opt == "galore" or d == "lowrank-restart":
        return "state-compressed"
    if fmt.startswith("lora"):
        return "adapter"
    if d == "ref-data" and "window" in mech:
        return "data"
    if d in ("ref-optim", "ref-data") and not mech.startswith("top"):
        return "full"
    return "partial"


def label(start: str, shape: dict) -> str:
    n = shape["parameters"]
    return f"{n / 1e6:.0f}M · {shape['tokens_per_param']:g} tok/p ({start})"


class Cycle:
    def __init__(self, rv, payload: dict, shapes: dict, launched: dict[str, str]):
        self.rv, self.shapes, self.launched = rv, shapes, launched
        self.records, self.cps = payload["records"], payload["checkpoints"]
        night = json.loads((S.STATE / "score.json").read_text())["starts"]
        self.l0 = {s: v["start_bpb"] for s, v in night.items()}
        self.starts = [s for s in self.l0 if s in shapes]
        self.ref_s, self.ref_k = rv.REFERENCE[0], rv.GRID_K[rv.REFERENCE[1]]
        self.w = {s: next(r["w"] for r in self.records.values() if r["start"] == s) for s in self.starts}
        self.grid = [(c, kn, k) for c in rv.GRID_C for kn, k in rv.GRID_K.items()]

    def rho(self, c: dict, s: float, k: float, mode: str) -> float:
        return float(self.rv.rho_cp(c, self.records[c["id"]], s, k, mode))

    def status(self, c: dict, s: float, k: float, mode: str) -> str:
        return self.rv.status(c, self.records[c["id"]], s, k, mode)

    def hit(self, start: str, target: float) -> list[dict]:
        ls = self.l0[start] * (1 - target)
        return [c for c in self.cps if c["start"] == start and c["bpb"] <= ls]

    def best(self, hit: list[dict], s: float, k: float, mode: str) -> dict | None:
        """W* over `hit` at (s, k): the checkpoint with the least |C| rho."""
        if not hit:
            return None
        w, c = min(((c["gates"] * self.rho(c, s, k, mode), c) for c in hit), key=lambda t: t[0])
        return {"W": w, "c": c, "rho": w / c["gates"], "status": self.status(c, s, k, mode)}

    def scores(self) -> dict:
        out = {}
        for start in self.starts:
            for t in self.rv.TARGETS:
                hit = self.hit(start, t)
                h = min(hit, key=lambda c: c["gates"]) if hit else None
                cell = {"l_star": self.l0[start] * (1 - t), "reached": len(hit), "H": h and self.cp_ref(h), "grid": {},
                        "marks": {}}
                for mode in READINGS:
                    for cw, kn, k in self.grid:
                        b = self.best(hit, cw, k, mode)
                        if b:
                            cell["grid"][f"{mode}|{cw}|{kn}"] = self.cell(b, h)
                    for m, v in self.shapes[start]["marks"].items():
                        b = self.best(hit, v["c"] / self.w[start], v["k"], mode)
                        if b:
                            cell["marks"][f"{mode}|{m}"] = self.cell(b, h) | {"c_gates": v["c"], "k_bits": v["k"],
                                                                              "s": v["c"] / self.w[start]}
                out[f"{start}|{t}"] = cell
        return out

    def cp_ref(self, c: dict) -> dict:
        r = self.records[c["id"]]
        return {"gates": c["gates"], "id": c["id"], "cp": c["cp"], "bpb": c["bpb"], "direction": r["direction"],
                "mechanism": r["mechanism"], "night2": self.night2(c["id"])}

    def cell(self, b: dict, h: dict) -> dict:
        r = self.records[b["c"]["id"]]
        return {"W*": b["W"], "S": b["W"] / h["gates"], "rho": b["rho"], "p": b["c"]["gates"] / h["gates"],
                "id": b["c"]["id"], "cp": b["c"]["cp"], "direction": r["direction"], "mechanism": r["mechanism"],
                "family": family(r), "status": b["status"], "delta": b["c"]["delta"], "delta_grad": r["delta_grad"]}

    def night2(self, rid: str) -> bool:
        return self.launched.get(rid, "") >= NIGHT2

    def per_record(self) -> dict:
        """Per checkpoint: Delta = u(1), Delta_grad, phi, and on the grid Fit (running, training) and rho by reading."""
        keys = [f"{cw}|{kn}" for cw, kn, _ in self.grid]
        cps = []
        for c in self.cps:
            r = self.records[c["id"]]
            g = {"fit_run": [], "fit_train": [], "rho_spec": [], "rho_witness": []}
            for cw, _, k in self.grid:
                run = c["delta"] + cw * self.rv.TAU <= k
                g["fit_run"].append(run)
                g["fit_train"].append(run and 2 * r["delta_grad"] + cw * self.rv.TAU <= k)
                for mode in READINGS:
                    g[f"rho_{mode}"].append(round(self.rho(c, cw, k, mode), 6))
            cps.append({"id": c["id"], "start": c["start"], "cp": c["cp"], "schedule_end": c["schedule_end"],
                        "gates": c["gates"], "bpb": c["bpb"], "delta_bits": c["delta"], "u1_bits": c["delta"],
                        "delta_grad_bits": r["delta_grad"], "phi": c["phi"], **g})
        recs = {rid: {k: v for k, v in r.items() if k != "note"} | {"family": family(r), "night2": self.night2(rid),
                                                                   "note": r["note"]}
                for rid, r in self.records.items()}
        return {"grid_keys": keys, "grid_c_over_w": list(self.rv.GRID_C), "grid_k_bits": self.rv.GRID_K,
                "tau_bits": self.rv.TAU, "u1": "u(1) = Delta, the best-known value from the record (an upper bound)",
                "records": recs, "checkpoints": cps}


def pareto(points):
    """(x, y, payload) sorted by x with y strictly decreasing: the best y reachable at each budget x."""
    out, best = [], math.inf
    for x, y, p in sorted(points, key=lambda t: (t[0], t[1])):
        if y < best:
            out.append((x, y, p))
            best = y
    return out


def deepest(l0: float, bpb: float, targets) -> float | None:
    hit = [t for t in targets if bpb <= l0 * (1 - t)]
    return -100 * max(hit) if hit else None


def panels(cy: Cycle, sc: dict, stamp: str, source: str) -> dict:
    rv, P = cy.rv, {}
    refpol = f"c={rv.REFERENCE[0]}w k={rv.REFERENCE[1]}"
    common = {"updated_at": stamp, "source": source, "kind": "table"}
    lab = {s: label(s, cy.shapes[s]) for s in cy.starts}
    size = {s: cy.shapes[s]["parameters"] for s in cy.starts}
    sat = {s: cy.shapes[s]["tokens_per_param"] for s in cy.starts}
    c0 = {s: cy.shapes[s]["c0_gates"] if cy.shapes[s]["c0_counted"] else None for s in cy.starts}

    def meth(r):
        return f"{r['direction']}/{r['mechanism']}"

    cols = ["start", "start_id", "size", "saturation", "curve", "policy", "reading", "basis", "target_pct",
            "compute_gates", "compute_frac", "loss", "loss_lo", "loss_hi", "record", "checkpoint", "method", "family"]
    rows = []
    for s in cy.starts:
        mine = [c for c in cy.cps if c["start"] == s]

        def emit(curve, policy, reading, pts, st=None, s=s):
            for x, y, c in pts:
                r = cy.records[c["id"]]
                rows.append([lab[s], s, size[s], sat[s], curve, policy, reading,
                             st(c) if st else None, deepest(cy.l0[s], y, rv.TARGETS), x,
                             x / c0[s] if c0[s] else None, y, None, None, c["id"], c["cp"], meth(r), family(r)])

        emit("honest", "", "", pareto([(c["gates"], c["bpb"], c) for c in mine]))
        for mode, reading in READINGS.items():
            for cw, kn, k in cy.grid:
                pol = f"c={cw}w k={kn}"
                pts = pareto([(c["gates"] * cy.rho(c, cw, k, mode), c["bpb"], c) for c in mine])
                emit("covert" if pol == refpol else "covert-policy", pol, reading, pts,
                     lambda c, cw=cw, k=k, mode=mode: cy.status(c, cw, k, mode))
    P["p1-loss-vs-compute"] = common | {
        "id": "training/p1-loss-vs-compute", "title": "Loss against compute spent after the start (every valid v2.3 "
        "checkpoint)", "columns": cols, "rows": rows,
        "note": "Each curve is the least compute at which a loss is reached (its Pareto points), over every checkpoint "
                "of every valid v2.3 record (the 19:50Z rule). honest = |C| (Boolean gates); covert = |C| rho at the "
                f"reference policy {refpol}; covert-policy = the other (c, k) of the grid (c in tokens per block, c/w). "
                "basis: fit-train = rho 1 by Fit for the training; fit-run = only running the updated model fits; "
                "unpriced = the Slowdown theorem's bound is 1 without Fit; priced = its bound is above 1 (a target: stated, "
                "not proved in Lean, with the best-known u and a, upper bounds). reading: evenly spread or frozen tail "
                "(the tied head is a frozen tail for every run that leaves the embedding frozen). loss = held-out bits "
                "per byte. target_pct = the deepest target the point reaches. compute_frac is null: C0 is a stub for "
                "these starts (" + ", ".join(f"{s} {cy.shapes[s]['c0_gates']:.3g}" for s in cy.starts) + " gates, "
                "te_core.policy). No scratch curve until the ladder's F. One seed (search slices)."}

    cols = ["start", "start_id", "size", "saturation", "l0", "target_pct", "g", "S", "S_lo", "S_hi", "basis",
            "reading", "policy", "method", "family", "record", "checkpoint", "H_gates", "W_gates", "rho", "p"]
    rows = []
    for s in cy.starts:
        for t in rv.TARGETS:
            cell = sc[f"{s}|{t}"]
            for mode, reading in READINGS.items():
                q = cell["grid"].get(f"{mode}|{rv.REFERENCE[0]}|{rv.REFERENCE[1]}")
                rows.append([lab[s], s, size[s], sat[s], cy.l0[s], -100 * t, None, q and q["S"], None, None,
                             q and q["status"], reading, refpol, q and f"{q['direction']}/{q['mechanism']}",
                             q and q["family"], q and q["id"], q and q["cp"], cell["H"] and cell["H"]["gates"],
                             q and q["W*"], q and q["rho"], q and q["p"]])
    P["p2-slowdown-vs-gain"] = common | {
        "id": "training/p2-slowdown-vs-gain", "title": "Slowdown S against the target (g waits for the ladder's F)",
        "columns": cols, "rows": rows,
        "note": f"S = W*/H at the reference policy {refpol}, the best known attack over every checkpoint of every valid "
                "v2.3 record; H and W* read every checkpoint, so S >= 1. g = F(l*)/C0 is null until the ladder's "
                "from-scratch curve F: plot S against target_pct. basis as in P1. One seed: S_lo and S_hi null."}

    cols = ["start", "start_id", "target_pct", "g", "reading", "c_tokens_per_block", "c_gates", "k_bits", "k_bytes",
            "S", "basis", "fit_train", "route", "method", "record", "checkpoint", "rho", "p", "mark"]
    rows = []
    for s in cy.starts:
        for t in rv.TARGETS:
            cell = sc[f"{s}|{t}"]
            for mode, reading in READINGS.items():
                for cw, kn, k in cy.grid:
                    q = cell["grid"].get(f"{mode}|{cw}|{kn}")
                    mark = "reference" if (cw, kn) == tuple(rv.REFERENCE) else ""
                    rows.append([lab[s], s, -100 * t, None, reading, cw, cw * cy.w[s], k, k / 8, q and q["S"],
                                 q and q["status"], q and q["status"] == "fit-train", q and q["direction"],
                                 q and f"{q['direction']}/{q['mechanism']}", q and q["id"], q and q["cp"],
                                 q and q["rho"], q and q["p"], mark])
                for m in ("inference", "server"):
                    q = cell["marks"].get(f"{mode}|{m}")
                    v = cy.shapes[s]["marks"][m]
                    rows.append([lab[s], s, -100 * t, None, reading, v["c"] / cy.w[s], v["c"], v["k"], v["k"] / 8,
                                 q and q["S"], q and q["status"], q and q["status"] == "fit-train",
                                 q and q["direction"], q and f"{q['direction']}/{q['mechanism']}", q and q["id"],
                                 q and q["cp"], q and q["rho"], q and q["p"], f"{m} block"])
    P["p3-policy-plane"] = common | {
        "id": "training/p3-policy-plane", "title": "Slowdown S over the policy (c, k)", "columns": cols, "rows": rows,
        "note": "S = W*/H at each (c, k) of the grid and at the start's inference and server blocks (te_core.policy: one "
                "token's forward at full context with its K, V; the start's training in one block, C0 a stub, with its "
                "state beyond theta). c_tokens_per_block = c/w, w the start's forward gates per token; k in bits. "
                "fit_train: Fit holds for the training (2 Delta_grad + s tau <= k as well as Delta + s tau <= k). The "
                "reference target is -1%; every target is listed. basis as in P1. One seed."}

    cols = ["start", "start_id", "target_pct", "algorithm", "family", "route", "record", "checkpoint", "p", "s", "S",
            "basis", "fit_train", "reading", "g", "seeds", "delta_bits", "delta_grad_bits", "compute_gates"]
    rows = []
    for s in cy.starts:
        for t in rv.TARGETS:
            h = sc[f"{s}|{t}"]["H"]
            hit = cy.hit(s, t)
            by = defaultdict(list)
            for c in hit:
                by[c["id"]].append(c)
            for mode, reading in READINGS.items():
                for rid, cs in by.items():
                    b = cy.best(cs, cy.ref_s, cy.ref_k, mode)
                    r = cy.records[rid]
                    rows.append([lab[s], s, -100 * t, meth(r), family(r), r["direction"], rid, b["c"]["cp"],
                                 b["c"]["gates"] / h["gates"], b["rho"], b["W"] / h["gates"], b["status"],
                                 b["status"] == "fit-train", reading, None, 1, b["c"]["delta"], r["delta_grad"],
                                 b["c"]["gates"]])
    P["p4-decomposition"] = common | {
        "id": "training/p4-decomposition", "title": "p against s per record at the reference policy",
        "columns": cols, "rows": rows,
        "note": f"One point per valid v2.3 record reaching the target: its checkpoint with the least |C| rho at {refpol}; "
                "p = |C|/H, s = rho (the Slowdown theorem's bound, floored at 1), S = p s. H over every checkpoint. "
                "basis as in P1. g null until the ladder's F. One seed per point."}

    cols = ["start", "start_id", "size", "saturation", "target_pct", "axis", "delta_bin", "delta_bits",
            "delta_grad_bits", "compute_gates", "method", "family", "route", "record", "checkpoint", "night2",
            "fit_run_k_bits", "fit_train_k_bits"]
    rows = []
    sref_tau = cy.ref_s * rv.TAU
    for s in cy.starts:
        for t in rv.TARGETS:
            hit = cy.hit(s, t)
            for axis in ("delta", "delta_grad"):
                def x_of(c, axis=axis):
                    return c["delta"] if axis == "delta" else cy.records[c["id"]]["delta_grad"]
                pts = pareto([(x_of(c), c["gates"], c) for c in hit])
                for _, g, c in pts:
                    r = cy.records[c["id"]]
                    rows.append([lab[s], s, size[s], sat[s], -100 * t, axis,
                                 next(n for n, b in BINS if c["delta"] <= b), c["delta"], r["delta_grad"], g, meth(r),
                                 family(r), r["direction"], c["id"], c["cp"], cy.night2(c["id"]),
                                 c["delta"] + sref_tau, max(c["delta"], 2 * r["delta_grad"]) + sref_tau])
    P["a2-efficiency-floor"] = common | {
        "id": "training/a2-efficiency-floor", "title": "Efficiency floor: least compute to each target against the "
        "update's size", "columns": cols, "rows": rows,
        "note": "Per start and target, the Pareto points of (update size, least compute to the target) over every "
                "checkpoint of every valid v2.3 record: axis delta = the total update Delta (Fit's quantity; also u(1), "
                "the best-known value, an upper bound), axis delta_grad = the gradient and optimizer state passed between "
                "blocks (from the optimizer's widths). compute in Boolean gates. fit_run_k_bits = Delta + s tau and "
                "fit_train_k_bits = max(Delta, 2 Delta_grad) + s tau at the reference c (s = 4096 tokens, tau = 16 "
                "bits): the least k at which Fit passes. One seed (search slices)."}
    return P


def frame(p: dict):
    ix = {c: i for i, c in enumerate(p["columns"])}
    return [{c: r[i] for c, i in ix.items()} for r in p["rows"]]


def plots(cy: Cycle, P: dict, sc: dict, out: Path) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rv, paths = cy.rv, {}
    refpol = f"c={rv.REFERENCE[0]}w k={rv.REFERENCE[1]}"
    starts = cy.starts
    tgt_colors = {-0.5: "tab:blue", -1.0: "tab:red", -2.0: "tab:green", -5.0: "tab:purple"}

    rows = frame(P["p1-loss-vs-compute"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, s in zip(axes.flat, starts):
        mine = [r for r in rows if r["start_id"] == s]
        for pol in sorted({r["policy"] for r in mine if r["curve"] == "covert-policy"}):
            for reading in READINGS.values():
                pts = [(r["compute_gates"], r["loss"]) for r in mine if r["policy"] == pol and r["reading"] == reading]
                if pts:
                    ax.step(*zip(*pts), where="post", color="tab:red", alpha=0.12, lw=0.7)
        for curve, reading, kw in (("honest", "", {"color": "k", "lw": 1.6, "label": "H, honest"}),
                                   ("covert", "evenly spread", {"color": "tab:red", "lw": 1.4,
                                                                "label": f"W* at {refpol}, evenly spread"}),
                                   ("covert", "frozen tail", {"color": "darkorange", "lw": 1.2, "ls": "--",
                                                              "label": f"W* at {refpol}, frozen tail"})):
            pts = [(r["compute_gates"], r["loss"]) for r in mine if r["curve"] == curve and r["reading"] == reading]
            if pts:
                ax.step(*zip(*pts + [(pts[-1][0] * 30, pts[-1][1])]), where="post", **kw)
        for t in rv.TARGETS:
            ax.axhline(cy.l0[s] * (1 - t), color="0.8", lw=0.6, ls=":")
            ax.text(0.99, cy.l0[s] * (1 - t), f"−{100 * t:g}%", fontsize=7, color="0.5", va="bottom", ha="right",
                    transform=ax.get_yaxis_transform())
        ax.axhline(cy.l0[s], color="0.6", lw=0.8)
        ax.set_xscale("log")
        if mine:
            ax.set_xlim(min(r["compute_gates"] for r in mine) / 2, max(r["compute_gates"] for r in mine
                                                                      if r["curve"] != "covert-policy") * 30)
        ax.set_ylim(cy.l0[s] * 0.93, cy.l0[s] * 1.005)
        c0 = cy.shapes[s]["c0_gates"]
        sec = ax.secondary_xaxis("top", functions=(lambda x, c0=c0: x / c0, lambda f, c0=c0: f * c0))
        sec.set_xlabel("compute / C0 stub (te_core.policy)", fontsize=8)
        cell = sc[f"{s}|{REF_TARGET}"]
        q = cell["grid"].get(f"spec|{rv.REFERENCE[0]}|{rv.REFERENCE[1]}")
        ax.set_title(f"{label(s, cy.shapes[s])}, start {cy.l0[s]:.4f} bpb\n−1%: H {fmt(cell['H']['gates'])} gates, "
                     f"S {fmt(q['S'])} ({q['mechanism']}, {q['status']})" if q else label(s, cy.shapes[s]), fontsize=9)
        ax.set_xlabel("compute after θ (Boolean gates)")
        ax.set_ylabel("held-out bpb")
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("P1: loss against compute after θ, every checkpoint of every valid v2.3 record "
                 "(light red: W* at the grid's other policies)")
    paths["p1"] = out / "p1_loss_vs_compute.png"
    fig.savefig(paths["p1"], dpi=105)
    plt.close(fig)

    rows = frame(P["p2-slowdown-vs-gain"])
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    for i, s in enumerate(starts):
        for reading, mk, ls in (("evenly spread", "o", "-"), ("frozen tail", "s", "--")):
            pts = [(r["target_pct"], r["S"]) for r in rows if r["start_id"] == s and r["reading"] == reading and r["S"]]
            if pts:
                ax.plot(*zip(*pts), marker=mk, ls=ls, color=f"C{i}", label=f"{s}, {reading}")
    ax.set_yscale("log")
    ax.set_xlabel("target (% of the start's bpb)")
    ax.set_ylabel(f"S = W*/H at {refpol}")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("P2 (until the ladder's F): S against the target, every checkpoint of every valid v2.3 record")
    paths["p2"] = out / "p2_slowdown_vs_target.png"
    fig.savefig(paths["p2"], dpi=105)
    plt.close(fig)

    rows = frame(P["p3-policy-plane"])
    cs, ks = list(rv.GRID_C), list(rv.GRID_K.values())
    fig, axes = plt.subplots(2, len(starts), figsize=(4.4 * len(starts), 8.6), constrained_layout=True, squeeze=False)
    for ri, reading in enumerate(READINGS.values()):
        for ax, s in zip(axes[ri], starts):
            mine = [r for r in rows if r["start_id"] == s and r["reading"] == reading and r["target_pct"] == -100 * REF_TARGET]
            z = [[math.nan] * len(cs) for _ in ks]
            for r in mine:
                if not r["mark"].endswith("block") and r["S"]:
                    z[ks.index(r["k_bits"])][cs.index(r["c_tokens_per_block"])] = math.log10(r["S"])
            m = ax.pcolormesh(range(len(cs) + 1), range(len(ks) + 1), z, vmin=0, vmax=6, cmap="viridis")
            for yi, row in enumerate(z):
                for xi, v in enumerate(row):
                    if not math.isnan(v):
                        ax.text(xi + 0.5, yi + 0.5, fmt(10 ** v), ha="center", va="center", fontsize=7,
                                color="w" if v < 4 else "k")
            ax.scatter([cs.index(rv.REFERENCE[0]) + 0.5], [0.5 + list(rv.GRID_K).index(rv.REFERENCE[1])], marker="*",
                       s=160, facecolors="none", edgecolors="red", linewidths=1.4)
            marks = {r["mark"]: r for r in mine if r["mark"].endswith("block")}
            ax.set_xticks([x + 0.5 for x in range(len(cs))], [f"{c}w" for c in cs], fontsize=7)
            ax.set_yticks([y + 0.5 for y in range(len(ks))], list(rv.GRID_K), fontsize=7)
            ax.set_xlabel("c (tokens per block, c/w)", fontsize=8)
            ax.set_ylabel("k", fontsize=8)
            mk = "; ".join(f"{n.split()[0]} S {fmt(v['S'])}" for n, v in marks.items() if v["S"])
            ax.set_title(f"{s}, −1%, {reading}\n{mk}", fontsize=8)
        fig.colorbar(m, ax=axes[ri].tolist(), label="log10 S")
    fig.suptitle("P3: S = W*/H over (c, k) at −1% (red star: the reference policy; inference and server blocks in "
                 "the titles)")
    paths["p3"] = out / "p3_policy_plane.png"
    fig.savefig(paths["p3"], dpi=105)
    plt.close(fig)

    rows = frame(P["p4-decomposition"])
    fig, axes = plt.subplots(2, len(starts), figsize=(4.6 * len(starts), 9), constrained_layout=True, squeeze=False)
    for ri, reading in enumerate(READINGS.values()):
        for ax, s in zip(axes[ri], starts):
            mine = [r for r in rows if r["start_id"] == s and r["reading"] == reading
                    and r["target_pct"] == -100 * REF_TARGET]
            for r in mine:
                ax.scatter(r["s"], r["p"], s=90 if r["family"] == "full" else 30,
                           marker="*" if r["family"] == "full" else "o", color=FAMILY_COLORS.get(r["family"], "gray"),
                           edgecolors="red" if r["fit_train"] else "k", linewidths=1.6 if r["fit_train"] else 0.3,
                           zorder=3)
            xs = [10 ** (e / 10) for e in range(-5, 91)]
            for e in range(10):
                ax.plot(xs, [10 ** e / x for x in xs], color="0.85", lw=0.7, zorder=1)
            if mine:
                b = min(mine, key=lambda r: r["S"])
                ax.scatter(b["s"], b["p"], s=260, facecolors="none", edgecolors="red", zorder=4)
                ax.set_title(f"{s}, −1%, {reading}: S {fmt(b['S'])} ({b['algorithm']})", fontsize=8)
                ax.set_xlim(0.7, max(r["s"] for r in mine) * 3)
                ax.set_ylim(0.7, max(r["p"] for r in mine) * 3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("s = ρ at the reference policy", fontsize=8)
            ax.set_ylabel("p = |C| / H", fontsize=8)
    handles = [Line2D([], [], marker="o", ls="", color=v, label=k) for k, v in FAMILY_COLORS.items()]
    handles += [Line2D([], [], marker="o", ls="", mfc="white", mec="red", mew=1.6, label="Fit holds for training"),
                Line2D([], [], marker="o", ls="", mfc="none", mec="red", ms=13, label="sets W*")]
    fig.legend(handles=handles, loc="outside right center", fontsize=8)
    fig.suptitle(f"P4: p against s per record at −1% and {refpol} (gray: iso-S, S = p s)")
    paths["p4"] = out / "p4_decomposition.png"
    fig.savefig(paths["p4"], dpi=100)
    plt.close(fig)

    rows = frame(P["a2-efficiency-floor"])
    fig, axes = plt.subplots(2, len(starts), figsize=(4.6 * len(starts), 8.6), constrained_layout=True, squeeze=False)
    for ri, axis in enumerate(("delta", "delta_grad")):
        for ax, s in zip(axes[ri], starts):
            for t in rv.TARGETS:
                tp = -100 * t
                pts = [(r["delta_bits"] if axis == "delta" else r["delta_grad_bits"], r["compute_gates"])
                       for r in rows if r["start_id"] == s and r["axis"] == axis and r["target_pct"] == tp]
                if pts:
                    ax.step(*zip(*pts + [(3e10, pts[-1][1])]), where="post", color=tgt_colors[tp], label=f"{tp:g}%")
                    ax.scatter(*zip(*pts), s=10, color=tgt_colors[tp])
            for kn, k in rv.GRID_K.items():
                lim = k - cy.ref_s * rv.TAU if axis == "delta" else (k - cy.ref_s * rv.TAU) / 2
                if lim > 0:
                    ax.axvline(lim, color="0.75", ls="--", lw=0.7)
                    ax.text(lim * 1.05, 0.02, f"Fit k={kn}", rotation=90, fontsize=6, color="0.5", va="bottom",
                            transform=ax.get_xaxis_transform())
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e3, 3e10)
            ax.set_xlabel("Δ (bits)" if axis == "delta" else "Δ_grad (bits)", fontsize=8)
            ax.set_ylabel("least compute to the target (gates)", fontsize=8)
            ax.set_title(f"{s}, against {'Δ' if axis == 'delta' else 'Δ_grad'}", fontsize=9)
            ax.legend(fontsize=7)
    fig.suptitle("A2: least compute to each target against the update size, every checkpoint of every valid v2.3 "
                 "record (dashed: Fit's limit at s = 4096, running for Δ and training for Δ_grad)")
    paths["a2"] = out / "a2_efficiency_floor.png"
    fig.savefig(paths["a2"], dpi=105)
    plt.close(fig)
    return paths


def tables(cy: Cycle, sc: dict, rescore_md: str, meta: dict) -> str:
    rv, short = cy.rv, cy.rv.short
    L = [f"# Formalism v1 refresh {meta['stamp']}", "",
         (f"rescore_v1.py at {meta['rescore'][:10]}; {len(cy.records)} valid v2.3 records ({meta['night2']} from "
          f"night 2, launched at or after {NIGHT2}), {len(cy.cps)} checkpoints. H and W* read every checkpoint of every "
          "valid record (the coordinator's 19:50Z rule), so S = W*/H >= 1. rho is the Slowdown theorem's bound (a "
          "target, with the best-known u(1) = Delta and a = d_model x 16 bits, upper bounds) unless Fit holds for the "
          "training (rho = 1). Readings: evenly spread (`spec`) and frozen tail (`witness`). Gates are Boolean gates."),
         ""]
    L += ["## At the reference policy (c = 4096w, k = 10 kB)", "",
          ("| start | target | H (gates; record, cp) | evenly spread: S (route, ρ, p, basis) | frozen tail: S (route, "
           "ρ, p, basis) |"), "|---|---|---|---|---|"]
    for key, cell in sc.items():
        s, t = key.split("|")
        h = cell["H"]
        hs = f"{fmt(h['gates'])} ({h['mechanism']} {short(h['id'])}, {h['cp']:g}{', night 2' if h['night2'] else ''})" \
            if h else "not reached"
        vs = []
        for mode in READINGS:
            q = cell["grid"].get(f"{mode}|{rv.REFERENCE[0]}|{rv.REFERENCE[1]}")
            vs.append(f"{fmt(q['S'])} ({q['mechanism']} {short(q['id'])}, ρ {fmt(q['rho'])}, p {fmt(q['p'])}, "
                      f"{q['status']})" if q else "–")
        L.append(f"| {s} | −{100 * float(t):g}% | {hs} | " + " | ".join(vs) + " |")
    L.append("")
    for mode, reading in READINGS.items():
        for title, axis in ((f"S at c = 4096w, by k ({reading})", "k"), (f"S at k = 10 kB, by c ({reading})", "c")):
            cols = list(rv.GRID_K) if axis == "k" else [f"{c}w" for c in rv.GRID_C]
            L += [f"## {title}", "", "| start | target | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
            for key, cell in sc.items():
                s, t = key.split("|")
                vals = []
                for col in (rv.GRID_K if axis == "k" else rv.GRID_C):
                    q = cell["grid"].get(f"{mode}|{4096 if axis == 'k' else col}|{col if axis == 'k' else '10kB'}")
                    vals.append(f"{fmt(q['S'])} {q['mechanism']} [{q['status']}]" if q else "–")
                L.append(f"| {s} | −{100 * float(t):g}% | " + " | ".join(vals) + " |")
            L.append("")
    L += ["## Marked blocks (te_core.policy), S by reading", "", "| start | target | inference block | server block |",
          "|---|---|---|---|"]
    for key, cell in sc.items():
        s, t = key.split("|")
        vals = []
        for m in ("inference", "server"):
            vals.append(" / ".join(f"{fmt(q['S'])} {q['mechanism']} [{q['status']}]" if (q := cell["marks"].get(
                f"{mode}|{m}")) else "–" for mode in READINGS))
        L.append(f"| {s} | −{100 * float(t):g}% | " + " | ".join(vals) + " |")
    L.append("")
    L += ["## A2: least compute (gates) to each target per Δ bin, every checkpoint", "",
          "| start | target | " + " | ".join(n for n, _ in BINS) + " |", "|---|---|" + "---|" * len(BINS)]
    for key in sc:
        s, t = key.split("|")
        hit = cy.hit(s, float(t))
        vals = []
        for _, b in BINS:
            ok = [c for c in hit if c["delta"] <= b]
            c = min(ok, key=lambda c: c["gates"]) if ok else None
            vals.append(f"{fmt(c['gates'])} {cy.records[c['id']]['mechanism']} (Δ {fmt(c['delta'])}, Δ_grad "
                        f"{fmt(cy.records[c['id']]['delta_grad'])})" if c else "none")
        L.append(f"| {s} | −{100 * float(t):g}% | " + " | ".join(vals) + " |")
    L.append("")
    per_record = rescore_md.split("\n### S = W*/H")[0]
    L += ["## Per record, from rescore_v1.py (independent of H)", "", per_record.replace("### ", "#### "), ""]
    return "\n".join(L)


def put_data(tree: Path, kind: str, meta: dict) -> str:
    r = subprocess.run(["research", "data", "put", "--kind", kind, "--tree", str(tree), "--preserve", "--json",
                        "--meta", json.dumps(meta)], capture_output=True, text=True, check=False)
    if r.returncode:
        raise RuntimeError(f"data put {tree}: {r.stderr[-400:]}")
    return json.loads(r.stdout)["id"]


def put_panel(panel: dict) -> str:
    import urllib.error
    import urllib.request

    from research.slack import oidc_token
    req = urllib.request.Request(f"{PANELS_URL}/{panel['id']}", json.dumps(panel).encode(), method="PUT",
                                 headers={"Authorization": f"Bearer {oidc_token(PANELS_URL)}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return f"{r.status}"
    except urllib.error.HTTPError as e:
        return f"{e.code} {e.read()[:400].decode(errors='replace')}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="default /tmp/scorer/v1/<stamp>")
    ap.add_argument("--gates", type=Path, default=Path("/tmp/scorer/v1/trial/gates.json"))
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--no-put", action="store_true")
    ap.add_argument("--no-post", action="store_true")
    args = ap.parse_args()
    stamp = now()
    out = args.out or Path("/tmp/scorer/v1") / stamp.replace(":", "")
    out.mkdir(parents=True, exist_ok=True)
    led = ledger(Path("/tmp/scorer/v1/ledger"))
    commit = rescore_head()
    r = subprocess.run([sys.executable, str(RESCORE_WT / RESCORE_PATH), "--scorer", str(S.STATE), "--ledger",
                        "/tmp/scorer/v1/ledger", "--out", str(out / "rescore"), "--gates", str(args.gates)],
                       capture_output=True, text=True, check=False)
    if r.returncode:
        print(r.stderr[-2000:], file=sys.stderr)
        return 1
    rv = load_rescore()
    payload = json.loads((out / "rescore" / "rescore_v1.json").read_text())
    unjoined = sum(c["source"] != "ledger" for c in payload["checkpoints"])
    launched = {}
    for line in (S.STATE / "index.jsonl").open():
        x = json.loads(line)
        if x.get("source") == "launch.py":
            launched[x["id"]] = x.get("launched") or ""
    cy = Cycle(rv, payload, policy_marks(), launched)
    sc = cy.scores()
    n2 = sum(cy.night2(i) for i in cy.records)
    meta = {"stamp": stamp, "rescore": commit, "night2": n2, "records": len(cy.records), "checkpoints": len(cy.cps),
            "ledger_rows": led, "unjoined": unjoined}
    plots_dir = out / "plots"
    plots_dir.mkdir(exist_ok=True)
    trees = {}
    for name in ("tables", "p1", "p2", "p3", "p4", "a2"):
        trees[name] = out / "publish" / name
        trees[name].mkdir(parents=True, exist_ok=True)
    (trees["tables"] / "tables.md").write_text(tables(cy, sc, (out / "rescore" / "tables.md").read_text(), meta))
    (trees["tables"] / "scores.json").write_text(json.dumps(meta | {"rule": "H and W* over every checkpoint (19:50Z)",
                                                                    "scores": sc}, indent=1, default=float) + "\n")
    (trees["tables"] / "records_v1.json").write_text(json.dumps(cy.per_record(), default=float) + "\n")
    (trees["tables"] / "rescore_v1.json").write_text((out / "rescore" / "rescore_v1.json").read_text())
    ids = {}
    if not args.no_publish:
        ids["tables"] = put_data(trees["tables"], "evidence/v1", {"what": "te formalism v1 refresh: tables, scores, "
                                                                          "per-record grid", **meta})
    source = (f"{ids.get('tables', 'unpublished')} (scorer v1cycle {stamp}; rescore_v1.py {commit[:10]}; "
              f"{len(cy.records)} valid v2.3 records, {len(cy.cps)} checkpoints)")
    P = panels(cy, sc, stamp, source)
    paths = plots(cy, P, sc, plots_dir)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    plot_of = {"p1": "p1-loss-vs-compute", "p2": "p2-slowdown-vs-gain", "p3": "p3-policy-plane",
               "p4": "p4-decomposition", "a2": "a2-efficiency-floor"}
    for key, pid in plot_of.items():
        (trees[key] / f"{pid}.json").write_text(json.dumps(P[pid], default=float) + "\n")
        (trees[key] / paths[key].name).write_bytes(paths[key].read_bytes())
        (PANELS_DIR / f"{pid}.json").write_text(json.dumps(P[pid], default=float) + "\n")
        (PANELS_DIR / paths[key].name).write_bytes(paths[key].read_bytes())
        if not args.no_publish:
            ids[key] = put_data(trees[key], "evidence/v1", {"what": f"te formalism v1 {pid}", **meta})
    puts = {}
    if not args.no_put:
        for pid in plot_of.values():
            puts[pid] = put_panel(json.loads(json.dumps(P[pid], default=float)))
    summary = meta | {"art": ids, "panels_put": puts, "rows": {k: len(v["rows"]) for k, v in P.items()},
                      "out": str(out)}
    (out / "cycle.json").write_text(json.dumps(summary, indent=1) + "\n")
    line = digest(cy, sc, meta, ids, puts)
    (out / "digest.txt").write_text(line + "\n")
    print(json.dumps(summary))
    print(line)
    if not args.no_post and not args.no_publish:
        subprocess.run([sys.executable, str(HERE / "scorer.py"), "post", "DIGEST", line], check=True)
    return 0


def digest(cy: Cycle, sc: dict, meta: dict, ids: dict, puts: dict) -> str:
    rv, short = cy.rv, cy.rv.short
    ref = f"{rv.REFERENCE[0]}|{rv.REFERENCE[1]}"
    parts = []
    for s in ("s100", "m100", "s50", "m50"):
        if s not in cy.starts:
            continue
        cell = sc[f"{s}|{REF_TARGET}"]
        h = cell["H"]
        if not h:
            parts.append(f"{s} not reached")
            continue
        q = {m: cell["grid"].get(f"{m}|{ref}") for m in READINGS}
        hit = cy.hit(s, REF_TARGET)
        floor = []
        for n, b in BINS[:3]:
            ok = [c for c in hit if c["delta"] <= b]
            floor.append(f"{n} {fmt(min(c['gates'] for c in ok))}" if ok else f"{n} none")
        parts.append(f"{s}: H {fmt(h['gates'])} ({h['mechanism']} {short(h['id'])}{', night 2' if h['night2'] else ''}); "
                     f"S {fmt(q['spec']['S'])} evenly spread ({q['spec']['mechanism']} {short(q['spec']['id'])}, ρ "
                     f"{fmt(q['spec']['rho'])}, {q['spec']['status']}) / {fmt(q['witness']['S'])} frozen tail "
                     f"({q['witness']['mechanism']}, ρ {fmt(q['witness']['rho'])}); A2 least gates at "
                     + ", ".join(floor))
    arts = ", ".join(f"{k} {v[:16]}" for k, v in ids.items())
    ok = sum(str(v).startswith("2") for v in puts.values())
    return (f"v1 refresh (rescore_v1.py {meta['rescore'][:10]}; {meta['records']} valid v2.3 records, {meta['night2']} "
            f"from night 2, {meta['checkpoints']} checkpoints; H and W* over every checkpoint, so S >= 1; rho is the "
            "Slowdown theorem's target unless Fit holds for training) at −1% and the reference (4096w, 10 kB): "
            + "; ".join(parts) + f". art: {arts}. Console panels PUT {ok}/{len(puts)} (p1-p4, a2).")


if __name__ == "__main__":
    sys.exit(main())

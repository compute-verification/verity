"""Night 2's formalism-v1 refresh (note:training-20261102T1817Z-report-training-formalism-v1), every 90 minutes.

    python v1cycle.py [--out DIR] [--no-publish] [--no-put] [--no-post]
    python v1cycle.py --reput OUT

The score is te_core's alone (coordinator 20:15Z: one codebase, `te_core.rescore` with its readings). One refresh:

1. Checks out, detached in RESCORE_WT, the newest commit touching te_core or analysis/rescore_v1.py on
   cursor/covert-witness-4207 or cursor/training-efficiency-4207.
2. Runs its `python -m te_core.rescore` on the scorer's ledger at tonight's pin (PINS): per checkpoint of every valid
   v2.3 record, rescore_v1.py's quantities (Delta, u(1), Delta_grad, phi, the frozen tail's share t), Fit (running and
   training) and the Slowdown theorem's rho (a target) on te_core.policy's (c, k) grid with its frozen tail charged
   (`--reading tail-charged`, the one reading: Daniel's ruling, coordinator 22:00Z), H and W* over every checkpoint of
   every valid record (19:50Z), S = W*/H; frontier.md/json, formalism.jsonl and the console's five panels with their
   PNGs. Then v1runs.py gives A2 and P4 one row per valid run and target from te_core's same rows (coordinator 20:52Z),
   which replace te_core's staircase and per-algorithm rows in those two panels (kept beside them as
   te_core-<plot>.json), and Daniel's headline panel, frontier-branches (coordinator 22:30Z).
3. Publishes the tables and each panel (its JSON and PNG) to the evidence store, writes each panel's console cut (the
   route's body and note limits, `console`) to /opt/cursor/artifacts/te-panels/, PUTs them, and posts one DIGEST line
   with the art ids. `--reput OUT` re-cuts a finished refresh's panels and PUTs them, with nothing re-scored.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import scorer as S

HERE = Path(__file__).resolve().parent
RESCORE_WT = Path(os.environ.get("TE_RESCORE_WT", "/tmp/scorer/rescore"))
RESCORE_PATHS = ("benchmarks/covert_witness/te_core", "benchmarks/covert_witness/analysis/rescore_v1.py")
RESCORE_REFS = ("cursor/covert-witness-4207", "cursor/training-efficiency-4207")
PINS = os.environ.get("TE_V1_PINS", "11d96c1b8745d8093746ea11e3e33fe6f07f7d3b,"
                                    "36aac280a62be7a04cbf510b50636b5ad875ba54").split(",")
PANELS_DIR = Path(os.environ.get("TE_PANELS_DIR", "/opt/cursor/artifacts/te-panels"))
PANELS_URL = "https://website-docs-sage.vercel.app/api/panels"
PLOT_OF = {"p1": "p1-loss-vs-compute", "p2": "p2-slowdown-vs-gain", "p3": "p3-policy-plane", "p4": "p4-decomposition",
           "a2": "a2-efficiency-floor", "fb": "frontier-branches"}
FB = "frontier-branches"
READINGS = ("tail-charged",)
REFERENCE = "c=4096w k=10kB"
REF_TARGET = -1.0
NIGHT2 = "2026-11-02T19:15:00Z"
BINS = (("<=10kB", 8e4), ("<=100kB", 8e5), ("<=1MB", 8e6))
BODY_MAX = 65536
NOTE_MAX = 500
COMMON = ("Reading: tail-charged, the frozen tail's work charged (Daniel, 22:00Z). basis: fit = rho 1 by Fit for the "
          "training; target = Slowdown's bound, unproved. Delta, u(1), t, a: witness upper bounds; tau a stub. One "
          "seed. Full panels: evidence store.")
A2_COMMON = "Delta: a witness upper bound. One seed. Full panels: evidence store."
PER_RUN = "Each valid v2.3 run's least-compute checkpoint per target (19:50Z); variant: its launch note's label."
SHORT = {
    "p1-loss-vs-compute": "Loss (val bpb) against compute after theta, every checkpoint of every valid v2.3 record: "
                          "honest |C| (gates), covert |C| rho at c=4096w k=10kB, covert-policy at other (c, k).",
    "p2-slowdown-vs-gain": "S = W*/H at c=4096w k=10kB per start and target; H and W* over every checkpoint of every "
                           "valid v2.3 record, so S >= 1. g null until the ladder's F.",
    "p3-policy-plane": "S = W*/H over te_core.policy's (c, k) grid and each start's marked shapes, k in bits; H and W* "
                       "over every checkpoint of every valid v2.3 record.",
    "p4-decomposition": f"{PER_RUN} At c=4096w k=10kB: p = |C|/H, s = rho, S = p s.",
    "a2-efficiency-floor": f"{PER_RUN} The floor is their frontier; fit_*_k_bits: least k where Fit holds at c=4096w.",
    FB: "Headline: loss against compute staircases; frontier = the starts' points (the ladder's to come); honest after "
        "each start; throttled = |C| rho; S = W*/H; C0 a stub.",
}
BAND_FEW = ("c1e15/k1e5", "c1e15/k1e6", "c1e14/k1e4")
LEAN = ("start_id", "checkpoint", "method", "family", "reading")
RUNS_LEAN = ("start_id", "record", "checkpoint", "compute_frac", "g", "seeds", "reading")
FB_LEAN = ("start_id", "policy", "record", "checkpoint")
REF_C_TOKENS, REF_K_BITS = 4096, 80_000
SIZE_SAT = ("size", "saturation")
FIT_K = ("fit_run_k_bits", "fit_train_k_bits")
PER_RUN_PANELS = ("a2-efficiency-floor", "p4-decomposition")
FINAL_TITLE = "FINAL (sealed shard: FINAL entries' 3-seed means, other records at the same drop): "
VARIANT_LABEL = 24


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
    """Check out, detached, the newest commit touching te_core or the analysis on either branch; returns it."""
    for ref in RESCORE_REFS:
        subprocess.run(["git", "-C", str(RESCORE_WT), "fetch", "-q", "origin", ref], check=False, capture_output=True)
    commit = git("log", "-1", "--format=%H", *(f"origin/{r}" for r in RESCORE_REFS), "--", *RESCORE_PATHS)
    git("checkout", "-q", "--detach", commit)
    return commit


def rescore(out: Path, gates: Path, final: Path | None = None) -> str:
    """te_core.rescore into `out` (module docstring, 2), through v1final.py with FINAL_DIR's values for `final`, and
    v1runs.py's per-run A2 and P4 (in final mode also the search's, prefixed search-runs-); returns its last line."""
    out.mkdir(parents=True, exist_ok=True)
    if gates.exists() and not (out / "analysis-gates.json").exists():
        (out / "analysis-gates.json").write_bytes(gates.read_bytes())
    ledger = os.environ.get("TE_LEDGER", str(Path.home() / ".research" / "te-ledger"))
    args = ["--out", str(out), "--root", ledger, "--scorer", str(S.STATE), *(x for p in PINS for x in ("--pin", p))]
    if '"--reading"' in (RESCORE_WT / "benchmarks" / "covert_witness" / "te_core" / "rescore.py").read_text():
        args += [x for m in READINGS for x in ("--reading", m)]
    runs = [sys.executable, str(HERE / "v1runs.py"), *args]
    cmds = ([[sys.executable, str(HERE / "v1final.py"), str(final), *args], [*runs, "--final", str(final)],
             [*runs, "--prefix", "search-runs-"]] if final else [[sys.executable, "-m", "te_core.rescore", *args], runs])
    logs = []
    for c in cmds:
        r = subprocess.run(c, cwd=RESCORE_WT / "benchmarks" / "covert_witness", capture_output=True, text=True,
                           check=False)
        logs.append(r.stdout + r.stderr)
        (out / "rescore.log").write_text("\n".join(logs))
        if r.returncode:
            raise SystemExit(f"v1cycle: {' '.join(c[1:3])} failed (rc {r.returncode}):\n{r.stderr[-2000:]}")
    return logs[0].strip().splitlines()[-1]


def body(p: dict) -> bytes:
    return json.dumps(p, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode()


def frame(p: dict) -> list[dict]:
    ix = {c: i for i, c in enumerate(p["columns"])}
    return [{c: r[i] for c, i in ix.items()} for r in p["rows"]]


def compact(p: dict, keep=lambda r: True, drop: tuple = (), fix=None, pick=None) -> dict:
    """`p` with the rows `keep` takes, of those the ones `pick` picks (each through `fix`), without the columns `drop`,
    booleans as yes/no and numbers to 4 significant digits (losses 6)."""
    cols = [c for c in p["columns"] if c not in drop]

    def val(c, v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return round(v, 6) if c in ("loss", "loss_lo", "loss_hi", "l0") else float(f"{v:.4g}")
        return v
    ds = [d for d in frame(p) if keep(d)]
    ds = pick(ds) if pick else ds
    rows = [[val(c, d[c]) for c in cols] for d in (map(fix, ds) if fix else ds)]
    return p | {"columns": cols, "rows": rows}


def variant_label(d: dict) -> dict:
    """The row with its variant cut to the launch note's label (the text before its first colon), at most
    VARIANT_LABEL characters."""
    v = d.get("variant")
    return d | {"variant": v.split(":")[0][:VARIANT_LABEL]} if v else d


def thinned(step: float):
    """A pick: P1's curves at the points that lower the loss by `step` or reach a deeper target, each curve's ends
    kept."""
    def pick(ds: list[dict]) -> list[dict]:
        groups: dict[tuple, list] = defaultdict(list)
        for d in ds:
            groups[(d["start_id"], d["curve"], d.get("policy"), d.get("reading"))].append(d)
        out = []
        for g in groups.values():
            kept = [g[0]]
            for i, d in enumerate(g[1:], 1):
                if i == len(g) - 1 or d.get("target_pct") != kept[-1].get("target_pct") or \
                        d["loss"] <= kept[-1]["loss"] * (1 - step):
                    kept.append(d)
            out += kept
        return out
    return pick


def frontier(x: str, y: str, *by: str):
    """A pick: per start, target, reading and `by`, the rows no other row beats in both `x` and `y` (smaller is
    better, null worst), in their order."""
    inf = float("inf")

    def pick(ds: list[dict]) -> list[dict]:
        groups: dict[tuple, list] = defaultdict(list)
        for i, d in enumerate(ds):
            groups[tuple(d.get(c) for c in ("start_id", "target_pct", "reading", *by))].append(i)
        kept = []
        for ix in groups.values():
            best = inf
            for i in sorted(ix, key=lambda i: tuple(inf if ds[i][c] is None else ds[i][c] for c in (x, y))):
                if ds[i][y] is not None and ds[i][y] < best:
                    best = ds[i][y]
                    kept.append(i)
        return [ds[i] for i in sorted(kept)]
    return pick


def dial(d: dict) -> float:
    """A throttled row's distance from the reference policy, in decades of c or k (the larger; the console cut rounds
    it up to half decades)."""
    return max(abs(math.log10(d["c_tokens_per_block"] / REF_C_TOKENS)), abs(math.log10(d["k_bits"] / REF_K_BITS)))


def union(*picks):
    """A pick: the rows any of `picks` picks, in their order."""
    def pick(ds: list[dict]) -> list[dict]:
        kept = {id(d) for f in picks for d in f(ds)}
        return [d for d in ds if id(d) in kept]
    return pick


def console(P: dict, final: bool = False) -> dict:
    """Each panel cut to the route's limits (a body of BODY_MAX bytes, a note of NOTE_MAX characters): the short note,
    and the first of its cuts that fits, which the note names. Every cut keeps both readings, except A2's, whose rows
    agree under both (A2 reads no rho) and so are one row each. No cut has a `reading` column: one reading is scored
    (Daniel, 22:00Z). A2's and P4's rows grow with every run, so their last cuts keep every target and, per start and
    target, first each method's runs on its own frontier, then the panel's frontier (A2's floor, Delta against
    compute; P4's, p against s) with each method's best run, then the frontier alone. frontier-branches keeps its
    frontier and honest rows and the throttled ones at the (c, k) nearest the reference that fit."""
    def every(r):
        return True

    def ref(r):
        return r["target_pct"] == REF_TARGET

    def one(r):
        return r["reading"] == READINGS[0]

    def band(policies):
        return lambda r: r["curve"] != "covert-policy" or r["policy"] in policies

    def near(x):
        return lambda r: r["curve"] != "throttled" or math.ceil(2 * dial(r)) / 2 <= x
    out = {}
    for pid, p in P.items():
        tries = [("", every, LEAN, None, None)]
        if pid == "p1-loss-vs-compute":
            fine = thinned(0.0025)
            tries = [("", every, LEAN, None, fine),
                     (f" The band here: {', '.join(BAND_FEW)}.", band(BAND_FEW), LEAN, None, fine),
                     (f" The band here: {BAND_FEW[1]}.", band(BAND_FEW[1:2]), LEAN, None, fine),
                     (" No band here.", band(()), LEAN, None, fine),
                     (" No band; points 0.5% apart in loss.", band(()), LEAN, None, thinned(0.005)),
                     (" No band; points 1% apart in loss.", band(()), LEAN, None, thinned(0.01))]
        elif pid == "p3-policy-plane":
            tries += [(" Here -1% only.", ref, LEAN, None, None),
                      (" Here -1% only, no record column.", ref, (*LEAN, "record"), None, None),
                      (" Here -1% only, no record, g, k_bytes or p (= S/rho).", ref,
                       (*LEAN, "record", "g", "k_bytes", "p"), None, None)]
        elif pid == "a2-efficiency-floor":
            lean = (*RUNS_LEAN, *SIZE_SAT)
            floor = frontier("delta_bits", "compute_gates")
            tries = [("", one, RUNS_LEAN, variant_label, None),
                     (" No size, saturation (in start).", one, lean, variant_label, None),
                     (" No size/saturation; each method's own floor only.", one, lean, variant_label,
                      frontier("delta_bits", "compute_gates", "method")),
                     (" No size/saturation; the floor and each method's least-compute run.", one, lean, variant_label,
                      union(floor, frontier("compute_gates", "compute_gates", "method"))),
                     (" No size/saturation; the floor only.", one, lean, variant_label, floor)]
        elif pid == "p4-decomposition":
            lean = (*RUNS_LEAN, *SIZE_SAT, *FIT_K)
            front = frontier("p", "s")
            tries = [("", every, RUNS_LEAN, variant_label, None),
                     (" No fit_*_k_bits (in A2).", every, (*RUNS_LEAN, *FIT_K), variant_label, None),
                     (" No fit_*_k_bits (in A2), size, saturation (in start).", every, lean, variant_label, None),
                     (" No fit_*_k_bits, size/saturation; each method's (p, s) front only.", every, lean,
                      variant_label, frontier("p", "s", "method")),
                     (" No fit_*_k_bits, size/saturation; (p, s) front + each method's least-S run.", every, lean,
                      variant_label, union(front, frontier("S", "S", "method"))),
                     (" No fit_*_k_bits, size/saturation; the (p, s) front only.", every, lean, variant_label,
                      front)]
        elif pid == FB:
            cells = defaultdict(set)
            for d in frame(p):
                if d["curve"] == "throttled":
                    cells[math.ceil(2 * dial(d)) / 2].add((d["start_id"], d["policy"]))
            far = sorted(cells, reverse=True)
            per = {x: max(Counter(s for y in far if y <= x for s, _ in cells[y]).values()) for x in far}
            tries = [((f" Throttled within {x:g} decades of the reference (<= {per[x]} (c, k) a start), points "
                       f"{100 * step:g}% apart."), near(x), FB_LEAN, None, thinned(step))
                     for step in (0.0025, 0.01) for x in far]
        common = A2_COMMON if pid == "a2-efficiency-floor" else COMMON
        common = common.replace(" One seed.", "") if final else common
        for why, keep, drop, fix, pick in tries:
            q = compact(p, keep, drop, fix, pick)
            note = f"{SHORT[pid]}{why} {common}"
            q["note"] = note if len(note) <= NOTE_MAX else note[:NOTE_MAX - 1] + "…"
            if len(body(q)) <= BODY_MAX:
                break
        out[pid] = q
    return out


def put_data(tree: Path, kind: str, meta: dict) -> str:
    r = subprocess.run(["research", "data", "put", "--kind", kind, "--tree", str(tree), "--preserve", "--json",
                        "--meta", json.dumps(meta)], capture_output=True, text=True, check=False)
    if r.returncode:
        raise RuntimeError(f"data put {tree}: {r.stderr[-400:]}")
    return json.loads(r.stdout)["id"]


def put_panels(C: dict) -> dict[str, str]:
    """PUT each console cut with one identity token (the identity service rate-limits: HTTP 429 on a token per panel);
    each panel's status, or why it wasn't PUT."""
    import urllib.error
    import urllib.request

    from research.slack import SlackError, oidc_token
    try:
        token = oidc_token(PANELS_URL)
    except SlackError as e:
        return {pid: f"not PUT: {e}" for pid in C}
    out = {}
    for pid, panel in C.items():
        req = urllib.request.Request(f"{PANELS_URL}/{panel['id']}", body(panel), method="PUT",
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out[pid] = f"{r.status}"
        except urllib.error.HTTPError as e:
            out[pid] = f"{e.code} {e.read()[:400].decode(errors='replace')}"
        except (urllib.error.URLError, OSError) as e:
            out[pid] = f"not PUT: {e}"
    return out


def night2_runs() -> set[str]:
    out = set()
    for line in (S.STATE / "index.jsonl").open():
        x = json.loads(line)
        if x.get("source") == "launch.py" and (x.get("launched") or "") >= NIGHT2:
            out.add(x.get("run"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="default /tmp/scorer/v1/<stamp>")
    ap.add_argument("--gates", type=Path, default=Path("/tmp/scorer/v1/trial/gates.json"),
                    help="the analysis's gate factors, seeded into the re-score's cache")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--no-put", action="store_true")
    ap.add_argument("--no-post", action="store_true")
    ap.add_argument("--reput", type=Path, metavar="OUT", help="re-cut a refresh's full panels (OUT/publish) and PUT them")
    ap.add_argument("--final", type=Path, metavar="FINAL_DIR",
                    help="the final table: te_core.rescore with FINAL_DIR's values (scorer.py final table; v1final.py); "
                         "posts FINAL-TABLE, not DIGEST")
    args = ap.parse_args()
    if args.reput:
        return reput(args.reput)
    stamp, final = now(), args.final
    out = args.out or Path("/tmp/scorer/v1") / f"{'final-' if final else ''}{stamp.replace(':', '')}"
    commit = rescore_head()
    last = rescore(out / "rescore", args.gates, final)
    rs = out / "rescore"
    what = "final table" if final else "refresh"
    fr = json.loads((rs / "frontier.json").read_text())
    scored = [json.loads(line) for line in (rs / "formalism.jsonl").open()]
    runs, n2 = {r["run_id"] for r in scored}, night2_runs()
    meta = {"stamp": stamp, "te_core": commit, "pin": fr["pin"], "readings": fr["readings"],
            "checkpoints": len(scored), "records": len(runs), "night2_records": len(runs & n2),
            "not_analysed": len(fr["not_analysed"]), "rule": "H and W* over every checkpoint (19:50Z)"}
    trees = {k: out / "publish" / k for k in ("tables", *PLOT_OF)}
    for t in trees.values():
        t.mkdir(parents=True, exist_ok=True)
    for name in ("frontier.md", "frontier.json", "formalism.jsonl", "rescore.log"):
        (trees["tables"] / name).write_bytes((rs / name).read_bytes())
    fin = json.loads((rs / "final_rows.json").read_text()) if final else None
    if final:
        ents = fin["entries"]
        meta |= {"final": str(final), "entries": len(ents), "entries_matched": sum(e["matched"] > 0 for e in ents.values()),
                 "final_checkpoints": len(fin["rows"])}
        (trees["tables"] / "final.md").write_text(final_md(rs, fin, json.loads((final / "table.json").read_text())))
        (trees["tables"] / "final_rows.json").write_bytes((rs / "final_rows.json").read_bytes())
        for name in ("table.md", "table.json", "final_submissions.json", "results.jsonl", "plan.json"):
            if (final / name).exists():
                (trees["tables"] / f"scorer-final-{name}").write_bytes((final / name).read_bytes())
    (trees["tables"] / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    ids = {}
    if not args.no_publish:
        ids["tables"] = put_data(trees["tables"], "evidence/v1", {"what": f"te formalism v1 {what}: te_core.rescore's "
                                                                          "frontier and per-checkpoint blocks", **meta})
    source = (f"{ids.get('tables', 'unpublished')[:16]} (scorer v1cycle {what} {stamp}: te_core.rescore "
              f"{commit[:10]}, {len(scored)} valid v2.3 checkpoints)")
    P = {}
    for pid in PLOT_OF.values():
        own = rs / (f"runs-{pid}.json" if pid in (*PER_RUN_PANELS, FB) else f"{pid}.json")
        P[pid] = json.loads(own.read_text()) | {"source": source}
        if final:
            P[pid]["title"] = FINAL_TITLE + P[pid]["title"]
    if final:
        P["p1-loss-vs-compute"] = seeds_range(P["p1-loss-vs-compute"], fin["rows"])
    C = console(P, bool(final))
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    for key, pid in PLOT_OF.items():
        if pid in PER_RUN_PANELS:
            (trees[key] / f"te_core-{pid}.json").write_bytes((rs / f"{pid}.json").read_bytes())
        (trees[key] / f"{pid}.json").write_text(json.dumps(P[pid]) + "\n")
        (trees[key] / f"{pid}.console.json").write_bytes(body(C[pid]) + b"\n")
        png = (rs / (f"runs-{pid}.png" if pid == FB else f"{pid}.png")).read_bytes()
        (trees[key] / f"{pid}.png").write_bytes(png)
        (PANELS_DIR / f"{pid}.json").write_bytes(body(C[pid]) + b"\n")
        (PANELS_DIR / f"{pid}.png").write_bytes(png)
        if not args.no_publish:
            ids[key] = put_data(trees[key], "evidence/v1", {"what": f"te formalism v1 {what} {pid}", **meta})
    puts = {} if args.no_put else put_panels(C)
    summary = meta | {"art": ids, "panels_put": puts, "rows": {k: len(v["rows"]) for k, v in P.items()},
                      "console_rows": {k: len(v["rows"]) for k, v in C.items()},
                      "console_bytes": {k: len(body(v)) for k, v in C.items()}, "rescore": last, "out": str(out)}
    (out / "cycle.json").write_text(json.dumps(summary, indent=1) + "\n")
    line = digest(P, meta, ids, puts)
    (out / "digest.txt").write_text(line + "\n")
    print(json.dumps(summary))
    print(line)
    if not args.no_post and not args.no_publish:
        subprocess.run([sys.executable, str(HERE / "scorer.py"), "post", "FINAL-TABLE" if final else "DIGEST", line],
                       check=True)
    return 0


def seeds_range(p: dict, rows: dict) -> dict:
    """P1 with each final checkpoint's seeds' range as loss_lo and loss_hi."""
    ix = {c: i for i, c in enumerate(p["columns"])}
    out = []
    for r in p["rows"]:
        f = rows.get(r[ix["record"]])
        out.append([*r[:ix["loss_lo"]], f["bpb_min"], f["bpb_max"], *r[ix["loss_hi"] + 1:]] if f else r)
    return p | {"rows": out}


def final_md(rs: Path, fin: dict, tab: dict) -> str:
    """Per FINAL entry and target: its least compute reaching the target on the sealed shard (the seeds' mean) beside
    the search's, and its S at that checkpoint under each reading (v1runs.py's per-run P4, final and search)."""
    p4 = frame(json.loads((rs / "runs-p4-decomposition.json").read_text()))
    s4 = frame(json.loads((rs / "search-runs-p4-decomposition.json").read_text()))
    L = ["## FINAL entries on the sealed shard", "",
         (f"{len(fin['entries'])} entries; targets from the final starts' bpb ("
          + ", ".join(f"{s} {v:.4f}" for s, v in sorted(fin["final_starts"].items())) + "). A checkpoint reaches a "
          "target when its seeds' mean bpb does (all: the worst seed does too). Gates and every quantity but the bpb "
          "are the search checkpoint's at the nearest MACs (v1final.py); S = p s at that checkpoint, at c = 4096w, "
          "k = 10 kB, over the final H (te_core.rescore)."), "",
         ("| entry | start | target | final: least gates (frac; bpb mean [lo, hi]; seeds) | search: least gates (frac) "
          "| final/search | " + " | ".join(f"S {m} (ρ, basis)" for m in READINGS) + " |"),
         "|---" * (6 + len(READINGS)) + "|"]
    for e, x in sorted(fin["entries"].items(), key=lambda t: str(t[1]["run"])):
        mine = [r for r in p4 if r["run"] == x["run"]]
        name = f"{mine[0]['method'] if mine else '?'} {e[4:16]} ({x['run']})"
        if not x["matched"] or not mine:
            why = "no final row at a search checkpoint's MACs" if not x["matched"] else "reaches no target"
            L.append(f"| {name} | | | {why} |" + " |" * (2 + len(READINGS)))
            continue
        for tp in sorted({r["target_pct"] for r in mine}, reverse=True):
            at = {r["reading"]: r for r in mine if r["target_pct"] == tp}
            f = next(iter(at.values()))
            rec = fin["rows"].get(f["record"], {})
            lf = fin["final_starts"][f["start_id"]] * (1 + tp / 100)
            g = next((r for r in s4 if r["run"] == x["run"] and r["target_pct"] == tp), None)
            seeds = rec.get("seeds") or []
            fs = (f"{fmt(f['compute_gates'])} ({fmt(f['checkpoint_frac'])}; {rec.get('final_mean_bpb', 0):.4f} "
                  f"[{rec.get('bpb_min', 0):.4f}, {rec.get('bpb_max', 0):.4f}]; "
                  f"{'all' if rec.get('bpb_max', 9) <= lf else 'not all'} of {len(seeds)})" if rec else
                  f"{fmt(f['compute_gates'])} (search value)")
            gs = f"{fmt(g['compute_gates'])} ({fmt(g['checkpoint_frac'])})" if g else "not reached"
            ratio = fmt(f["compute_gates"] / g["compute_gates"]) if g else "–"
            vs = [f"{fmt(at[m]['S'])} ({fmt(at[m]['s'])}, {at[m]['basis']})" if m in at else "–" for m in READINGS]
            L.append(f"| {name} | {f['start_id']} | {tp:g}% | {fs} | {gs} | {ratio} | " + " | ".join(vs) + " |")
    L.append("")
    bad = [x for x in tab.get("reruns", []) if x.get("problems")]
    if bad:
        L += ["### Reruns left out", "", "| entry | seed | run | problems |", "|---|---|---|---|"]
        L += [f"| {x['id'][4:16]} | {x.get('seed')} | {x.get('run')} | {'; '.join(x['problems'])} |" for x in bad]
        L.append("")
    return "\n".join(L)


def reput(out: Path) -> int:
    """The console's cut of a finished refresh's full panels, written to PANELS_DIR and PUT, with nothing re-scored."""
    C = console({pid: json.loads((out / "publish" / key / f"{pid}.json").read_text()) for key, pid in PLOT_OF.items()
                 if (out / "publish" / key / f"{pid}.json").exists()})
    for pid, q in C.items():
        (PANELS_DIR / f"{pid}.json").write_bytes(body(q) + b"\n")
    puts = put_panels(C)
    print(json.dumps({"panels_put": puts, "console_bytes": {k: len(body(v)) for k, v in C.items()}}))
    return 0 if all(str(v).startswith("2") for v in puts.values()) else 1


def digest(P: dict, meta: dict, ids: dict, puts: dict) -> str:
    """At the reference target and policy, per start: H, each reading's S (rho, route) from P2's rows, and A2's least
    gates at the first Delta bins."""
    p2, a2 = frame(P["p2-slowdown-vs-gain"]), frame(P["a2-efficiency-floor"])
    parts = []
    for s in ("s100", "m100", "s50", "m50"):
        at = {r["reading"]: r for r in p2 if r["start_id"] == s and r["target_pct"] == REF_TARGET
              and r["policy"] == REFERENCE}
        if not at:
            continue
        h = next(iter(at.values()))["H_gates"]
        if h is None:
            parts.append(f"{s} not reached")
            continue
        q = " / ".join(f"{fmt(at[m]['S'])} {m} (ρ {fmt(at[m]['rho'])}, {at[m]['basis']}; {at[m]['method']} "
                       f"{at[m]['record']}@{at[m]['checkpoint']})" if m in at and at[m]["S"] is not None else f"– {m}"
                       for m in READINGS)
        fl = [r for r in a2 if r["start_id"] == s and r["target_pct"] == REF_TARGET and r["reading"] == READINGS[0]]
        floor = []
        for n, b in BINS:
            ok = [r["compute_gates"] for r in fl if r["delta_bits"] <= b]
            floor.append(f"{n} {fmt(min(ok))}" if ok else f"{n} none")
        parts.append(f"{s}: H {fmt(h)}; S {q}; A2 least gates at " + ", ".join(floor))
    arts = ", ".join(f"{k} {v[:16]}" for k, v in ids.items())
    ok = sum(str(v).startswith("2") for v in puts.values())
    head = (f"v1 final table (sealed shard: {meta['entries']} FINAL entries' reruns, {meta['entries_matched']} matched "
            f"at {meta['final_checkpoints']} checkpoints, 3 seeds' mean; every other record's search value at the same "
            "drop; " if "entries" in meta else "v1 refresh (")
    return (f"{head}te_core.rescore {meta['te_core'][:10]}: {meta['records']} valid v2.3 records, "
            f"{meta['night2_records']} from night 2, {meta['checkpoints']} checkpoints, {meta['not_analysed']} not "
            "analysed yet; H and W* over every checkpoint, so S >= 1; Slowdown's frozen tail charged, the one reading "
            "(Daniel, 22:00Z); rho a target unless Fit holds for the training) at −1% and "
            f"{REFERENCE}: " + "; ".join(parts) + f". art: {arts}. Console panels PUT {ok}/{len(puts)} (p1-p4, a2, "
            "frontier-branches).")


if __name__ == "__main__":
    sys.exit(main())

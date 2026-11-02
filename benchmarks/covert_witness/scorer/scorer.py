"""The training-efficiency search's scorer: one ledger of the launcher's records, ρ and W* from te_core, promotions, digests.

    python scorer.py collect                  # the store's new manifests -> STATE/index.jsonl and index.md
    python scorer.py score                    # STATE/score.json and score.md: ρ per (k, r), W*(l*), honest(l*), W*/honest
    python scorer.py board [--promote] [--digest] [--append]   # this hour's PROMOTED and DIGEST lines (appended to BOARD.md)

STATE is `$TE_SCORER_STATE` (default ~/.research/te-scorer); everything in it is derived from the evidence store and
rebuilt by `collect`. Records are the evidence/v1 artifacts whose meta has `covert_witness.class` (launch.py's), and the
te-core/submission/v1 ledger records (te_core.ledger's), written since `$TE_SCORER_SINCE`.

A launch.py record carries the witness (training FLOPs, φ over forward weight-matmul FLOPs, |E|, search bpb, the
FLOPs-to-target curve), not te_core's count. Its te_core annotation here is analytic, from the record's declared trainable
set and step-log tokens over the start's tensors: U is every forward multiplication whose weight operand is trained (a
full tensor's matmul or norm gain, an adapter's two factors; a frozen base W·x is θ's), each node's upstream is the
updated MACs per token of the nodes before it in (layer, stage) order, |C| = training FLOPs / 2, p = 16 (bf16 autocast)
and c = 4096 times the start's forward MACs per token (weight matmuls plus 2·d·ctx attention MACs a layer). ρ, W*,
honest and the effective slowdown are te_core's (`score.score_grid`, `w_star`, `honest`, `effective_slowdown`) on that
annotation, with no counterexample coverage. A te_core ledger record carries its counted ρ, used as is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STATE = Path(os.environ.get("TE_SCORER_STATE", Path.home() / ".research" / "te-scorer"))
SINCE = os.environ.get("TE_SCORER_SINCE", "2026-11-02T06:30:00Z")
SEARCH_START = "2026-11-02T08:30:00Z"  # the board's opening: earlier records are the integrator's seed smokes
TE_CORE = Path(os.environ.get("TE_CORE_PATH", "/tmp/scorer/te-core/benchmarks/covert_witness"))
BOARD = Path(os.environ.get("TE_BOARD", "/tmp/scorer/board/benchmarks/covert_witness/BOARD.md"))
SMALL_FILES = ("witness.json", "candidate.json", "meta.json", "timing.json")
STARTS = ("s50", "s100", "m50", "m100")
P_BITS = 16
E_EDGES = (1e4, 1e5, 1e6, 1e7, 1e8)
E_LABELS = ("<10kB", "10-100kB", "100kB-1MB", "1-10MB", "10-100MB", ">=100MB")
PHI_LABELS = ("<0.1", "0.1-0.25", "0.25-0.5", "0.5-0.75", ">=0.75")
THIN = 2
SHORT = 12

sys.path.insert(0, str(TE_CORE))
from te_core import VERSION as TE_CORE_VERSION  # noqa: E402
from te_core.annotate import Annotation, StepAnnotation  # noqa: E402
from te_core.frontier import PHI_EDGES, TARGETS, bin_of  # noqa: E402
from te_core.score import (C_PER_FORWARD_TOKEN, K_BITS, R, block_c, cell, effective_slowdown, honest,  # noqa: E402
                           score_grid, w_star)

KEYS = [cell(k, r) for k in K_BITS for r in R]


def short(art: str) -> str:
    return art[: 4 + SHORT] if art.startswith("art:") else art


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- the store ---------------------------------------------------------------------------------------------------------

def store_remote():
    from research.store import remote
    r = remote.load_remote()
    if r is None:
        raise SystemExit("scorer: no store remote (R2_ENDPOINT/R2_BUCKET or ~/.research/store.toml)")
    return r


def fetch(r, key: str, dest: Path, sha256: str | None = None) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    r.get(key, tmp)
    if sha256 and hashlib.sha256(tmp.read_bytes()).hexdigest() != sha256:
        tmp.unlink()
        raise RuntimeError(f"{key}: sha256 mismatch")
    tmp.rename(dest)
    return dest


def blob(r, sha256: str, dest: Path) -> Path:
    return fetch(r, f"objects/sha256/{sha256}", dest, sha256)


def new_manifests(r, since: str) -> list[tuple[str, str]]:
    with ThreadPoolExecutor(32) as ex:
        parts = ex.map(lambda i: r.list_times(f"manifests/{i:02x}"), range(256))
        return sorted((k, m) for part in parts for k, m in part if m >= since)


def manifests(r, since: str) -> list[tuple[str, str, dict]]:
    keys = new_manifests(r, since)
    d = STATE / "manifests"
    with ThreadPoolExecutor(32) as ex:
        paths = list(ex.map(lambda km: fetch(r, km[0], d / Path(km[0]).name), keys))
    return [("art:" + Path(k).stem, m, json.loads(p.read_text())) for (k, m), p in zip(keys, paths)]


def attempts(r, since: str) -> list[dict]:
    """The candidate runs (candidate.sh or te_core.run) whose attempts the store has since `since`."""
    after = "attempts/r" + since[:10].replace("-", "") + "-" + since[11:19].replace(":", "")
    keys = r.list_times("attempts/", start_after=after)
    d = STATE / "attempts"
    with ThreadPoolExecutor(16) as ex:
        paths = list(ex.map(lambda km: fetch(r, km[0], d / Path(km[0]).name), keys))
    out = []
    for p in paths:
        a = json.loads(p.read_text())
        argv = " ".join(a.get("argv") or [])
        m = re.search(r"covert_witness/candidate\.sh (\S+) (\S+)", argv)
        if not m and "te_core.run" not in argv:
            continue
        ex_ = a.get("execution") or {}
        out.append({"run": a["id"], "state": a["status"]["state"], "rc": a["status"].get("rc"),
                    "machine": ex_.get("machine"), "start_utc": ex_.get("start_utc"), "end_utc": ex_.get("end_utc"),
                    "commit": a["source"]["commit"], "class": m.group(1) if m else "te_core",
                    "start": m.group(2) if m else "", "te_core": not m})
    return out


# ---- commits to branches -----------------------------------------------------------------------------------------------

def branches(commits: set[str]) -> dict[str, list[str]]:
    path = STATE / "commits.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    todo = [c for c in commits if c and not known.get(c)]
    if todo:
        subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin",
                        "+refs/heads/cursor/te-*:refs/remotes/origin/cursor/te-*",
                        "+refs/heads/cursor/covert-witness-4207:refs/remotes/origin/cursor/covert-witness-4207"],
                       capture_output=True, text=True)
    for c in todo:
        out = subprocess.run(["git", "-C", str(REPO), "branch", "-r", "--contains", c, "--format=%(refname:short)"],
                             capture_output=True, text=True)
        names = [b.removeprefix("origin/") for b in out.stdout.split() if "/te-" in b or "covert-witness" in b]
        known[c] = sorted(names)
    path.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
    return known


def direction_of(names: list[str], klass: str) -> tuple[str, str]:
    """(agent branch, direction): the te-<direction>-4207 branch holding the commit, else the class."""
    own = [b for b in names if re.fullmatch(r"cursor/te-.+-4207", b) and b not in (
        "cursor/te-board-4207", "cursor/te-core-4207", "cursor/te-capture-4207")]
    if own:
        b = sorted(own, key=len)[0]
        return b, b.removeprefix("cursor/te-").removesuffix("-4207")
    if "cursor/covert-witness-4207" in names:
        return "cursor/covert-witness-4207", klass
    return "", klass


# ---- collect -----------------------------------------------------------------------------------------------------------

def mechanism_of(note: str) -> str:
    w = re.split(r"[\s,;:(]+", note.strip().lower())
    return (w[0] if w and w[0] else "unnamed").strip(".")


def screen_key(note: str) -> str:
    """The note without its learning rates, seeds and step counts: one key per mechanism's screen."""
    s = note.lower()
    s = re.sub(r"\b(lr|muon[_-]?lr|adam[_-]?lr|norm[_-]?lr|seed|data[_-]?seed|eta|step[s]?)\s*[=: ]?\s*[-+.\deE]+", "", s)
    s = re.sub(r"\b\d+(\.\d+)?e-?\d+\b", "", s)
    return re.sub(r"\s+", " ", s).strip(" ,;")


def starts_info(r, recs: list[tuple[str, str, dict]]) -> dict:
    """Each start's config, tensors and search bpb, from the launcher's start-scores record."""
    for art, _, man in sorted(recs, key=lambda x: x[1], reverse=True):
        cw = (man.get("meta") or {}).get("covert_witness") or {}
        if not str(cw.get("what", "")).startswith("start scores"):
            continue
        files = {f["path"]: f["sha256"] for f in man["payload"].get("files", [])}
        d = STATE / "starts" / art.removeprefix("art:")
        out = {}
        for s in STARTS:
            st = json.loads(blob(r, files[f"{s}.start.json"], d / f"{s}.start.json").read_text())
            sc = json.loads(blob(r, files[f"{s}.score.json"], d / f"{s}.score.json").read_text())
            out[s] = {"config": st["config"], "tensors": st["tensors"], "bpb": sc["score"]["bpb"], "art": art}
        return out
    raise SystemExit("scorer: no start-scores record in the store since SINCE")


def cw_row(r, art: str, recorded: str, man: dict, commits: dict) -> dict:
    cw = man["meta"]["covert_witness"]
    files = {f["path"]: f for f in man["payload"].get("files", [])}
    d = STATE / "records" / art.removeprefix("art:")
    got = {}
    for name in SMALL_FILES:
        if name in files:
            got[name] = json.loads(blob(r, files[name]["sha256"], d / name).read_text())
    w, cand, meta = got.get("witness.json"), got.get("candidate.json", {}), got.get("meta.json", {})
    agent, direction = direction_of(commits.get(cand.get("commit", ""), []), cw.get("class", ""))
    checks = (w or {}).get("checks", {})
    bad = sorted(k for k, v in checks.items() if v is not True)
    curve = (w or {}).get("curve") or []
    return {
        "id": art, "source": "launch.py", "run": cw.get("run"), "class": cw.get("class"), "start": cw.get("start"),
        "note": cw.get("note", ""), "mechanism": mechanism_of(cw.get("note", "")), "screen": screen_key(cw.get("note", "")),
        "agent": agent, "direction": direction, "commit": cand.get("commit", ""), "args": cand.get("args", ""),
        "launched": run_time(cw.get("run", "")), "recorded": recorded,
        "valid": bool(w and w.get("valid")), "failed_checks": bad if w else ["no witness"],
        "train_flops": (w or {}).get("train", {}).get("flops"), "tokens": (w or {}).get("train", {}).get("tokens"),
        "bpb": (w or {}).get("score", {}).get("bpb"), "E_bytes": (w or {}).get("E", {}).get("bytes"),
        "E_format": (w or {}).get("E", {}).get("format"),
        "phi": (w or {}).get("structure", {}).get("phi"),
        "lowest_updated_layer": (w or {}).get("structure", {}).get("lowest_updated_layer"),
        "gpu_seconds": (w or {}).get("gpu_seconds"), "curve": curve,
        "trainable": meta.get("trainable", {}), "method": ((w or {}).get("notes") or {}).get("method"),
        "train_args": ((w or {}).get("notes") or {}).get("args"),
        "has_E": "E.bin" in files,
    }


def te_row(r, art: str, recorded: str, man: dict) -> dict | None:
    f = man["payload"]
    sha = f.get("sha256") if f.get("type") == "file" else None
    if not sha:
        return None
    rec = json.loads(blob(r, sha, STATE / "te" / (art.removeprefix("art:") + ".json")).read_text())
    if rec.get("kind") != "submission":
        return {"id": art, "source": "te_core", "te_kind": rec.get("kind"), "te_id": rec.get("id"), "agent": rec.get("agent"),
                "recorded": recorded, "body": {k: rec.get(k) for k in ("entry", "tier", "reason", "test_bpb")}}
    return {
        "id": art, "source": "te_core", "te_kind": "submission", "te_id": rec.get("id"), "run": rec.get("run_id"),
        "class": "te_core", "start": rec.get("start"), "note": rec.get("mechanism", ""),
        "mechanism": mechanism_of(rec.get("mechanism", "")), "screen": screen_key(rec.get("mechanism", "")),
        "agent": rec.get("agent", ""), "direction": rec.get("direction", ""), "commit": rec.get("commit", ""),
        "launched": rec.get("written_at"), "recorded": recorded, "valid": not rec.get("invalid"),
        "failed_checks": rec.get("invalid", []), "train_flops": 2 * rec["macs"], "macs": rec["macs"],
        "bpb": rec.get("val_bpb"), "test_bpb": rec.get("final_bpb"), "E_bytes": (rec.get("output") or {}).get("e_bytes"),
        "phi": rec.get("phi"), "D": rec.get("depth"), "rho": rec.get("rho"), "beta": rec.get("beta"),
        "reference": rec.get("reference"), "tier": rec.get("tier"), "fraction": rec.get("fraction"),
        "start_bpb": rec.get("start_bpb"),
    }


def run_time(run: str) -> str:
    m = re.fullmatch(r"r(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})-[0-9a-f]{4}", run or "")
    return f"{m[1]}-{m[2]}-{m[3]}T{m[4]}:{m[5]}:{m[6]}Z" if m else ""


def collect(since: str = SINCE) -> dict:
    r = store_remote()
    t0 = time.time()
    recs = [x for x in manifests(r, since) if x[2].get("kind") == "evidence/v1"]
    cw = [x for x in recs if "class" in ((x[2].get("meta") or {}).get("covert_witness") or {})]
    te = [x for x in recs if str((x[2].get("meta") or {}).get("schema", "")).startswith("te-core/")]
    starts = starts_info(r, recs)
    (STATE / "starts.json").write_text(json.dumps(starts) + "\n")
    commit_of = {}
    for art, _, man in cw:
        files = {f["path"]: f for f in man["payload"].get("files", [])}
        if "candidate.json" in files:
            p = blob(r, files["candidate.json"]["sha256"], STATE / "records" / art.removeprefix("art:") / "candidate.json")
            commit_of[art] = json.loads(p.read_text()).get("commit", "")
    runs = attempts(r, since)
    commits = branches(set(commit_of.values()) | {a["commit"] for a in runs})
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(lambda x: cw_row(r, x[0], x[1], x[2], commits), cw))
        rows += [t for t in ex.map(lambda x: te_row(r, x[0], x[1], x[2]), te) if t]
    recorded_runs = {x.get("run") for x in rows}
    for a in runs:
        a["agent"], a["direction"] = direction_of(commits.get(a["commit"], []), a["class"])
        a["record"] = a["run"] in recorded_runs
    rows.sort(key=lambda x: (x.get("launched") or "", x["id"]))
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "index.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    (STATE / "runs.json").write_text(json.dumps(runs, indent=1) + "\n")
    (STATE / "index.md").write_text(index_md(rows, runs, starts))
    out = {"at": now(), "records": len([x for x in rows if x["source"] == "launch.py"]),
           "te_core_records": len([x for x in rows if x["source"] == "te_core"]), "runs": len(runs),
           "runs_without_record": [a["run"] for a in runs if not a["record"] and run_time(a["run"]) >= SEARCH_START],
           "seconds": round(time.time() - t0, 1)}
    (STATE / "collect.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def fmt(x, spec=".3g"):
    return "—" if x is None else format(x, spec)


def index_md(rows: list[dict], runs: list[dict], starts: dict) -> str:
    lines = [f"# The search's ledger, {now()}", "",
             f"{len(rows)} records since {SINCE}; starts' search bpb: " +
             ", ".join(f"{s} {starts[s]['bpb']:.4f}" for s in STARTS), "",
             "| record | agent | direction | class | mechanism note | start | launched | train FLOPs | bpb | Δ% | |E| bytes |"
             " φ | D (analytic) | valid |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        if x["source"] != "launch.py":
            continue
        s0 = starts.get(x["start"], {}).get("bpb")
        drop = 100 * (s0 - x["bpb"]) / s0 if s0 and x["bpb"] is not None else None
        ann = analytic(x, starts) if x["valid"] and x["start"] in starts else None
        D = ann["depth"] if ann else None
        lines.append(f"| `{short(x['id'])}` | {x['agent'].removeprefix('cursor/') or '—'} | {x['direction']} | {x['class']} | "
                     f"{x['note']} | {x['start']} | {x['launched'][11:16]} | {fmt(x['train_flops'])} | {fmt(x['bpb'], '.4f')} | "
                     f"{fmt(drop, '.2f')} | {fmt(x['E_bytes'], ',')} | {fmt(x['phi'], '.3f')} | {fmt(D, '.3g')} | "
                     f"{'yes' if x['valid'] else 'no: ' + ','.join(x['failed_checks'])} |")
    pend = [a for a in runs if not a["record"] and run_time(a["run"]) >= SEARCH_START]
    if pend:
        lines += ["", "Ended candidate runs with no record yet (launch.py collects a record when its run ends): " + ", ".join(
            f"{a['run']} ({a['direction']} {a['class']} {a['start']}, {a['state']})" for a in pend)]
    return "\n".join(lines) + "\n"


# ---- the analytic annotation and ρ ----------------------------------------------------------------------------------

LF_STAGE = {"attn_norm": 0, "wq": 1, "wk": 1, "wv": 1, "wo": 2, "mlp_norm": 3, "w_gate": 4, "w_up": 4, "w_down": 5}


def nodes_of(start: dict) -> list[dict]:
    """The forward's multiplications with a weight operand, per token: {param, key (layer, stage), macs, d_in, d_out}."""
    cfg, tensors = start["config"], start["tensors"]
    if cfg["family"] != "lf":
        raise ValueError("scorer: only lf starts")
    L = cfg["n_layer"]
    out = []
    for name, (_, shape) in tensors.items():
        if name == "embed":
            out.append({"param": name, "key": (L + 1, 1), "macs": shape[0] * shape[1], "d_in": shape[1], "d_out": shape[0]})
            continue
        if name == "norm":
            out.append({"param": name, "key": (L + 1, 0), "macs": shape[0], "d_in": shape[0], "d_out": shape[0]})
            continue
        m = re.fullmatch(r"blocks\.(\d+)\.(\w+?)(?:\.weight)?", name)
        two = len(shape) == 2
        out.append({"param": name, "key": (int(m[1]) + 1, LF_STAGE[m[2]]), "macs": shape[0] * shape[1] if two else shape[0],
                    "d_in": shape[1] if two else shape[0], "d_out": shape[0]})
    return sorted(out, key=lambda n: (n["key"], n["param"]))


def forward_macs_per_token(start: dict) -> float:
    """Every weight matmul and norm gain, and each layer's QKᵀ and AV at the full context (2·d·ctx MACs a token)."""
    cfg = start["config"]
    return sum(n["macs"] for n in nodes_of(start)) + cfg["n_layer"] * 2 * cfg["d_model"] * cfg["ctx"]


def analytic(row: dict, starts: dict, fraction: float = 1.0) -> dict:
    """The record's annotation at `fraction` of its run: |C| in MACs, the updated nodes' (upstream, MACs) profile, D."""
    start = starts[row["start"]]
    tr = row.get("trainable") or {}
    full, adapters = set(tr.get("full", [])), {k: int(v) for k, v in (tr.get("adapters") or {}).items()}
    upd = []  # (key, param, sub, per-token MACs): an adapter is x·Aᵀ (sub 0) then its product with Bᵀ (sub 1)
    for n in nodes_of(start):
        if n["param"] in full:
            upd.append((n["key"], n["param"], 0, n["macs"]))
        elif n["param"] in adapters:
            r = adapters[n["param"]]
            upd += [(n["key"], n["param"], 0, r * n["d_in"]), (n["key"], n["param"], 1, r * n["d_out"])]
    profile = [(sum(mm for k, _, _, mm in upd if k < key) + sum(mm for k, p, s, mm in upd if k == key and p == param and s < sub), m)
               for key, param, sub, m in upd]
    tokens = row.get("tokens") or {}
    t_upd = fraction * (int(tokens.get("train", 0)) + int(tokens.get("forward", 0)))
    macs = fraction * row["train_flops"] / 2
    per_token = sum(m for _, m in profile)
    depth = max((u for u, _ in profile), default=0)
    return {"macs": macs, "tokens": t_upd, "profile": [(u, m * t_upd) for u, m in profile], "depth": depth,
            "updated_per_token": per_token, "c": block_c(forward_macs_per_token(start))}


def annotation_of(a: dict) -> Annotation:
    upd = sum(m for _, m in a["profile"])
    step = StepAnnotation(step=0, tokens=int(a["tokens"]), macs=int(a["macs"]), updated_macs=int(upd),
                          updated_per_token=a["updated_per_token"], depth=int(a["depth"]),
                          profile=tuple(sorted((int(u), int(m)) for u, m in a["profile"])), bits=P_BITS if upd else 0,
                          weights={}, shapes={})
    return Annotation(macs=int(a["macs"]), steps=(step,))


def submissions(rows: list[dict], starts: dict) -> list[dict]:
    """Every checkpoint of every valid record as a te_core submission: {id, cp, start, macs, val_bpb, rho, phi, depth}."""
    out = []
    for x in rows:
        if not x.get("valid") or x.get("start") not in starts:
            continue
        if x["source"] == "te_core":
            if x.get("te_kind") == "submission" and x.get("rho"):
                out.append({"id": x["id"], "cp": x.get("fraction"), "start": x["start"], "macs": x["macs"],
                            "val_bpb": x["bpb"], "rho": x["rho"], "phi": x["phi"], "depth": x["D"],
                            "reference": bool(x.get("reference")), "direction": x["direction"], "counted": True,
                            "E_bytes": x.get("E_bytes")})
            continue
        total = x["train_flops"] or 0
        for i, cp in enumerate(x.get("curve") or []):
            if not isinstance(cp.get("bpb"), (int, float)) or not total:
                continue
            f = cp["train_flops"] / total
            a = analytic(x, starts, f)
            grid = score_grid(annotation_of(a), {}, P_BITS, a["c"])
            out.append({"id": x["id"], "cp": round(f, 3), "start": x["start"], "macs": a["macs"], "val_bpb": cp["bpb"],
                        "rho": {k: v["rho"] for k, v in grid.items()}, "phi": {k: v["phi"] for k, v in grid.items()},
                        "depth": a["depth"], "reference": x["class"] == "reference", "direction": x["direction"],
                        "counted": False, "E_bytes": cp.get("e_bytes")})
    return out


def pareto(points: list[tuple[float, float, dict]]) -> list[dict]:
    out, best = [], math.inf
    for c, b, s in sorted(points, key=lambda p: (p[0], p[1])):
        if b < best:
            out.append(s)
            best = b
    return out


def score(write: bool = True) -> dict:
    rows = [json.loads(x) for x in (STATE / "index.jsonl").read_text().splitlines() if x.strip()]
    starts = json.loads((STATE / "starts.json").read_text())
    subs = submissions(rows, starts)
    rep = {"at": now(), "te_core": TE_CORE_VERSION, "annotation": "analytic for launch.py records (module docstring)",
           "p_bits": P_BITS, "c_per_forward_token": C_PER_FORWARD_TOKEN, "starts": {}, "submissions": len(subs)}
    for s in STARTS:
        on = [x for x in subs if x["start"] == s]
        refs, cov = [x for x in on if x["reference"]], [x for x in on if not x["reference"]]
        l0 = starts[s]["bpb"]
        st = {"start_bpb": l0, "c": block_c(forward_macs_per_token(starts[s])), "targets": [],
              "proxy_frontier_flops": [{"id": x["id"], "cp": x["cp"], "flops": 2 * x["macs"], "bpb": x["val_bpb"]}
                                       for x in pareto([(x["macs"], x["val_bpb"], x) for x in on])],
              "proxy_frontier_E": [{"id": x["id"], "cp": x["cp"], "E_bytes": x["E_bytes"], "bpb": x["val_bpb"]}
                                   for x in pareto([(x["E_bytes"] or 0, x["val_bpb"], x) for x in on])]}
        for f in TARGETS:
            l_star = l0 * (1 - f)
            h = honest(refs, l_star)
            h_id = min((x for x in refs if x["val_bpb"] <= l_star), key=lambda x: x["macs"], default=None)
            t = {"drop": f, "l_star": l_star, "honest_macs": h, "honest_id": h_id and h_id["id"],
                 "honest_cp": h_id and h_id["cp"], "cells": {}}
            for key in KEYS:
                w = w_star(cov, l_star, key)
                arg = min((x for x in cov if x["val_bpb"] <= l_star), key=lambda x: x["macs"] * x["rho"][key], default=None)
                t["cells"][key] = {"w_star": w, "effective_slowdown": effective_slowdown(cov, refs, l_star, key),
                                   "id": arg and arg["id"], "cp": arg and arg["cp"], "rho": arg and arg["rho"][key],
                                   "direction": arg and arg["direction"]}
            st["targets"].append(t)
        rep["starts"][s] = st
    if write:
        (STATE / "score.json").write_text(json.dumps(rep, indent=1) + "\n")
        (STATE / "submissions.json").write_text(json.dumps(subs) + "\n")
        (STATE / "score.md").write_text(score_md(rep))
    return rep


def score_md(rep: dict) -> str:
    lines = [f"# Scores, {rep['at']} (te_core {rep['te_core']}; {rep['annotation']}; p = {rep['p_bits']})", ""]
    for s, st in rep["starts"].items():
        lines += [f"## {s} (start {st['start_bpb']:.4f} bpb, c = {st['c']:.3g} MACs)", "",
                  "| target | l* | honest MACs | " + " | ".join(KEYS) + " |", "|---|---|---|" + "---|" * len(KEYS)]
        for t in st["targets"]:
            cells = []
            for key in KEYS:
                c = t["cells"][key]
                cells.append("—" if c["w_star"] is None else
                             f"{c['w_star']:.3g} ({fmt(c['effective_slowdown'], '.3g')}x, {short(c['id'])}@{c['cp']})")
            lines.append(f"| -{100 * t['drop']:g}% | {t['l_star']:.4f} | {fmt(t['honest_macs'])} | " + " | ".join(cells) + " |")
        lines += ["", "Proxy frontier, bpb against training FLOPs: " + "; ".join(
            f"{short(x['id'])}@{x['cp']} {x['flops']:.3g} -> {x['bpb']:.4f}" for x in st["proxy_frontier_flops"]),
                  "", "Proxy frontier, bpb against |E|: " + "; ".join(
            f"{short(x['id'])}@{x['cp']} {fmt(x['E_bytes'], ',')} B -> {x['bpb']:.4f}" for x in st["proxy_frontier_E"]), ""]
    return "\n".join(lines) + "\n"


# ---- cells, promotions, digest -----------------------------------------------------------------------------------------

def e_bin(e) -> str:
    return E_LABELS[bin_of(e or 0, E_EDGES)]


def phi_bin(phi) -> str:
    return PHI_LABELS[bin_of(phi or 0.0, PHI_EDGES)]


def cell_of(x: dict) -> tuple:
    return (x["mechanism"], e_bin(x["E_bytes"]), phi_bin(x["phi"]), x["start"])


def drop_of(x: dict, starts: dict) -> float | None:
    s0 = starts.get(x["start"], {}).get("bpb")
    return (s0 - x["bpb"]) / s0 if s0 and isinstance(x.get("bpb"), (int, float)) else None


def promotions(rows: list[dict], starts: dict, promoted: dict) -> dict:
    """This hour's promotions: each direction's top third by relative bpb drop, and the best of each thin cell (at most
    THIN valid records), among valid records of the search (launched from SEARCH_START) not promoted before; plus
    re-screens of mechanisms already screened."""
    valid = [x for x in rows if x["source"] == "launch.py" and x["valid"] and drop_of(x, starts) is not None
             and (x["launched"] or "") >= SEARCH_START]
    new, why = [], {}
    by_dir: dict[str, list[dict]] = {}
    for x in valid:
        by_dir.setdefault(x["direction"], []).append(x)
    for d, xs in by_dir.items():
        xs = sorted(xs, key=lambda x: -drop_of(x, starts))
        for x in xs[: math.ceil(len(xs) / 3)]:
            why.setdefault(x["id"], []).append(f"top third of {d}")
    cells: dict[tuple, list[dict]] = {}
    for x in valid:
        cells.setdefault(cell_of(x), []).append(x)
    for c, xs in cells.items():
        if len(xs) <= THIN:
            best = max(xs, key=lambda x: drop_of(x, starts))
            why.setdefault(best["id"], []).append(f"thin cell {c[0]} |E| {c[1]} phi {c[2]} {c[3]}")
    for i, reasons in why.items():
        if i not in promoted:
            new.append({"id": i, "reasons": reasons})
    return {"new": new, "cells": cells, "rescreens": rescreens(rows)}


def rescreens(rows: list[dict]) -> list[dict]:
    """Records screening a mechanism (note less its rates and seeds) on a start that an earlier record of another
    direction screened, or that its own direction had already swept three times."""
    seen: dict[tuple, list[dict]] = {}
    out = []
    for x in sorted((x for x in rows if x["source"] == "launch.py"), key=lambda x: x["launched"] or ""):
        k = (x["screen"], x["class"], x["start"])
        prev = seen.setdefault(k, [])
        other = [p for p in prev if p["direction"] != x["direction"]]
        own = [p for p in prev if p["direction"] == x["direction"]]
        if other:
            out.append({"id": x["id"], "of": other[0]["id"], "why": f"{other[0]['direction']} screened it first"})
        elif len(own) >= 3 and x["screen"] not in ("seed smoke",):
            out.append({"id": x["id"], "of": own[0]["id"], "why": f"{len(own)} earlier runs of the same note"})
        prev.append(x)
    return out


def empty_cells(rows: list[dict], starts: dict) -> list[str]:
    """E range x φ range x start cells no valid record fills, within the E ranges each class's cap allows."""
    valid = [x for x in rows if x["source"] == "launch.py" and x["valid"]]
    filled = {(e_bin(x["E_bytes"]), phi_bin(x["phi"]), x["start"]) for x in valid}
    out = []
    for s in STARTS:
        for e in E_LABELS[:4]:
            for p in PHI_LABELS:
                if (e, p, s) not in filled:
                    out.append(f"{s}:{e}:{p}")
    return out


def board(promote: bool, digest: bool, append: bool, hhmm: str | None = None) -> list[str]:
    rows = [json.loads(x) for x in (STATE / "index.jsonl").read_text().splitlines() if x.strip()]
    starts = json.loads((STATE / "starts.json").read_text())
    rep = json.loads((STATE / "score.json").read_text())
    ppath = STATE / "promoted.json"
    promoted = json.loads(ppath.read_text()) if ppath.exists() else {}
    hhmm = hhmm or datetime.now(timezone.utc).strftime("%H:%MZ")
    lines = []
    pr = promotions(rows, starts, promoted)
    if promote:
        if pr["new"]:
            parts = [f"{short(p['id'])} ({'; '.join(p['reasons'])})" for p in pr["new"]]
            lines.append(f"{hhmm} scorer [PROMOTED] " + ", ".join(parts))
        old = set(json.loads((STATE / "rescreens.json").read_text())) if (STATE / "rescreens.json").exists() else set()
        rs = [x for x in pr["rescreens"] if x["id"] not in old]
        if rs:
            lines.append(f"{hhmm} scorer [PROMOTED] re-screens flagged (not promoted for it): " + ", ".join(
                f"{short(x['id'])} repeats {short(x['of'])} ({x['why']})" for x in rs))
    if digest:
        lines += digest_lines(rows, starts, rep, pr, hhmm)
    if append and lines:
        append_board(lines)
        if promote:
            for p in pr["new"]:
                promoted[p["id"]] = {"at": hhmm, "reasons": p["reasons"]}
            ppath.write_text(json.dumps(promoted, indent=1) + "\n")
            (STATE / "rescreens.json").write_text(json.dumps(sorted({x["id"] for x in pr["rescreens"]})) + "\n")
    return lines


def digest_lines(rows, starts, rep, pr, hhmm) -> list[str]:
    out = []
    cw = [x for x in rows if x["source"] == "launch.py"]
    n, nv = len(cw), len([x for x in cw if x["valid"]])
    out.append(f"{hhmm} scorer [DIGEST] {n} records ({nv} valid). Frontier per target below: honest = least reference "
               f"FLOPs reaching l*; W*/honest per (k, r) in the order 10kB r1,r10,r100 | 100kB | 1MB, te_core rho on the "
               f"analytic annotation (p 16), with the W* entry; full tables in the scorer's score.md.")
    for s, st in rep["starts"].items():
        parts = []
        for t in st["targets"]:
            h = t["honest_macs"]
            cells = t["cells"]
            if h is None and all(c["w_star"] is None for c in cells.values()):
                continue
            eff = [cells[k]["effective_slowdown"] for k in KEYS]
            vec = " | ".join(", ".join(fmt(e, ".3g") for e in eff[i: i + 3]) for i in (0, 3, 6))
            ids = sorted({short(c["id"]) for c in cells.values() if c["id"]})
            parts.append(f"-{100 * t['drop']:g}% (l* {t['l_star']:.4f}): honest {fmt(h and 2 * h)}"
                         + (f" ({short(t['honest_id'])}@{t['honest_cp']})" if t["honest_id"] else "")
                         + (f"; W*/honest [{vec}] via {', '.join(ids)}" if ids else "; no covert entry reaches it"))
        if parts:
            out.append(f"{hhmm} scorer [DIGEST] frontier {s}: " + "; ".join(parts) + ".")
    if pr["new"]:
        out.append(f"{hhmm} scorer [DIGEST] newly promoted: " + ", ".join(short(p["id"]) for p in pr["new"]) + ".")
    empty = empty_cells(rows, starts)
    by_start: dict[str, list[str]] = {}
    for e in empty:
        s, eb, pb = e.split(":")
        by_start.setdefault(s, []).append(f"{eb}/{pb}")
    out.append(f"{hhmm} scorer [DIGEST] empty cells (|E| range x phi range, per start, |E| under 10 MB, any mechanism): " +
               "; ".join(f"{s} {len(by_start.get(s, []))}/20 empty, filled {', '.join(sorted(filled_of(rows, s))) or 'none'}"
                         for s in STARTS) + ".")
    for d in sorted({x["direction"] for x in cw}):
        out.append(f"{hhmm} scorer [DIGEST] {d}: " + direction_sentence(d, rows, starts))
    return out


def filled_of(rows, s) -> set[str]:
    return {f"{e_bin(x['E_bytes'])}/{phi_bin(x['phi'])}" for x in rows
            if x["source"] == "launch.py" and x["valid"] and x["start"] == s and (x["E_bytes"] or 0) < 1e7}


def direction_sentence(d: str, rows: list[dict], starts: dict) -> str:
    xs = [x for x in rows if x["source"] == "launch.py" and x["direction"] == d]
    v = [x for x in xs if x["valid"] and drop_of(x, starts) is not None]
    mechs = sorted({x["mechanism"] for x in xs})
    if not v:
        return f"{len(xs)} records, none valid yet ({', '.join(mechs)})."
    best = max(v, key=lambda x: drop_of(x, starts))
    return (f"{len(xs)} records, {len(v)} valid, {len(mechs)} mechanisms ({', '.join(mechs)}); best "
            f"{short(best['id'])} '{best['note']}' on {best['start']}: {best['bpb']:.4f} bpb "
            f"({100 * drop_of(best, starts):+.2f}% from the start) at {best['train_flops']:.3g} FLOPs, |E| "
            f"{fmt(best['E_bytes'], ',')} B, phi {fmt(best['phi'], '.2f')}.")


def append_board(lines: list[str]) -> None:
    wt = BOARD.parents[2]
    for attempt in range(5):
        subprocess.run(["git", "-C", str(wt), "fetch", "-q", "origin", "cursor/te-board-4207"], check=True)
        subprocess.run(["git", "-C", str(wt), "reset", "-q", "--hard", "origin/cursor/te-board-4207"], check=True)
        text = BOARD.read_text()
        BOARD.write_text(text + ("" if text.endswith("\n") else "\n") + "\n".join(lines) + "\n")
        tags = sorted({m[1] for m in (re.search(r"\[(\w+)\]", x) for x in lines) if m})
        subprocess.run(["git", "-C", str(wt), "commit", "-q", "-am", f"board: scorer {lines[0][:6]} {', '.join(tags)}"],
                       check=True)
        if subprocess.run(["git", "-C", str(wt), "push", "-q", "origin", "HEAD:cursor/te-board-4207"]).returncode == 0:
            return
        time.sleep(2 ** attempt)
    raise SystemExit("scorer: board push failed five times")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--since", default=SINCE)
    sub.add_parser("score")
    b = sub.add_parser("board")
    b.add_argument("--promote", action="store_true")
    b.add_argument("--digest", action="store_true")
    b.add_argument("--append", action="store_true", help="append the lines to the board and push (else print them)")
    b.add_argument("--hhmm", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        print(json.dumps(collect(a.since)))
    elif a.cmd == "score":
        print(score_md(score()))
    else:
        print("\n".join(board(a.promote, a.digest, a.append, a.hhmm)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

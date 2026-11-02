"""The training-efficiency search's scorer: one ledger of the launcher's records, ρ and W* from te_core, promotions, digests.

    python scorer.py collect                  # the store's new manifests -> STATE/index.jsonl and index.md
    python scorer.py score                    # STATE/score.json and score.md: ρ per (k, r), W*(l*), honest(l*), W*/honest
    python scorer.py board [--promote] [--digest] [--append]   # this hour's PROMOTED and DIGEST lines (--append pushes them)
    python scorer.py post TAG TEXT            # one line onto the scorer's board (islands/scorer/BOARD.md), pushed

STATE is `$TE_SCORER_STATE` (default ~/.research/te-scorer); everything in it is derived from the evidence store and
rebuilt by `collect`. Records are the evidence/v1 artifacts whose meta has `covert_witness.class` (launch.py's), and the
te-core/submission/v1 ledger records (te_core.ledger's), written since `$TE_SCORER_SINCE`.

A record is scored only if its witness and run.json are the node's (`custody_problems`: launch.py uploads records from an
agent's VM, the node's runner publishes the run's out/) and its commit's fixed side is the harness's (`fixed_side`). A
record made inside te_core's harness (witness v2, covert_witness v2.1) is priced by the harness's count: `collect` ingests
the node's run.json with te_core at its branch head (`te_current`), one submission per checkpoint carrying |C|, U, D, φ
and ρ; the te_core records agents publish themselves are listed, never scored. A witness v1 record carries the witness
(training FLOPs, φ over forward weight-matmul FLOPs, |E|, search bpb, the FLOPs-to-target curve), not te_core's count. Its
annotation here is analytic, from the record's declared trainable set and step-log tokens over the start's tensors: U is
every forward multiplication whose weight operand is trained (a full tensor's matmul or norm gain, an adapter's two
factors; a frozen base W·x is θ's, except in a run that merges its pieces, whose targets are full), each node's upstream
is the updated MACs per token of the nodes before it in (layer, stage) order, |C| = training FLOPs / 2, p = 16 (bf16
autocast) and c = 4096 times the start's forward MACs per token (weight matmuls plus 2·d·ctx attention MACs a layer).
ρ, W*, honest and the effective slowdown are te_core's (`score.score_grid`, `w_star`, `honest`, `effective_slowdown`) on
that annotation, with the coverage `credit` reads off a seed-log E.

honest(l*) reads the reference directions (ref-*), W* every other direction's records whatever their class, and the
measurement auditor's records (direction audit) are in the ledger only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import struct
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STATE = Path(os.environ.get("TE_SCORER_STATE", Path.home() / ".research" / "te-scorer"))
SINCE = os.environ.get("TE_SCORER_SINCE", "2026-11-02T06:30:00Z")
SEARCH_START = "2026-11-02T08:30:00Z"  # the board's opening: earlier records are the integrator's seed smokes
TE_CORE = Path(os.environ.get("TE_CORE_PATH", Path(__file__).resolve().parents[1]))
TE_CURRENT = Path(os.environ.get("TE_CURRENT", "/tmp/scorer/te-core"))  # a worktree of this repository, any branch
BOARD = Path(os.environ.get("TE_BOARD", Path(__file__).resolve().parents[1] / "islands" / "scorer" / "BOARD.md"))
BOARD_HEAD = ("# The scorer's board lines\n\nAppend-only, one line per entry: `HH:MMZ scorer [TAG] text (record ids)`. The "
              "coordinator merges them into benchmarks/covert_witness/BOARD.md on cursor/te-digest-4207. The scorer's lines "
              "through 09:00Z are on cursor/te-board-4207.\n\n## Entries\n")
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
                    "cancelled": str(a["status"].get("class", "")).startswith("CANCELLED"),
                    "run_record": (a.get("outputs") or {}).get("run_record"),
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
        # local branches too: agents on this VM share its object store, and some never push their branch
        out = subprocess.run(["git", "-C", str(REPO), "branch", "-a", "--contains", c, "--format=%(refname:short)"],
                             capture_output=True, text=True)
        names = {b.removeprefix("origin/") for b in out.stdout.split() if "/te-" in b or "covert-witness" in b}
        known[c] = sorted(names)
    path.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
    return known


def direction_of(names: list[str], klass: str) -> tuple[str, str]:
    """(agent branch, direction): the harness's for a commit on cursor/covert-witness-4207 (every te-* branch holds those),
    else the te-<direction>-4207 branch holding it, else the class."""
    if "cursor/covert-witness-4207" in names:
        return "cursor/covert-witness-4207", klass
    own = [b for b in names if re.fullmatch(r"cursor/te-.+-4207", b) and b not in (
        "cursor/te-board-4207", "cursor/te-core-4207", "cursor/te-capture-4207", "cursor/te-digest-4207")]
    if own:
        b = sorted(own, key=len)[0]
        return b, b.removeprefix("cursor/te-").removesuffix("-4207")
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


def seedlog_header(r, files: dict, d: Path) -> dict | None:
    """A seed-log E's header (names, seed, steps = its coefficient count, coeff): magic, u32 length, zlib JSON."""
    f = files.get("E.bin")
    if not f:
        return None
    b = blob(r, f["sha256"], d / "E.bin").read_bytes()
    if b[:8] != b"CWSEED1\n":
        return None
    (n,) = struct.unpack("<I", b[8:12])
    return json.loads(zlib.decompress(b[12:12 + n]))


def custody_problems(files: dict, custody: dict | None) -> list[str]:
    """The record's witness and harness record against the node's own copy of the run's out/ (its run-record, which the
    node's runner publishes): launch.py uploads the record from an agent's VM, so anything it holds the node didn't write is
    the agent's."""
    names = [n for n in ("witness.json", "run/run.json") if n in files]
    if not names:
        return []
    if custody is None:
        return ["custody: no run-record from the node for this run"]
    cfiles = {f["path"]: f["sha256"] for f in custody["payload"].get("files", [])}
    return [f"custody: {n} is not the node's out/{n}" for n in names if cfiles.get(f"out/{n}") != files[n]["sha256"]]


OFFICIAL = ("cursor/covert-witness-4207", "cursor/te-core-4207")
FREE = re.compile(r"^benchmarks/covert_witness/(islands/|scorer/)|\.md$")  # what a direction may change


def tree(commit: str) -> list[tuple[str, str]] | None:
    """(path, blob) of benchmarks/covert_witness at `commit`, or None if the commit isn't in this repository or origin."""
    g = ["git", "-C", str(REPO)]
    for attempt in range(2):
        kind = subprocess.run(g + ["cat-file", "-t", commit], capture_output=True, text=True).stdout.strip()
        out = subprocess.run(g + ["ls-tree", "-r", commit, "--", "benchmarks/covert_witness"], capture_output=True, text=True)
        if kind == "commit" and out.returncode == 0:
            return [(ln.split("\t", 1)[1], ln.split()[2]) for ln in out.stdout.splitlines()]
        if attempt == 0:
            subprocess.run(g + ["fetch", "-q", "origin", commit], capture_output=True, text=True)
    return None


def official_blobs() -> set[tuple[str, str]]:
    """Every (path, blob) of the fixed side (benchmarks/covert_witness outside FREE) that the harness's branches held since
    30 Oct."""
    g = ["git", "-C", str(REPO)]
    subprocess.run(g + ["fetch", "-q", "origin", *(f"+refs/heads/{b}:refs/remotes/origin/{b}" for b in OFFICIAL)],
                   capture_output=True, text=True)
    revs = subprocess.run(g + ["rev-list", "--since=2026-10-30", *(f"origin/{b}" for b in OFFICIAL), "--",
                               "benchmarks/covert_witness"], capture_output=True, text=True, check=True).stdout.split()
    return {pb for c in revs for pb in tree(c) or [] if not FREE.search(pb[0])}


def fixed_side(commit: str, official: set) -> list[str]:
    """A record's evaluator, harness and starts are the harness's own: every fixed-side file at the candidate's commit is
    one the official branches held. A direction changes only its island (and docs)."""
    t = tree(commit) if commit else None
    if t is None:
        return [f"fixed_side_unverified: commit {commit[:10] or '(none)'} not found"]
    if not {"benchmarks/covert_witness/candidate.sh", "benchmarks/covert_witness/evaluate.py"} <= {p for p, _ in t}:
        return [f"fixed_side_unverified: commit {commit[:10]} holds no harness"]
    diff = sorted(p.removeprefix("benchmarks/covert_witness/") for p, b in t if not FREE.search(p) and (p, b) not in official)
    return [f"fixed side differs from the harness's: {', '.join(diff[:4])}" + (" …" if len(diff) > 4 else "")] if diff else []


def te_current() -> str:
    """TE_CURRENT checked out at te_core's branch head, the te_core every harness record is re-ingested with: an agent's
    branch can carry its own te_core, so its ledger records are listed and never scored."""
    g = ["git", "-C", str(TE_CURRENT)]
    subprocess.run(g + ["fetch", "-q", "origin", "cursor/te-core-4207"], capture_output=True, text=True)
    subprocess.run(g + ["checkout", "-q", "--detach", "FETCH_HEAD"], capture_output=True, text=True, check=True)
    return subprocess.run(g + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def te_ingest(run_dir: Path, root: Path, mechanism: str, run: str) -> list[dict] | str:
    """te_core's submission records (TE_CURRENT's) for the run.json in `run_dir`, kept under `root`; an error string if
    te_core refuses it."""
    if not root.exists():
        tmp = root.with_name(root.name + ".part")
        subprocess.run(["rm", "-rf", str(tmp)], check=True)
        env = {k: v for k, v in os.environ.items() if k != "TE_LEDGER"}
        p = subprocess.run([sys.executable, "-m", "te_core.ledger", "--root", str(tmp), "--agent", "scorer-ingest", "ingest",
                            str(run_dir), "--direction", "scored", "--mechanism", mechanism or "unnamed", "--run-id", run or ""],
                           cwd=TE_CURRENT / "benchmarks" / "covert_witness", env=env, capture_output=True, text=True)
        if p.returncode:
            return f"ingest: te_core.ledger rc {p.returncode}: {(p.stderr.strip().splitlines() or [''])[-1][-200:]}"
        tmp.rename(root)
    return [json.loads(p.read_text()) for p in sorted((root / "records" / "scorer-ingest").glob("*.json"))]


def ingest_run(r, art: str, recorded: str, sha: str, run: str, mechanism: str, te: str) -> list[dict] | str:
    """te_core's submissions for a record's run, one per checkpoint, ingested here by te_core `te` from the run.json the
    node wrote (custody checked)."""
    h = art.removeprefix("art:")
    d = blob(r, sha, STATE / "records" / h / "run" / "run.json").parent
    recs = te_ingest(d, STATE / "ingest" / te[:12] / h, mechanism, run)
    if isinstance(recs, str):
        return recs
    return [te_rec_row(x, f"{art}#{x['checkpoint_step']}", recorded) | {"derived": True, "record": art} for x in recs]


def cw_row(r, art: str, recorded: str, man: dict, commits: dict, custody: dict | None = None, te: str = "",
           official: set | None = None) -> dict:
    cw = man["meta"]["covert_witness"]
    files = {f["path"]: f for f in man["payload"].get("files", [])}
    d = STATE / "records" / art.removeprefix("art:")
    got = {}
    for name in SMALL_FILES:
        if name in files:
            got[name] = json.loads(blob(r, files[name]["sha256"], d / name).read_text())
    w, cand, meta = got.get("witness.json"), got.get("candidate.json", {}), got.get("meta.json", {})
    agent, direction = direction_of(commits.get(cand.get("commit", ""), []), cw.get("class", ""))
    if not agent and cw.get("direction"):
        direction = cw["direction"]
    checks = (w or {}).get("checks", {})
    bad = sorted(k for k, v in checks.items() if v is not True) + custody_problems(files, custody)
    if w and official is not None:
        bad += fixed_side(cand.get("commit", ""), official)
    curve = (w or {}).get("curve") or []
    harness = (w or {}).get("harness")
    notes = (w or {}).get("notes") or {}
    derived = []
    if harness and not bad:
        if "run/run.json" not in files:
            bad = ["harness: the record holds no run/run.json"]
        else:
            derived = ingest_run(r, art, recorded, files["run/run.json"]["sha256"], cw.get("run", ""),
                                 cw.get("mechanism") or mechanism_of(cw.get("note", "")), te)
        if isinstance(derived, str):
            bad, derived = [derived], []
        elif not derived or abs(derived[-1]["macs"] - harness.get("macs", -1)) > 1e-9 * derived[-1]["macs"]:
            bad = [f"harness: te_core's |C| {derived and derived[-1]['macs']} is not the witness's {harness.get('macs')}"]
            derived = []
        elif any(min(t["rho"].values()) < 1 for t in derived):
            bad, derived = ["harness: rho below 1"], []
    for t in derived:
        t |= {"agent": agent, "direction": direction, "start": cw.get("start"), "note": cw.get("note", ""),
              "mechanism": mechanism_of(cw.get("note", "")), "screen": screen_key(cw.get("note", ""))}
    return {
        "id": art, "source": "launch.py", "run": cw.get("run"), "class": cw.get("class"), "start": cw.get("start"),
        "note": cw.get("note", ""), "mechanism": mechanism_of(cw.get("note", "")), "screen": screen_key(cw.get("note", "")),
        "agent": agent, "direction": direction, "commit": cand.get("commit", ""), "args": cand.get("args", ""),
        "launched": run_time(cw.get("run", "")), "recorded": recorded,
        "valid": bool(w and w.get("valid")) and not bad, "failed_checks": bad if w else ["no witness"],
        "derived": derived,
        "witness": (w or {}).get("schema", "covert-witness/witness/v1") if w else None, "counted": bool(harness),
        "train_flops": (harness or {}).get("flops") or (w or {}).get("train", {}).get("flops"),
        "declared_flops": (w or {}).get("train", {}).get("flops"), "tokens": (w or {}).get("train", {}).get("tokens"),
        "bpb": (w or {}).get("score", {}).get("bpb"), "E_bytes": (w or {}).get("E", {}).get("bytes"),
        "E_format": (w or {}).get("E", {}).get("format"),
        "phi": ((w or {}).get("structure_actual") or (w or {}).get("structure") or {}).get("phi"),
        "lowest_updated_layer": (w or {}).get("structure", {}).get("lowest_updated_layer"),
        "gpu_seconds": (w or {}).get("gpu_seconds"), "curve": curve,
        "trainable": meta.get("trainable", {}), "method": notes.get("method"),
        "train_args": notes.get("args"), "steps": notes.get("steps"), "mech": notes.get("mech"),
        "pieces": notes.get("pieces"), "has_E": "E.bin" in files,
        "seedlog": seedlog_header(r, files, d) if (w or {}).get("E", {}).get("format") == "seedlog" else None,
    }


def te_row(r, art: str, recorded: str, man: dict) -> dict | None:
    f = man["payload"]
    sha = f.get("sha256") if f.get("type") == "file" else None
    if not sha:
        return None
    rec = json.loads(blob(r, sha, STATE / "te" / (art.removeprefix("art:") + ".json")).read_text())
    return te_rec_row(rec, art, recorded)


def te_ledger_rows(known: set[str]) -> list[dict]:
    """The te_core ledger's records on this VM (`$TE_LEDGER`, every agent's directory) that the store hasn't shown yet."""
    root = Path(os.environ.get("TE_LEDGER", Path.home() / ".research" / "te-ledger"))
    out = []
    for p in sorted((root / "records").glob("*/*.json")):
        rec = json.loads(p.read_text())
        if rec.get("agent") != p.parent.name or rec.get("id") in known:
            continue
        art = p.with_suffix(".art").read_text().strip() if p.with_suffix(".art").exists() else f"te:{rec['id']}"
        out.append(te_rec_row(rec, art, rec.get("written_at", "")))
    return out


def te_rec_row(rec: dict, art: str, recorded: str) -> dict:
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
        "families": [{"name": f.get("family"), "covers": len(f.get("covers") or []), "declared": len(f.get("declared") or [])}
                     for f in rec.get("families") or []],
    }


def run_time(run: str) -> str:
    m = re.fullmatch(r"r(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})-[0-9a-f]{4}", run or "")
    return f"{m[1]}-{m[2]}-{m[3]}T{m[4]}:{m[5]}:{m[6]}Z" if m else ""


def collect(since: str = SINCE) -> dict:
    r = store_remote()
    t0 = time.time()
    te = te_current()
    every = manifests(r, since)
    by_art = {art: man for art, _, man in every}
    recs = [x for x in every if x[2].get("kind") == "evidence/v1"]
    cw = [x for x in recs if "class" in ((x[2].get("meta") or {}).get("covert_witness") or {})]
    tes = [x for x in recs if str((x[2].get("meta") or {}).get("schema", "")).startswith("te-core/")]
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
    custody = {a["run"]: by_art.get(a["run_record"]) for a in runs if a.get("run_record")}
    official = official_blobs()
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(lambda x: cw_row(r, x[0], x[1], x[2], commits, custody.get(
            x[2]["meta"]["covert_witness"].get("run")), te, official), cw))
        rows += [d for x in rows for d in x.pop("derived")]
        rows += [t for t in ex.map(lambda x: te_row(r, x[0], x[1], x[2]), tes) if t]
    rows += te_ledger_rows({x.get("te_id") for x in rows if x["source"] == "te_core" and not x.get("derived")})
    recorded_runs = {x.get("run") for x in rows}
    for a in runs:
        a["agent"], a["direction"] = direction_of(commits.get(a["commit"], []), a["class"])
        a["record"] = a["run"] in recorded_runs
    by_run = {a["run"]: a for a in runs}
    for x in rows:
        a = by_run.get(x.get("run"))
        if x["source"] == "launch.py" and not x.get("commit") and a:
            x["commit"], x["agent"], x["direction"] = a["commit"], a["agent"], a["direction"]
        if x["source"] == "launch.py":
            x["cancelled"] = bool(a and a.get("cancelled") and x["failed_checks"] == ["no witness"])
    rows.sort(key=lambda x: (x.get("launched") or "", x["id"]))
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "index.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    (STATE / "runs.json").write_text(json.dumps(runs, indent=1) + "\n")
    (STATE / "index.md").write_text(index_md(rows, runs, starts))
    out = {"at": now(), "records": len([x for x in rows if x["source"] == "launch.py"]),
           "te_core_records": len([x for x in rows if x["source"] == "te_core"]), "runs": len(runs),
           "runs_without_record": [a["run"] for a in runs if not a["record"] and not a.get("cancelled")
                                    and run_time(a["run"]) >= SEARCH_START],
           "cancelled": len([a for a in runs if a.get("cancelled")]),
           "te_core_commit": te, "seconds": round(time.time() - t0, 1)}
    (STATE / "collect.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def fmt(x, spec=".3g"):
    return "—" if x is None else format(x, spec)


def index_md(rows: list[dict], runs: list[dict], starts: dict) -> str:
    lines = [f"# The search's ledger, {now()}", "",
             f"{len(rows)} records since {SINCE}; starts' search bpb: " +
             ", ".join(f"{s} {starts[s]['bpb']:.4f}" for s in STARTS), "",
             "| record | agent | direction | class | mechanism note | start | launched | train FLOPs | bpb | Δ% | |E| bytes |"
             " φ | D (c counted, a analytic) | valid |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in entries(rows, audit=True):
        s0 = starts.get(x["start"], {}).get("bpb")
        drop = 100 * (s0 - x["bpb"]) / s0 if s0 and x["bpb"] is not None else None
        if x["source"] == "te_core" or x.get("counted"):
            D, how = x.get("D"), "c"
        else:
            ann = analytic(x, starts) if x["valid"] and x["start"] in starts else None
            D, how = ann and ann["depth"], "a"
        d = x["direction"] + (" (not scored)" if is_audit(x) else " (cancelled unrun)" if x.get("cancelled") else
                              " (not scored: no launch.py record)" if x.get("unrecorded") else "")
        lines.append(f"| `{short(x['id'])}` | {(x['agent'] or '').removeprefix('cursor/') or '—'} | {d} | {x['class']} | "
                     f"{x['note']} | {x['start']} | {(x['launched'] or '')[11:16]} | {fmt(x['train_flops'])} | "
                     f"{fmt(x['bpb'], '.4f')} | {fmt(drop, '.2f')} | {fmt(x['E_bytes'], ',')} | {fmt(x['phi'], '.3f')} | "
                     f"{fmt(D, '.3g')} {how if D is not None else ''} | "
                     f"{'yes' if x['valid'] else 'no: ' + ','.join(x['failed_checks'])} |")
    pend = [a for a in runs if not a["record"] and not a.get("cancelled") and run_time(a["run"]) >= SEARCH_START]
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
    if merges_pieces(row):
        full |= set(adapters)  # every merged piece makes the target's own W·x a multiplication by an updated weight
    upd = []  # (key, param, sub, per-token MACs): an adapter is x·Aᵀ (sub 0) then its product with Bᵀ (sub 1)
    for n in nodes_of(start):
        if n["param"] in full:
            upd.append((n["key"], n["param"], 0, n["macs"]))
        if n["param"] in adapters:
            r = adapters[n["param"]]
            upd += [(n["key"], n["param"], 0, r * n["d_in"]), (n["key"], n["param"], 1, r * n["d_out"])]
    profile = [(sum(mm for k, _, _, mm in upd if k < key) + sum(mm for k, p, s, mm in upd if k == key and p == param and s < sub), m)
               for key, param, sub, m in upd]
    tokens = row.get("tokens") or {}
    t_upd = fraction * (int(tokens.get("train", 0)) + int(tokens.get("forward", 0)))
    macs = fraction * row["train_flops"] / 2
    per_token = sum(m for _, m in profile)
    depth = max((u for u, _ in profile), default=0)
    c = block_c(forward_macs_per_token(start))
    return {"macs": macs, "tokens": t_upd, "profile": [(u, m * t_upd) for u, m in profile], "depth": depth,
            "updated_per_token": per_token, "c": c, "credit": credit(row, start, fraction, per_token, c)}


def merges_pieces(row: dict) -> bool:
    """A ReLoRA-style run (notes.mech relora, notes.pieces > 1) merges each piece into the running weights."""
    return row.get("mech") == "relora" and int(row.get("pieces") or 0) > 1


def credit(row: dict, start: dict, fraction: float, per_token: float, c: float) -> dict | None:
    """The counterexample family a seed-log E gives, for a record whose witness carries no harness-verified family.

    A log with one coefficient a training step (MeZO, each step's perturbation `seedlog_step` itself): E_t, its first t
    coefficients with ε and the sign, rebuilds step t's weights (masters after t `seedlog_step`s, plus ±εz_t) in (t + 1) n
    MACs for n trained values, so it covers the steps with (t + 1) n ≤ c, at β_t = bits(E_t) / updated MACs per token,
    bounded here by the smallest snapshot's E. A log of fewer coefficients than steps (E's header) is a trained subspace
    whose steps the decoder rebuilds only if they ran its own sequential fp32 adds, which a v1 record doesn't show, so it
    gets none (the harness's `declare_step` credits such steps). Adapter and full E's rebuild no step's fp32 values bit for
    bit, and their bits per updated MAC are at least p, so they get none."""
    if row.get("E_format") != "seedlog" or not per_token:
        return None
    names = (row.get("trainable") or {}).get("full", [])
    n = sum(math.prod(start["tensors"][k][1]) for k in names if k in start["tensors"])
    steps = int(row.get("steps") or 0)
    k = int((row.get("seedlog") or {}).get("steps") or steps)
    if not n or not steps or k != steps:
        return None
    t_max = int(c // n) - 1
    covered = min(1.0, max(0, t_max + 1) / max(1.0, fraction * steps))
    e_min = min((cp["e_bytes"] for cp in row.get("curve") or [] if cp.get("e_bytes")), default=row.get("E_bytes") or 0)
    return {"coefficients": k, "covered_share": covered, "beta_t": (8 * e_min + 33) / per_token,
            "rebuild_macs_last": (fraction * steps + 1) * n, "values": n}


def annotation_of(a: dict) -> tuple[Annotation, dict[int, float]]:
    """Two synthetic steps, the covered steps' share and the rest, and the coverage te_core's score reads."""
    cr = a.get("credit")
    q = cr["covered_share"] if cr else 0.0
    steps, covered = [], {}
    for i, share in enumerate((q, 1.0 - q)):
        if share <= 0:
            continue
        prof = tuple(sorted((int(u), int(m * share)) for u, m in a["profile"]))
        upd = sum(m for _, m in prof)
        steps.append(StepAnnotation(step=i, tokens=int(a["tokens"] * share), macs=int(a["macs"] * share),
                                    updated_macs=upd, updated_per_token=a["updated_per_token"], depth=int(a["depth"]),
                                    profile=prof, bits=P_BITS if upd else 0, weights={}, shapes={}))
        if i == 0:
            covered[0] = cr["beta_t"]
    return Annotation(macs=int(a["macs"]), steps=tuple(steps)), covered


def is_reference(x: dict) -> bool:
    """honest(l*) reads the reference directions (ref-*, and the harness's own reference smoke), whatever the class; every
    other direction's records, a reference-class one included, are W*'s at their ρ."""
    return x["direction"].startswith("ref-") or x["direction"] == "reference"


def is_audit(x: dict) -> bool:
    """The measurement auditor's records exercise measurement gaps on purpose: in the ledger, out of every score."""
    return x.get("direction") == "audit" or str(x.get("note", "")).lower().startswith("audit:")


def run_rows(rows: list[dict]) -> dict[str, dict]:
    return {x["run"]: x for x in rows if x["source"] == "launch.py" and x.get("run")}


def submissions(rows: list[dict], starts: dict) -> list[dict]:
    """Every checkpoint of every valid record as a te_core submission: {id, cp, start, macs, val_bpb, rho, phi, depth}.
    A record made inside te_core's harness (witness v2) is scored from the te_core submissions `collect` ingested from the
    node's run.json (one per checkpoint, the harness's counted annotation); the analytic annotation is only for witness
    v1 records. The te_core records agents publish themselves are listed, never scored."""
    out = []
    launched = run_rows(rows)
    for x in rows:
        if x["source"] == "te_core" and not x.get("derived"):
            continue
        rec = launched.get(x.get("run")) if x["source"] == "te_core" else None
        if x["source"] == "te_core" and not (rec and rec["valid"]):
            continue
        if not x.get("valid") or x.get("start") not in starts or is_audit(x):
            continue
        if x["source"] == "te_core":
            if x.get("te_kind") == "submission" and x.get("rho"):
                out.append({"id": rec["id"] if rec else x["id"], "te_id": x["te_id"], "cp": x.get("fraction"),
                            "start": x["start"], "macs": x["macs"], "val_bpb": x["bpb"], "rho": x["rho"], "phi": x["phi"],
                            "depth": x["D"], "reference": is_reference(x), "direction": x["direction"], "counted": True,
                            "E_bytes": x.get("E_bytes")})
            continue
        if x.get("counted"):
            continue
        total = x["train_flops"] or 0
        for i, cp in enumerate(x.get("curve") or []):
            if not isinstance(cp.get("bpb"), (int, float)) or not total:
                continue
            f = cp["train_flops"] / total
            a = analytic(x, starts, f)
            ann, cov = annotation_of(a)
            grid = score_grid(ann, cov, P_BITS, a["c"])
            out.append({"id": x["id"], "cp": round(f, 3), "start": x["start"], "macs": a["macs"], "val_bpb": cp["bpb"],
                        "rho": {k: v["rho"] for k, v in grid.items()}, "phi": {k: v["phi"] for k, v in grid.items()},
                        "option": {k: v["option"] for k, v in grid.items()}, "depth": a["depth"],
                        "reference": is_reference(x), "direction": x["direction"], "counted": False,
                        "E_bytes": cp.get("e_bytes"), "credit": a["credit"]})
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
    col = STATE / "collect.json"
    rep = {"at": now(), "te_core": TE_CORE_VERSION,
           "te_core_commit": json.loads(col.read_text()).get("te_core_commit", "") if col.exists() else "",
           "annotation": "the harness's count for witness v2 records, analytic for witness v1 (module docstring)",
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
                                   "direction": arg and arg["direction"], "counted": arg and arg["counted"]}
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


def entries(rows: list[dict], audit: bool = False) -> list[dict]:
    """One row per candidate: a launch.py record, or (listed with `audit` only) a te_core run no record of launch.py's
    holds. A witness v2 record takes |C|, φ and D (read at 10kB/r1) from the last te_core checkpoint `collect` ingested
    from its node's run.json, and the te id of its agent's own last checkpoint, if any, for the ledger's promotion. The
    auditor's records and the records of runs cancelled unrun only with `audit` (the ledger lists them; nothing scores or
    counts them)."""
    last: dict[str, dict] = {}
    own: dict[str, dict] = {}
    for x in rows:
        if x["source"] == "te_core" and x.get("te_kind") == "submission":
            m, k = (last, x["record"]) if x.get("derived") else (own, x.get("run") or x["te_id"])
            if k not in m or (x.get("fraction") or 0) > (m[k].get("fraction") or 0):
                m[k] = x
    out = []
    for x in rows:
        if x["source"] != "launch.py":
            continue
        t = last.get(x["id"])
        o = own.pop(x["run"], None) if x.get("run") else None
        out.append(x | ({"phi": at10(t["phi"]), "D": at10(t["D"]), "train_flops": 2 * t["macs"], "te_id": o and o["te_id"],
                         "tier": t.get("tier"), "families": t.get("families"), "ingested": True} if t else {}))
    out += [x | {"phi": at10(x.get("phi")), "D": at10(x.get("D")), "unrecorded": True} for x in own.values()]
    return sorted((x for x in out if audit or not (is_audit(x) or x.get("cancelled") or x.get("unrecorded"))),
                  key=lambda x: (x.get("launched") or "", x["id"]))


def at10(v):
    return v.get("10kB/r1") if isinstance(v, dict) else v


def promotions(rows: list[dict], starts: dict, promoted: dict) -> dict:
    """This hour's promotions: each direction's top third by relative bpb drop, and the best of each thin cell (at most
    THIN valid records), among valid records of the search (launched from SEARCH_START) not promoted before; plus
    re-screens of mechanisms already screened."""
    valid = [x for x in entries(rows) if x["valid"] and drop_of(x, starts) is not None
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
    for x in entries(rows):
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
    valid = [x for x in entries(rows) if x["valid"]]
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
            ledger_promote(pr["new"], rows)
            for p in pr["new"]:
                promoted[p["id"]] = {"at": hhmm, "reasons": p["reasons"]}
            ppath.write_text(json.dumps(promoted, indent=1) + "\n")
            (STATE / "rescreens.json").write_text(json.dumps(sorted({x["id"] for x in pr["rescreens"]})) + "\n")
    return lines


def ledger_promote(new: list[dict], rows: list[dict]) -> list[str]:
    """A promoted te_core entry at tier 0 gets the scorer's promotion record to tier 1 in the te_core ledger (published),
    which tier 1's launch gate reads; tier 2 is final scoring's."""
    from te_core.ledger import Ledger

    ents = {x["id"]: x for x in entries(rows)}
    led = Ledger(agent="scorer", publish=True)
    out = []
    for p in new:
        x = ents.get(p["id"])
        if x and x.get("te_id") and int(x.get("tier") or 0) < 1:
            led.promote(x["te_id"], 1, "; ".join(p["reasons"]))
            out.append(x["te_id"])
    return out


def digest_lines(rows, starts, rep, pr, hhmm) -> list[str]:
    out = []
    cw = entries(rows)
    n, nv = len(cw), len([x for x in cw if x["valid"]])
    nc = len([x for x in rows if x["source"] == "launch.py" and x.get("cancelled")])
    out.append(f"{hhmm} scorer [DIGEST] {n} records ({nv} valid; direction audit's and {nc} of runs cancelled unrun not "
               f"counted). Frontier per target "
               f"below: honest = least |C| (MACs) of the reference directions (ref-*) reaching l*; W*/honest per (k, r) in the "
               f"order 10kB r1,r10,r100 | 100kB | 1MB, te_core rho, with the W* entries, (c) priced by the harness's "
               f"count, (a) by the analytic annotation (p 16); full tables in the scorer's score.md.")
    for s, st in rep["starts"].items():
        parts = []
        for t in st["targets"]:
            h = t["honest_macs"]
            cells = t["cells"]
            if h is None and all(c["w_star"] is None for c in cells.values()):
                continue
            eff = [cells[k]["effective_slowdown"] for k in KEYS]
            vec = " | ".join(", ".join(fmt(e, ".3g") for e in eff[i: i + 3]) for i in (0, 3, 6))
            ids = sorted({short(c["id"]) + (" (c)" if c.get("counted") else " (a)") for c in cells.values() if c["id"]})
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
    cred = [(x, analytic(x, starts)["credit"]) for x in cw
            if x["source"] == "launch.py" and not x.get("counted") and x["valid"] and x["start"] in starts]
    cred = [(x, c) for x, c in cred if c and c["covered_share"] > 0]
    fam = [x for x in cw if x.get("ingested") and any(f["covers"] for f in x.get("families") or [])]
    out.append(f"{hhmm} scorer [DIGEST] E credited as a T1 counterexample family (its decoder rebuilds each covered "
               f"step's weights from theta within c): witness v1 seed logs as the witness states them, " + (", ".join(
                   f"{short(x['id'])} (prefix log of {c['coefficients']} coefficients, "
                   f"{100 * c['covered_share']:.0f}% of steps, beta_t {c['beta_t']:.2g}, last rebuild "
                   f"{c['rebuild_macs_last']:.2g} MACs)" for x, c in cred) or "none")
               + "; harness-verified families (witness v2, te_core.run): "
               + (", ".join(f"{short(x['id'])} ({families_of(x)})" for x in fam) or "none")
               + "; no other entry's E is credited.")
    v2 = [x for x in cw if x["source"] == "launch.py" and x.get("counted")]
    refused = [x for x in v2 if not x.get("ingested")]
    unrec = [x for x in entries(rows, audit=True) if x.get("unrecorded")]
    out.append(f"{hhmm} scorer [DIGEST] counted: {len(v2)} witness v2 records priced by the harness's count, re-ingested "
               f"here by te_core {rep.get('te_core_commit', '')[:10]} from the run.json the node wrote (custody: the "
               f"record's witness and run.json are the node's out/)"
               + (f" ({len(refused)} refused: " + ", ".join(f"{short(x['id'])} {';'.join(x['failed_checks'])}"
                                                            for x in refused) + ")" if refused else "")
               + f"; {len([x for x in cw if x['source'] == 'launch.py' and not x.get('counted')])} witness v1 records "
               "priced by the analytic annotation (the declared trainable set, which the witness checks covers every "
               "changed tensor, so U, phi and D are never understated; the step log's tokens are self-reported); "
               + (f"{len(unrec)} te_core runs with no launch.py record listed, not scored." if unrec else
                  "te_core ledger records agents publish are listed, never scored."))
    for d in sorted({x["direction"] for x in cw}):
        out.append(f"{hhmm} scorer [DIGEST] {d}: " + direction_sentence(d, rows, starts))
    return out


def families_of(x: dict) -> str:
    return ", ".join(f"{f['name']} covers {f['covers']}/{f['declared']} steps" for f in x.get("families") or [])


def filled_of(rows, s) -> set[str]:
    return {f"{e_bin(x['E_bytes'])}/{phi_bin(x['phi'])}" for x in entries(rows)
            if x["valid"] and x["start"] == s and (x["E_bytes"] or 0) < 1e7}


def direction_sentence(d: str, rows: list[dict], starts: dict) -> str:
    xs = [x for x in entries(rows) if x["direction"] == d]
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
    """Append to the scorer's own board file and push its branch; the coordinator merges it into the shared board."""
    BOARD.parent.mkdir(parents=True, exist_ok=True)
    wt = subprocess.run(["git", "-C", str(BOARD.parent), "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                        check=True).stdout.strip()
    text = BOARD.read_text() if BOARD.exists() else BOARD_HEAD
    BOARD.write_text(text + ("" if text.endswith("\n") else "\n") + "\n".join(lines) + "\n")
    tags = sorted({m[1] for m in (re.search(r"\[(\w+)\]", x) for x in lines) if m})
    subprocess.run(["git", "-C", wt, "add", str(BOARD)], check=True)
    subprocess.run(["git", "-C", wt, "commit", "-q", "-m", f"board: scorer {lines[0][:6]} {', '.join(tags)}", "--", str(BOARD)],
                   check=True)
    for attempt in range(5):
        if subprocess.run(["git", "-C", wt, "push", "-q", "origin", "HEAD"]).returncode == 0:
            return
        time.sleep(2 ** attempt)
    raise SystemExit("scorer: board push failed five times")


# ---- final scoring: reruns of an entry at fresh seeds, scored on the final set ----------------------------------------

FINAL = STATE / "final"
FINAL_SRC = Path(os.environ.get("TE_FINAL_SRC", "/tmp/scorer/final-src"))
FINAL_SH = Path(__file__).resolve().parent / "final.sh"
FINAL_NODES = tuple(os.environ.get("TE_FINAL_NODES", "vy-nebius-2").split(","))
FINAL_SEEDS = (101, 102, 103)
RUNS = Path.home() / ".research" / "runs"


def source_at(commit: str) -> Path:
    """A clean detached checkout of `commit` to ship (`research run --source`); only covert_witness and the research package
    the machine runs (the tree's own, as for a candidate) are checked out."""
    if not (FINAL_SRC / ".git").exists():
        subprocess.run(["git", "-C", str(REPO), "worktree", "add", "-q", "--no-checkout", "--detach", str(FINAL_SRC), commit],
                       check=True)
        subprocess.run(["git", "-C", str(FINAL_SRC), "sparse-checkout", "set", "--no-cone", "/benchmarks/covert_witness/",
                        "/tools/research/"], check=True)
    subprocess.run(["git", "-C", str(FINAL_SRC), "checkout", "-q", "--detach", commit], check=True)
    if subprocess.run(["git", "-C", str(FINAL_SRC), "status", "--porcelain"], capture_output=True, text=True).stdout.strip():
        raise SystemExit(f"scorer: {FINAL_SRC} is not clean at {commit}")
    return FINAL_SRC


def research(*args: str) -> subprocess.CompletedProcess:
    """This tree's research package: the VM's install predates pools a node keeps of its own (gpu)."""
    env = os.environ | {"PYTHONPATH": str(REPO / "tools" / "research" / "src")}
    return subprocess.run([sys.executable, "-m", "research", *args], capture_output=True, text=True, cwd=REPO, env=env)


def final_run(commit: str, args: list[str], what: dict, dry_run: bool = False, node: str = FINAL_NODES[0]) -> dict:
    src = source_at(commit)
    argv = ["research", "run", "--on", node, "--project", "verity", "--source", str(src), "--cwd", "source",
            "--pool", "gpu", "--as", "training", "--pool-wait", "7200", "--timeout", "120m", "--declared-output", "out/*",
            "--send", str(FINAL_SH), "--", "bash", "-c", 'bash "$RESEARCH_RUN_DIR/inputs/final.sh" ' + shlex.join(args)]
    row = what | {"commit": commit, "args": args, "at": now(), "argv": argv, "run": "", "node": node}
    if not dry_run:
        r = research(*argv[1:])
        m = re.search(r"run (r\d{8}-\d{6}-[0-9a-f]{4}) launched", r.stdout + r.stderr)
        row |= {"run": m[1] if m else "", "rc": r.returncode, "err": "" if m else (r.stdout + r.stderr)[-800:]}
        FINAL.mkdir(parents=True, exist_ok=True)
        with (FINAL / "launches.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
    return row


def seed_args(row: dict, seed: int) -> list[str]:
    """The seed flags the entry's script takes (its witness's recorded args), set to `seed`."""
    have = row.get("train_args") or {}
    return [x for k, flag in (("seed", "--seed"), ("data_seed", "--data-seed")) if k in have for x in (flag, str(seed))]


def final_launch(ids: list[str], seeds=FINAL_SEEDS, dry_run: bool = False) -> list[dict]:
    rows = {x["id"]: x for x in (json.loads(l) for l in (STATE / "index.jsonl").read_text().splitlines() if l.strip())}
    out = []
    for i in ids:
        x = next((r for k, r in rows.items() if k.removeprefix("art:").startswith(i.removeprefix("art:"))), None)
        if x is None or x["source"] != "launch.py" or not x.get("commit"):
            raise SystemExit(f"scorer: {i} is not a launch.py record with a commit")
        for seed in seeds:
            args = ["--seed", str(seed), "--direction", re.sub(r"[^a-z0-9_-]", "-", x["direction"].lower()) or "scorer-final",
                    "--mechanism", x["mechanism"], *(["--reference"] if is_reference(x) else []), x["class"], x["start"],
                    *shlex.split(x.get("args") or ""), *seed_args(x, seed)]
            out.append(final_run(x["commit"], args, {"kind": "rerun", "id": x["id"], "seed": seed}, dry_run,
                                 FINAL_NODES[len(out) % len(FINAL_NODES)]))
    return out


def final_collect() -> list[dict]:
    """Every launched final run's state and, once it ended, its scores (from the run's out/)."""
    path = FINAL / "launches.jsonl"
    launches = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    out, te = [], ""
    for la in launches:
        run = la.get("run")
        if not run:
            continue
        on = ["--on", la["node"]] if la.get("node") else []
        research("fetch", run, *on)
        st = (json.loads((RUNS / run / "status.json").read_text()).get("state")
              if (RUNS / run / "status.json").exists() else "unknown")
        row = {k: la.get(k) for k in ("kind", "id", "seed", "run", "commit", "args")} | {"state": st}
        if st in ("done", "failed", "cancelled", "refused", "timeout", "lost"):
            o = RUNS / run / "out"
            if not o.is_dir():
                research("fetch", run, "--all", *on)
            for f in sorted(o.glob("*.json")) if o.is_dir() else []:
                d = json.loads(f.read_text())
                if f.name.startswith("witness."):
                    sp = f.name.split(".")[1]
                    row[sp] = {"valid": d.get("valid"), "bpb": (d.get("score") or {}).get("bpb"), "curve": d.get("curve"),
                               "train_flops": (d.get("train") or {}).get("flops"),
                               "tokens": (d.get("train") or {}).get("tokens"), "E_bytes": (d.get("E") or {}).get("bytes"),
                               "E_format": (d.get("E") or {}).get("format"), "phi": (d.get("structure") or {}).get("phi"),
                               "steps": (d.get("notes") or {}).get("steps"), "gpu_seconds": d.get("gpu_seconds"),
                               "failed_checks": sorted(k for k, v in (d.get("checks") or {}).items() if v is not True)}
                elif f.name.startswith("start."):
                    _, s, sp, _ = f.name.split(".")
                    row.setdefault("starts", {}).setdefault(s, {})[sp] = (d.get("score") or {}).get("bpb")
            if (o / "run" / "run.json").exists():
                te = te or te_current()
                a = la.get("args") or []
                m = a[a.index("--mechanism") + 1] if "--mechanism" in a[:-1] else ""
                recs = te_ingest(o / "run", FINAL / "ingest" / te[:12] / run, m, run)
                row["ledger"] = [] if isinstance(recs, str) else [te_rec_row(x, "", "") | {"derived": True} for x in recs]
                row["ingest_error"] = recs if isinstance(recs, str) else None
        out.append(row)
    FINAL.mkdir(parents=True, exist_ok=True)
    (FINAL / "results.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out))
    return out


def seed_mean(subs: list[dict]) -> list[dict]:
    """One submission per (entry, checkpoint) over its seeds: mean |C| and bpb, and ρ such that |C|·ρ is the mean |C|·ρ."""
    by: dict[tuple, list[dict]] = {}
    for s in subs:
        by.setdefault((s["id"], round(s["cp"], 2)), []).append(s)
    out = []
    for (i, cp), ss in sorted(by.items()):
        macs = sum(s["macs"] for s in ss) / len(ss)
        bpbs = [s["val_bpb"] for s in ss]
        out.append(ss[0] | {"cp": cp, "macs": macs, "val_bpb": sum(bpbs) / len(bpbs), "bpb_min": min(bpbs),
                            "bpb_max": max(bpbs), "seeds": sorted(s.get("seed") for s in ss),
                            "rho": {k: sum(s["macs"] * s["rho"][k] for s in ss) / len(ss) / macs for k in KEYS}})
    return out


def table(subs: list[dict], l0: dict[str, float]) -> dict:
    """W*(l*), honest(l*) and W*/honest per start, target and (k, r), with the entries that attain them."""
    out = {}
    for s in STARTS:
        on = [x for x in subs if x["start"] == s]
        if s not in l0 or not on:
            continue
        refs, cov = [x for x in on if x["reference"]], [x for x in on if not x["reference"]]
        rows = []
        for f in TARGETS:
            l_star = l0[s] * (1 - f)
            h = min((x for x in refs if x["val_bpb"] <= l_star), key=lambda x: x["macs"], default=None)
            cells = {}
            for key in KEYS:
                arg = min((x for x in cov if x["val_bpb"] <= l_star), key=lambda x: x["macs"] * x["rho"][key], default=None)
                cells[key] = {"w_star": w_star(cov, l_star, key), "effective_slowdown": effective_slowdown(cov, refs, l_star, key),
                              "id": arg and arg["id"], "cp": arg and arg["cp"], "rho": arg and arg["rho"][key],
                              "direction": arg and arg["direction"], "seeds": arg and arg.get("seeds")}
            rows.append({"drop": f, "l_star": l_star, "honest_macs": honest(refs, l_star), "honest_id": h and h["id"],
                         "honest_cp": h and h["cp"], "honest_seeds": h and h.get("seeds"), "cells": cells})
        out[s] = {"start_bpb": l0[s], "targets": rows, "entries": sorted({x["id"] for x in on})}
    return out


def table_md(tab: dict, split: str, seeds: str) -> str:
    lines = [f"W* and the effective slowdown W*/honest on the {split} set (te_core {TE_CORE_VERSION} ρ on the scorer's analytic "
             f"annotation, p = {P_BITS}; FLOPs = 2 MACs; {seeds})", ""]
    for s, st in tab.items():
        lines += [f"### {s} (start {st['start_bpb']:.4f} bpb)", "",
                  "| target | l* | honest FLOPs (entry) | k | W* r1 | W* r10 | W* r100 | W*/honest r1 | r10 | r100 | W* entries |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in st["targets"]:
            h = t["honest_macs"]
            for k in K_BITS:
                cs = [t["cells"][f"{k}/r{r}"] for r in R]
                ids = ", ".join(sorted({f"{short(c['id'])}@{c['cp']}" for c in cs if c["id"]})) or "—"
                lines.append(f"| -{100 * t['drop']:g}% | {t['l_star']:.4f} | "
                             + (f"{fmt(h and 2 * h)} ({short(t['honest_id'])}@{t['honest_cp']})" if h else "—")
                             + f" | {k} | " + " | ".join(fmt(c["w_star"] and 2 * c["w_star"]) for c in cs) + " | "
                             + " | ".join(fmt(c["effective_slowdown"]) for c in cs) + f" | {ids} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def plot(subs: list[dict], l0: dict[str, float], title: str, path: Path) -> None:
    """Per start, bpb against FLOPs: the reference checkpoints at |C|, every other at |C|·ρ for k = 10 kB, 100 kB, 1 MB (r = 1,
    solid; r = 10, dashed), each with its frontier, and the targets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    starts = [s for s in STARTS if s in l0 and any(x["start"] == s for x in subs)]
    if not starts:
        return
    fig, axes = plt.subplots(len(starts), 3, figsize=(15, 4.2 * len(starts)), squeeze=False)
    dirs = sorted({x["direction"] for x in subs if not x["reference"]})
    color = {d: plt.cm.tab20(i % 20) for i, d in enumerate(dirs)}
    for row, s in zip(axes, starts):
        on = [x for x in subs if x["start"] == s]
        refs, cov = [x for x in on if x["reference"]], [x for x in on if not x["reference"]]
        for ax, k in zip(row, K_BITS):
            for f in TARGETS:
                ax.axhline(l0[s] * (1 - f), color="0.8", lw=0.8, zorder=0)
                ax.text(1.0, l0[s] * (1 - f), f" -{100 * f:g}%", transform=ax.get_yaxis_transform(), fontsize=7, va="center")
            if refs:
                ax.scatter([2 * x["macs"] for x in refs], [x["val_bpb"] for x in refs], c="k", marker="s", s=18,
                           label="reference |C|")
                fr = pareto([(x["macs"], x["val_bpb"], x) for x in refs])
                ax.step([2 * x["macs"] for x in fr], [x["val_bpb"] for x in fr], where="post", c="k", lw=1)
            for r, ls in ((1, "-"), (10, "--")):
                key = f"{k}/r{r}"
                for d in dirs:
                    xs = [x for x in cov if x["direction"] == d]
                    if xs and r == 1:
                        ax.scatter([2 * x["macs"] * x["rho"][key] for x in xs], [x["val_bpb"] for x in xs], color=color[d],
                                   s=14, label=d)
                fr = pareto([(x["macs"] * x["rho"][key], x["val_bpb"], x) for x in cov])
                if fr:
                    ax.step([2 * x["macs"] * x["rho"][key] for x in fr], [x["val_bpb"] for x in fr], where="post",
                            c="tab:red", ls=ls, lw=1, label=f"covert frontier r{r}")
            ax.set_xscale("log")
            ax.set_ylim(l0[s] * 0.93, l0[s] * 1.01)
            ax.set_title(f"{s}, k = {k}", fontsize=9)
            ax.set_xlabel("FLOPs (reference: |C|; covert: |C| rho)", fontsize=8)
            ax.set_ylabel("bpb", fontsize=8)
            ax.tick_params(labelsize=7)
    h, lab = axes[0][0].get_legend_handles_labels()
    seen = dict(zip(lab, h))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=min(8, len(seen)), fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def final_table(write: bool = True) -> dict:
    """The final-set table and plot from the reruns, beside the search-set ones from every record."""
    rows = {x["id"]: x for x in (json.loads(l) for l in (STATE / "index.jsonl").read_text().splitlines() if l.strip())}
    starts = json.loads((STATE / "starts.json").read_text())
    res = [json.loads(l) for l in (FINAL / "results.jsonl").read_text().splitlines() if l.strip()] \
        if (FINAL / "results.jsonl").exists() else []
    l0 = {}
    for r in res:
        for s, v in (r.get("starts") or {}).items():
            if v.get("final") is not None:
                l0.setdefault(s, v["final"])
    subs = []
    for r in res:
        x, w = rows.get(r.get("id")), r.get("final")
        if r.get("kind") != "rerun" or not x or not w or not w.get("valid"):
            continue
        if r.get("ledger"):
            # the harness's tier-2 checkpoints, each scored on the final set (final_bpb) and priced by its count
            rec = x | {"run": r["run"], "counted": True}
            led = [t | {"run": r["run"], "valid": not t["failed_checks"], "bpb": t["test_bpb"], "direction": x["direction"]}
                   for t in r["ledger"] if t.get("test_bpb") is not None]
            subs += [s | {"seed": r["seed"]} for s in submissions([rec, *led], starts)]
            continue
        rr = x | {k: w[k] for k in ("train_flops", "tokens", "curve", "E_bytes", "phi") if w.get(k) is not None} | {
            "steps": w.get("steps") or x.get("steps")}
        subs += [s | {"seed": r["seed"]} for s in submissions([rr], starts)]
    fin = seed_mean(subs)
    search_subs = json.loads((STATE / "submissions.json").read_text())
    l0_search = {s: starts[s]["bpb"] for s in STARTS}
    rep = {"at": now(), "te_core": TE_CORE_VERSION, "final": table(fin, l0), "search": table(search_subs, l0_search),
           "final_starts": l0, "reruns": len(subs), "entries": sorted({x["id"] for x in fin})}
    if write:
        FINAL.mkdir(parents=True, exist_ok=True)
        (FINAL / "table.json").write_text(json.dumps(rep, indent=1) + "\n")
        (FINAL / "final_submissions.json").write_text(json.dumps(fin) + "\n")
        (FINAL / "table.md").write_text(table_md(rep["final"], "final", "seeds averaged per checkpoint") + "\n"
                                         + table_md(rep["search"], "search", "the search run's one seed"))
        plot(fin, l0, "final set, seeds averaged", FINAL / "frontier-final.png")
        plot(search_subs, l0_search, "search set, one seed", FINAL / "frontier-search.png")
    return rep


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
    p = sub.add_parser("post", help="append one line to the board and push")
    p.add_argument("tag")
    p.add_argument("text")
    f = sub.add_parser("final", help="final-set reruns: starts, launch IDS, collect")
    f.add_argument("what", choices=("starts", "launch", "collect", "table"))
    f.add_argument("ids", nargs="*", help="launch: record ids (12-hex prefixes); starts: start names")
    f.add_argument("--commit", default=None, help="starts: the commit to run in (default: HEAD's fixed side)")
    f.add_argument("--seeds", default=",".join(map(str, FINAL_SEEDS)))
    f.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "final":
        if a.what == "starts":
            commit = a.commit or subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                                text=True, check=True).stdout.strip()
            res = [final_run(commit, ["--starts", *(a.ids or STARTS)], {"kind": "starts"}, a.dry_run)]
        elif a.what == "launch":
            res = final_launch(a.ids, tuple(int(s) for s in a.seeds.split(",")), a.dry_run)
        elif a.what == "table":
            final_table()
            print((FINAL / "table.md").read_text())
            return 0
        else:
            res = final_collect()
        for r in res:
            print(json.dumps({k: v for k, v in r.items() if k != "argv"} if not a.dry_run else r))
        return 0
    if a.cmd == "post":
        append_board([f"{datetime.now(timezone.utc).strftime('%H:%MZ')} scorer [{a.tag}] {a.text}"])
        return 0
    if a.cmd == "collect":
        print(json.dumps(collect(a.since)))
    elif a.cmd == "score":
        print(score_md(score()))
    else:
        print("\n".join(board(a.promote, a.digest, a.append, a.hhmm)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

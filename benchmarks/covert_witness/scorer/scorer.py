"""The training-efficiency search's scorer: one ledger of the launcher's records, ρ and W* from te_core, promotions, digests.

    python scorer.py collect                  # the store's new manifests -> STATE/index.jsonl and index.md
    python scorer.py score                    # STATE/score.json and score.md: ρ per (k, r), W*(l*), honest(l*), W*/honest
    python scorer.py board [--promote] [--digest] [--append]   # this hour's PROMOTED and DIGEST lines (--append pushes them)
    python scorer.py post TAG TEXT            # one line onto the scorer's board (islands/scorer/BOARD.md), pushed
    python scorer.py final launch ID...|collect|table   # FINAL entries rerun in final mode (launch.py --final), the table

STATE is `$TE_SCORER_STATE` (default ~/.research/te-scorer); everything in it is derived from the evidence store and
rebuilt by `collect`. Records are the evidence/v1 artifacts whose meta has `covert_witness.class` (launch.py's), and the
te-core/submission/v1 ledger records (te_core.ledger's), written since `$TE_SCORER_SINCE`.

A record is scored only if its witness and run.json are the node's (`custody_problems`: launch.py uploads records from an
agent's VM, the node's runner publishes the run's out/) and its fixed side is the harness's (a pin, or `fixed_side`). A
record made inside te_core's harness (witness v2, covert_witness v2.1) is priced by the harness's count: `collect` ingests
the node's run.json with the newest official te_core (`te_current`), one submission per checkpoint carrying |C|, U, D, φ
and ρ; the te_core records agents publish themselves are listed, never scored. Only the pin's records (`regime_of`)
make the frontier, the promotions (decided by the sealed set, `sealed_bits`) and the final table. A witness v1 record carries the witness
(training FLOPs, φ over forward weight-matmul FLOPs, |E|, search bpb, the FLOPs-to-target curve), not te_core's count. Its
annotation here is analytic, from the record's declared trainable set and step-log tokens over the start's tensors: U is
every forward multiplication whose weight operand is trained (a full tensor's matmul or norm gain, an adapter's two
factors; a frozen base W·x is θ's, except in a run that merges its pieces, whose targets are full), each node's upstream
is the updated MACs per token of the nodes before it in (layer, stage) order, |C| = training FLOPs / 2, p = 16 (bf16
autocast) and c = 4096 times the start's forward MACs per token (weight matmuls plus 2·d·ctx attention MACs a layer).
ρ is te_core's (`score.score_grid`) on that annotation, with the coverage `credit` reads off a seed-log E.

honest(l*), W*(l*) and the effective slowdown are te_core.frontier.target's (TE_PIN's te_core 3.5.3; coordinator 14:57Z):
honest is the least |C| over every valid record's complete schedules, of any class or direction, W* the least |C|ρ over
the same records, so W*/honest >= 1; the reference directions' honest alone is kept as context. Records cw.readout_leaks
lists are invalid (15:00Z), and the measurement auditor's records (direction audit) are in the ledger only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STATE = Path(os.environ.get("TE_SCORER_STATE", Path.home() / ".research" / "te-scorer"))
SINCE = os.environ.get("TE_SCORER_SINCE", "2026-11-02T06:30:00Z")
SEARCH_START = "2026-11-02T08:30:00Z"  # the board's opening: earlier records are the integrator's seed smokes
TE_CORE = Path(os.environ.get("TE_CORE_PATH", Path(__file__).resolve().parents[1]))
TE_CURRENT = Path(os.environ.get("TE_CURRENT", "/tmp/scorer/te-core"))  # a worktree of this repository, any branch
#: a worktree at cursor/te-core-313-on-pin-3a8f (te_core 3.5.3, the pin's counter): its frontier.target prices honest and
#: W* (coordinator 14:57Z) and its cw.readout_leaks judges read-outs (15:00Z), for the hourly scores and the final table
TE_PIN = Path(os.environ.get("TE_PIN", "/tmp/scorer/te-core-pin"))
TE_BRANCH = os.environ.get("TE_BRANCH", "cursor/covert-witness-4207")  # the harness's tools (launch.py, promote.py)
#: te_core 2.2's commit on the harness's branch: from it a checkpoint's val bpb is the paired score on a fresh slice of the
#: search pool (its record's val_slice), against the start's whole-pool bpb (witness v3's score.start_bpb). Version
#: numbers don't mark it (te-core-4207's 2.8 predates it), so the scorer reads the record's own fields
PAIRED = "4cc610c6766"
#: the integrator's measurement pins: every commit this file named on TE_BRANCH, the head's being the pin whose records
#: alone make the frontier, honest and the finals (coordinator 12:55Z); a record names its own (witness v4)
PIN_FILE = "benchmarks/covert_witness/pinned.json"
COUNTED_FLOOR = 0.5  # the harness's count over the run's own step log, below which the counter missed work
#: records under COUNTED_FLOOR whose run.json op table (each step's `by` and `shapes`) the scorer read and found to
#: account for the gap op by op (fewer head rows, a smaller matmul), so they score (coordinator 13:42Z): {id: reason}
FLOOR_ACCOUNTED: dict[str, str] = {}
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
from te_core.annotate import Annotation, StepAnnotation  # noqa: E402
from te_core.frontier import PHI_EDGES, TARGETS, bin_of  # noqa: E402
from te_core.score import C_PER_FORWARD_TOKEN, K_BITS, R, block_c, cell, score_grid  # noqa: E402

KEYS = [cell(k, r) for k in K_BITS for r in R]


def te_pin(name: str):
    """TE_PIN's te_core module `name` (the package itself for ""), imported as package `te_pin` beside the scorer's own
    te_core."""
    import importlib
    import importlib.util

    if "te_pin" not in sys.modules:
        d = TE_PIN / "benchmarks" / "covert_witness" / "te_core"
        spec = importlib.util.spec_from_file_location("te_pin", d / "__init__.py", submodule_search_locations=[str(d)])
        sys.modules["te_pin"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules["te_pin"])
    return importlib.import_module(f"te_pin.{name}") if name else sys.modules["te_pin"]


def short(art: str) -> str:
    return art[: 4 + SHORT] if art.startswith("art:") else art


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(text: str):
    """A record file's JSON. candidate.sh writes its args unescaped, so a regex in them (`blocks\\.2`) leaves a
    backslash JSON refuses: such a backslash is read as itself."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text))


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


def pins() -> dict:
    """{"pin": the head's pinned.json, "pins": {commit: pin} for every pin the file has named on TE_BRANCH}."""
    g = ["git", "-C", str(REPO)]
    subprocess.run(g + ["fetch", "-q", "origin", f"+refs/heads/{TE_BRANCH}:refs/remotes/origin/{TE_BRANCH}"],
                   capture_output=True, text=True)
    out = {}
    for c in subprocess.run(g + ["log", "--reverse", "--format=%H", f"origin/{TE_BRANCH}", "--", PIN_FILE],
                            capture_output=True, text=True, check=True).stdout.split():
        p = subprocess.run(g + ["show", f"{c}:{PIN_FILE}"], capture_output=True, text=True)
        if p.returncode == 0:
            pin = json.loads(p.stdout)
            out[pin["commit"]] = pin
    head = json.loads(subprocess.run(g + ["show", f"origin/{TE_BRANCH}:{PIN_FILE}"], capture_output=True, text=True,
                                     check=True).stdout)
    return {"pin": head, "pins": out | {head["commit"]: head}}


def pin_of() -> dict:
    """The pin `collect` last read."""
    p = STATE / "pin.json"
    return json.loads(p.read_text())["pin"] if p.exists() else {}


def regime_name() -> str:
    p = STATE / "pin.json"
    pinned = json.loads(p.read_text()) if p.exists() else {"pin": {}, "pins": {}}
    pin = pinned["pin"]
    same = [c[:12] for c, x in pinned["pins"].items() if x.get("version") == pin.get("version")]
    return (f"v{pin.get('version', '?')}-pin ({', '.join(same) or '?'}, te_core {pin.get('te_core', '?')}, data "
            f"{pin.get('data', '?')})")


def regime_of(mc: str, pinned: dict, paired: bool) -> str:
    """`current` for the records of a commit pinned under the pin's version (a re-pin that keeps the version fixes the
    harness without changing what it measures, as v2.2's second pin did), `pin V` for an earlier version's, `paired` for
    te_core 2.2's paired-slice records before the pins, else `earlier`."""
    if mc and mc in pinned["pins"]:
        v = pinned["pins"][mc].get("version", "?")
        return "current" if v == pinned["pin"].get("version") else f"pin {v}"
    return "paired" if paired else "earlier"


def te_current() -> str:
    """TE_CURRENT checked out at the official branch head with the highest te_core VERSION among those holding PAIRED, the
    te_core every harness record is re-ingested with (a pricing fix re-prices every record): an agent's branch can carry
    its own te_core, so its ledger records are listed and never scored."""
    g = ["git", "-C", str(TE_CURRENT)]
    subprocess.run(g + ["fetch", "-q", "origin", *(f"+refs/heads/{b}:refs/remotes/origin/{b}" for b in OFFICIAL)],
                   capture_output=True, text=True)
    if pin := os.environ.get("TE_CURRENT_COMMIT"):
        subprocess.run(g + ["checkout", "-q", "--detach", pin], capture_output=True, text=True, check=True)
        return subprocess.run(g + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    best = None
    for b in OFFICIAL:
        if subprocess.run(g + ["merge-base", "--is-ancestor", PAIRED, f"origin/{b}"], capture_output=True).returncode:
            continue
        init = subprocess.run(g + ["show", f"origin/{b}:benchmarks/covert_witness/te_core/__init__.py"],
                              capture_output=True, text=True).stdout
        v = re.search(r'^VERSION = "([\d.]+)"', init, re.M)
        if v and (best is None or tuple(map(int, v[1].split("."))) > best[0]):
            best = (tuple(map(int, v[1].split("."))), b)
    if best is None:
        raise SystemExit(f"scorer: no official branch holds te_core's paired slices ({PAIRED})")
    subprocess.run(g + ["checkout", "-q", "--detach", f"origin/{best[1]}"], capture_output=True, text=True, check=True)
    return subprocess.run(g + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def te_version() -> str:
    """The VERSION of TE_CURRENT's te_core, which prices every record (`te_current`)."""
    v = re.search(r'^VERSION = "([\d.]+)"', (TE_CURRENT / "benchmarks/covert_witness/te_core/__init__.py").read_text(), re.MULTILINE)
    return v[1] if v else "?"


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


def island_commit(w: dict | None, cand: dict) -> str:
    """The commit the candidate's program came from: witness v4's island (the pin measured it), else the candidate's."""
    return ((w or {}).get("island") or {}).get("commit") or cand.get("commit", "")


def cw_row(r, art: str, recorded: str, man: dict, commits: dict, custody: dict | None = None, te: str = "",
           official: set | None = None, pinned: dict | None = None) -> dict:
    cw = man["meta"]["covert_witness"]
    files = {f["path"]: f for f in man["payload"].get("files", [])}
    d = STATE / "records" / art.removeprefix("art:")
    got = {}
    for name in SMALL_FILES:
        if name in files:
            got[name] = read_json(blob(r, files[name]["sha256"], d / name).read_text())
    w, cand, meta = got.get("witness.json"), got.get("candidate.json", {}), got.get("meta.json", {})
    commit, island = island_commit(w, cand), (w or {}).get("island") or {}
    mc = (w or {}).get("measurement_commit") or ""
    agent, direction = direction_of(commits.get(commit, []), cw.get("class", ""))
    if not agent and cw.get("direction"):
        direction = cw["direction"]
    if cw.get("final"):
        direction = "scorer"
    checks = (w or {}).get("checks", {})
    bad = sorted(k for k, v in checks.items() if v is not True) + custody_problems(files, custody)
    if w and mc and pinned is not None:
        bad += [] if mc in pinned["pins"] else [f"measured by {mc[:12]}, which {TE_BRANCH}'s pinned.json never named"]
    elif w and official is not None:
        bad += fixed_side(commit, official)
    curve = (w or {}).get("curve") or []
    harness = (w or {}).get("harness")
    notes = (w or {}).get("notes") or {}
    over = (harness or {}).get("over_train_flops")
    if isinstance(over, (int, float)) and over < COUNTED_FLOOR and short(art) not in FLOOR_ACCOUNTED:
        bad += [(f"counted: the harness counted {over:.2g}x the FLOPs of the run's own steps (floor {COUNTED_FLOOR}), "
                 "and no op table read accounts for the gap (FLOOR_ACCOUNTED)")]
    changed = ((w or {}).get("changed") or {}).get("actual") or []
    layers = ((w or {}).get("structure") or {}).get("layers")
    below_top = [n for n in changed if (m := re.match(r"blocks\.(\d+)\.", n)) and layers and int(m[1]) < layers - 1]
    if "embed" in changed and not below_top:
        bad += [("priced wrong (tied lookup): the tied embedding is updated with no block below the top, and v2.3's "
                 "counter prices the lookup at 0 (coordinator 13:32Z)")]
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
        "agent": agent, "direction": direction, "commit": commit, "args": cand.get("args", ""),
        "island": island.get("name"), "program": island.get("program"), "island_tree": island.get("tree"),
        "measurement_commit": mc,
        "full_theta": (None if not (w or {}).get("eval_cost") else
                       (w.get("changed") or {}).get("actual_parameters") == w["eval_cost"].get("parameters")
                       and (w.get("E") or {}).get("format") == "full" and not (harness or {}).get("families")),
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
        "e_sha256": e_digest(r, files, d, harness), "final": bool(cw.get("final")),
        "te_core": (harness or {}).get("te_core"), "harness_version": (w or {}).get("version"),
        "start_bpb": (w or {}).get("score", {}).get("start_bpb"),
        "regime": regime_of(mc, pinned or {"pins": {}}, (w or {}).get("score", {}).get("start_bpb") is not None
                            and bool(derived) and all(t.get("val_slice") for t in derived)),
    }


def e_digest(r, files: dict, d: Path, harness: dict | None) -> str | None:
    """The submitted E's sha256: the harness's record of it, else the record's E.sha256 (sha256sum's line)."""
    if (harness or {}).get("submitted", {}).get("e_sha256"):
        return harness["submitted"]["e_sha256"]
    if "E.bin" in files:
        return files["E.bin"]["sha256"]
    if "E.sha256" in files:
        t = blob(r, files["E.sha256"]["sha256"], d / "E.sha256").read_text().split()
        return t[0] if t and re.fullmatch(r"[0-9a-f]{64}", t[0]) else None
    return None


def replays(rows: list[dict]) -> None:
    """A search record whose E is another direction's earlier record's E (by sha256) is unscored: its run didn't make
    that E (the auditor's 10:24Z replay). A direction's own relaunch of an E it had before counts, at any pin
    (coordinator 14:22Z: an agent's own relaunch counts, another agent's copy of an E doesn't)."""
    first: dict[str, list[dict]] = {}
    for x in sorted((x for x in rows if x["source"] == "launch.py" and x.get("e_sha256") and not x.get("final")),
                    key=lambda x: (x.get("launched") or "", x["id"])):
        e = x["e_sha256"]
        prior = [y for y in first.get(e, []) if y["direction"] != x["direction"]]
        if prior:
            x["failed_checks"] = [*x["failed_checks"], f"replay: E is {short(prior[0]['id'])}'s"]
            x["valid"] = False
        else:
            first.setdefault(e, []).append(x)


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
        "output_sha256": (rec.get("output") or {}).get("output_sha256"), "submitted": (rec.get("output") or {}).get("submitted"),
        "node": rec.get("node"), "run_te_core": rec.get("run_te_core"), "val_slice": rec.get("val_slice"),
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
        got = {n: read_json(blob(r, files[n]["sha256"], STATE / "records" / art.removeprefix("art:") / n).read_text())
               for n in ("candidate.json", "witness.json") if n in files}
        commit_of[art] = island_commit(got.get("witness.json"), got.get("candidate.json", {}))
    runs = attempts(r, since)
    commits = branches(set(commit_of.values()) | {a["commit"] for a in runs})
    custody = {a["run"]: by_art.get(a["run_record"]) for a in runs if a.get("run_record")}
    official, pinned = official_blobs(), pins()
    (STATE / "pin.json").write_text(json.dumps(pinned, indent=1) + "\n")
    with ThreadPoolExecutor(int(os.environ.get("TE_SCORER_WORKERS", "16"))) as ex:
        rows = list(ex.map(lambda x: cw_row(r, x[0], x[1], x[2], commits, custody.get(
            x[2]["meta"]["covert_witness"].get("run")), te, official, pinned), cw))
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
            x["node"] = a and a.get("machine")
    replays(rows)
    bad = {x["id"] for x in rows if x["source"] == "launch.py" and not x["valid"]}
    for x in rows:
        if x.get("derived") and x["record"] in bad:
            x["valid"] = False
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
           "te_core_commit": te, "pin": pinned["pin"], "seconds": round(time.time() - t0, 1)}
    (STATE / "collect.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def fmt(x, spec=".3g"):
    return "—" if x is None else format(x, spec)


def index_md(rows: list[dict], runs: list[dict], starts: dict) -> str:
    lines = [f"# The search's ledger, {now()}", "",
             f"{len(rows)} records since {SINCE}; starts' search bpb: " +
             ", ".join(f"{s} {starts[s]['bpb']:.4f}" for s in STARTS), "",
             "| record | agent | direction | class | mechanism note | start | launched | harness (witness, te_core) | "
             "train FLOPs | bpb | Δ% | |E| bytes | φ | D (c counted, a analytic) | valid |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in entries(rows, audit=True):
        s0 = starts.get(x["start"], {}).get("bpb")
        drop = 100 * (s0 - x["bpb"]) / s0 if s0 and x["bpb"] is not None else None
        if x["source"] == "te_core" or x.get("counted"):
            D, how = x.get("D"), "c"
        else:
            ann = analytic(x, starts) if x["valid"] and x["start"] in starts else None
            D, how = ann and ann["depth"], "a"
        d = x["direction"] + (" (not scored)" if is_audit(x) else " (cancelled unrun)" if x.get("cancelled") else
                              " (not scored: no launch.py record)" if x.get("unrecorded") else
                              " (final rerun)" if x.get("final") else "")
        hv = (f"{(x.get('witness') or '').rsplit('/', 1)[-1] or '—'} {x.get('harness_version') or ''}, "
              f"{x.get('te_core') or x.get('run_te_core') or '—'}" + (" (current)" if x.get("regime") == "current" else ""))
        lines.append(f"| `{short(x['id'])}` | {(x['agent'] or '').removeprefix('cursor/') or '—'} | {d} | {x['class']} | "
                     f"{x['note']} | {x['start']} | {(x['launched'] or '')[11:16]} | {hv} | {fmt(x['train_flops'])} | "
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


REF_DIRECTIONS = ("ref-optim", "ref-data")


def is_reference(x: dict) -> bool:
    """The reference-only honest (context since the coordinator's 14:57Z ruling) reads only runs that update every weight
    of the start (all of θ changed, E full, no declared family) launched by ref-optim or ref-data, or the harness's and
    the scorer's own reference smokes (13:42Z). A record whose witness can't show a full update (before witness v4) is
    read by its direction alone."""
    own = x["direction"] in REF_DIRECTIONS or (x["direction"] == "reference" and x.get("agent") == TE_BRANCH)
    return own and x.get("full_theta") is not False


def is_audit(x: dict) -> bool:
    """The measurement auditor's records exercise measurement gaps on purpose: in the ledger, out of every score."""
    return x.get("direction") == "audit" or str(x.get("note", "")).lower().startswith("audit:")


def run_rows(rows: list[dict]) -> dict[str, dict]:
    return {x["run"]: x for x in rows if x["source"] == "launch.py" and x.get("run")}


TORCH_PYTHON = os.environ.get("TE_TORCH_PYTHON", "/tmp/te-venv/bin/python")  # te_core.cw imports torch
JUDGE = ("import json, sys\nfrom te_core.cw import readout_leaks\n"
         "print(json.dumps({k: readout_leaks(json.load(open(p)), t) for k, (p, t) in json.load(sys.stdin).items()}))")


def readout_leaks(jobs: dict[str, tuple[Path, dict | None]]) -> dict[str, list | None]:
    """te_core's cw.readout_leaks (TE_PIN's; coordinator 15:00Z) on each job's run.json against its start's tensors:
    the steps whose read-outs came from index reads no counted work consumed (any makes the record invalid), None where
    the run.json or the start's tensors are missing (the record is unjudged)."""
    out = {k: None for k, (p, t) in jobs.items() if not p.exists() or not t}
    todo = {k: [str(p), t] for k, (p, t) in jobs.items() if k not in out}
    if todo:
        p = subprocess.run([TORCH_PYTHON, "-c", JUDGE], input=json.dumps(todo), cwd=TE_PIN / "benchmarks" / "covert_witness",
                           capture_output=True, text=True, check=False)
        if p.returncode:
            raise SystemExit(f"scorer: cw.readout_leaks rc {p.returncode}: {p.stderr.strip()[-300:]}")
        out |= json.loads(p.stdout)
    return out


def judge_readouts(rows: list[dict], starts: dict) -> dict[str, list | None]:
    """`readout_leaks` on every pin record of the launcher's (cached in STATE/readouts.json by record id), each record it
    lists marked invalid; the verdicts by record id."""
    path = STATE / "readouts.json"
    seen = json.loads(path.read_text()) if path.exists() else {}
    pin = [x for x in rows if x["source"] == "launch.py" and x.get("regime") == "current" and not is_audit(x)]
    seen |= readout_leaks({x["id"]: (STATE / "records" / x["id"].removeprefix("art:") / "run" / "run.json",
                                     (starts.get(x.get("start")) or {}).get("tensors"))
                           for x in pin if seen.get(x["id"]) is None})
    out = {}
    for x in pin:
        out[x["id"]] = seen[x["id"]]
        if out[x["id"]]:
            x["valid"] = False
            x["failed_checks"] = [*x["failed_checks"], "read-out leak: steps " + ", ".join(
                str(s["step"]) for s in out[x["id"]])]
    path.write_text(json.dumps(seen, indent=1) + "\n")
    return out


def submissions(rows: list[dict], starts: dict) -> list[dict]:
    """Every checkpoint of every valid record as a te_core submission: {id, cp, start, macs, val_bpb, rho, phi, depth}.
    A record made inside te_core's harness (witness v2) is scored from the te_core submissions `collect` ingested from the
    node's run.json (one per checkpoint, the harness's counted annotation); the analytic annotation is only for witness
    v1 records. The te_core records agents publish themselves are listed, never scored."""
    out = []
    launched = run_rows(rows)
    soft = {s["rec"]["id"] for s in soft_spots(rows)}
    for x in rows:
        if x["source"] == "te_core" and not x.get("derived"):
            continue
        rec = launched.get(x.get("run")) if x["source"] == "te_core" else None
        if x["source"] == "te_core" and not (rec and rec["valid"]):
            continue
        if not x.get("valid") or x.get("start") not in starts or is_audit(x) or (rec or x).get("final"):
            continue
        if x["source"] == "te_core":
            if x.get("te_kind") == "submission" and x.get("rho"):
                out.append({"id": rec["id"] if rec else x["id"], "te_id": x["te_id"], "cp": round(x.get("fraction") or 0, 3),
                            "start": x["start"], "macs": x["macs"], "val_bpb": x["bpb"], "rho": x["rho"], "phi": x["phi"],
                            "depth": x["D"], "reference": is_reference(rec or x), "direction": x["direction"], "counted": True,
                            "E_bytes": x.get("E_bytes"), "regime": (rec or {}).get("regime", "earlier"),
                            "start_bpb": x.get("start_bpb"), "output_sha256": x.get("output_sha256"),
                            "submitted": x.get("submitted"), "node": x.get("node"), "fork_ladder": fork_ladder(rec),
                            "phi_position": (rec or {}).get("phi"), "soft_spot": (rec or x)["id"] in soft})
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
                        "E_bytes": cp.get("e_bytes"), "credit": a["credit"], "regime": "earlier"})
    return schedule_ends(out)


def arg_value(args: str, flag: str) -> str | None:
    a = str(args or "").split()
    for i, t in enumerate(a):
        if t == flag:
            return "" if i + 1 == len(a) or a[i + 1].startswith("--") else a[i + 1]
        if t.startswith(flag + "="):
            return t[len(flag) + 1:]
    return None


def fork_ladder(x: dict | None) -> bool:
    """A reference island's train.py (ref-optim's, ref-data's) run with --forks and an empty --snap-at encodes only at
    its cooldown branches' ends (a branch decays the rate linearly to 0, is snapshotted, and the run returns to where it
    forked) and at its own end, so each of its checkpoints ends a complete schedule of that length. Its FLOPs include
    the earlier branches' passes, so honest is overstated there, never understated."""
    return bool(x and (x.get("island") or x.get("direction")) in REF_DIRECTIONS and arg_value(x.get("args"), "--forks")
                and arg_value(x.get("args"), "--snap-at") == "")


def schedule_ends(subs: list[dict], key=lambda s: s["id"]) -> list[dict]:
    """Marks each run's last checkpoint, the end of its own schedule: honest(l*) reads only those (coordinator 12:05Z, a
    checkpoint inside a longer reference schedule is not what an honest run of that length reaches), and every
    checkpoint of a fork ladder (`fork_ladder`)."""
    top: dict = {}
    for s in subs:
        top[key(s)] = max(top.get(key(s), 0), s["cp"])
    for s in subs:
        s["schedule_end"] = s["cp"] == top[key(s)] or bool(s.get("fork_ladder"))
    return subs


def regime_start_bpb(subs: list[dict], s: str) -> float | None:
    """The current regime's l0 on start `s`: the start's whole-pool bpb its records carry (the median, should they
    differ in the last digits)."""
    v = sorted(x["start_bpb"] for x in subs if x["start"] == s and isinstance(x.get("start_bpb"), (int, float)))
    return v[len(v) // 2] if v else None


def pareto(points: list[tuple[float, float, dict]]) -> list[dict]:
    out, best = [], math.inf
    for c, b, s in sorted(points, key=lambda p: (p[0], p[1])):
        if b < best:
            out.append(s)
            best = b
    return out


def target_row(on: list[dict], f: float, l_star: float) -> dict:
    """One target on one start by te_core.frontier.target (coordinator 14:57Z): honest is the least |C| over every
    submission in `on` that ends a complete schedule and reaches l*, of any class or direction, and W* per (k, r) the
    least |C|ρ over the same submissions, so W*/honest >= 1 is the effective slowdown; `bound` is te_core's 'upper bound
    (checkpoint)' where only a mid-schedule checkpoint reaches l*. honest_reference (the reference directions' complete
    schedules alone, the rule before) and W* over it (`vs_reference`) are context."""
    t = te_pin("frontier").target(on, l_star, KEYS, reference=lambda x: x["reference"])
    by = {(x["id"], x["cp"]): x for x in on}
    of = lambda o: o and by.get((o["id"], o["checkpoint"]))
    h, ref = of(t["honest_of"]), of(t["honest_reference_of"])
    row = {"drop": f, "l_star": l_star, "bound": t["bound"], "honest_macs": t["honest"], "honest_id": h and h["id"],
           "honest_cp": h and h["cp"], "honest_direction": h and h["direction"], "honest_seeds": h and h.get("seeds"),
           "honest_split": h and h.get("split"),
           "honest_reference_macs": t["honest_reference"], "honest_reference_id": ref and ref["id"],
           "honest_reference_cp": ref and ref["cp"], "cells": {}}
    for key, c in t["cells"].items():
        w = of(c["of"])
        row["cells"][key] = {
            "w_star": c["w_star"], "effective_slowdown": c["effective_slowdown"], "id": w and w["id"], "cp": w and w["cp"],
            "rho": c["rho"], "direction": w and w["direction"], "counted": w and w.get("counted"),
            "seeds": w and w.get("seeds"), "split": w and w.get("split"), "phi": w and (w.get("phi") or {}).get(key),
            "soft_spot": bool(w and w.get("soft_spot")),
            "phi_position": w and w.get("phi_position"),
            "vs_reference": c["w_star"] / t["honest_reference"] if c["w_star"] and t["honest_reference"] else None}
    return row


def start_table(on: list[dict], l0: float | None, c: float) -> dict:
    """W*, honest and W*/honest per target and (k, r) on one start (`target_row`), and the proxy frontiers, from its
    submissions."""
    st = {"start_bpb": l0, "c": c, "targets": [], "submissions": len(on),
          "proxy_frontier_flops": [{"id": x["id"], "cp": x["cp"], "flops": 2 * x["macs"], "bpb": x["val_bpb"]}
                                   for x in pareto([(x["macs"], x["val_bpb"], x) for x in on])],
          "proxy_frontier_E": [{"id": x["id"], "cp": x["cp"], "E_bytes": x["E_bytes"], "bpb": x["val_bpb"]}
                               for x in pareto([(x["E_bytes"] or 0, x["val_bpb"], x) for x in on])]}
    st["targets"] = [target_row(on, f, l0 * (1 - f)) for f in (TARGETS if l0 is not None else ())]
    return st


def score(write: bool = True) -> dict:
    rows = [json.loads(x) for x in (STATE / "index.jsonl").read_text().splitlines() if x.strip()]
    starts = json.loads((STATE / "starts.json").read_text())
    readouts = judge_readouts(rows, starts)
    subs = submissions(rows, starts)
    col = STATE / "collect.json"
    rep = {"at": now(), "te_core": te_version(), "regime": regime_name(), "pin": pin_of(),
           "te_core_commit": json.loads(col.read_text()).get("te_core_commit", "") if col.exists() else "",
           "annotation": "the harness's count for witness v2 records, analytic for witness v1 (module docstring)",
           "p_bits": P_BITS, "c_per_forward_token": C_PER_FORWARD_TOKEN, "starts": {}, "earlier": {},
           "submissions": len(subs), "readouts": {"judged": sum(v is not None for v in readouts.values()),
                                                  "leaks": {i: v for i, v in readouts.items() if v},
                                                  "unjudged": sorted(i for i, v in readouts.items() if v is None)}}
    cur = [x for x in subs if x["regime"] == "current"]
    for s in STARTS:
        c = block_c(forward_macs_per_token(starts[s]))
        rep["starts"][s] = start_table([x for x in cur if x["start"] == s], regime_start_bpb(cur, s), c)
    for reg in sorted({x["regime"] for x in subs} - {"current"}):
        old = [x for x in subs if x["regime"] == reg]
        rep["earlier"][reg] = {s: start_table([x for x in old if x["start"] == s], starts[s]["bpb"] if reg == "earlier"
                                              else regime_start_bpb(old, s), block_c(forward_macs_per_token(starts[s])))
                               for s in STARTS}
    if write:
        (STATE / "score.json").write_text(json.dumps(rep, indent=1) + "\n")
        (STATE / "submissions.json").write_text(json.dumps(subs) + "\n")
        (STATE / "score.md").write_text(score_md(rep))
    return rep


def score_md(rep: dict) -> str:
    lines = [f"# Scores, {rep['at']} (te_core {rep['te_core']}; {rep['annotation']}; p = {rep['p_bits']})", ""]
    parts = [(rep["regime"] + " records (paired slice scores against the start's whole search-pool bpb)", rep["starts"])]
    parts += [(f"{reg} records (context: not comparable with the above, and out of the final table)", part)
              for reg, part in rep.get("earlier", {}).items()]
    for name, part in parts:
        lines += [f"# {name}", ""]
        lines += start_md(part)
    return "\n".join(lines) + "\n"


def start_md(part: dict) -> list[str]:
    lines = []
    for s, st in part.items():
        if not st["submissions"]:
            lines += [f"## {s}: no submission", ""]
            continue
        lines += [f"## {s} (start {fmt(st['start_bpb'], '.4f')} bpb, c = {st['c']:.3g} MACs)", "",
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
    return lines


# ---- cells, promotions, digest -----------------------------------------------------------------------------------------

def e_bin(e) -> str:
    return E_LABELS[bin_of(e or 0, E_EDGES)]


def phi_bin(phi) -> str:
    return PHI_LABELS[bin_of(phi or 0.0, PHI_EDGES)]


def cell_of(x: dict) -> tuple:
    return (x["mechanism"], e_bin(x["E_bytes"]), phi_bin(x["phi"]), x["start"])


def drop_of(x: dict, starts: dict) -> float | None:
    """The relative bpb drop from the start: a current-regime record against the whole-pool start bpb it carries."""
    s0 = x.get("start_bpb") if x.get("regime", "earlier") != "earlier" else starts.get(x["start"], {}).get("bpb")
    return (s0 - x["bpb"]) / s0 if s0 and isinstance(x.get("bpb"), (int, float)) else None


def current(xs: list[dict]) -> list[dict]:
    return [x for x in xs if x.get("regime") == "current"]


def entries(rows: list[dict], audit: bool = False) -> list[dict]:
    """One row per candidate: a launch.py record, or (listed with `audit` only) a te_core run no record of launch.py's
    holds. A witness v2 record takes |C|, φ and D (read at 10kB/r1) from the last te_core checkpoint `collect` ingested
    from its node's run.json, and the te id of its agent's own last checkpoint, if any, for the ledger's promotion. The
    auditor's records and the records of runs cancelled unrun only with `audit` (the ledger lists them; nothing scores or
    counts them)."""
    last: dict[str, dict] = {}
    own: dict[str, dict] = {}
    subm: dict[str, dict] = {}
    for x in rows:
        if x["source"] == "te_core" and x.get("te_kind") == "submission":
            m, k = (last, x["record"]) if x.get("derived") else (own, x.get("run") or x["te_id"])
            if k not in m or (x.get("fraction") or 0) > (m[k].get("fraction") or 0):
                m[k] = x
            if x.get("derived") and x.get("submitted"):
                subm[x["record"]] = x
    out = []
    for x in rows:
        if x["source"] != "launch.py":
            continue
        t = last.get(x["id"])
        o = own.pop(x["run"], None) if x.get("run") else None
        s = subm.get(x["id"]) or t
        out.append(x | ({"phi": at10(t["phi"]), "D": at10(t["D"]), "train_flops": 2 * t["macs"], "te_id": o and o["te_id"],
                         "tier": t.get("tier"), "families": t.get("families"), "ingested": True,
                         "output_sha256": s.get("output_sha256"), "node": t.get("node") or x.get("node")} if t else {}))
    out += [x | {"phi": at10(x.get("phi")), "D": at10(x.get("D")), "unrecorded": True} for x in own.values()]
    return sorted((x for x in out if audit or not (is_audit(x) or x.get("cancelled") or x.get("unrecorded") or x.get("final"))),
                  key=lambda x: (x.get("launched") or "", x["id"]))


def at10(v):
    return v.get("10kB/r1") if isinstance(v, dict) else v


def promotions(rows: list[dict], starts: dict, promoted: dict) -> dict:
    """This hour's promotion candidates: each direction's top third by relative bpb drop, and the best of each thin cell
    (at most THIN valid records), among the current regime's valid records not promoted before; plus re-screens of
    mechanisms already screened. Which candidates are promoted is the sealed set's (`sealed_bits`). The harness's own
    smokes (its branch's commits, note `smoke ...`) check the harness and are no direction's entry."""
    valid = [x for x in current(entries(rows)) if x["valid"] and drop_of(x, starts) is not None
             and (x["launched"] or "") >= SEARCH_START
             and not (x["agent"] == "cursor/covert-witness-4207" and x["mechanism"] == "smoke")]
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
    """E range x φ range x start cells no valid record of the current regime fills, within the E ranges each class's cap
    allows."""
    valid = [x for x in current(entries(rows)) if x["valid"]]
    filled = {(e_bin(x["E_bytes"]), phi_bin(x["phi"]), x["start"]) for x in valid}
    out = []
    for s in STARTS:
        for e in E_LABELS[:4]:
            for p in PHI_LABELS:
                if (e, p, s) not in filled:
                    out.append(f"{s}:{e}:{p}")
    return out


OFFICIAL_SRC = Path(os.environ.get("TE_OFFICIAL_SRC", "/tmp/scorer/official"))
SHIM = OFFICIAL_SRC.with_name("bin")


def official_src() -> Path:
    """A clean sparse checkout of TE_BRANCH's head (covert_witness and the research package) for the harness's own tools
    (promote.py, nodes.txt), and a `research` in SHIM that runs that package (the VM's install predates pools)."""
    if not (OFFICIAL_SRC / ".git").exists():
        subprocess.run(["git", "-C", str(REPO), "worktree", "add", "-q", "--no-checkout", "--detach", str(OFFICIAL_SRC),
                        f"origin/{TE_BRANCH}"], check=True)
        subprocess.run(["git", "-C", str(OFFICIAL_SRC), "sparse-checkout", "set", "--no-cone", "/benchmarks/covert_witness/",
                        "/tools/research/"], check=True)
    subprocess.run(["git", "-C", str(OFFICIAL_SRC), "fetch", "-q", "origin", TE_BRANCH], capture_output=True, text=True)
    subprocess.run(["git", "-C", str(OFFICIAL_SRC), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    SHIM.mkdir(parents=True, exist_ok=True)
    (SHIM / "research").write_text(f'#!/usr/bin/env bash\nPYTHONPATH="{OFFICIAL_SRC}/tools/research/src" '
                                   f'exec "{sys.executable}" -m research "$@"\n')
    (SHIM / "research").chmod(0o755)
    return OFFICIAL_SRC


#: the coordinator's shared node list (11:55Z: every launch reads it as $CW_NODES), else the harness's nodes.txt
TE_NODES = Path(os.environ.get("TE_NODES", "/tmp/te-nodes.txt"))


def nodes_file() -> Path:
    return TE_NODES if TE_NODES.exists() else official_src() / "benchmarks" / "covert_witness" / "nodes.txt"


def listed_nodes() -> list[str]:
    """The nodes launch.py and promote.py take (`nodes_file`), in order."""
    return [ln.split()[0] for ln in nodes_file().read_text().splitlines() if ln.split("#", 1)[0].strip()]


def tool_env() -> dict:
    official_src()
    return os.environ | {"PATH": f"{SHIM}:{os.environ.get('PATH', '')}", "CW_NODES": str(nodes_file())}


def sealed_bits(cands: list[dict]) -> dict:
    """{id: {promote, value_sha256, node, run}} from the harness's promote.py: one bit per candidate output, decided on
    its node from the sealed promotion value (search gain > 0 and promotion gain >= 0.5x it); nothing else returns."""
    src = official_src()
    d = STATE / "sealed"
    d.mkdir(parents=True, exist_ok=True)
    got, queued = queued_bits(d / "pending.json")
    waiting = {i for q in queued for i in q["ids"]}
    cands = [c for c in cands if c["id"] not in got and c["id"] not in waiting]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    inp, out = d / f"candidates-{stamp}.json", d / f"bits-{stamp}.json"
    inp.write_text(json.dumps(cands, indent=1) + "\n")
    p = subprocess.run([sys.executable, "benchmarks/covert_witness/promote.py", "--in", str(inp), "--out", str(out)],
                       cwd=src, env=tool_env(), capture_output=True, text=True) if cands else None
    if p and (p.returncode or not out.exists()):
        text = p.stdout + p.stderr
        going = set(re.findall(r"run (r\d{8}-\d{6}-[0-9a-f]{4}) \(waiting\) is still going", text))
        launched = dict(re.findall(r"run (r\d{8}-\d{6}-[0-9a-f]{4}) launched on (\S+)", text))
        if not going:
            raise SystemExit(f"scorer: promote.py failed ({p.returncode}): {text[-800:]}")
        default = listed_nodes()[0]
        queued += [{"run": r, "node": n, "at": stamp, "ids": [c["id"] for c in cands if (c.get("node") or default) == n]}
                    for r, n in launched.items() if r in going]
    (d / "pending.json").write_text(json.dumps(queued, indent=1) + "\n")
    return got | (json.loads(out.read_text()) if p else {}) | {
        i: {"queued": q["run"], "node": q["node"]} for q in queued for i in q["ids"]}


def queued_bits(path: Path) -> tuple[dict, list[dict]]:
    """The bits of decision runs promote.py stopped waiting on (research run's 20-minute cap: on a node whose CPU pool is
    its gpu line, `sealed.py decide` waits behind training runs) that have since ended, and the ones still queued."""
    got, still = {}, []
    for q in json.loads(path.read_text()) if path.exists() else []:
        research("fetch", q["run"], "--all")
        f = RUNS / q["run"] / "out" / "bits.json"
        if f.is_file():
            got |= {k: v | {"node": q["node"], "run": q["run"]} for k, v in json.loads(f.read_text()).items()}
        else:
            still.append(q)
    return got, still


def board(promote: bool, digest: bool, append: bool, hhmm: str | None = None) -> list[str]:
    rows = [json.loads(x) for x in (STATE / "index.jsonl").read_text().splitlines() if x.strip()]
    starts = json.loads((STATE / "starts.json").read_text())
    rep = json.loads((STATE / "score.json").read_text())
    ppath = STATE / "promoted.json"
    promoted = json.loads(ppath.read_text()) if ppath.exists() else {}
    hhmm = hhmm or datetime.now(timezone.utc).strftime("%H:%MZ")
    lines = []
    pr = promotions(rows, starts, promoted)
    bits = None
    if promote:
        ents = {x["id"]: x for x in entries(rows)}
        cands = [{"id": p["id"], "output_sha256": ents[p["id"]]["output_sha256"], "node": ents[p["id"]].get("node")}
                 for p in pr["new"] if ents.get(p["id"], {}).get("output_sha256")]
        listed = listed_nodes()
        pending = {c["id"]: c.get("node") for c in cands if (c.get("node") or listed[0]) not in listed}
        cands = [c for c in cands if c["id"] not in pending]
        bits = (sealed_bits(cands) if cands else {}) if append else None
        why = {p["id"]: p["reasons"] for p in pr["new"]}
        if bits is None and pr["new"]:
            lines.append(f"{hhmm} scorer [PROMOTED] (dry run; the sealed set decides on --append) candidates: " + ", ".join(
                f"{short(i)} ({'; '.join(r)})" for i, r in why.items()))
        elif bits is not None and why:
            yes = [i for i in why if (bits.get(i) or {}).get("promote")]
            runs = sorted({b["run"] for b in bits.values() if b.get("run")})
            if yes:
                lines.append(f"{hhmm} scorer [PROMOTED] " + ", ".join(f"{short(i)} ({'; '.join(why[i])})" for i in yes)
                             + f"; promoted by the sealed promotion set ({', '.join(runs)}).")
            queued = {i: bits[i] for i in why if (bits.get(i) or {}).get("queued")}
            if queued:
                lines.append(f"{hhmm} scorer [DIGEST] promotion candidates the sealed set hasn't decided yet: " + ", ".join(
                    f"{short(i)} ({'; '.join(why[i])}; decision run {b['queued']} waits in {b['node']}'s gpu line)"
                    for i, b in queued.items()) + "; read next hour.")
            no = [i for i in why if i not in yes and i not in pending and i not in queued]
            if no:
                lines.append(f"{hhmm} scorer [DIGEST] promotion candidates the sealed set did not promote: " + ", ".join(
                    f"{short(i)} ({'no sealed value' if (bits.get(i) or {}).get('promote') is None else 'gain not held'})"
                    for i in no) + ".")
            if pending:
                lines.append(f"{hhmm} scorer [DIGEST] promotion candidates not yet decided: " + ", ".join(
                    f"{short(i)} ({'; '.join(why[i])}; its value is on {n}, which {nodes_file()} doesn't list, "
                    f"so promote.py can't run there)" for i, n in pending.items()) + "; candidates again next hour.")
            pr["new"] = [p for p in pr["new"] if p["id"] in yes]
            pr["decided"] = {i: bits.get(i) for i in why if i not in pending and i not in queued}
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
            for i, b in (pr.get("decided") or {}).items():
                promoted[i] = {"at": hhmm, "promote": (b or {}).get("promote"), "sealed": b}
            ppath.write_text(json.dumps(promoted, indent=1) + "\n")
            (STATE / "rescreens.json").write_text(json.dumps(sorted({x["id"] for x in pr["rescreens"]})) + "\n")
    return lines


def frontier_lines(part: dict) -> list[str]:
    """One `START: target; target; ...` per start with an honest or W* entry at some target."""
    out = []
    for s, st in part.items():
        parts = []
        for t in st["targets"]:
            h = t["honest_macs"]
            cells = t["cells"]
            if h is None and all(c["w_star"] is None for c in cells.values()):
                continue
            eff = [cells[k]["effective_slowdown"] for k in KEYS]
            vec = " | ".join(", ".join(fmt(e, ".3g") for e in eff[i: i + 3]) for i in (0, 3, 6))
            ids = sorted({short(c["id"]) for c in cells.values() if c["id"] and (c["effective_slowdown"] or 0) > 1})
            hr = t.get("honest_reference_macs")
            parts.append(f"-{100 * t['drop']:g}% (l* {t['l_star']:.4f}): honest {fmt(2 * h)} ({short(t['honest_id'])}"
                         f"@{t['honest_cp']}, {t.get('honest_direction')}{', ' + t['bound'] if t.get('bound') else ''})"
                         + (f", reference-only {fmt(2 * hr)}" if hr and hr != h else "" if hr else ", no reference reaches it")
                         + f"; W*/honest [{vec}]" + (f" via {', '.join(ids)}" if ids else ""))
        if parts:
            out.append(f"{s}: " + "; ".join(parts) + ".")
    return out


def context_lines(earlier: dict) -> list[str]:
    """The frontier of each earlier pin's records and of the paired records (the fixed search set's only if neither
    reaches a target)."""
    order = sorted((r for r in earlier if r.startswith("pin ")), key=lambda r: tuple(map(int, re.findall(r"\d+", r))),
                   reverse=True) + [r for r in ("paired",) if r in earlier]
    out = [f"Context, the {reg} records' frontier (not comparable, and out of the final table): " + " ".join(lines)
           for reg in order if (lines := frontier_lines(earlier[reg]))]
    if not out and (lines := frontier_lines(earlier.get("earlier", {}))):
        out = ["Context, the earlier records' frontier (the fixed search set; not comparable, and out of the final "
               "table): " + " ".join(lines)]
    return out


def digest_lines(rows, starts, rep, pr, hhmm) -> list[str]:
    out = []
    cw = entries(rows)
    n, nv = len(cw), len([x for x in cw if x["valid"]])
    nc = len([x for x in rows if x["source"] == "launch.py" and x.get("cancelled")])
    ncur = len([x for x in current(cw) if x["valid"]])
    reg = rep.get("regime") or regime_name()
    out.append(f"{hhmm} scorer [DIGEST] {n} records ({nv} valid, {ncur} of them {reg} records; direction audit's, "
               f"the scorer's final reruns and {nc} of runs cancelled unrun not counted). Frontier per target below, "
               f"{reg} records only (l* from the start's whole-pool bpb they carry), by the coordinator's 14:57Z ruling "
               f"(te_core 3.5.3's frontier.target): honest = least training FLOPs (2|C|) over every valid record, of any "
               f"class or direction, whose complete schedule (a fork ladder's branch ends included) reaches l*; W* = least "
               f"2|C| rho over the same records, so W*/honest >= 1, per (k, r) in the order 10kB r1,r10,r100 | 100kB | "
               f"1MB, te_core {rep.get('te_core')} rho, with the entries setting a ratio above 1; the reference "
               f"directions' honest alone where it differs, as context; read-outs judged by cw.readout_leaks "
               f"({(rep.get('readouts') or {}).get('judged', 0)} records, "
               f"{len((rep.get('readouts') or {}).get('leaks') or {})} invalid); full tables in the scorer's score.md.")
    lines = frontier_lines(rep["starts"])
    nsub = sum(st["submissions"] for st in rep["starts"].values())
    out += [f"{hhmm} scorer [DIGEST] frontier {x}" for x in lines] or [
        f"{hhmm} scorer [DIGEST] frontier: " + (f"the {reg} records' {nsub} checkpoints reach no target yet." if nsub
                                                else f"no {reg} record yet.")]
    out += [f"{hhmm} scorer [DIGEST] {x}" for x in context_lines(rep.get("earlier", {}))]
    if pr["new"] and "decided" in pr:
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
    flagged = [x for x in current(cw) if any(f.startswith(("replay:", "counted:")) for f in x["failed_checks"])]
    if flagged:
        out.append(f"{hhmm} scorer [DIGEST] unscored as counted-compute escapes (coordinator 10:35Z): " + ", ".join(
            f"{short(x['id'])} ({'; '.join(f for f in x['failed_checks'] if f.startswith(('replay:', 'counted:')))})"
            for x in flagged) + ".")
    tied = [x for x in current(cw) if any(f.startswith("priced wrong (tied lookup)") for f in x["failed_checks"])]
    if tied:
        out.append(f"{hhmm} scorer [DIGEST] priced wrong (tied lookup), out of the frontier (coordinator 13:32Z: the "
                   f"tied embedding updated with no block below the top): " + ", ".join(short(x["id"]) for x in tied) + ".")
    if soft := soft_spot_line(rows, hhmm):
        out.append(soft)
    out.append(f"{hhmm} scorer [DIGEST] counted: {len(v2)} witness v2/v3 records priced by the harness's count, re-ingested "
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


SOFT_PHI = 0.05


def soft_spots(rows: list[dict]) -> list[dict]:
    """Records whose harness-counted φ is under a tenth of their witness's (prepare.deep_share): the count prices an
    unmerged adapter at its own matmuls, though the rebuilt model reads W + sBA at every adapted matrix. The coordinator
    (13:42Z) ruled the count's ρ a real result and the route a soft spot. Each: {rec, phi_w, phi_c, rho} over the
    record's checkpoints."""
    out: dict[str, dict] = {}
    runs = run_rows(rows)
    fam = {runs[x["run"]]["id"] for x in rows if x["source"] == "te_core" and x.get("run") in runs
           and any(f["covers"] for f in x.get("families") or [])}
    for x in rows:
        rec = runs.get(x.get("run")) if x["source"] == "te_core" and x.get("rho") else None
        if not rec or is_audit(rec) or rec.get("final") or (rec.get("phi") or 0) < SOFT_PHI:
            continue
        if rec["id"] in fam or any(
                f.startswith(("priced wrong", "replay:", "counted:")) for f in rec["failed_checks"]):
            continue
        s = out.setdefault(rec["id"], {"rec": rec, "phi_w": rec["phi"], "phi_c": 0.0, "rho": 1.0})
        s["phi_c"] = max(s["phi_c"], *x["phi"].values())
        s["rho"] = max(s["rho"], *x["rho"].values())
    return [s for s in out.values() if s["phi_c"] < s["phi_w"] / 10]


def soft_spot_line(rows: list[dict], hhmm: str) -> str | None:
    spots = soft_spots(rows)
    cur = [s for s in spots if s["rec"].get("regime") == "current" and s["rec"]["valid"]]
    extreme = [s for s in spots if s["rho"] == 1 and s["phi_w"] >= 0.3]
    newest = max((s["rec"].get("launched") or "" for s in extreme), default="")
    extreme = [s for s in extreme if (s["rec"].get("launched") or "") == newest]
    if not cur and not extreme:
        return None
    return (f"{hhmm} scorer [DIGEST] soft spot (coordinator 13:42Z: priced by the harness's count, a real result): the "
            "count prices an unmerged adapter at its own matmuls, r(d_in + d_out) MACs a token, though the rebuilt model "
            "reads W + sBA at every adapted matrix, so these records' counted phi is under a tenth of prepare.deep_share's"
            + (": " + ", ".join(f"{short(s['rec']['id'])} {s['rec']['direction']} {s['rec']['mechanism']} (phi "
                                f"{s['phi_w']:.2f} by deep_share, {s['phi_c']:.4f} counted, rho <= {s['rho']:.3g})"
                                for s in sorted(cur, key=lambda s: -s["phi_w"])) if cur else "")
            + "." + "".join(f" The extreme is {short(s['rec']['id'])} ({s['rec']['regime']}"
                            f"{', invalid: ' + '; '.join(s['rec']['failed_checks']) if not s['rec']['valid'] else ''}): "
                            f"'{s['rec']['note'][:120]}', rho 1 in every cell at phi {s['phi_w']:.2f} by deep_share."
                            for s in extreme))


def families_of(x: dict) -> str:
    return ", ".join(f"{f['name']} covers {f['covers']}/{f['declared']} steps" for f in x.get("families") or [])


def filled_of(rows, s) -> set[str]:
    return {f"{e_bin(x['E_bytes'])}/{phi_bin(x['phi'])}" for x in current(entries(rows))
            if x["valid"] and x["start"] == s and (x["E_bytes"] or 0) < 1e7}


def direction_sentence(d: str, rows: list[dict], starts: dict) -> str:
    xs = [x for x in entries(rows) if x["direction"] == d]
    v = [x for x in xs if x["valid"] and drop_of(x, starts) is not None]
    mechs = sorted({x["mechanism"] for x in xs})
    if not v:
        return f"{len(xs)} records, none valid yet ({', '.join(mechs)})."
    best = max(current(v) or v, key=lambda x: drop_of(x, starts))
    return (f"{len(xs)} records, {len(v)} valid ({len(current(v))} {regime_name()}), {len(mechs)} mechanisms "
            f"({', '.join(mechs)}); best{'' if best.get('regime') == 'current' else ' (earlier regime)'} "
            f"{short(best['id'])} '{best['note']}' on {best['start']}: {best['bpb']:.4f} bpb "
            f"({abs(100 * drop_of(best, starts)):.2f}% {'below' if drop_of(best, starts) > 0 else 'above'} the start) at {best['train_flops']:.3g} FLOPs, |E| "
            f"{fmt(best['E_bytes'], ',')} B, phi {fmt(best['phi'], '.2f')}.")


def append_board(lines: list[str]) -> None:
    """Append to the scorer's own board file and push its branch; the coordinator merges it into the shared board. One
    writer at a time (a promotion run waits on the sealed set while digests are posted)."""
    import fcntl

    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "board.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _append_board(lines)


def _append_board(lines: list[str]) -> None:
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


# ---- final scoring: each FINAL entry's island rerun at fresh seeds in final mode, under the harness's head -------------

FINAL = Path(os.environ.get("TE_FINAL_DIR", STATE / "final"))
NIGHT_START = os.environ.get("TE_NIGHT_START", "2026-11-02T08:30:00Z")
FINAL_SRC = Path(os.environ.get("TE_FINAL_SRC", "/tmp/scorer/final-src"))
FINAL_SEEDS = (101, 102, 103)
RUNS = Path.home() / ".research" / "runs"
ENDED = {"done", "failed", "cancelled", "refused", "timeout", "lost"}
ISLANDS = "benchmarks/covert_witness/islands"


def git(*args: str, env: dict | None = None, stdin: str | None = None) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True, env=env,
                          input=stdin).stdout.strip()


def overlay(commit: str, harness: str) -> str:
    """The commit a FINAL entry reruns from: `harness`'s tree (the fixed side every final rerun is measured by) with
    islands/ as the entry's `commit` holds it, its parents both. Dated, so the same pair gives the same commit."""
    if tree(commit) is None:
        raise SystemExit(f"scorer: commit {commit[:10]} not found")
    FINAL.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"GIT_INDEX_FILE": str(FINAL / "overlay.index"), "GIT_AUTHOR_DATE": "2026-11-02T14:30:00Z",
                        "GIT_COMMITTER_DATE": "2026-11-02T14:30:00Z"}
    git("read-tree", harness, env=env)
    git("update-index", "--force-remove", "--stdin", env=env, stdin=git("ls-files", "--", ISLANDS, env=env) + "\n")
    git("read-tree", f"--prefix={ISLANDS}/", f"{commit}:{ISLANDS}", env=env)
    t = git("write-tree", env=env)
    return git("commit-tree", t, "-p", harness, "-p", commit, "-m",
               f"scorer final: {harness[:10]}'s fixed side with {commit[:10]}'s islands/", env=env)


def source_at(commit: str) -> Path:
    """A clean detached checkout of `commit` for launch.py, which ships its own checkout's HEAD (`git archive`); only
    covert_witness and the research package are checked out."""
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


def seed_args(row: dict, seed: int) -> list[str]:
    """The seed flags the entry's train.py takes (its witness's recorded args), set to `seed`: the harness seeds nothing
    for a --script run, so an entry without them is rerun once."""
    have = row.get("train_args") or {}
    return [x for k, flag in (("seed", "--seed"), ("data_seed", "--data-seed")) if k in have for x in (flag, str(seed))]


def entry_argv(row: dict) -> list[str]:
    """The entry's train args as its program got them. The record's `args` is the argv joined by spaces, which loses an
    empty value (ref-optim's `--snap-at ''`) and splits one with spaces (shallow's `--arms`); a flag whose parsed value
    (`train_args`) is a string takes that value whole, and every other flag keeps its tokens."""
    toks, have = (row.get("args") or "").split(), row.get("train_args") or {}
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        i += 1
        v = have.get(t[2:].replace("-", "_")) if t.startswith("--") and "=" not in t else None
        if isinstance(v, str):
            out += [t, v]
            i += len(v.split())
        else:
            out.append(t)
    return out


def superseded(la: dict, rows: dict[str, dict]) -> str | None:
    """Why a launched rerun doesn't stand for its entry: its train args aren't the entry's (`entry_argv`), as the
    launches before entry_argv lost an empty or spaced value. Its entry is relaunched at the same seed."""
    x = rows.get(la.get("id"))
    if la.get("kind") != "rerun" or x is None or "--" not in (la.get("argv") or []):
        return None
    got = la["argv"][la["argv"].index("--") + 1:]
    want = [*unseeded(entry_argv(x)), *seed_args(x, la["seed"] if la.get("seed") is not None else 0)]
    return None if got == want else "its train args aren't the entry's (an empty or spaced value lost)"


def unseeded(args: list[str]) -> list[str]:
    """`args` without the seed flags seed_args sets."""
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in ("--seed", "--data-seed"):
            skip = True
        elif not a.startswith(("--seed=", "--data-seed=")):
            out.append(a)
    return out


def final_launch(ids: list[str], seeds=FINAL_SEEDS, dry_run: bool = False, node: str | None = None,
                 kind: str = "rerun") -> list[dict]:
    """launch.py --final --no-wait for each seed of each entry (a valid search record of the pin), from the overlay of
    its island on the harness's head (launch.py ships the pin and takes only HEAD's islands/ISLAND/, the entry's tree);
    one line each into FINAL/launches.jsonl. A `smoke` (the pipeline's check before the final) is collected like a rerun
    and never enters the table."""
    rows = [json.loads(l) for l in (STATE / "index.jsonl").read_text().splitlines() if l.strip()]
    harness = subprocess.run(["git", "-C", str(official_src()), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
    out = []
    for i in ids:
        x = next((r for r in rows if r["source"] == "launch.py" and r["id"].removeprefix("art:").startswith(
            i.removeprefix("art:"))), None)
        if x is None or not x["valid"] or x.get("final") or x.get("regime") != "current" or not x.get("island"):
            raise SystemExit(f"scorer: {i} is not a valid {regime_name()} search record of launch.py's")
        commit = overlay(x["commit"], harness)
        src = source_at(commit)
        got = git("rev-parse", f"{commit}:{ISLANDS}/{x['island']}")
        if got != x["island_tree"]:
            raise SystemExit(f"scorer: {short(x['id'])}'s island tree is {x['island_tree']}, its commit holds {got}")
        for seed in seeds if seed_args(x, 0) else seeds[:1]:
            tag = "final" if kind == "rerun" else "final smoke"
            note = f"{tag} {short(x['id'])} seed {seed}" if seed_args(x, 0) else f"{tag} {short(x['id'])} (no seed flag)"
            argv = [sys.executable, "benchmarks/covert_witness/launch.py", x["class"], x["start"], "--direction", "scorer",
                    "--final", "--no-wait", "--note", note, "--mechanism", x["mechanism"], "--island", x["island"],
                    "--program", x["program"], *(["--node", node] if node else []),
                    "--", *unseeded(entry_argv(x)), *seed_args(x, seed)]
            row = {"kind": kind, "id": x["id"], "seed": seed if seed_args(x, 0) else None, "commit": commit,
                   "entry_commit": x["commit"], "harness": harness, "class": x["class"], "start": x["start"],
                   "mechanism": x["mechanism"], "note": note, "at": now(), "argv": argv, "run": ""}
            if not dry_run:
                p = subprocess.run(argv, cwd=src, env=tool_env() | {"CW_LEDGER": str(FINAL / "launcher.jsonl")},
                                   capture_output=True, text=True)
                m = next((json.loads(l) for l in p.stdout.splitlines() if l.startswith('{"run"')), None)
                row |= {"run": m["run"] if m else "", "node": m["node"] if m else "", "rc": p.returncode,
                        "measurement_commit": (m or {}).get("measurement_commit"),
                        "err": "" if m else (p.stdout + p.stderr)[-800:]}
                with (FINAL / "launches.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
            out.append(row)
    return out


GPUS_PER_NODE = 8  # each node's gpu pool, g0-g7
FINAL_LINE = re.compile(r"^(\d\d:\d\dZ) (\S+) \[FINAL\] (.*)$")


def final_entries() -> dict[str, dict]:
    """Each direction's last FINAL line of the night ({at, ids, text}): from its own board (islands/*/BOARD.md on every
    cursor/te-* branch) and the merged board (BOARD.md on cursor/te-digest-4207), the later line winning. Only lines
    appended after NIGHT_START count (the boards are append-only, and a line's stamp has no date), and a stamp earlier
    in the day than NIGHT_START's sorts as the next day's."""
    heads = git("ls-remote", "--heads", "origin", "cursor/te-*").split()[1::2]
    subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin", *(f"+{h}:refs/remotes/origin/{h[11:]}" for h in heads)],
                   capture_output=True, text=True)
    t0 = int(NIGHT_START[11:13]) * 60 + int(NIGHT_START[14:16])

    def order(at: str) -> int:
        m = int(at[:2]) * 60 + int(at[3:5])
        return m + 1440 * (m < t0)

    out: dict[str, dict] = {}
    for h in heads:
        ref = f"origin/{h[11:]}"
        paths = [p for p in git("ls-tree", "-r", "--name-only", ref, "--", "benchmarks/covert_witness").splitlines()
                 if p.endswith("BOARD.md") and (p.startswith(f"{ISLANDS}/") or ref == "origin/cursor/te-digest-4207")]
        for p in paths:
            base = git("log", "-1", "--format=%H", f"--before={NIGHT_START}", ref, "--", p)
            was = subprocess.run(["git", "-C", str(REPO), "show", f"{base}:{p}"], capture_output=True, text=True,
                                 check=False).stdout if base else ""
            old = len(was.splitlines())
            for ln in git("show", f"{ref}:{p}").splitlines()[old:]:
                m = FINAL_LINE.match(ln.strip())
                if m and (m[2] not in out or order(m[1]) >= order(out[m[2]]["at"])):
                    out[m[2]] = {"at": m[1], "text": m[3], "board": f"{h[11:]}:{p}", "ids": list(dict.fromkeys(
                        re.findall(r"\b(r\d{8}-\d{6}-[0-9a-f]{4}|(?<=art:)[0-9a-f]{12,64}|[0-9a-f]{12,64})\b", m[3])))}
    return out


FINAL_PER_DIRECTION = 3


def final_plan(begin: str = "15:00", deadline: str = "17:00", seeds=FINAL_SEEDS) -> dict:
    """The reruns in launch order, and which of them end by `deadline` if they start at `begin` (UTC today, or now if
    later) on the listed nodes' GPUs (coordinator 12:55Z: finals 15:00-17:00Z, ~2 h of 24 GPUs): the entries that set
    honest or W* at some target in the pin's table, shortest first, then the directions' other FINAL entries in their
    direction's order, shortest first. A direction's FINAL entries are the first FINAL_PER_DIRECTION distinct records
    its line names (an art id and its run id are one), the rest being its notes. A rerun's time is its record's GPU
    seconds plus 10 minutes."""
    rows = [json.loads(l) for l in (STATE / "index.jsonl").read_text().splitlines() if l.strip()]
    rep = json.loads((STATE / "score.json").read_text())
    sets: dict[str, int] = {}
    for st in rep["starts"].values():
        for t in st["targets"]:
            for i in [t["honest_id"], *(c["id"] for c in t["cells"].values())]:
                if i:
                    sets[i] = sets.get(i, 0) + 1
    fin, named, refused = final_entries(), [], []
    for d, f in fin.items():
        picked: dict[str, dict | None] = {}
        for i in f["ids"]:
            x = next((r for r in rows if r["source"] == "launch.py" and (r["id"].removeprefix("art:").startswith(i)
                                                                         or r.get("run") == i)), None)
            if len(picked) == FINAL_PER_DIRECTION and (x["id"] if x else i) not in picked:
                break
            picked.setdefault(x["id"] if x else i, x)
        for k, (i, x) in enumerate(picked.items()):
            if x is None or not x["valid"] or x.get("final") or x.get("regime") != "current":
                refused.append({"direction": d, "id": i, "why": "no record" if x is None else "invalid" if not x["valid"]
                                else "earlier regime" if x.get("regime") != "current" else "a final rerun"})
            elif x["id"] not in {y["id"] for y, _, _ in named}:
                named.append((x, d, k))
    by_id = {x["id"]: (x, d, k) for x, d, k in named}
    setters = [by_id.get(x["id"], (x, x["direction"], -1)) for x in rows
               if x["source"] == "launch.py" and sets.get(x["id"]) and x.get("regime") == "current"]
    budget = lambda e: e[0].get("gpu_seconds") or 3000
    order = sorted(setters, key=budget) + sorted((e for e in named if not sets.get(e[0]["id"])),
                                                 key=lambda e: (e[2], budget(e), e[1]))
    nodes = listed_nodes()
    now_ = datetime.now(timezone.utc)
    hm = lambda s: now_.replace(hour=int(s[:2]), minute=int(s[3:]), second=0, microsecond=0)
    t0, end = max(now_, hm(begin)), hm(deadline)
    free = [0.0] * (GPUS_PER_NODE * len(nodes))
    plan = []
    for x, d, _ in order:
        dur = budget((x,)) + 600
        ss = list(seeds if seed_args(x, 0) else seeds[:1])
        fit = []
        for s in ss:
            j = min(range(len(free)), key=free.__getitem__)
            if t0.timestamp() + free[j] + dur <= end.timestamp():
                free[j] += dur
                fit.append(s)
        plan.append({"id": x["id"], "direction": d, "named": x["id"] in by_id, "start": x["start"],
                     "sets_cells": sets.get(x["id"], 0), "gpu_seconds": x.get("gpu_seconds"), "seeds": fit,
                     "dropped_seeds": [s for s in ss if s not in fit]})
    return {"at": now(), "begin": t0.strftime("%H:%MZ"), "deadline": deadline, "nodes": nodes, "gpus": len(free),
            "gpu_hours_planned": round(sum(free) / 3600, 1), "finals": fin, "refused": refused, "plan": plan}


WAITING_PER_NODE = 4


def rerun_state(run: str) -> str:
    """A launched rerun's state, `waiting` while its pool hasn't given it a GPU slot (slot.py's slot.json not `held`)."""
    d = RUNS / run
    if (d / "status.json").exists() and (st := json.loads((d / "status.json").read_text()).get("state")) in ENDED:
        return st
    research("fetch", run)
    st = json.loads((d / "status.json").read_text()).get("state") if (d / "status.json").exists() else "submitted"
    slot = json.loads((d / "slot.json").read_text()) if (d / "slot.json").exists() else {}
    return "waiting" if st in ("submitted", "running") and slot.get("state") != "held" else st


def final_dispatch(nodes: list[str], deadline: str = "17:00", dry_run: bool = False) -> list[dict]:
    """One pass: launch the plan's next reruns (final_plan's order, recomputed each pass so a record collected since
    enters it) onto `nodes` in their order, each node taking one while fewer than WAITING_PER_NODE of the scorer's reruns
    wait there for a GPU slot, so a node takes reruns as its search runs free it. A rerun that would end after `deadline`
    (its record's GPU seconds plus 5 minutes from now) is not launched."""
    path = FINAL / "launches.jsonl"
    launches = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    rows = {r["id"]: r for r in (json.loads(l) for l in (STATE / "index.jsonl").read_text().splitlines() if l.strip())
            if r["source"] == "launch.py"}
    launches = [la for la in launches if la.get("kind") == "rerun" and la.get("run") and not superseded(la, rows)]
    launched = {(la["id"], la["seed"]) for la in launches}
    waiting = {n: 0 for n in nodes}
    for la in launches:
        if la.get("node") in waiting and rerun_state(la["run"]) in ("waiting", "submitted"):
            waiting[la["node"]] += 1
    now_ = datetime.now(UTC)
    end = now_.replace(hour=int(deadline[:2]), minute=int(deadline[3:]), second=0, microsecond=0)
    todo = [(e["id"], s) for e in final_plan(deadline=deadline)["plan"]
            for s in (e["seeds"] + e["dropped_seeds"] if seed_args(rows[e["id"]], 0) else [None])
            if (e["id"], s) not in launched]
    out = []
    for n in nodes:
        while waiting[n] < WAITING_PER_NODE and todo:
            i, s = todo.pop(0)
            if now_.timestamp() + (rows[i].get("gpu_seconds") or 3000) + 300 > end.timestamp():
                out.append({"id": i, "seed": s, "skipped": f"would end after {deadline}Z"})
                continue
            out += final_launch([i], (s if s is not None else FINAL_SEEDS[0],), dry_run, n)
            waiting[n] += 1
    return out


def final_collect() -> list[dict]:
    """Every launched rerun's state; once it ended, its record put by launch.py --collect (meta final: true, not ingested
    into the search's ledger), its witness (every score the final set's) and te_core's submissions from its run.json."""
    path = FINAL / "launches.jsonl"
    launches = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    done = json.loads((FINAL / "collected.json").read_text()) if (FINAL / "collected.json").exists() else {}
    src, env = official_src(), tool_env() | {"CW_LEDGER": str(FINAL / "launcher.jsonl")}
    rows = {r["id"]: r for r in (json.loads(l) for l in (STATE / "index.jsonl").read_text().splitlines() if l.strip())
            if r["source"] == "launch.py"}
    out, te = [], ""
    for la in launches:
        run = la.get("run")
        if not run or la.get("kind") not in ("rerun", "smoke"):
            continue
        if why := superseded(la, rows):
            out.append({k: la.get(k) for k in ("kind", "id", "seed", "run", "node", "start")} | {"state": "superseded",
                                                                                                "superseded": why})
            continue
        research("fetch", run)
        st = (json.loads((RUNS / run / "status.json").read_text()).get("state")
              if (RUNS / run / "status.json").exists() else "unknown")
        row = {k: la.get(k) for k in ("kind", "id", "seed", "run", "node", "commit", "entry_commit", "harness", "start")} | {
            "state": st}
        if st in ENDED:
            if run not in done:
                p = subprocess.run([sys.executable, "benchmarks/covert_witness/launch.py", la["class"], la["start"],
                                    "--direction", "scorer", "--final", "--collect", run, "--note", la["note"],
                                    "--mechanism", la["mechanism"]], cwd=src, env=env, capture_output=True, text=True)
                put = next((json.loads(l) for l in p.stdout.splitlines() if l.startswith('{"run"')), None)
                done[run] = put["art"] if put else ""
                FINAL.mkdir(parents=True, exist_ok=True)
                (FINAL / "collected.json").write_text(json.dumps(done, indent=1) + "\n")
            row["art"] = done[run]
            o = RUNS / run / "out"
            if not o.is_dir():
                research("fetch", run, "--all")
            w = json.loads((o / "witness.json").read_text()) if (o / "witness.json").exists() else None
            if w:
                sc = w.get("score") or {}
                row["witness"] = {"valid": w.get("valid"), "split": sc.get("split"), "bpb": sc.get("bpb"),
                                  "start_bpb": sc.get("start_bpb"), "E_bytes": (w.get("E") or {}).get("bytes"),
                                  "e_sha256": ((w.get("harness") or {}).get("submitted") or {}).get("e_sha256"),
                                  "phi": ((w.get("structure_actual") or w.get("structure") or {}).get("phi")),
                                  "macs": (w.get("harness") or {}).get("macs"), "te_core": (w.get("harness") or {}).get(
                                      "te_core"), "gpu_seconds": w.get("gpu_seconds"),
                                  "over": (w.get("harness") or {}).get("over_train_flops"),
                                  "measurement_commit": w.get("measurement_commit"),
                                  "failed_checks": sorted(k for k, v in (w.get("checks") or {}).items() if v is not True)}
            if (o / "run" / "run.json").exists():
                te = te or te_current()
                recs = te_ingest(o / "run", FINAL / "ingest" / te[:12] / run, la["mechanism"], run)
                row["ledger"] = [] if isinstance(recs, str) else [te_rec_row(x, f"{run}#{x['checkpoint_step']}", "")
                                                                   for x in recs]
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
        out[s] = {"start_bpb": l0[s], "targets": [target_row(on, f, l0[s] * (1 - f)) for f in TARGETS],
                  "entries": sorted({x["id"] for x in on})}
    return out


def soft_phi(c: dict) -> bool:
    """A W* entry that is a soft spot (`soft_spots`: an unmerged adapter, its counted φ under a tenth of its position φ,
    prepare.deep_share), shown with both (coordinator 14:22Z; the counted one sets the frontier, 13:42Z)."""
    return bool(c.get("soft_spot")) and c.get("phi") is not None and c.get("phi_position") is not None


def table_md(tab: dict, split: str, seeds: str) -> str:
    lines = [(f"W* and the effective slowdown W*/honest on the {split} set ({regime_name()} records, te_core's ρ on the "
              f"harness's count; FLOPs = 2 MACs; {seeds}). honest and W* read every valid record's complete schedules, "
              "of any direction (coordinator 14:57Z, te_core.frontier.target), so W*/honest >= 1; 'upper bound' marks a "
              "target only a mid-schedule checkpoint reaches; the reference directions' honest alone is context."), ""]
    for s, st in tab.items():
        lines += [f"### {s} (start {st['start_bpb']:.4f} bpb)", "",
                  ("| target | l* | honest FLOPs (entry) | reference-only honest | k | W* r1 | W* r10 | W* r100 "
                   "| W*/honest r1 | r10 | r100 | W* entries |"), "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in st["targets"]:
            h, hr = t["honest_macs"], t.get("honest_reference_macs")
            hon = (f"{fmt(2 * h)} ({short(t['honest_id'])}@{t['honest_cp']}, {t.get('honest_direction')})"
                   + (f", {t['bound']}" if t.get("bound") else "")) if h else "—"
            ref = f"{fmt(2 * hr)} ({short(t['honest_reference_id'])}@{t['honest_reference_cp']})" if hr else "—"
            for k in K_BITS:
                cs = [t["cells"][f"{k}/r{r}"] for r in R]
                ids = ", ".join(sorted({f"{short(c['id'])}@{c['cp']}" + (
                    f" (φ {c['phi']:.3g} counted, {c['phi_position']:.2f} position)" if soft_phi(c) else "")
                                        for c in cs if c["id"]})) or "—"
                lines.append(f"| -{100 * t['drop']:g}% | {t['l_star']:.4f} | {hon} | {ref} | {k} | "
                             + " | ".join(fmt(c["w_star"] and 2 * c["w_star"]) for c in cs) + " | "
                             + " | ".join(fmt(c["effective_slowdown"]) for c in cs) + f" | {ids} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def plot(subs: list[dict], l0: dict[str, float], title: str, path: Path) -> None:
    """Per start, bpb against FLOPs: the reference checkpoints at |C|, every other at |C|·ρ for k = 10 kB, 100 kB, 1 MB
    (r = 1), the targets, and the two frontiers W*/honest compares (coordinator 14:57Z), over every record's complete
    schedules: honest's at |C| (black) and W*'s at |C|·ρ (red; r = 1 solid, r = 10 dashed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    starts = [s for s in STARTS if s in l0 and any(x["start"] == s for x in subs)]
    if not starts:
        return
    fig, axes = plt.subplots(len(starts), 3, figsize=(15, 4.2 * len(starts)), squeeze=False)
    dirs = sorted({x["direction"] for x in subs if not x["reference"]})
    palette = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.tab20(2 * i + 1) for i in range(10)]
    color = {d: palette[i % 20] for i, d in enumerate(dirs)}
    for row, s in zip(axes, starts):
        on = [x for x in subs if x["start"] == s]
        refs, cov = [x for x in on if x["reference"]], [x for x in on if not x["reference"]]
        ends = [x for x in on if x.get("schedule_end")]
        for ax, k in zip(row, K_BITS):
            for f in TARGETS:
                ax.axhline(l0[s] * (1 - f), color="0.8", lw=0.8, zorder=0)
                ax.text(1.0, l0[s] * (1 - f), f" -{100 * f:g}%", transform=ax.get_yaxis_transform(), fontsize=7, va="center")
            if refs:
                ax.scatter([2 * x["macs"] for x in refs], [x["val_bpb"] for x in refs], c="k", marker="s", s=18,
                           label="reference |C|")
            fr = pareto([(x["macs"], x["val_bpb"], x) for x in ends])
            if fr:
                ax.step([2 * x["macs"] for x in fr], [x["val_bpb"] for x in fr], where="post", c="k", lw=1,
                        label="honest frontier: every complete schedule at |C|")
            for r, ls in ((1, "-"), (10, "--")):
                key = f"{k}/r{r}"
                for d in dirs:
                    xs = [x for x in cov if x["direction"] == d]
                    if xs and r == 1:
                        ax.scatter([2 * x["macs"] * x["rho"][key] for x in xs], [x["val_bpb"] for x in xs], color=color[d],
                                   s=14, label=d)
                fr = pareto([(x["macs"] * x["rho"][key], x["val_bpb"], x) for x in ends])
                if fr:
                    ax.step([2 * x["macs"] * x["rho"][key] for x in fr], [x["val_bpb"] for x in fr], where="post",
                            c="tab:red", ls=ls, lw=1, label=f"W* frontier r{r}: every complete schedule at |C| rho")
            ax.set_xscale("log")
            ax.set_ylim(l0[s] * 0.93, l0[s] * 1.01)
            ax.set_title(f"{s}, k = {k}", fontsize=9)
            ax.set_xlabel("FLOPs (reference: |C|; others: |C| rho)", fontsize=8)
            ax.set_ylabel("bpb", fontsize=8)
            ax.tick_params(labelsize=7)
    seen: dict = {}
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=min(8, len(seen)), fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def rerun_problems(r: dict) -> list[str]:
    """Why a rerun's checkpoints don't enter the final table: the checks a search record passes, on the final split."""
    w, led = r.get("witness"), r.get("ledger") or []
    if r.get("state") != "done":
        return [f"run {r.get('state')}"]
    if not w:
        return ["no witness"]
    bad = list(w["failed_checks"]) + ([] if w["valid"] else ["witness invalid"])
    bad += [] if w["split"] == "final" else [f"scored on the {w['split']} split, not the final set"]
    bad += [] if isinstance(w["start_bpb"], (int, float)) else ["no start bpb"]
    pin = pin_of().get("commit")
    if w.get("measurement_commit") != pin:
        bad.append(f"measured by {(w.get('measurement_commit') or 'no commit')[:12]}, not the pin {(pin or '?')[:12]}")
    if r.get("ingest_error") or not led:
        return bad + [r.get("ingest_error") or "no te_core submission"]
    if abs(led[-1]["macs"] - (w["macs"] or -1)) > 1e-9 * led[-1]["macs"]:
        bad.append(f"te_core's |C| {led[-1]['macs']} is not the witness's {w['macs']}")
    if any(t.get("rho") and min(t["rho"].values()) < 1 for t in led):
        bad.append("rho below 1")
    if isinstance(w.get("over"), (int, float)) and w["over"] < COUNTED_FLOOR and short(r.get("id") or "") not in FLOOR_ACCOUNTED:
        bad.append(f"counted {w['over']:.2g}x the run's own steps")
    return bad


def final_table(write: bool = True) -> dict:
    """The final table (`table`) and its plot: every valid pin record's checkpoints, an entry's final-set values (its
    reruns, each checkpoint priced by te_core from the rerun's run.json and scored on the final set, seeds averaged)
    replacing its search values where they exist (coordinator 15:27Z), a search value carried onto the final set's start
    bpb (the reruns' median per start) at the same drop from its search start. Beside it the reruns alone (`final`) and
    the search set alone (`search`)."""
    rows = {x["id"]: x for x in (json.loads(l) for l in (STATE / "index.jsonl").read_text().splitlines() if l.strip())}
    starts = json.loads((STATE / "starts.json").read_text())
    res = [json.loads(l) for l in (FINAL / "results.jsonl").read_text().splitlines() if l.strip()] \
        if (FINAL / "results.jsonl").exists() else []
    subs, reruns = [], []
    soft = {s["rec"]["id"] for s in soft_spots(list(rows.values()))}
    judged = readout_leaks({r["run"]: (RUNS / r["run"] / "out" / "run" / "run.json",
                                       (starts.get(rows[r["id"]]["start"]) or {}).get("tensors"))
                            for r in res if r.get("kind") == "rerun" and r.get("id") in rows and r.get("run")
                            and not r.get("superseded")})
    for r in res:
        x = rows.get(r.get("id"))
        if r.get("kind") != "rerun" or not x or r.get("superseded"):
            continue
        bad = rerun_problems(r)
        leaks = judged.get(r["run"])
        bad += ["read-out leak: steps " + ", ".join(str(s["step"]) for s in leaks)] if leaks else []
        bad += ["read-outs unjudged"] if leaks is None and not bad else []
        reruns.append({"id": r["id"], "seed": r.get("seed"), "run": r["run"], "art": r.get("art"), "state": r["state"],
                       "bpb": (r.get("witness") or {}).get("bpb"), "problems": bad, "readout_leaks": leaks})
        if bad:
            continue
        subs += [{"id": x["id"], "run": r["run"], "seed": r.get("seed"), "cp": round(t.get("fraction") or 0, 3),
                  "start": x["start"], "macs": t["macs"], "val_bpb": t["bpb"], "rho": t["rho"], "phi": t["phi"],
                  "depth": t["D"], "reference": is_reference(x), "direction": x["direction"], "counted": True,
                  "E_bytes": t.get("E_bytes"), "start_bpb": r["witness"]["start_bpb"], "fork_ladder": fork_ladder(x),
                  "phi_position": x.get("phi"), "soft_spot": x["id"] in soft, "split": "final"}
                 for t in r["ledger"] if t.get("te_kind") == "submission" and t.get("rho") and t.get("valid")]
    schedule_ends(subs, key=lambda s: s["run"])
    l0 = {s: v for s in STARTS if (v := regime_start_bpb(subs, s)) is not None}
    fin = seed_mean(subs)
    search_subs = [x for x in json.loads((STATE / "submissions.json").read_text()) if x.get("regime") == "current"]
    l0_search = {s: v for s in STARTS if (v := regime_start_bpb(search_subs, s)) is not None}
    rerun = {x["id"] for x in fin}
    merged = fin + [x | {"val_bpb": x["val_bpb"] * l0[x["start"]] / l0_search[x["start"]], "search_bpb": x["val_bpb"],
                         "split": "search"}
                    for x in search_subs if x["id"] not in rerun and x["start"] in l0 and x["start"] in l0_search]
    rep = {"at": now(), "te_core": te_version(), "frontier": f"te_core {te_pin('frontier') and te_pin('').VERSION} ({TE_PIN})",
           "regime": regime_name(), "pin": pin_of(), "table": table(merged, l0), "final": table(fin, l0),
           "search": table(search_subs, l0_search), "final_starts": l0, "search_starts": l0_search, "reruns": reruns,
           "entries": sorted({x["id"] for x in fin}),
           "superseded": [{k: r.get(k) for k in ("id", "seed", "run", "superseded")} for r in res if r.get("superseded")]}
    if write:
        FINAL.mkdir(parents=True, exist_ok=True)
        (FINAL / "table.json").write_text(json.dumps(rep, indent=1) + "\n")
        (FINAL / "final_submissions.json").write_text(json.dumps(fin) + "\n")
        (FINAL / "table_submissions.json").write_text(json.dumps(merged) + "\n")
        (FINAL / "table.md").write_text(
            "## The final table\n\n" + table_md(rep["table"], "final", "an entry's reruns, seeds averaged per checkpoint, "
                                                "where it was rerun, else its search value at the same drop") + "\n"
            + "## The reruns alone\n\n" + table_md(rep["final"], "final", "seeds averaged per checkpoint") + "\n"
            + "## The search set alone\n\n" + table_md(rep["search"], "search", "the search run's one seed"))
        plot(merged, l0, "final table: reruns on the final set (seeds averaged), else search values at the same drop",
             FINAL / "frontier-final.png")
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
    f = sub.add_parser("final", help="final-set reruns (launch.py --final): entries, plan, launch IDS, collect, table")
    f.add_argument("what", choices=("entries", "plan", "launch", "dispatch", "collect", "table"))
    f.add_argument("ids", nargs="*", help="launch: record ids (12-hex prefixes)")
    f.add_argument("--seeds", default=",".join(map(str, FINAL_SEEDS)))
    f.add_argument("--node", default=None, help="launch: one of nodes.txt's nodes (default: launch.py's turn)")
    f.add_argument("--nodes", default="vy-nebius-1,vy-nebius-3,vy-nebius-2",
                   help="dispatch: the nodes, in the order they take reruns (coordinator 12:55Z: n1, then n3 and n2)")
    f.add_argument("--begin", default="15:00", help="plan: HH:MM UTC the reruns start (or now, if later)")
    f.add_argument("--deadline", default="17:00", help="plan: HH:MM UTC by which a planned rerun ends")
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--smoke", action="store_true", help="launch: a check of the pipeline, never in the table")
    a = ap.parse_args(argv)
    if a.cmd == "final":
        if a.what in ("entries", "plan"):
            rep = final_entries() if a.what == "entries" else final_plan(
                a.begin, a.deadline, tuple(int(s) for s in a.seeds.split(",")))
            if a.what == "plan":
                FINAL.mkdir(parents=True, exist_ok=True)
                (FINAL / "plan.json").write_text(json.dumps(rep, indent=1) + "\n")
            print(json.dumps(rep, indent=1))
            return 0
        if a.what == "launch":
            res = final_launch(a.ids, tuple(int(s) for s in a.seeds.split(",")), a.dry_run, a.node,
                               "smoke" if a.smoke else "rerun")
        elif a.what == "dispatch":
            res = final_dispatch(a.nodes.split(","), a.deadline, a.dry_run)
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

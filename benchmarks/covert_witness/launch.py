"""Launch one candidate on a node and preserve its record (fixed; run from an agent VM's checkout).

    python benchmarks/covert_witness/launch.py START [--island NAME] [--note TEXT] [--direction D] [--mechanism M] \\
        [--node NODE] [--program FILE] [--reference] [--long 'HH:MMZ who'] [-- TRAIN_ARGS...]

The candidate is measured by the pinned measurement commit (`pin.py`, fetched from the integrator's branch): the run ships
that commit (a clean checkout of it beside this one), and takes from this checkout only the island,
`islands/NAME/` as HEAD has it (committed; NAME defaults to the direction, when HEAD has `islands/<direction>/FILE`),
sent as a tar; FILE (default train.py) is the program, and nothing caps what it does. The node is the next of the
pinned commit's `nodes.txt` in turn (`$CW_NODES`; `--node` names one of them), each line `NODE pool=GPU_POOL
cpu=CPU_POOL` for a node `prep.sh` has prepared. The candidate runs as `research run --on NODE --pool GPU_POOL` (one slot
per GPU; candidate.sh leases the GPU with gpu-lease) and records its node (`CW_NODE`): candidate.sh runs the program in a
sandboxed training job that writes only its E files and step log, and scores them in a separate evaluation job. When it
ends, its `out/` (the scored run.json, the witness record, the island and its record, the process tree, the step log, E
when at most 1 MB, logs) is fetched and put in the evidence store as `evidence/v1`, preserved; the run is ingested into
the te_core ledger as agent D (`python -m te_core.ledger --agent D --publish ingest`, one submission record per
checkpoint, each put in the store), and the frontier of the local ledger goes to stderr. The art id, the ledger ids and
the witness summary are printed as one JSON line, and appended to `$CW_LEDGER` (default ~/.research/covert-witness.jsonl).

D, the direction, defaults to the branch's `cursor/te-<D>-NNNN` (else `integrator`); M, the mechanism, to the note's first
word (else the island). A candidate of a reference direction (`te_core.frontier.REF_DIRECTIONS`), or `--reference`, is
ingested as a reference run; honest is the least |C| of every valid record either way (te_core.frontier).

At a v3 pin, START must be one of v3's starts: a start the pin's `starts_store.json` records with a tokenizer other
than data v3's (the pilot's FineWeb-Edu starts) is refused (`refuse_pilot_start`), and scoring leaves out any v3 record
from one (`pilot-start`, te_core.frontier).

`--long 'HH:MMZ who'` runs the candidate at the long tier (`CW_TIER=long`, `prepare.budget`: up to LONG_C0 times the
start's C0 and LONG_GPU_SECONDS of steps), only for a run the board assigns as long, whose line it names; the witness
records it, and is valid at that tier only with it.

`--final` (the scorer's, direction `scorer`, after the search) runs the candidate in final mode (`CW_FINAL=1`): the
harness and the checker score the sealed final set, a sealed start (`pythia160m`) may be named, and the run is preserved
but not ingested into the search's ledger.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ISLANDS = "benchmarks/covert_witness/islands"
RUNS = Path.home() / ".research" / "runs"
LEDGER = Path(os.environ.get("CW_LEDGER", Path.home() / ".research" / "covert-witness.jsonl"))
#: this VM's launches, one line each, and each turn a launch took (a `turn` line, written under `.lock` before the
#: launch starts, so launches started together take consecutive turns); the next launch's node follows the turns in order
LAUNCHES = LEDGER.with_name(LEDGER.stem + "-launches.jsonl")
sys.path.insert(0, str(HERE))
from pin import source  # noqa: E402
from te_core.frontier import REF_DIRECTIONS  # noqa: E402
from te_core.launch import Node, nodes  # noqa: E402
#: the harness this launcher drives (prepare.VERSION, read without importing prepare's torch)
VERSION = re.search(r'^VERSION = "([^"]+)"', (HERE / "prepare.py").read_text(), re.M).group(1)
DONE = {"done", "failed", "cancelled", "refused", "timeout", "lost"}
#: candidate.sh's GPU lease (1.2x prepare.GPU_SECONDS_CAP plus 30 minutes for both jobs) and a margin. The long tier's
#: is capped by run custody instead: research run caps a custody key at 24 hours and counts the pool wait in the run's
#: lifetime, so a long run's timeout plus POOL_WAIT stays under LONG_CUSTODY_TTL. A long run that needs more is split.
TIMEOUT = "110m"
POOL_WAIT = 7200
LONG_TIMEOUT = "1260m"
LONG_CUSTODY_TTL = "24h"


def research(*args: str, check: bool = True, stdout_only: bool = False) -> str:
    r = subprocess.run(["research", *args], capture_output=True, text=True, cwd=REPO)
    if check and r.returncode:
        raise SystemExit(f"launch: research {' '.join(args[:2])} failed ({r.returncode}): {r.stderr[-2000:]}")
    return r.stdout if stdout_only else r.stdout + r.stderr


def tag(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", text.lower())[:64]


def branch_direction() -> str:
    b = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    m = re.fullmatch(r"cursor/te-(.+)-[0-9a-f]{4}", b)
    return tag(m.group(1)) if m else "integrator"


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO)
    if r.returncode:
        raise SystemExit(f"launch: git {' '.join(args[:2])} failed: {r.stderr.strip()[-500:]}")
    return r.stdout.strip()


def island_of(name: str | None, program: str, direction: str) -> dict:
    """The island HEAD has (committed, unedited): `name`, else the direction's."""
    def has(n: str) -> bool:
        return subprocess.run(["git", "cat-file", "-e", f"HEAD:{ISLANDS}/{n}/{program}"], cwd=REPO,
                              capture_output=True).returncode == 0
    if not name and not has(direction):
        raise SystemExit(f"launch: HEAD has no {ISLANDS}/{direction}/{program}: name the island (--island NAME)")
    name = name or direction
    if not re.fullmatch(r"[a-z0-9_-]+", name) or ".." in program or program.startswith("/"):
        raise SystemExit(f"launch: island {name!r} and program {program!r} must be names inside {ISLANDS}/")
    if not has(name):
        raise SystemExit(f"launch: HEAD has no {ISLANDS}/{name}/{program}: commit the island first (the run takes "
                         f"only {ISLANDS}/{name}/ from HEAD)")
    if git("status", "--porcelain", "--untracked-files=all", "--", f"{ISLANDS}/{name}"):
        raise SystemExit(f"launch: {ISLANDS}/{name}/ has uncommitted changes: commit them (the run ships HEAD's island, "
                         "and its record names that commit)")
    return {"name": name, "program": program, "commit": git("rev-parse", "HEAD"),
            "tree": git("rev-parse", f"HEAD:{ISLANDS}/{name}")}


def refuse_pilot_start(start: str, pin: dict, src: Path) -> None:
    """At a v3 pin, a start the pin's `starts_store.json` records with a tokenizer other than data v3's (the pilot's
    FineWeb-Edu starts record none) is refused: its loss on v3's text would make it a v3 branch point. A start the
    store doesn't have is prepare's to refuse (a sealed start in final mode)."""
    if pin.get("data") != "v3":
        return
    cw = src / "benchmarks/covert_witness"
    store = json.loads((cw / "starts_store.json").read_text())["starts"]
    want = re.search(r'^TOKENIZER = "([^"]+)"', (cw / "data_v3.py").read_text(), re.M).group(1)
    if start in store and store[start]["shape"].get("tokenizer") != want:
        v3 = sorted(n for n, s in store.items() if s["shape"].get("tokenizer") == want)
        raise SystemExit(f"launch: {start} is a pilot start, and v3 branches start from the v3 ladder's starts; "
                         f"FineWeb-Edu starts are v2.3-pilot. Launch from one of the pin's v3 starts ("
                         + (", ".join(v3) if v3 else "none published yet: prepare your port, or run a from-scratch "
                            "frontier point on v3") + ")")


def pick_node(name: str | None, path: Path, launches: Path = LAUNCHES) -> Node:
    """`name` if the node list has it, else the list's next node after the turns this VM's launches took so far."""
    known = nodes(path)
    if name:
        node = next((n for n in known if n.name == name), None)
        if node is None:
            raise SystemExit(f"launch: {name} is not in {path} ({', '.join(n.name for n in known)})")
    else:
        launches.parent.mkdir(parents=True, exist_ok=True)
        with open(launches.with_suffix(".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            lines = launches.read_text().splitlines() if launches.exists() else []
            turns = sum("turn" in json.loads(line) for line in lines if line.strip())
            node = known[turns % len(known)]
            with launches.open("a") as f:
                f.write(json.dumps({"turn": turns, "node": node.name, "at": time.time()}) + "\n")
    if not node.pool:
        raise SystemExit(f"launch: {path} names no gpu pool for {node.name} (`{node.name} pool=gpu cpu=research`)")
    return node


def main() -> int:
    argv = sys.argv[1:]
    train_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("start")
    ap.add_argument("--note", default="")
    ap.add_argument("--direction", default=None, help="the ledger agent and direction (default: from the branch)")
    ap.add_argument("--mechanism", default=None, help="the mechanism tag (default: the note's first word)")
    ap.add_argument("--lane", default=os.environ.get("RESEARCH_LANE", "training"))
    ap.add_argument("--no-wait", action="store_true", help="print the run id and return; `--collect RUN` later")
    ap.add_argument("--collect", default=None, metavar="RUN", help="collect an ended candidate run")
    ap.add_argument("--final", action="store_true", help="final scoring, the scorer's (module docstring)")
    ap.add_argument("--node", default=None, help="one of nodes.txt's nodes (default: the next in turn)")
    ap.add_argument("--island", default=None, help="the island, islands/NAME/ (default: the direction's)")
    ap.add_argument("--program", default="train.py", help="the island's program (default train.py)")
    ap.add_argument("--reference", action="store_true", help="ingest as a reference run (default: a reference direction's)")
    ap.add_argument("--long", default=None, metavar="BOARD_LINE",
                    help="the long tier (prepare.budget), for a run the board assigns as long: name its line ('HH:MMZ who')")
    a = ap.parse_args(argv)
    if a.long is not None and not re.fullmatch(r"\d\d:\d\dZ [a-z0-9-]+.*", a.long):
        raise SystemExit("launch: --long names the board line that assigns the run, as 'HH:MMZ who'")
    direction = tag(a.direction) if a.direction else branch_direction()
    reference = a.reference or direction in REF_DIRECTIONS
    if a.final and direction != "scorer":
        raise SystemExit("launch: --final is the scorer's (direction scorer), after the search")
    if a.collect:
        mechanism = tag(a.mechanism or (a.note.split() or [a.island or direction])[0])
        return collect(a.collect, a.start, a.note, direction, mechanism, a.final, reference)
    island = island_of(a.island, a.program, direction)
    mechanism = tag(a.mechanism or (a.note.split() or [island["name"]])[0]) or island["name"]
    pin, src = source()
    if str(pin.get("version", "")).split(".")[0] != VERSION.split(".")[0]:
        raise SystemExit(f"launch: this launcher is harness v{VERSION} and the pin is v{pin.get('version')} "
                         f"({pin['commit'][:11]}): launch from a checkout at the pin's version until the integrator "
                         f"pins v{VERSION} (the board's [TOOL] line), then merge origin/cursor/covert-witness-4207")
    refuse_pilot_start(a.start, pin, src)
    node = pick_node(a.node, Path(os.environ.get("CW_NODES", src / "benchmarks/covert_witness/nodes.txt")))
    with tempfile.TemporaryDirectory() as tmp:
        tar = Path(tmp) / "island.tar"
        git("archive", "--format=tar", "-o", str(tar), f"HEAD:{ISLANDS}/{island['name']}")
        env = ["--env", f"CW_NODE={node.name}", "--env", f"CW_ISLAND={island['name']}",
               "--env", f"CW_PROGRAM={island['program']}", "--env", f"CW_ISLAND_COMMIT={island['commit']}",
               "--env", f"CW_ISLAND_TREE={island['tree']}", *(["--env", "CW_FINAL=1"] if a.final else []),
               *(["--env", "CW_TIER=long", "--env", f"CW_TIER_BY={a.long}"] if a.long else [])]
        cmd = ["run", "--on", node.name, "--project", "verity", "--source", str(src), "--send", str(tar),
               "--pool", node.pool, "--as", a.lane, "--pool-wait", str(POOL_WAIT),
               "--timeout", LONG_TIMEOUT if a.long else TIMEOUT,
               *(["--custody-ttl", LONG_CUSTODY_TTL] if a.long else []),
               "--declared-output", "out/*", *env, "--",
               "bash", "benchmarks/covert_witness/candidate.sh", a.start, *train_args]
        out = research(*cmd)
    m = re.search(r"run (r\d{8}-\d{6}-[0-9a-f]{4}) launched", out)
    if not m:
        raise SystemExit(f"launch: no run id in research run's output:\n{out[-2000:]}")
    run = m.group(1)
    launched = {"run": run, "node": node.name, "start": a.start, "direction": direction, "mechanism": mechanism,
                "reference": reference, "measurement_commit": pin["commit"], "island": island,
                "tier": "long" if a.long else "search", "tier_assigned_by": a.long}
    LAUNCHES.parent.mkdir(parents=True, exist_ok=True)
    with LAUNCHES.open("a") as f:
        f.write(json.dumps(launched) + "\n")
    print(json.dumps(launched), flush=True)
    if a.no_wait:
        return 0
    while True:
        research("fetch", run, check=False)
        st = json.loads((RUNS / run / "status.json").read_text()) if (RUNS / run / "status.json").exists() else {}
        if st.get("state") in DONE:
            break
        time.sleep(30)
    return collect(run, a.start, a.note, direction, mechanism, a.final, reference)


def ingest(run: str, out: Path, direction: str, mechanism: str, reference: bool) -> dict:
    """The harness's record into the te_core ledger (each record published), then the local frontier to stderr."""
    if not (out / "run" / "run.json").exists():
        return {"ledger_error": "no run/run.json"}
    r = subprocess.run([sys.executable, "-m", "te_core.ledger", "--agent", direction, "--publish", "ingest",
                        str(out / "run"), "--direction", direction, "--mechanism", mechanism, "--run-id", run,
                        *(["--reference"] if reference else [])], capture_output=True, text=True, cwd=HERE)
    if r.returncode:
        return {"ledger_error": r.stderr.strip()[-500:]}
    recs = json.loads(r.stdout.strip().splitlines()[-1])
    f = subprocess.run([sys.executable, "-m", "te_core.frontier"], capture_output=True, text=True, cwd=HERE)
    print(f.stdout if f.returncode == 0 else f"launch: te_core.frontier failed: {f.stderr[-500:]}", file=sys.stderr)
    return {"ledger": [x["id"] for x in recs], "macs": recs[-1]["macs"] if recs else None,
            "rho": recs[-1].get("rho") if recs else None}


def collect(run: str, start: str, note: str, direction: str, mechanism: str, final: bool = False,
            reference: bool = False) -> int:
    research("fetch", run, "--all")
    out = RUNS / run / "out"
    w = json.loads((out / "witness.json").read_text()) if (out / "witness.json").exists() else None
    remote = RUNS / run / "remote.json"
    shipped = json.loads(remote.read_text()).get("source_sha") if remote.exists() else None
    agrees = not w or w.get("measurement_commit") == shipped
    if not agrees:
        print(f"launch: {run}'s witness names measurement commit {w.get('measurement_commit')}, the run shipped "
              f"{shipped}: preserved, not ingested", file=sys.stderr)
    put = json.loads(research("data", "put", "--kind", "evidence/v1", "--tree", str(out), "--preserve", "--json",
                              "--meta", json.dumps({"covert_witness": {
                                  "run": run, "island": w and (w.get("island") or {}).get("name"), "start": start,
                                  "note": note, "direction": direction, "mechanism": mechanism, "final": final}}),
                              stdout_only=True))
    row = {"run": run, "art": put["id"], "start": start, "note": note, "direction": direction,
           "mechanism": mechanism, "final": final, "reference": reference, "measurement_commit": shipped,
           "island": w and {k: (w.get("island") or {}).get(k) for k in ("name", "program", "commit", "digest")},
           "valid": w and w.get("valid"),
           "bpb": w and w.get("score", {}).get("bpb"), "split": w and w.get("score", {}).get("split"),
           "E_bytes": w and w["E"]["bytes"], "phi": w and w.get("structure", {}).get("phi"),
           "lowest_updated_layer": w and w.get("structure", {}).get("lowest_updated_layer"),
           "train_flops": w and w.get("harness", {}).get("flops"), "gpu_seconds": w and w.get("gpu_seconds"),
           "checks": w and w.get("checks")}
    if not final and agrees:
        row |= ingest(run, out, direction, mechanism, reference)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return 0 if w else 1


if __name__ == "__main__":
    sys.exit(main())

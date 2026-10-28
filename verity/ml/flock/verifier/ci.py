"""The CI agreement job (PROTOCOL.md §17.1): every replayable set of `vectors.json` through both verifiers.

    python ci.py --lean BIN --upstream PR83_COMMIT=BIN [--upstream ...] --work DIR [--inputs DIR] [--jobs J] [--session-jobs K]
        [--seed S] [--fuzz N] [--sets I,J] [--out DIR]
    python ci.py --live --lean BIN --work DIR [--sets I,J] ...    `vectors.json`'s `live.sets`, against the live verdicts

For each set (in `vectors.json` `replayable.sets`) whose PR #83 commit has an upstream binary, the job:
- fetches its inputs from the evidence store (`research data fetch`), among them the verifier's own registered reads
  (`statement.registered`, `agree.py --registered`) for a statement that reads registered values, and passes the verifier's
  own K (`statement.stratified`, `agree.py --stratified`) for sessions drawn by the stratified law, and a plain session's
  coins (`statement.coins`, `agree.py --coins`: `os` for a live server's OS coins);
- for a set that names `mutants`, regenerates the proof mutants (`redigest.py`), the retained-bytes sessions
  (`redigest.py --retain`) and `N` fuzz cases (`fuzz.py --seed S`) from its honest session;
- adds the set's stored forgeries, if it has any;
- puts every stored session's record (its vectors and forgeries) in the canonical text the writers emit
  (`record_text.session_text`, Lean's `Json.compress`), in a copy under the set's work directory;
- runs `agree.py` on all of them, with `--lean-refuses` for a set marked `lean_refuses` (D12: a statement form the soundness
  proof does not cover, which the Lean verifier refuses whatever the session, once every check has passed),
  `--lean-refuses-drawn` for one marked `lean_refuses_drawn` (D12 at its drawn sessions' law, which no soundness claim draws
  by, refused at setup), and `--lean-refuses-drawless` for one marked `lean_refuses_drawless` (D12 at its sessions without
  a draw, refused once every check has passed). A session refused last, and for nothing else, passed every check.

`--live` runs `live.sets` instead: sets with no upstream binary, whose upstream verdicts are the live server's in each
case's `case.json` (`agree.py --upstream live`).

`--inputs DIR` holds the stored inputs already fetched (`DIR/<art id prefix>/…`, as the store's fetch lays them
out), for a machine without store access. `--jobs` runs sets in parallel, and `--session-jobs` a set's sessions across that
many verifier processes at once over all sets (by default the cores, and the available memory at 8 GB a process): each set's
sessions are dealt into chunks of four or more (`agree.py --jobs`), and a set finishing early frees its slots for the
others. The verdicts and every output are the same whatever the parallelism. Writes DIR/result.json (one entry per set) and
exits 1 if any set has a disagreement or a missed expectation. With `--out` (a recorded run's directory), each set's
agreement outputs are copied there and `result.json` is written in the `research/result/v0.1` envelope. Run it as a
recorded job (`ci-pod.sh` on a CPU pod). Use a fixed seed per change and a rolling seed nightly.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agree  # noqa: E402
from record_text import session_text  # noqa: E402

#: the verifier processes all sets may run at once (main sets it: `--session-jobs`)
SLOTS = threading.Semaphore(1)


def store_env():
    """The store CLI's environment. Inside a recorded check's steps, which carry no store credentials, it is given the run's
    custody key that `$CHECK_STORE_CUSTODY` names (`tools/verity/check/check.py`, "The evidence store"), when no store config is set
    here, as integrations/vllm's `store_io` does: the key goes into that subprocess's environment only."""
    env = dict(os.environ)
    custody = Path(env["CHECK_STORE_CUSTODY"]) if env.get("CHECK_STORE_CUSTODY") else None
    if (custody is None or env.get("RESEARCH_STORE_CONFIG") or (Path.home() / ".research" / "store.toml").is_file()
            or not ((custody / "store.toml").is_file() and (custody / "cred.json").is_file())):
        return env
    cred = json.loads((custody / "cred.json").read_text())
    env["RESEARCH_STORE_CONFIG"] = str(custody / "store.toml")
    env.update({k: cred[k] for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN") if cred.get(k)})
    return env


def fetch(art, store):
    """`art:<id>[/path]` under store/<id>/ (fetched from the evidence store unless present); a file artifact is its one file."""
    ref, _, path = art.partition("/")
    d = store / ref.split(":", 1)[1]
    if not d.exists():
        subprocess.run([sys.executable, "-m", "research", "data", "fetch", ref, "--to", str(d)], check=True, capture_output=True,
                       env=store_env())
    if path:
        return d / path
    files = list(d.iterdir())
    return files[0] if len(files) == 1 and files[0].is_file() else d


def canonical(src, dst, records=("session.json",)):
    """The stored session directory `src` at `dst`, its record in the canonical text the writers emit
    (`record_text.session_text`): the stored vectors predate it. Its other files are hard links (copies across devices); a
    record that isn't JSON is kept as it is. `records`: the records to re-render, relative to `dst` (`canonical_tree`'s
    are every `session.json` under it)."""
    if not dst.exists():
        def link(a, b):
            try:
                os.link(a, b)
            except OSError:
                shutil.copy2(a, b)
        shutil.copytree(src, dst, copy_function=link)
        for rec in (dst / r for r in (records(dst) if callable(records) else records)):
            if not rec.is_file():
                continue
            try:
                v = json.loads(rec.read_text())
            except ValueError:
                continue
            rec.unlink()
            rec.write_text(session_text(v))
    return dst


def canonical_tree(src, dst):
    """`canonical` of a stored tree of sessions: every `session.json` under it re-rendered. Only for a tree whose records
    nothing binds by their bytes (a recursive set's inner record is: re-render the session being verified alone)."""
    return canonical(src, dst, lambda d: [p.relative_to(d) for p in d.rglob("session.json")])


def fetch_sessions(art, store):
    """`fetch`, as a `canonical_tree` copy under store/<id>.canonical/: a stored set whose sessions a test verifies."""
    ref, _, path = art.partition("/")
    d = canonical_tree(fetch(ref, store), store / (ref.split(":", 1)[1] + ".canonical"))
    return d / path if path else d


def py(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True, capture_output=True)


def run_set(i, s, a, ups, store):
    up = agree.LIVE if a.live else ups.get(s["pr83"])
    if up is None:
        return {"set": i, "pr83": s["pr83"], "skipped": "no upstream binary for this PR #83 commit"}, False
    w = a.work / f"set{i}"
    st = s["statement"]
    circuit, public = fetch(st["circuit"], store), fetch(st["public"], store)
    comp = fetch(st["comp"], store) if "comp" in st else None
    partition = fetch(st["partition"], store) if "partition" in st else None
    program = fetch(st["program"], store) if "program" in st else None
    tables = fetch(st["tables"], store) if "tables" in st else None
    registered = fetch(st["registered"], store) if "registered" in st else None
    checks = ",".join(s.get("upstream_checks", [])) or "none"
    sessions, expects = [], []
    if "sessions" in s:
        vectors = fetch(s["sessions"][0]["dir"].rsplit("/", 1)[0], store)
        expect = w / "expect.json"
        w.mkdir(parents=True, exist_ok=True)
        expect.write_text(json.dumps([{"name": x["dir"].rsplit("/", 1)[1], "expect": x["expect"]} for x in s["sessions"]]))
        expects = [expect]
        names = [x["dir"].rsplit("/", 1)[1] for x in s["sessions"]]
        sessions = [str(canonical(vectors / n, w / "stored" / n)) for n in names]
    if "mutants" in s:
        honest = vectors / "honest"
        py(HERE / "redigest.py", honest, w / "mutants")
        py(HERE / "redigest.py", "--retain", honest, w / "retained", "--upstream-checks", checks)
        py(HERE / "fuzz.py", honest, w / "fuzz", "--seed", a.seed, "--n", a.fuzz, "--upstream-checks", checks)
        expects = [w / "mutants" / "mutants.json", w / "fuzz" / "fuzz.json", w / "retained" / "retained.json", expect]
        sessions += sorted(str(p) for p in (w / "mutants").iterdir() if p.is_dir())
        sessions += [str(w / "retained" / n) for n in ("retained", "retained-edited")]
        if "retained_bytes" in checks.split(","):
            sessions.append(str(w / "retained" / "retained-not-digest"))
        sessions += sorted(str(p) for p in (w / "fuzz").iterdir() if p.is_dir())
    if "forgeries" in s:
        f = fetch(s["forgeries"], store)
        sessions += sorted(str(canonical(p, w / "forgeries" / p.name)) for p in f.iterdir() if p.is_dir())
    for k, rd in enumerate(s.get("selftest", [])):
        d = w / "selftest" / str(k)
        py(HERE / "selftest_records.py", fetch(rd, store), d)
        sessions += sorted(str(p) for p in d.iterdir() if p.is_dir())
    cmd = [sys.executable, HERE / "agree.py", "--lean", a.lean, "--upstream", up, "--statement", s["lean_statement"],
           "--circuit", circuit, *(("--comp", comp) if comp else ()), *(("--partition", partition) if partition else ()), *(("--program", program) if program else ()),
           "--public", public, *(("--registered", registered) if registered else ()), *(x for e in expects for x in ("--expect", e)),
           "--out", w / "agree", "--jobs", a.session_jobs, "--sessions", *sessions]
    if tables:
        cmd += ["--tables", tables]
    if "stratified" in st:
        cmd += ["--stratified", str(st["stratified"])]
    if "coins" in st:
        cmd += ["--coins", st["coins"]]
    if "lean_refuses" in s:
        cmd += ["--lean-refuses", s["lean_refuses"]]
    if "lean_refuses_drawn" in s:
        cmd += ["--lean-refuses-drawn", s["lean_refuses_drawn"]]
    if "lean_refuses_drawless" in s:
        cmd += ["--lean-refuses-drawless", s["lean_refuses_drawless"]]
    # in-process: every set's verifier processes draw on one budget (SLOTS), so a long set takes the slots short ones free
    rc = agree.main([str(x) for x in cmd[2:]], slots=SLOTS, quiet=True)
    r = json.loads((w / "agree" / "result.json").read_text())
    r.update({"set": i, "pr83": s["pr83"], "lean_statement": s["lean_statement"], "detail": st.get("detail", "")})
    return r, rc != 0


#: the memory one verifier process may take (a statement of m <= 26: upstream's replay builds it, and leaks it)
GB_PER_PROCESS = 8


def auto_session_jobs():
    """Verifier processes at once, over all sets: the cores, and the available memory at `GB_PER_PROCESS` each."""
    mem = GB_PER_PROCESS
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                mem = int(line.split()[1]) // (1 << 20)
    except OSError:
        pass
    return max(1, min(os.cpu_count() or 1, mem // GB_PER_PROCESS))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--upstream", action="append", default=[], metavar="COMMIT=BIN")
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--inputs", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=1, help="sets at once")
    ap.add_argument("--session-jobs", type=int, default=0,
                    help="verifier processes at once over all sets; 0: the cores, and the available memory at 8 GB a process")
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--fuzz", type=int, default=20)
    ap.add_argument("--live", action="store_true", help="vectors.json's live.sets (the live server's verdicts), not replayable.sets")
    ap.add_argument("--sets", default=None, help="comma-separated set indices (default: every set with an upstream binary)")
    ap.add_argument("--out", type=Path, default=None, help="a recorded run's directory: outputs and result.json go there")
    a = ap.parse_args()
    ups = dict(u.split("=", 1) for u in a.upstream)
    store = a.inputs or (a.work / "store")
    sets = json.loads((HERE / "vectors.json").read_text())["live" if a.live else "replayable"]["sets"]
    pick = [int(i) for i in a.sets.split(",")] if a.sets else list(range(len(sets)))
    global SLOTS
    if a.session_jobs <= 0:
        a.session_jobs = auto_session_jobs()
    SLOTS = threading.Semaphore(a.session_jobs)
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        done = list(ex.map(lambda i: run_set(i, sets[i], a, ups, store), pick))
    results = [r for r, _ in done]
    bad = any(b for _, b in done)
    (a.work / "result.json").write_text(json.dumps({"seed": a.seed, "sets": results}, indent=1))
    print(json.dumps(results))
    if a.out:
        out = a.out / "ci"
        for r in results:
            if "skipped" in r:
                continue
            w = a.work / f"set{r['set']}"
            shutil.copytree(w / "agree", out / f"set{r['set']}", dirs_exist_ok=True)
            for f in ("mutants/mutants.json", "fuzz/fuzz.json", "retained/retained.json", "expect.json"):
                if (w / f).exists():
                    shutil.copy(w / f, out / f"set{r['set']}" / Path(f).name)
        (out / "sets.json").write_text(json.dumps({"seed": a.seed, "sets": results}, indent=1))
        ran = [r for r in results if "skipped" not in r]
        doc = {
            "schema": "research/result/v0.1",
            "run_id": os.environ.get("RESEARCH_RUN_ID", ""),
            "kind": "verification-verdict/v1",
            "workload_fingerprint": {"lean": sha256(a.lean), "upstream": {k: sha256(v) for k, v in ups.items()},
                                     "vectors_json": sha256(HERE / "vectors.json"), "seed": a.seed, "fuzz": a.fuzz,
                                     "bundle": os.environ.get("CI_BUNDLE_SHA256"),
                                     "sets": [r["set"] for r in ran]},
            "validation": {"status": "failed" if bad else "passed", "checker": "verity/ml/flock/verifier/agree.py",
                           "detail": "; ".join(f"set {r['set']} ({r['pr83']}): {r['agree']}/{r['sessions']} agree, "
                                               f"{r['accepted']} accepted, {r['passed_every_check']} passed every check"
                                               for r in ran),
                           "evidence": ["ci/sets.json"]},
            "measurements": [{"name": "sessions", "value": sum(r["sessions"] for r in ran), "unit": "count"},
                             {"name": "agree", "value": sum(r["agree"] for r in ran), "unit": "count"},
                             {"name": "disagree", "value": sum(len(r["disagree"]) for r in ran), "unit": "count"},
                             {"name": "accepted", "value": sum(r["accepted"] for r in ran), "unit": "count"},
                             {"name": "passed_every_check", "value": sum(r["passed_every_check"] for r in ran),
                              "unit": "count"}],
            "measurement_files": ["ci/sets.json"],
            "artifacts": ["ci/"],
        }
        tmp = a.out / "result.json.tmp"
        tmp.write_text(json.dumps(doc, indent=1))
        os.replace(tmp, a.out / "result.json")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

"""The CI agreement job (PROTOCOL.md §17.1): every replayable set of `vectors.json` through both verifiers.

    python ci.py --lean BIN --upstream PR83_COMMIT=BIN [--upstream ...] --work DIR [--seed S] [--fuzz N] [--sets I,J]

For each set (in `vectors.json` `replayable.sets`) whose PR #83 commit has an upstream binary, the job:
- fetches its inputs from the evidence store (`research data fetch`);
- regenerates the proof mutants (`redigest.py`), the retained-bytes sessions (`redigest.py --retain`) and `N` fuzz cases
  (`fuzz.py --seed S`) from its honest session;
- adds the set's stored forgeries, if it has any;
- runs `agree.py` on all of them.

Writes DIR/result.json (one entry per set) and exits 1 if any set has a disagreement or a missed expectation. Run it as a
recorded job: `research run --tool flock-agreement --campaign flock-verifier -- python ci.py ...`. Use a fixed seed per change
and a rolling seed nightly.
"""
import argparse, json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fetch(art, work):
    """Materialise `art:<id>[/path]` under work/<id>/ and return the local path."""
    ref, _, path = art.partition("/")
    d = work / "store" / ref.split(":", 1)[1]
    if not d.exists():
        subprocess.run([sys.executable, "-m", "research", "data", "fetch", ref, "--to", str(d)], check=True, capture_output=True)
    return d / path if path else d


def py(*args):
    subprocess.run([sys.executable, *map(str, args)], check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lean", required=True)
    ap.add_argument("--upstream", action="append", default=[], metavar="COMMIT=BIN")
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--fuzz", type=int, default=20)
    ap.add_argument("--sets", default=None, help="comma-separated set indices (default: every set with an upstream binary)")
    a = ap.parse_args()
    ups = dict(u.split("=", 1) for u in a.upstream)
    sets = json.loads((HERE / "vectors.json").read_text())["replayable"]["sets"]
    pick = [int(i) for i in a.sets.split(",")] if a.sets else range(len(sets))
    results, bad = [], False
    for i in pick:
        s = sets[i]
        up = ups.get(s["pr83"])
        if up is None:
            results.append({"set": i, "pr83": s["pr83"], "skipped": "no upstream binary for this PR #83 commit"})
            continue
        w = a.work / f"set{i}"
        st = s["statement"]
        circuit, public, comp = fetch(st["circuit"], a.work), fetch(st["public"], a.work), fetch(st["comp"], a.work)
        tables = fetch(st["tables"], a.work) if "tables" in st else None
        vectors = fetch(s["sessions"][0]["dir"].rsplit("/", 1)[0], a.work)
        honest = vectors / "honest"
        py(HERE / "redigest.py", honest, w / "mutants")
        py(HERE / "redigest.py", "--retain", honest, w / "retained")
        py(HERE / "fuzz.py", honest, w / "fuzz", "--seed", a.seed, "--n", a.fuzz)
        expect = w / "expect.json"
        expect.write_text(json.dumps([{"name": x["dir"].rsplit("/", 1)[1], "expect": x["expect"]} for x in s["sessions"]]))
        sessions = [str(vectors / x["dir"].rsplit("/", 1)[1]) for x in s["sessions"]]
        sessions += sorted(str(p) for p in (w / "mutants").iterdir() if p.is_dir())
        sessions += [str(w / "retained" / "retained"), str(w / "retained" / "retained-edited")]
        sessions += sorted(str(p) for p in (w / "fuzz").iterdir() if p.is_dir())
        if "forgeries" in s:
            f = fetch(s["forgeries"], a.work)
            sessions += sorted(str(p) for p in f.iterdir() if p.is_dir())
        cmd = [sys.executable, HERE / "agree.py", "--lean", a.lean, "--upstream", up, "--statement", s["lean_statement"],
               "--circuit", circuit, "--comp", comp, "--public", public, "--expect", w / "mutants" / "mutants.json",
               "--expect", w / "fuzz" / "fuzz.json", "--expect", expect, "--out", w / "agree", "--sessions", *sessions]
        if tables:
            cmd += ["--tables", tables]
        p = subprocess.run(list(map(str, cmd)), capture_output=True, text=True)
        r = json.loads((w / "agree" / "result.json").read_text())
        r.update({"set": i, "pr83": s["pr83"], "detail": st.get("detail", "")})
        results.append(r)
        bad = bad or p.returncode != 0
    (a.work / "result.json").write_text(json.dumps({"seed": a.seed, "sets": results}, indent=1))
    print(json.dumps(results))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

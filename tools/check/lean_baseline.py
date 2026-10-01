"""`lean_baseline`: what `check`'s Lean step costs one package, built cold and then after edits that change a module's `.olean`.

    python tools/check/lean_baseline.py PKG --out DIR [--edit MODULE]... [--deps DIR] [--urls FILE]

It sets up as `lean_audit` does: a scratch tree of the package's inputs, its fetched dependencies warm (`WarmDeps`), Lean given
this machine's CPUs. It runs `tools/lean/audit.py --build PKG` once with nothing of the package's own built, then once after each
`--edit`, which appends a theorem to MODULE so that MODULE's `.olean` and every importer's change; the edits accumulate. After an
edit it also replays through the kernel only the modules that pass rebuilt, which is what a per-module replay cache would leave
to do. `DIR/baseline.json` holds, per pass, `audit.py`'s seconds per step, each module's build time from Lake's log, and that
replay's seconds, rewritten after every pass; each pass's reports and logs are beside it. The dependencies go back only when
every pass passed, as after an audit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LA = _module(ROOT / "tools" / "check" / "lean_audit.py", "lean_audit_for_baseline")

#: Lake's line for a job it ran, `✔ [i/n] Built M (850ms)`, `(1.2s)` or `(42s)` (`formatTime` in `Lake.Build.Run`)
BUILT = re.compile(r"\] (?:\(Optional\) )?Built (\S+) \((\d+(?:\.\d)?)(ms|s)\)")


def built(log: str) -> dict[str, float]:
    """Seconds per job Lake built, slowest first."""
    got = {m: float(v) / (1000 if u == "ms" else 1) for m, v, u in BUILT.findall(log)}
    return dict(sorted(got.items(), key=lambda kv: -kv[1]))


def source(pkg: Path, module: str) -> Path:
    return pkg / (module.replace(".", "/") + ".lean")


def one_pass(work: Path, pkg: str, out: Path, env: dict, replay: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, str(work / "tools" / "lean" / "audit.py"), "--build", str(work / pkg), "--out", str(out)],
                       cwd=work, env=env)
    got = {"exit": r.returncode, "wall": round(time.monotonic() - t0, 1)}
    doc = json.loads((out / "audit.json").read_text()) if (out / "audit.json").exists() else {}
    rep = (doc.get("packages") or [{}])[0]
    got |= {"seconds": rep.get("seconds", {}), "failures": rep.get("failures") or ([] if r.returncode == 0 else ["audit.py"])}
    adir = next(out.glob("audit-*"), out)
    log = "".join((adir / f).read_text() for f in ("build.log", "build-roots.log") if (adir / f).exists())
    got["built"] = built(log)
    facts = adir / "facts.json"
    own = [m for m in json.loads(facts.read_text())["modules"] if m in got["built"]] if facts.exists() else []
    got["rebuilt_own"] = len(own)
    if replay and own:
        t0 = time.monotonic()
        rr = subprocess.run(["lake", "env", "lean", "--run", str(work / "tools" / "lean" / "Replay.lean"),
                             str(out / "replay-rebuilt.json"), *own], cwd=work / pkg, env=env, capture_output=True, text=True)
        (out / "replay-rebuilt.log").write_text(rr.stdout + rr.stderr)
        got["replay_rebuilt"] = {"modules": len(own), "seconds": round(time.monotonic() - t0, 1), "exit": rr.returncode}
        if rr.returncode:
            got["failures"].append(f"replay of the rebuilt modules: exit {rr.returncode}")
    return got


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lean_baseline.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("package", help="the package's directory, relative to the repository root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--edit", action="append", default=[], metavar="MODULE",
                    help="a module of the package to append a theorem to before another pass (repeatable; the edits accumulate)")
    ap.add_argument("--deps", type=Path, default=LA.COMMON.cache_dir() / "lean-deps")
    ap.add_argument("--urls", type=Path, help=f"{LA.URLS}: presigned URLs for the pinned bundles ({LA.PIN})")
    a = ap.parse_args(argv)
    files = LA.COMMON.tracked_files(ROOT)
    pkg = a.package.rstrip("/")
    if files is None or pkg not in LA.packages(files):
        ap.error(f"{pkg} is not an audited package of this checkout")
    if missing := [m for m in a.edit if not source(ROOT / pkg, m).is_file()]:
        ap.error(f"no source file for {', '.join(missing)} in {pkg}")
    urls = json.loads(a.urls.read_text()) if a.urls and a.urls.is_file() else None
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    a.deps.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="lean-baseline-", dir=a.deps.parent))    # beside the dependencies: moves, not copies
    warm = LA.WarmDeps(a.deps, urls)
    union = LA.inputs(pkg, files)
    pkgs = LA.packages(union)
    doc: dict = {"package": pkg, "cores": LA.COMMON.cores(), "edits": a.edit, "passes": []}
    ok = False
    try:
        LA.scratch_tree(work, union)
        for p in pkgs:
            warm.take(work / p, lambda s: print(s, flush=True))
        doc["dependencies"] = ("none" if LA.deps_key(work / pkg) is None
                               else "warm" if (work / pkg / ".lake" / "packages").is_dir() else "cold")
        env = {**os.environ, "PATH": f"{Path.home() / '.elan' / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}",
               "LEAN_NUM_THREADS": str(doc["cores"])}
        ok = True
        for i, module in enumerate([None, *a.edit]):
            name = "cold" if module is None else f"edit-{i}-{module}"
            if module is not None:
                f = source(work / pkg, module)
                f.write_text(f.read_text() + f"\n\ntheorem leanBaselineProbe{i} : True := trivial\n")
            got = {"pass": name, "edit": module, **one_pass(work, pkg, out / name, env, replay=module is not None)}
            doc["passes"].append(got)
            (out / "baseline.json").write_text(json.dumps(doc, indent=1) + "\n")
            print(f"lean-baseline: {name}: {'PASS' if not got['failures'] else 'FAIL'} in {got['wall']} s; "
                  f"{got['rebuilt_own']} own modules built; {got['seconds']}"
                  + (f"; replay of those {got['replay_rebuilt']['seconds']} s" if "replay_rebuilt" in got else ""), flush=True)
            ok = ok and not got["failures"]
            if got["failures"]:
                break
    finally:
        warm.give_back({work / p for p in pkgs} if ok else set())
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

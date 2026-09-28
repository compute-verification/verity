"""`check`: the repository's one check, what a merge into `main` requires (`research merge`).

    uv run python tools/check/check.py              run every step here, in the checkout
    uv run python tools/check/check.py --record     the same through `research run --tool check`: published as an Attempt of the
                                                    commit checked out (a clean tree), the record `research merge` reads
    research run --on MACHINE --project verity --source . --cwd source --tool check -- python tools/check/check.py
                                                    the same on a machine (a CPU pod: about 40 minutes on one core)

Steps, in order; `check` stops at the first that fails:

    pytest          every test under the root `pyproject.toml` testpaths, except the `circuit_suite` ones (the next step runs them)
    circuit-check   `circuit-check --all`: every registered Definition and template (`tools/circuit_check`), its report beside
                    the run's result
    lean-build      `lake build` of the Lean verifier (`backends/flock/verifier/lean`), then
    lean-unit-cut   its partition check against `verity.ir.partition` on random cuts (`unit_cut_agree.py`), then
    lean-audit      `tools/lean/audit.py --all --build`: every Lake package with a `lean-audit.json` built and audited
                    (axioms, sorry, declared axioms, escapes, files, kernel bypasses, pinned statements, a kernel replay),
                    after the controls; its reports under `lean-audit/`, then
    lean-agreement  its agreement with upstream's Rust verifier (`ci-pod.sh`) when the run is sent the input bundle
                    (`--send ci-bundle.tar.gz`); skipped by name without one (`lean_steps`)

Inside a recorded run it writes `$RESEARCH_RUN_DIR/result.json` (kind `check/v1`): each step's status and seconds.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: the Lean verifier of record, and `ci.py`, its agreement job against upstream's Rust verifier
FLOCK_VERIFIER = ROOT / "backends" / "flock" / "verifier"


#: the toolchain `lean-toolchain` pins, through elan when `lake` is missing (as `ci-pod.sh` does)
ELAN = ('command -v lake >/dev/null || command -v "$HOME/.elan/bin/lake" >/dev/null || { curl -sSf '
        'https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null; }; '
        'export PATH="$HOME/.elan/bin:$PATH"; ')


def lean_steps(out: Path) -> list[tuple[str, list[str] | str, Path]]:
    """The Lean steps: nothing on a tree without the verifier (a skip, named); then `lake build`, the Lean partition check's
    conformance with `verity.ir.partition` (`unit_cut_agree.py`), and the agreement job against upstream's Rust verifier
    (`ci-pod.sh`) when the run was sent its input bundle (`--send ci-bundle.tar.gz`, made by `ci-bundle.sh` on a machine with
    store credentials); without one that step is skipped by name, and the cross-check stays the recorded `flock_agreement` job."""
    lean = FLOCK_VERIFIER / "lean"
    if not (lean / "lakefile.toml").exists():
        return [("lean", "skipped: backends/flock/verifier/lean is not on this tree", ROOT)]
    verify = lean / ".lake" / "build" / "bin" / "flock-verify"
    bundles = sorted((out / "inputs").glob("ci-bundle*.tar.gz"))
    agreement: list[str] | str = "skipped: no upstream bundle sent with this run (--send ci-bundle.tar.gz); the cross-check is `flock_agreement`"
    if bundles:
        work = out / "flock-agreement"
        agreement = ["bash", "-c", f'mkdir -p "{work}/inputs" && ln -sf "{bundles[0]}" "{work}/inputs/" && '
                     f'RESEARCH_RUN_DIR="{work}" bash "{FLOCK_VERIFIER / "ci-pod.sh"}"']
    return [("lean-build", ["bash", "-c", ELAN + "lake build"], lean),
            ("lean-unit-cut", [sys.executable, str(FLOCK_VERIFIER / "unit_cut_agree.py"), "--lean", str(verify),
                               "--out", str(out / "lean-unit-cut")], ROOT),
            ("lean-audit", [sys.executable, str(ROOT / "tools" / "lean" / "audit.py"), "--all", "--build", "--root", str(ROOT),
                            "--out", str(out / "lean-audit")], ROOT),
            ("lean-agreement", agreement, ROOT)]


def steps(out: Path) -> list[tuple[str, list[str] | str, Path]]:
    """(name, argv or the reason it is skipped / cannot run, cwd), in order."""
    py = sys.executable
    return [("pytest", [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not circuit_suite"], ROOT),
            ("circuit-check", [py, "-m", "circuit_check", "--all", "-q", "--out", str(out / "circuit-check.json")], ROOT),
            *lean_steps(out)]


def env() -> dict[str, str]:
    """The steps' environment, as a plain shell's: none of what the recording run gives its workload (`RESEARCH_*`, the
    machine's PYTHONPYCACHEPREFIX; the tests start runs of their own, which would inherit them), PYTHONPATH entries inside this
    tree only (a machine's runner puts its shipped `research` first), and a git identity when none is set."""
    out = {k: v for k, v in os.environ.items() if not k.startswith("RESEARCH_") and k != "PYTHONPYCACHEPREFIX"}
    keep = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p and Path(p).resolve().is_relative_to(ROOT)]
    if keep:
        out["PYTHONPATH"] = os.pathsep.join(keep)
    else:
        out.pop("PYTHONPATH", None)
    for k in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):          # tests commit in throwaway repos; a fresh machine has no identity
        out.setdefault(k, "check")
    for k in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        out.setdefault(k, "check@localhost")
    return out


def run(out: Path) -> tuple[bool, list[dict]]:
    done: list[dict] = []
    for name, argv, cwd in steps(out):
        if isinstance(argv, str):
            skipped = argv.startswith("skipped:")
            done.append({"step": name, "status": "skipped" if skipped else "failed", "detail": argv, "seconds": 0.0})
            print(f"check: {name}: {argv}", flush=True)
            if skipped:
                continue
            return False, done
        print(f"check: {name}: {' '.join(argv)}", flush=True)
        t0 = time.monotonic()
        rc = subprocess.run(argv, cwd=cwd, env=env()).returncode
        done.append({"step": name, "status": "passed" if rc == 0 else "failed", "rc": rc, "seconds": round(time.monotonic() - t0, 1)})
        print(f"check: {name}: {'passed' if rc == 0 else f'failed (exit {rc})'} in {done[-1]['seconds']} s", flush=True)
        if rc:
            return False, done
    return True, done


def write_result(run_dir: Path, ok: bool, done: list[dict]) -> None:
    doc = {"schema": "research/result/v0.1", "kind": "check/v1", "run_id": os.environ["RESEARCH_RUN_ID"],
           "workload_fingerprint": {"steps": [s["step"] for s in done]},
           "validation": {"status": "passed" if ok else "failed", "checker": "tools/check/check.py",
                          "detail": "; ".join(f"{s['step']} {s['status']}" for s in done)},
           "measurements": [{"name": f"{s['step']}_wall", "value": s["seconds"], "unit": "s"} for s in done],
           "steps": done, "artifacts": [p for p in ("circuit-check.json", "lean-audit/") if (run_dir / p).exists()]}
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, run_dir / "result.json")


def main(argv: list[str]) -> int:
    if "--record" in argv:
        if os.environ.get("RESEARCH_RUN_ID"):
            print("check: --record inside a recorded run", file=sys.stderr)
            return 2
        cmd = [sys.executable, "-m", "research", "run", "--tool", "check", "--project", "verity", "--cwd", "source", "--",
               sys.executable, "tools/check/check.py"]
        return subprocess.run(cmd, cwd=ROOT).returncode
    if argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    run_dir = Path(os.environ["RESEARCH_RUN_DIR"]) if os.environ.get("RESEARCH_RUN_DIR") else None
    out = run_dir or Path(tempfile.mkdtemp(prefix="check-"))
    ok, done = run(out)
    if run_dir is not None:
        write_result(run_dir, ok, done)
    print(f"check: {'passed' if ok else 'FAILED'} ({', '.join(s['step'] + ' ' + s['status'] for s in done)}); outputs in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

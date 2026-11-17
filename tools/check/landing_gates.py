"""`landing_gates`: the check's landing-gates step, every gate drive.toml lists in `landing_gates` at BASE run on the
candidate's own change.

    python3 tools/check/landing_gates.py BASE OUT

Each gate runs as BASE has it, as `python3 GATE BASE COMMIT --json` (exit 0 passes, 1 fails with `{"failures": [{"why", "fix"}]}`),
where COMMIT is made here: HEAD's tree, with each lock the run's lean-audit regenerated (OUT's `lean-audit/audit.json`) written
in, whose one parent is HEAD's merge base with BASE. So a gate that judges a change against its first parent, its merge base or
its date judges the candidate's whole change from that merge base, whatever merges or dates its commits carry. Exit 0 when every
gate passes, 1 when one fails or can't judge, each named with its lines.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRIVE = "tools/check/drive.toml"
KEY = "landing_gates"
GATE_TIMEOUT_S = 900
FIX_LINES = 8
IDENTITY = {"GIT_AUTHOR_NAME": "check", "GIT_AUTHOR_EMAIL": "check@verity.invalid", "GIT_COMMITTER_NAME": "check",
            "GIT_COMMITTER_EMAIL": "check@verity.invalid"}


def git(*args: str, env: dict | None = None, repo: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True, env=env).stdout.strip()


def gates(base: str, repo: Path = ROOT) -> list[str]:
    r = subprocess.run(["git", "show", f"{base}:{DRIVE}"], cwd=repo, capture_output=True, text=True)
    if r.returncode:
        return []
    got = tomllib.loads(r.stdout).get(KEY) or []
    if not (isinstance(got, list) and all(isinstance(g, str) and g.endswith(".py") and not g.startswith("/")
                                          and ".." not in g.split("/") for g in got)):
        raise ValueError(f"{DRIVE} at {base[:12]}: {KEY} = {got!r}: a list of the gates' paths in the tree")
    return got


def regenerated(out: Path, repo: Path = ROOT) -> dict[str, Path]:
    p = out / "lean-audit" / "audit.json"
    if not p.is_file():
        return {}
    return {f"{pkg}/lean-audit.json": repo / f for pkg, f in (json.loads(p.read_text()).get("regenerated") or {}).items()}


def change(base: str, head: str, locks: dict[str, Path], repo: Path = ROOT) -> tuple[str, str]:
    mb = git("merge-base", base, head, repo=repo)
    with tempfile.TemporaryDirectory(prefix="landing-gates-") as d:
        env = os.environ | IDENTITY | {"GIT_INDEX_FILE": str(Path(d) / "index")}
        git("read-tree", head, env=env, repo=repo)
        for path, f in sorted(locks.items()):
            blob = git("hash-object", "-w", str(f), env=env, repo=repo)
            git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env, repo=repo)
        tree = git("write-tree", env=env, repo=repo)
    msg = f"landing gates: {head[:12]}'s change from {mb[:12]}" + (f", with {len(locks)} regenerated locks" if locks else "")
    return git("commit-tree", tree, "-p", mb, "-m", msg, env=os.environ | IDENTITY, repo=repo), mb


def run_gate(base: str, commit: str, path: str, repo: Path = ROOT) -> tuple[int | None, str]:
    r = subprocess.run(["git", "show", f"{base}:{path}"], cwd=repo, capture_output=True, text=True, errors="surrogateescape")
    if r.returncode:
        return None, f"{path} isn't at {base[:12]}"
    with tempfile.TemporaryDirectory(prefix="landing-gate-", ignore_cleanup_errors=True) as d:
        script = Path(d) / Path(path).name
        script.write_text(r.stdout, errors="surrogateescape")
        try:
            p = subprocess.run([sys.executable, str(script), base, commit, "--json"], cwd=repo, capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=GATE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return None, f"ran over {GATE_TIMEOUT_S} s"
    return p.returncode, p.stdout if p.returncode in (0, 1) else (p.stderr or p.stdout).strip()[-400:]


def judge(base: str, head: str, out: Path, repo: Path = ROOT) -> list[str]:
    commit, mb = change(base, head, regenerated(out, repo), repo)
    said = []
    for path in gates(base, repo):
        rc, text = run_gate(base, commit, path, repo)
        if rc == 0:
            print(f"landing-gates: {path} passes {head[:12]}'s change from {mb[:12]}")
            continue
        failures = []
        if rc == 1:
            try:
                failures = [(str(f["why"]), [str(x) for x in (f.get("fix") or [])][:FIX_LINES]) for f in json.loads(text)["failures"]]
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                text = f"its --json isn't {{\"failures\": [{{\"why\", \"fix\"}}]}} ({type(e).__name__}: {e}): {text}"
        if not failures:
            said.append(f"{path} at {base[:12]} can't judge {head[:12]}'s change from {mb[:12]} "
                        f"({text.strip()[-400:] if rc is None else f'exit {rc}: {text.strip()[-400:]}'})")
        said += [f"{path} at {base[:12]} fails {head[:12]}'s change from {mb[:12]}: {why}" + "".join(f"; {x}" for x in fix)
                 + f" (`python3 {path} origin/main HEAD` runs it)" for why, fix in failures]
    return said


def main(argv: list[str] | None = None) -> int:
    base, out = (argv if argv is not None else sys.argv[1:])[:2]
    said = judge(git("rev-parse", f"{base}^{{commit}}"), git("rev-parse", "HEAD"), Path(out))
    for line in said:
        print(f"landing-gates: {line}")
    return 1 if said else 0


if __name__ == "__main__":
    sys.exit(main())

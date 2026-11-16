"""Every exact check of this directory, one after another.

  uv run python benchmarks/pouw/theory/run.py [--quick] [--mutations] [--verdict PATH] [NAME ...]

A check is a script here that records its claims on exact_kit's registry: one that imports its `Checks` (`checks()`;
NAME ... runs those alone). Each runs in its own process as `NAME.py [--quick] [--mutations]`, with its output passed
through, and counts here as one check: it passes when the script exits 0, and its detail is the script's summary line.
`--quick` reaches the scripts that read it (`takes_quick`); without it `fold_wide32.py` searches for CPU-hours.
`--verdict PATH` writes this run's verdict there and each script's beside it as `NAME.verdict.json`; under `research
run` every verdict goes to `$OUT` without it, and the run's result is this script's. The scripts that drive a C
enumerator need gcc.

On a node it is the command of a `research run`, `uv run python benchmarks/pouw/theory/run.py --quick --mutations`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from exact_kit.checks import Checks

HERE = Path(__file__).resolve().parent
REGISTRY = "from exact_kit.checks import Checks"


def checks() -> list[str]:
    """The scripts here on exact_kit's registry, by name, in order."""
    return sorted(p.stem for p in HERE.glob("*.py")
                  if p.name != Path(__file__).name and REGISTRY in p.read_text(encoding="utf-8"))


def takes_quick(name: str) -> bool:
    """Whether script `name` reads `--quick` from its arguments."""
    return '"--quick" in ' in (HERE / f"{name}.py").read_text(encoding="utf-8")


def command(name: str, quick: bool, mutations: bool, verdict: Path | None) -> list[str]:
    """The process that runs check `name`: this interpreter on its script, with the flags it takes."""
    cmd = [sys.executable, "-B", str(HERE / f"{name}.py")]
    if quick and takes_quick(name):
        cmd.append("--quick")
    if mutations:
        cmd.append("--mutations")
    if verdict is not None:
        cmd += ["--verdict", str(verdict)]
    return cmd


def run_one(cmd: list[str]) -> tuple[int, str]:
    """Run `cmd` with its output passed through; its exit code and the last line it printed that isn't blank."""
    last = ""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            last = line.strip() or last
    return proc.returncode, last


def main(argv: list[str] | None = None) -> int:
    registry = Checks(argv, script=__file__)
    quick = "--quick" in registry.args
    names, known = [a for a in registry.args if a != "--quick"], checks()
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"run.py: no check named {', '.join(unknown)}; the checks are {', '.join(known)}")
    beside = Path(registry.verdict_path).parent if registry.verdict_path else None
    for name in names or known:
        verdict = beside / f"{name}.verdict.json" if beside else None
        code, last = run_one(command(name, quick, registry.mutations, verdict))
        registry.check(f"{name}.py", code == 0, last if code == 0 else f"exit {code}: {last}")
    return registry.finish()


if __name__ == "__main__":
    sys.exit(main())

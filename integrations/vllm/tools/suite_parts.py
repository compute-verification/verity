"""verity-vllm's tests in N parts, each a `suites.py verity-vllm` run of its own, so N jobs run the suite at once (VLLM-112).

    python3 integrations/vllm/tools/suite_parts.py 2/6             part 2 of 6: its test modules, from integrations/vllm
    python3 integrations/vllm/tools/suite_parts.py --list 6        each part's modules and estimated seconds
    python3 integrations/vllm/tools/suite_parts.py launch 6        the 6 parts as `research run --on auto` jobs; their run ids
    python3 integrations/vllm/tools/suite_parts.py launch 6 -- --fresh     with suites.py options (--fresh, --quick)

The suite is CPU-bound at about 7000 CPU-s (r20261025-164354-e533: 12926 tests, 6931 s of test time, its 5 xdist workers'
7194 CPU-s), so one 4-CPU job can't run it in under about 29 min, and `tests/program` alone, one flat directory, is half of
it. A part is a list of test modules handed to pytest (`suites.py verity-vllm -- <modules>`), so the suite, its inputs, its
guard and its configuration stay as they are. suites.py keys every pass on the pytest arguments (the suite's, each module's
and the whole run's traced pass), so a part's pass answers only for that part, never for the suite or for another part.

The parts are a function of the tree alone, so N jobs on N machines agree on them: every test module pytest collects under
the suite's `testpaths` (`modules`) goes to exactly one part, longest first, each to the part with the least so far
(`parts`). A module's estimate is its seconds in `SECONDS` (measured; a module renamed since gets `DEFAULT_S`, which only
unbalances the parts), so a stale table costs balance, never a test.

`launch` starts each part as `research run --on auto` does for any suite run, with `--jobs` the CPUs the job holds:
suites.py's default counts each process at its `TEST_PROCESS_GB` of 9 GB, which runs the suite in one process in a 16 GB job.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
ROOT = SUITE.parents[1]
TESTPATHS = ("tests",)
#: pytest's default `python_files` and `norecursedirs` (the suite's pyproject.toml sets neither)
PYTHON_FILES = ("test_*.py", "*_test.py")
NORECURSE = ("*.egg", ".*", "_darcs", "build", "CVS", "dist", "node_modules", "venv", "{arch}", "__pycache__")
DEFAULT_PARTS = 6
#: a module SECONDS doesn't name: about the median of those it leaves out
DEFAULT_S = 4.0
#: each test module's seconds of test time in a full run, the modules over 20 s (r20261025-200148-c421: main 272d1a6f74 on
#: 8 workers); `--seconds REPORT` prints the table from a suites.py report (`<logs>/integrations_vllm.json`)
SECONDS: dict[str, float] = {}
#: what each part's job runs, from the repository's root; {k}/{n} names the part and {opts} the suites.py options
JOB = ("uv sync -q --all-packages --extra torch-cpu && uv run tools/verity/check/suites.py verity-vllm --jobs \"$(nproc)\"{opts}"
       " -- $(python3 integrations/vllm/tools/suite_parts.py {k}/{n})")
#: the memory each part's job asks for: 4 CPUs' workers at the suite's worker_gb
JOB_MEM_GB = 32


def _skipped(name: str) -> bool:
    return any(Path(name).match(p) for p in NORECURSE)


def modules(suite: Path = SUITE) -> list[str]:
    """Every test module pytest collects from the suite's `testpaths`, by its path from the suite's directory."""
    out = []
    for top in TESTPATHS:
        stack = [suite / top]
        while stack:
            d = stack.pop()
            for p in d.iterdir():
                if p.is_dir():
                    if not _skipped(p.name):
                        stack.append(p)
                elif any(p.match(g) for g in PYTHON_FILES):
                    out.append(p.relative_to(suite).as_posix())
    return sorted(out)


def parts(names: list[str], n: int, seconds: dict[str, float] = SECONDS) -> list[list[str]]:
    """``names`` in ``n`` parts: longest first (by ``seconds``, else `DEFAULT_S`; ties by name), each to the part with the fewest
    seconds so far (the lowest-numbered of those tied); each part's names sorted."""
    if n < 1:
        raise ValueError(f"{n} parts: there must be at least one")
    load, out = [0.0] * n, [[] for _ in range(n)]
    for m in sorted(names, key=lambda m: (-seconds.get(m, DEFAULT_S), m)):
        k = min(range(n), key=lambda i: (load[i], i))
        load[k] += seconds.get(m, DEFAULT_S)
        out[k].append(m)
    return [sorted(p) for p in out]


def part(spec: str, names: list[str] | None = None) -> list[str]:
    """Part ``spec`` (``K/N``, K from 1) of `modules`."""
    m = re.fullmatch(r"(\d+)/(\d+)", spec)
    if not m or not 1 <= int(m[1]) <= int(m[2]):
        raise ValueError(f"{spec!r}: a part is K/N with 1 <= K <= N")
    return parts(modules() if names is None else names, int(m[2]))[int(m[1]) - 1]


def table(report: dict, floor_s: float = 20.0) -> dict[str, float]:
    """{module: seconds} from a suites.py report's per-test seconds (``passed``), the modules of at least ``floor_s``."""
    total: dict[str, float] = {}
    for node, s in report.get("passed", {}).items():
        total[node.split("::", 1)[0]] = total.get(node.split("::", 1)[0], 0.0) + s
    return {m: round(s) for m, s in sorted(total.items(), key=lambda x: (-x[1], x[0])) if s >= floor_s}


def jobs(n: int, opts: list[str] = ()) -> list[list[str]]:
    """The `research run` command of each of the ``n`` parts, with suites.py's ``opts``."""
    o = "".join(f" {x}" for x in opts)
    return [["research", "run", "--on", "auto", "--mem-gb", str(JOB_MEM_GB), "--cwd", "source", "--",
             "bash", "-lc", JOB.format(k=k, n=n, opts=o)] for k in range(1, n + 1)]


def launch(n: int, opts: list[str]) -> int:
    """Start the ``n`` parts at once, from the repository's root, and print each one's run id (or why it didn't start)."""
    procs = [subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for cmd in jobs(n, opts)]
    rc = 0
    for k, p in enumerate(procs, 1):
        out = p.communicate()[0]
        run = re.search(r"run (r\d{8}-\d{6}-[0-9a-f]+) launched", out)
        print(f"part {k}/{n}: {run[1] if run else 'not started: ' + out.strip()[-500:]}")
        rc = rc or (0 if run and p.returncode == 0 else 1)
    if rc == 0:
        print("each run's suites.py summary: `research fetch RUN` and its stdout.log; the parts add up to the suite")
    return rc


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    opts = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(prog="suite_parts", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="?", help="K/N (part K of N's test modules), or `launch`")
    ap.add_argument("n", nargs="?", type=int, default=DEFAULT_PARTS, help=f"launch: the number of parts (default {DEFAULT_PARTS})")
    ap.add_argument("--list", type=int, metavar="N", help="each of N parts' modules and estimated seconds")
    ap.add_argument("--seconds", type=Path, metavar="REPORT", help="SECONDS from a suites.py report of a full run")
    a = ap.parse_args(argv)
    if a.seconds:
        import json
        for m, s in table(json.loads(a.seconds.read_text())).items():
            print(f"    {m!r}: {s},")
        return 0
    if a.list:
        for k, p in enumerate(parts(modules(), a.list), 1):
            print(f"part {k}/{a.list}: {len(p)} modules, about {sum(SECONDS.get(m, DEFAULT_S) for m in p):.0f} s")
        return 0
    if a.what == "launch":
        return launch(a.n, opts)
    if a.what is None:
        ap.error("name a part (K/N), `launch` or --list N")
    try:
        print("\n".join(part(a.what)))
    except ValueError as e:
        ap.error(str(e))
    return 0


if __name__ == "__main__":
    sys.exit(main())

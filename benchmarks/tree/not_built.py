"""What the architecture requires that `main` doesn't do yet, as a number: the rows of `ARCHITECTURE.md`'s table of code
that has to be written (its header `| What | Today |`), each a thing the architecture states beside what `main` does
instead, measured as `top/architecture-not-built-rows`. Lower is better; the architecture is built at zero.

    research run --on auto --cwd source -- uv run python benchmarks/tree/not_built.py

It reads the one file of the checkout it runs in and writes under `--out` (`out/` outside a research run).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from verity_bench import kit

DOC = Path(__file__).resolve().parents[2] / "ARCHITECTURE.md"
HEADER = ("What", "Today")


def _cells(line: str) -> tuple[str, ...]:
    return tuple(c.strip() for c in line.strip().strip("|").split("|"))


def rows(text: str) -> int:
    """The data rows of the one table whose header is `HEADER`; it refuses a document with none or with two."""
    lines = text.splitlines()
    at = [i for i, s in enumerate(lines) if s.lstrip().startswith("|") and _cells(s) == HEADER]
    if len(at) != 1:
        raise SystemExit(f"ARCHITECTURE.md holds {len(at)} tables headed `| {' | '.join(HEADER)} |`, not one: the "
                         "table of code that has to be written is what this counts; restore its header or update HEADER")
    body = []
    for s in lines[at[0] + 2:]:
        if not s.lstrip().startswith("|"):
            break
        body.append(s)
    return len(body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out", help="where the result goes outside a research run")
    out = ap.parse_args().out
    n = rows(DOC.read_text(encoding="utf-8"))
    w = kit.writer(out)
    w.measurement("top/architecture-not-built-rows", n, "rows", target=0, mark="measured")
    w.validation("passed", "benchmarks/tree/not_built.py",
                 f"read {n} rows of ARCHITECTURE.md's table headed `| {' | '.join(HEADER)} |`")


if __name__ == "__main__":
    main()

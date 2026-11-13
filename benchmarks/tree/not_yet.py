"""The layout's not-yet list, as a number: the lines of `tests/tree.toml`'s [[not_yet]], each a path `main` holds outside
the layout of record, measured as `top/tree-not-yet-lines`, with the unowned lines among them as
`top/tree-unowned-lines`. Lower is better; the tree holds to its layout at zero.

    research run --on auto --cwd source -- uv run python benchmarks/tree/not_yet.py

It reads the one file of the checkout it runs in and writes under `--out` (`out/` outside a research run).
"""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from verity_bench import kit

LAYOUT = Path(__file__).resolve().parents[2] / "tests" / "tree.toml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out", help="where the result goes outside a research run")
    out = ap.parse_args().out
    lines = tomllib.loads(LAYOUT.read_text(encoding="utf-8")).get("not_yet", [])
    w = kit.writer(out)
    w.measurement("top/tree-not-yet-lines", len(lines), "lines", target=0)
    w.measurement("top/tree-unowned-lines", sum(r.get("owner") == "unowned" for r in lines), "lines", target=0)
    w.validation("passed", "benchmarks/tree/not_yet.py", f"read {len(lines)} [[not_yet]] lines of tests/tree.toml")


if __name__ == "__main__":
    main()

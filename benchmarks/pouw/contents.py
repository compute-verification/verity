"""What is in benchmarks/pouw, as the Markdown tables the README points to, read from `contents/`:

    uv run python benchmarks/pouw/contents.py [paths|filler|kernels]...

`contents/TABLE/NAME.toml` is one entry: `order` (entries sort by it, then by NAME) and the table's columns. A new
directory, filler tool or GEMM kernel adds its own file and edits no other; the README holds none of these tables."""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTENTS = HERE / "contents"
TABLES = {"paths": ("What is here", {"path": "Path", "runs_on": "Runs on", "what": "What"}),
          "filler": ("Where the filler's tools live", {"tool": "Tool", "files": "Files"}),
          "kernels": ("Where the GEMM kernels and the committer live",
                      {"tool": "Tool", "files": "Files", "includes": "Includes"})}


def entries(name: str) -> list[dict]:
    cols = TABLES[name][1]
    out = []
    for p in sorted((CONTENTS / name).glob("*.toml")):
        t = tomllib.loads(p.read_text())
        if set(t) != {"order", *cols}:
            raise SystemExit(f"{p.relative_to(HERE)}: its keys are order, {', '.join(cols)}; it has {', '.join(t)}")
        out.append((t["order"], p.stem, t))
    return [t for _, _, t in sorted(out, key=lambda e: e[:2])]


def table(name: str) -> str:
    title, cols = TABLES[name]
    body = ["| " + " | ".join(e[k].replace("|", "\\|") for k in cols) + " |" for e in entries(name)]
    return f"## {title}\n\n" + "\n".join(["| " + " | ".join(cols.values()) + " |", "|" + "---|" * len(cols), *body])


def main(argv: list[str]) -> int:
    if bad := [n for n in argv if n not in TABLES]:
        raise SystemExit(f"contents: no table {', '.join(bad)}; the tables are {', '.join(TABLES)}")
    print("\n\n".join(table(n) for n in argv or TABLES))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""`python3 tools/verity/lean/claim_ids.py [--write]`: `verity.claims`' table of each claim id's Lean premises (`LEAN`),
generated from the Lean locks.

Each lock's `ids` gives a premise its claim id, the one `verity.claims` cites it by; the Lean check requires one of every
named assumption and every premise `trust` names (`check.id_failures`). This reads every tracked lock, the auditor's
fixtures aside (`FIXTURES`, `check.FIXTURES`), and writes `LEAN` in `verity/claims/__init__.py`: {claim id: the premises the locks give
it, sorted}, ids sorted. Without `--write` it only compares, and exits 1 naming the fix where the file differs.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CLAIMS = ROOT / "verity" / "claims" / "__init__.py"
FIXTURES = "tools/verity/lean/tests/fixtures/"
#: the table's lines in `CLAIMS`: its assignment through the closing brace at the start of a line
BLOCK = re.compile(r"^LEAN: dict\[str, tuple\[str, \.\.\.\]\] = \{\n.*?^\}\n", re.S | re.M)


def locks(root: Path = ROOT) -> list[str]:
    files = subprocess.run(["git", "-C", str(root), "ls-files", "*lean-audit.json"], check=True, capture_output=True,
                           text=True).stdout.split()
    return sorted(f for f in files if f.endswith("lean-audit.json") and not f.startswith(FIXTURES))


def table(root: Path = ROOT) -> dict[str, tuple[str, ...]]:
    """{claim id: its premises}, over every lock's `ids`."""
    out: dict[str, set[str]] = {}
    for f in locks(root):
        for premise, claim in (json.loads((root / f).read_text()).get("ids") or {}).items():
            out.setdefault(claim, set()).add(premise)
    return {claim: tuple(sorted(ps)) for claim, ps in sorted(out.items())}


def block(t: dict[str, tuple[str, ...]]) -> str:
    lines = ["LEAN: dict[str, tuple[str, ...]] = {"]
    for claim, ps in t.items():
        one = f"    {json.dumps(claim)}: ({', '.join(json.dumps(p) for p in ps)}{',' * (len(ps) == 1)}),"
        if len(one) <= 120:
            lines.append(one)
        else:
            lines += [f"    {json.dumps(claim)}: (", *(f"        {json.dumps(p)}," for p in ps), "    ),"]
    return "\n".join([*lines, "}"]) + "\n"


def written(text: str, t: dict[str, tuple[str, ...]]) -> str:
    """``text`` (`CLAIMS`') with its table replaced by ``t``'s; `ValueError` if it holds no table."""
    if not BLOCK.search(text):
        raise ValueError(f"{CLAIMS.relative_to(ROOT)} holds no `LEAN: dict[str, tuple[str, ...]] = {{` ... `}}` table")
    return BLOCK.sub(lambda _: block(t), text, count=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true", help="write the table rather than only compare")
    args = ap.parse_args(argv)
    text = CLAIMS.read_text()
    new = written(text, table())
    if new == text:
        return 0
    if args.write:
        CLAIMS.write_text(new)
        print(f"claim-ids: wrote {CLAIMS.relative_to(ROOT)}'s LEAN")
        return 0
    print(f"claim-ids: {CLAIMS.relative_to(ROOT)}'s LEAN differs from the locks' `ids`: run "
          "`python3 tools/verity/lean/claim_ids.py --write` and commit it", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

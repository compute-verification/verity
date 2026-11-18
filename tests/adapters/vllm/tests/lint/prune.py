"""Each lint's allowlist here pruned to what its scan finds in this tree (`_ratchet.pruned`): an entry lowered to what is found
of its key, one found nowhere deleted, nothing added or raised.  It reads what each lint's stale test reads, P10's sizes as
that lint does (a size stays while its key is found), and leaves mypy's list to its own ``--record``.

    uv run python tests/adapters/vllm/tests/lint/prune.py [NAME...]   # every list, or those named; from the repository root
"""
from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent))

from tests.lint import _ratchet as R
from tests.lint import merge_allowlist as MA


def found(m) -> tuple[Counter, bool] | None:
    """What lint module ``m``'s stale test holds its allowlist to, and whether its counts are sizes; None: it has none."""
    if hasattr(m, "sizes"):
        return Counter(dict.fromkeys(m.sizes(), 1)), True
    if (test := getattr(m, "test_allowlist_has_no_stale_entries", None)) is None:
        return None
    seen: list = []
    stale, R.assert_no_stale = R.assert_no_stale, lambda name, violations, rule: seen.append(Counter(v.key for v in violations))
    try:
        test()
    finally:
        R.assert_no_stale = stale
    return seen[0], False


def main(names: list[str]) -> int:
    done = []
    for p in sorted(R.HERE.glob("test_*.py")):
        m = importlib.import_module(f"tests.lint.{p.stem}")
        name = getattr(m, "NAME", None)
        if not isinstance(name, str) or names and name not in names or (got := found(m)) is None:
            continue
        path = R.ALLOWLISTS / f"{name}.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        after = R.pruned(doc, *got)
        if after != doc:
            path.write_text(MA.dump(after), encoding="utf-8")
        done.append(name)
        print(f"allowlists/{name}.json: {len(doc['entries'])} entries, {len(after['entries'])} kept")
    if missing := sorted(set(names) - set(done)):
        print(f"no lint here prunes {', '.join(missing)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

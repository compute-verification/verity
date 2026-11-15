"""Write the generated tables of the import order's listing, `tests/import_order.toml`, from the tracked tree.

The listing's head is kept by hand: the order of the tree's directories, the distributions under `verity/`, and each
place's `paths` and `trusted` mark.  Its tables are the tree's own, and this writes them: `against`, the ties against
the order by file, and each place's `loads`, with a trusted place's `reaches` and `loaded_by`.
`tests/test_boundaries.py` holds the tree to the listing and the listing to what this writes.

To let a place of `verity` import another, write the import and run this: it adds the line to the place's `loads`, and
the listing's diff shows what the change opens.  The lists that only shrink gain nothing by default: a tie against the
order and a line of a trusted place (its `loads`, `reaches` and `loaded_by`) are printed, not written, and the exit is
1.  `--grow` writes those lines too, for a change that is meant to put more on the soundness path and says so in its
pull request.  `--check` writes nothing and exits 1 when the listing differs from what would be written.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def _checks(root: Path) -> Any:
    """`tests/test_boundaries.py` of ``root``, which reads the tree's loads and holds the listing to them."""
    spec = importlib.util.spec_from_file_location("import_order_checks", root / "tests" / "test_boundaries.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tables(checks: Any, text: str, loads: Any, grow: bool = False) -> tuple[str, list[str]]:
    """The listing ``text`` with its tables written from ``loads``, and the lines held back: what the tree adds to a
    list that only shrinks, which is written only with ``grow``."""
    listing = checks._listing(text)
    held = [] if grow else [
        f"against {a} -> {b}: {f}" for (a, b), files in sorted(checks._against(loads, listing.steps).items())
        for f in sorted(files) if f not in listing.against.get((a, b), ())]
    held += [] if grow else [
        f"{place}: {kind} {line}" for place, lists in checks._place_loads(loads, listing).items()
        if checks._trusted_place(listing.places[place])
        for kind, lines in zip(checks.LISTS, lists) for line in sorted(lines) if line not in listing.places[place][kind]]
    return checks._generated(text, loads, grow=grow), held


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="import_order", description=__doc__.split("\n\n")[0])
    p.add_argument("--check", action="store_true", help="write nothing; exit 1 when the listing differs from what would be written")
    p.add_argument("--grow", action="store_true",
                   help="also write what the tree adds to the lists that only shrink: `against` and a trusted place's lists")
    args = p.parse_args(argv)
    checks = _checks(ROOT)
    text = checks.ORDER.read_text(encoding="utf-8")
    new, held = tables(checks, text, checks._tree_loads()[1], grow=args.grow)
    if held:
        print("lines of the lists that only shrink, not written without --grow (cut the tie; for a trusted place, state "
              "it in Lean or take the value as an argument):")
        print("\n".join(held))
    if new != text and not args.check:
        checks.ORDER.write_text(new, encoding="utf-8")
    print(f"{checks.ORDER.relative_to(ROOT)}: {'the same' if new == text else 'differs' if args.check else 'written'}")
    return 1 if held or (args.check and new != text) else 0


if __name__ == "__main__":
    sys.exit(main())

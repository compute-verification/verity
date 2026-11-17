"""`claim_ids`: the table from each claim id to the Lean names it names, written from the locks.

A claim id names a security claim that code and documents cite: a named assumption (`cr/sha-512`, `trusted-device`), a
model (`live-verifier`) or a guarantee (`designated-verifier`).  It is kebab-case or `<property>/<instance>`
(`check.CLAIM_ID`), and once cited it is never renamed.  A premise entry of a lock (`assumptions`, `domain` or `open` of a
`lean-audit.json`) carries the id that names it as `id`, which `check.py --update` keeps.  The table (`TABLE`, which
`verity.commitments.claim_ids` reads) holds each id a lock carries as ``{"kind": "assumption", "lean": [its Lean
names]}``, and each id no lock carries, which only a reader in code makes a row, as ``{"kind": ..., "lean": [], "name":
...}``, written by hand and kept by the writer.  A Lean name that carries two ids is refused.

    uv run tools/verity/lean/claim_ids.py            exit 1 naming each row the locks would change
    uv run tools/verity/lean/claim_ids.py --write    rewrite the table from the locks
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("lean_check_for_claim_ids", HERE / "check.py")
C = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = C
_spec.loader.exec_module(C)

ROOT = C.ROOT
TABLE = ROOT / "verity" / "commitments" / "claim_ids.json"
SECTIONS = ("assumptions", "domain", "open")
KINDS = ("assumption", "model", "guarantee")
WRITE = "uv run tools/verity/lean/claim_ids.py --write"


def carried(root: Path = ROOT) -> tuple[dict[str, list[str]], list[str]]:
    """{id: the Lean names whose premise entries carry it} over every lock under ``root``, and why the ids can't stand: one
    that is no claim id, or a Lean name carrying two."""
    ids: dict[str, set[str]] = {}
    of: dict[str, tuple[str, str]] = {}
    fails = []
    for pkg in C.packages(root):
        if not (lock := pkg / C.POLICY).is_file():
            continue
        where = lock.relative_to(root).as_posix()
        policy = json.loads(lock.read_text(encoding="utf-8"))
        for s in SECTIONS:
            for name, rec in (policy.get(s) or {}).items():
                if not isinstance(rec, dict) or (cid := rec.get("id")) is None:
                    continue
                if not (isinstance(cid, str) and C.CLAIM_ID.fullmatch(cid)):
                    fails.append(f"{where}: {s}[{name}]'s `id` {cid!r} is no claim id")
                    continue
                if (seen := of.setdefault(name, (cid, where)))[0] != cid:
                    fails.append(f"{name} carries `id` {seen[0]!r} in {seen[1]} and {cid!r} in {where}: one Lean name, one id")
                ids.setdefault(cid, set()).add(name)
    return {i: sorted(n) for i, n in sorted(ids.items())}, fails


def declared_failures(rows: dict) -> list[str]:
    """Why a row no lock carries (``"lean": []``) can't stand: it gives its `kind` and its `name`, and nothing else."""
    out = []
    for cid, r in rows.items():
        if isinstance(r, dict) and r.get("lean"):
            continue
        if not (isinstance(r, dict) and set(r) == {"kind", "lean", "name"} and r["lean"] == [] and r["kind"] in KINDS
                and isinstance(r["name"], str) and r["name"].strip() and C.CLAIM_ID.fullmatch(cid)):
            out.append(f"{cid}: a row no lock carries is {{\"kind\": one of {', '.join(KINDS)}, \"lean\": [], \"name\": "
                       f"what the id means}}, its id kebab-case or <property>/<instance>")
    return out


def table(old: dict, ids: dict[str, list[str]]) -> dict:
    """The table the locks give: each id they carry with its Lean names, and ``old``'s rows no lock carries."""
    rows = {i: r for i, r in old.items() if not (isinstance(r, dict) and r.get("lean")) and i not in ids}
    rows |= {i: {"kind": "assumption", "lean": n} for i, n in ids.items()}
    return dict(sorted(rows.items()))


def text(rows: dict) -> str:
    """The table as `TABLE` holds it: one row a line."""
    return "{\n" + ",\n".join(f" {json.dumps(i)}: {json.dumps(r, ensure_ascii=False)}" for i, r in rows.items()) + "\n}\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true", help="rewrite the table from the locks")
    args = ap.parse_args(argv)
    ids, fails = carried()
    old = json.loads(TABLE.read_text(encoding="utf-8"))
    fails += declared_failures(old)
    if fails:
        print("\n".join(fails), file=sys.stderr)
        return 1
    new = table(old, ids)
    if args.write:
        TABLE.write_text(text(new), encoding="utf-8")
        return 0
    if text(new) == TABLE.read_text(encoding="utf-8"):
        return 0
    changed = sorted(i for i in old.keys() | new.keys() if old.get(i) != new.get(i))
    print(f"{TABLE.relative_to(ROOT)}: the locks change {', '.join(changed) or 'its layout'}: run `{WRITE}`", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

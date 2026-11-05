"""Apply LEAN-186's one-time seed (`Security/lean-186-seed.json`) to the locks before the one regeneration.

    python3 tools/verity/lean/apply_seed.py PACKAGE_DIR...

For each package's `lean-audit.json` it moves each seeded `domain` entry the lock lists (under `assumptions` or `open`)
to `domain` with the seed's owner and `why`, adds one whose `lock` names the package (a premise no lock lists yet), and
writes an `open_seed`: the seed's issue and due date under every owner the lock names and under each seeded module
prefix with its owner. `check.py --update` then records the domain entries' digests,
moves a domain entry whose guarantee's honest witnesses don't conclude it to `open` under its owner's ticket, and consumes
`open_seed`. The regeneration's commit deletes this file and the seed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from check import text  # noqa: E402

SEED = Path(__file__).parents[3] / "Security" / "lean-186-seed.json"


def apply(policy: dict, seed: dict, pkg: str) -> list[str]:
    """Seed ``policy``, package ``pkg``'s lock, in place; the domain entries it moved or added."""
    ticket = {"issue": seed["issue"], "due": seed["due"]}
    moved = []
    for name, entry in sorted(seed["domain"].items()):
        if entry.get("lock") == pkg or any(name in (policy.get(s) or {}) for s in ("assumptions", "open")):
            for s in ("assumptions", "open"):
                (policy.get(s) or {}).pop(name, None)
            policy.setdefault("domain", {})[name] = {"owner": entry["owner"], "why": entry["why"]}
            moved.append(name)
    owners = {r.get("owner") for s in ("guarantees", "assumptions", "open", "domain")
              for r in (policy.get(s) or {}).values() if isinstance(r, dict)} - {None, ""}
    policy["open_seed"] = {o: dict(ticket) for o in sorted(owners)} | {
        m: ticket | {"owner": o} for m, o in sorted(seed["modules"].items())}
    return moved


def main(argv: list[str]) -> int:
    seed = json.loads(SEED.read_text())
    for pkg in argv:
        lock = Path(pkg) / "lean-audit.json"
        policy = json.loads(lock.read_text())
        moved = apply(policy, seed, Path(pkg).as_posix().rstrip("/"))
        lock.write_text(text(policy))
        print(f"{pkg}: {len(moved)} domain entries seeded, open_seed under {len(policy['open_seed'])} keys")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

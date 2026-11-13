#!/usr/bin/env python3
"""remake_list.py [--write] [OLD_REV]: the superseded-versions list made again from the registries alone, held to the tree's.

Run at the root of a checkout, in the workspace's environment.  It loads the registries the way the test does (its
`registered` and `superseded`), writes the list those give with nothing read from the list in the tree (every superseded
version an entry at the file that defines it, no `held` line), and compares:

- with the list in the tree: the versions only one of the two holds, and the versions whose `file` differs;
- with the list in the tree after `pruned` (what `--prune` writes): the two must be the same bytes, or the made-again list
  holds a version the carried one never had;
- with OLD_REV's list (a commit), when given: the versions that entered and left since, and those whose `file` moved.

It prints one JSON document.  `--write` puts the made-again list in the tree.
"""
import collections
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "tools/verity/circuit_check/tests")
import test_superseded_versions as t  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ids = t.registered()
now = t.superseded(ids)
made = t.text({i: now[i][0] for i in now}, {})
entries, held = t.listed()
kept = t.text(*t.pruned(now, entries, held))
found = t.findings(now, entries, held)


def files(e: dict[str, str]) -> int:
    return len(set(e.values()))


out = {
    "head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
    "registered": len(ids),
    "boolean_versions": sum(1 for _, b in ids.values() if b),
    "superseded": len(now),
    "superseded_files": len({f for f, _ in now.values()}),
    "tree_list": {"entries": len(entries), "files": files(entries), "held": len(held)},
    "tree_list_against_the_registries": found,
    "made_again": {"entries": len(now), "files": len({f for f, _ in now.values()})},
    "made_again_is_the_tree_list_pruned": made == kept,
    "made_again_is_the_tree_list": made == t.LIST.read_text(encoding="utf-8"),
    "by_place": dict(collections.Counter("/".join(f.split("/")[:2]) for f, _ in now.values()).most_common()),
}
if args:
    old, _ = t.parse(json.loads(subprocess.run(["git", "show", f"{args[0]}:tests/lint/allowlists/superseded_versions.json"],
                                                 capture_output=True, text=True, check=True).stdout))
    new = {i: now[i][0] for i in now}
    out["against_old"] = {
        "old_rev": args[0], "old_entries": len(old), "old_files": files(old),
        "entered": {i: new[i] for i in sorted(set(new) - set(old))},
        "left": {i: old[i] for i in sorted(set(old) - set(new))},
        "file_moved": len([i for i in old if i in new and old[i] != new[i]]),
        "file_same": len([i for i in old if i in new and old[i] == new[i]]),
        "moves": dict(collections.Counter(
            f"{'/'.join(old[i].split('/')[:3])} -> {'/'.join(new[i].split('/')[:4])}" for i in old if i in new and old[i] != new[i])),
    }
if "--write" in sys.argv:
    t.LIST.write_text(made, encoding="utf-8")
    out["written"] = str(t.LIST.relative_to(t.REPO))
print(json.dumps(out, indent=1))

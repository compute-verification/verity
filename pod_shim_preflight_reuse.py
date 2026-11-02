"""A member's preflight pass stands across a rebuild of `next`, on drive's `research.queue` as `main` has it, until `main` holds
`Queue._standing`.

`apply(research.queue)` before the tick builds its `Queue`.  The rule is `Queue._standing`'s: a head's newest record in the
sync's `flights` is a pass onto another base tree, and it stands for the head's merge onto the new base when (a) the head's
change is the same onto both bases (`git diff-tree -r -z --no-renames` from base to merge, modes and blobs on both sides), (b)
between the two bases no path of `RULES` changed and no path was deleted, and (c) the change deletes no path.  The sync's summary
line gains ", preflights' verdicts reused for N" where `main`'s own code puts it.  `apply` refuses a module that already has
`_standing` or whose `_preflighted` takes other arguments.
"""
from __future__ import annotations

import inspect
import re

#: what the sync's preflight runs rather than judges (`PREFLIGHT_RULES`); a directory ends in `/`
RULES = ("tests/test_module_strings.py", "tests/test_pins.py", "tests/test_doc_paths.py", "tests/test_repository.py",
         "tools/research/tests/test_pythonpath.py", "tests/conftest.py", "tools/research/tests/conftest.py",
         "tools/research/tests/waits.py", "tools/research/pyproject.toml", "tools/research/src/", "tools/verity/check/",
         "pyproject.toml")
#: `_preflighted`'s parameters on the `main` this patches
SIGNATURE = ["self", "pr", "tip", "commit", "members", "flights"]
#: where `main`'s summary line puts the clause: before the waiting PRs, else before the dry run, else at the end
_AFTER = re.compile(r", \d+ waiting off it for their preflight| \(dry run: not pushed\)")


def apply(mod) -> None:
    Queue = mod.Queue
    if hasattr(Queue, "_standing"):
        raise RuntimeError(f"{mod.__name__}.Queue already has _standing: this shim is obsolete, drop it")
    if (got := list(inspect.signature(Queue._preflighted).parameters)) != SIGNATURE:
        raise RuntimeError(f"{mod.__name__}.Queue._preflighted takes {got}, not {SIGNATURE}: this shim doesn't fit it")
    preflighted, sync = Queue._preflighted, Queue.sync

    def _standing(self, head: str, base_tree: str, tree: str, flights: dict) -> bool:
        old = next(((k, rec) for fl in (flights["now"], flights["before"]) for k, rec in reversed(fl.items())
                    if rec.get("head") == head), None)
        if old is None or old[1].get("why") is not None or not old[1].get("tree"):
            return False
        old_base, old_tree = old[0].rpartition(":")[2], old[1]["tree"]

        def rows(a: str, b: str) -> list[tuple[str, str]] | None:
            r = mod._git(self.workdir, "diff-tree", "-r", "-z", "--no-renames", a, b, check=False)
            parts = r.stdout.split("\0")
            return None if r.returncode else [(meta, path) for meta, path in zip(parts[::2], parts[1::2])]

        change = rows(old_base, old_tree)
        if change is None or change != rows(base_tree, tree) or (between := rows(old_base, base_tree)) is None:
            return False
        ruled = lambda path: any(path == r or (r.endswith("/") and path.startswith(r)) for r in RULES)  # noqa: E731
        return not any(meta.endswith(" D") for meta, _ in [*change, *between]) and not any(ruled(p) for _, p in between)

    def _preflighted_(self, pr, tip, commit, members, flights):
        base_tree, tree = mod._out(self.workdir, "rev-parse", f"{tip}^{{tree}}", f"{commit}^{{tree}}").split()
        key = f"{pr.head}:{base_tree}"
        if any(got is not None and got.get("tree") == tree for got in (flights["now"].get(key), flights["before"].get(key))):
            self._preflights_reused += 1
        elif _standing(self, pr.head, base_tree, tree, flights):
            flights["now"][key] = {"pr": pr.number, "head": pr.head, "tree": tree, "why": None}
            self._preflights_reused += 1
            return True, None
        return preflighted(self, pr, tip, commit, members, flights)

    def sync_(self, dry_run: bool = False):
        self._preflights_reused, log = 0, self.log

        def said(msg: str) -> None:
            if self._preflights_reused and msg.startswith(f"research queue: `{self.branch}` at ") and " PR(s) above " in msg:
                at = m.start() if (m := _AFTER.search(msg)) else len(msg)
                msg = f"{msg[:at]}, preflights' verdicts reused for {self._preflights_reused}{msg[at:]}"
            log(msg)

        self.log = said
        try:
            return sync(self, dry_run)
        finally:
            self.log = log

    Queue._preflights_reused = 0
    Queue._standing, Queue._preflighted, Queue.sync = _standing, _preflighted_, sync_

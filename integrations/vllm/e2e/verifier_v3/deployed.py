"""[E9, SCALE 2026-09-12 -- order 3] Deployed-mode openings for ``reverify.py`` (``--deployed-openings``, default OFF; the archive check
``V3.trees_rehashed`` stays the default for archive verification).

The question this answers (the user's, verbatim): the archive check "recomputes roots from archived leaves ... we need to distinguish that cost
from the deployed verifier's work authenticating sampled values".  MEASURED 2026-09-12 (out/scale/reports/e9.md): with the instrument on, the
shipped verifier rehashes every tree from its leaf BYTES twice per verification -- once in ``V3.trees_rehashed`` and once more when a deployed
check opens one sampled word through ``TreeCache.tree()`` / ``load_tree(recompute=True)`` (n4: 288 MB of leaf bytes for 468 openings whose
leaves are 1.7 MB).  Deployed mode removes the second walk and replaces the first by an O(N x 32 B) fold, and it is FAIL-CLOSED:

1. every tree the verifier touches is built from the record's STORED per-leaf hashes (RU ``index.json#entries[].leaf``, interior
   ``index.json#leaves[].sha256``, weights ``leaves.bin``): ``TreeCache(recompute_leaves=False)`` / ``load_tree(recompute=False)`` are FORCED
   for every cache the process creates (the Deriver's, ``V6.gather``'s, ``V8.output_copies``', ``V3``'s) -- 32 bytes per leaf are read, never
   the leaf bytes of an unsampled leaf;
2. ``V3.trees_from_leaf_hashes`` (replaces ``V3.trees_rehashed``, which is NOT_APPLICABLE by name in this mode): every committed tree's root
   folded from the stored leaf hashes == the root PINNED in ``commitments.json`` (whose sha256 the chronology pins: ``V3.interior`` /
   ``V3.boundary``), leaf_count and ids_digest too.  An altered leaf hash moves the root -> FAIL.  (Only the roots' derivation changes: the
   same ``steps.trees_rehashed`` runs, its trees now built from hashes.)
3. EVERY opening the verifier takes -- ``TreeCache.open`` (RU / weights / boundary leaves) and ``InteriorTree.open`` / ``open_gate`` /
   ``open_range`` (interior words) -- is authenticated here, in the wrapper, against the PINNED root (not the rebuilt tree's): the leaf hash is
   recomputed from the opened BYTES (id, dtype, shape, value all bind), the path is folded to the root, the root must equal the pin
   (``verity_vllm.commit.verify_opening_in`` for RU / weights / boundary; ``e2e.interior.tree.verify_opening`` for interior trees, whose roots
   live under ``commitments.json#interior_roots``).  A failure is the named refusal ``OPENING_UNAUTHENTICATED`` (a ``ck_elem.Refusal``: the
   Deriver / gather record it by name, the check FAILS, the record is REFUSED -- never a value used unauthenticated).  Interior words read by
   ``elem_builders.interior_leaf_words`` (bytes -> index row's sha256, refusal ``LEAF_HASH_MISMATCH``) are covered by (2): the row's hash is
   one of the N folded into the pinned root.

T1's altered-word attacks (D35 F1) under 1-3: an interior word flipped in ``leaf-*.bin`` with the index untouched -> the opened bytes hash to
another leaf hash -> the path folds to another root != pin -> ``OPENING_UNAUTHENTICATED`` (or ``LEAF_HASH_MISMATCH`` at the row); the index
row's sha256 edited too -> ``V3.trees_from_leaf_hashes`` FAIL (root moved); the root edited too -> ``V3.interior`` FAIL (chronology pin); a
boundary output word flipped -> the same chain on ``boundary/<rid>`` (opened in full by ``V8.step2`` / ``V8.output_copies``); the claimed
output y changed with the root re-derived -> ``V3.boundary`` FAIL.  NOT caught, by design: a word flipped in a leaf that no sampled unit,
input check or output check opens -- that is what sampling means; the archive check exists for that.

What the record format still lacks for a transfer-minimal deployed verifier: stored openings (leaf + 32 B x depth path) for the |Q| sampled
words -- today the paths are derived from ALL N stored leaf hashes (32 N bytes: n4 14.1 MB, s4 21.9 MB, n3 15.5 MB) instead of read (n4:
399 leaves, 1.68 MB leaves + 162 KB paths).  ``verify_accounting.json#deployed_ideal`` carries both numbers.
"""
from __future__ import annotations

import threading
from typing import Any, Mapping

OPENING_UNAUTHENTICATED = "OPENING_UNAUTHENTICATED"
INTERIOR_TREE_PREFIX = "interior/"


class DeployedMode:
    """Installed once per process by :func:`install`; counts what it authenticated."""

    def __init__(self, commitments: Mapping[str, Any]) -> None:
        self.commitments = commitments
        self.lock = threading.Lock()
        self.openings_authenticated = 0
        self.openings_refused = 0
        self.trees_from_hashes = 0
        self.installed = False

    def pinned_root(self, tree_id: str) -> tuple[bytes, int] | None:
        c = self.commitments
        rec = (c.get("trees") or {}).get(tree_id)
        if rec is None:
            key = tree_id[len(INTERIOR_TREE_PREFIX):] if tree_id.startswith(INTERIOR_TREE_PREFIX) else tree_id
            rec = (c.get("interior_roots") or {}).get(key)
        if not rec or "root" not in rec:
            return None
        try:
            return bytes.fromhex(str(rec["root"])), int(rec.get("leaf_count", 0))
        except (TypeError, ValueError):
            return None

    def authenticate(self, opening: Any) -> Any:
        """``opening`` unchanged when it verifies against the PINNED root; the named refusal otherwise."""
        from e2e.coverage_v1.ck_elem import Refusal
        from e2e.interior import tree as itree
        from verity_vllm import commit

        tree_id = str(getattr(opening, "tree", ""))
        pin = self.pinned_root(tree_id)
        ok = False
        if pin is not None:
            root, count = pin
            if tree_id.startswith(INTERIOR_TREE_PREFIX):
                if isinstance(opening, itree.InteriorRangeOpening):
                    ok = itree.verify_range_opening(root, count, opening)
                else:
                    ok = itree.verify_opening(root, count, opening)
            elif isinstance(opening, commit.RangeOpening):
                ok = commit.verify_range_opening(root, count, opening)
            else:
                ok = commit.verify_opening(root, count, opening)
        with self.lock:
            if ok:
                self.openings_authenticated += 1
            else:
                self.openings_refused += 1
        if not ok:
            leaf = getattr(opening, "canonical_id", None) or f"ranks[{getattr(opening, 'lo', '?')},{getattr(opening, 'hi', '?')})"
            why = "tree not pinned in commitments.json" if pin is None else "leaf bytes + path do not fold to the pinned root"
            raise Refusal(f"{OPENING_UNAUTHENTICATED}: {tree_id}: {leaf}: {why}")
        return opening

    def summary(self) -> dict[str, Any]:
        return {"mode": "deployed-openings/v1", "openings_authenticated": self.openings_authenticated, "openings_refused": self.openings_refused,
                "trees_from_leaf_hashes": self.trees_from_hashes, "refusal": OPENING_UNAUTHENTICATED}


_CURRENT: DeployedMode | None = None


def current() -> DeployedMode | None:
    return _CURRENT


def install(commitments: Mapping[str, Any]) -> DeployedMode:
    """Force stored-hash trees and authenticate every opening, process-wide (after ``accounting.install`` so the instrument sees the calls)."""
    global _CURRENT
    if _CURRENT is not None:
        return _CURRENT
    dm = DeployedMode(commitments)
    from e2e.interior import tree as itree
    from verity_vllm import commit

    real_init = commit.TreeCache.__init__

    def init(self: Any, reader: Any, *a: Any, **k: Any) -> None:
        k["recompute_leaves"] = False
        real_init(self, reader, *a, **k)

    real_build = commit.TreeCache._build

    def build(self: Any, tree_id: str):
        t = real_build(self, tree_id)
        with dm.lock:
            dm.trees_from_hashes += 1
        return t

    real_open = commit.TreeCache.open

    def open_(self: Any, canonical_id: str):
        return dm.authenticate(real_open(self, canonical_id))

    commit.TreeCache.__init__ = init  # type: ignore[method-assign]
    commit.TreeCache._build = build  # type: ignore[method-assign]
    commit.TreeCache.open = open_  # type: ignore[method-assign]

    real_load = itree.load_tree

    def load_tree(directory: Any, *a: Any, **k: Any):
        k["recompute"] = False
        t = real_load(directory, *a, **k)
        with dm.lock:
            dm.trees_from_hashes += 1
        return t

    itree.load_tree = load_tree  # type: ignore[assignment]
    real_iopen, real_igate, real_irange = itree.InteriorTree.open, itree.InteriorTree.open_gate, itree.InteriorTree.open_range

    def iopen(self: Any, rank: int):
        return dm.authenticate(real_iopen(self, rank))

    def igate(self: Any, g: int):
        o, flat = real_igate(self, g)
        return dm.authenticate(o), flat

    def irange(self: Any, lo: int, hi: int):
        return dm.authenticate(real_irange(self, lo, hi))

    itree.InteriorTree.open = iopen  # type: ignore[method-assign]
    itree.InteriorTree.open_gate = igate  # type: ignore[method-assign]
    itree.InteriorTree.open_range = irange  # type: ignore[method-assign]
    dm.installed = True
    _CURRENT = dm
    return dm

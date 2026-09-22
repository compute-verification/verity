"""[E9, SCALE wave 2026-09-12 -- the user's question, verbatim: "how much data does verification itself read and hash? The recent fix recomputes
roots from archived leaves. That may be appropriate for checking an archive, but we need to distinguish that cost from the deployed verifier's
work authenticating sampled values. Otherwise the headline 'sparse verification' could conceal substantial whole-record processing."]

The verifier's DATA ACCOUNTING: per check (V1..V8, each named sub-check) and per named section, the bytes READ from the record (files opened,
bytes), the bytes HASHED (every ``hashlib.sha256`` update: Merkle leaf + node bytes, file digests, statement hashes), the leaves / nodes / id lists
hashed, the openings taken, wall and RSS -- written as ``verify_accounting.json`` (schema ``veritor/verify-accounting/v1``) next to the verdict by
``reverify.py`` (and by the live driver when it installs the same instrument).  Nothing here changes a predicate: the instrument observes.

CLASSIFICATION RULE (the one rule, applied to every counted event; ``bucket_map`` in the JSON lists the outcome per row):

* **ARCHIVE** (``archive_check``) = whole-record work -- work whose size is the RECORD's, not the SAMPLE's:
  (a) everything inside ``V3.trees_rehashed`` (every shipped tree rehashed from its leaf bytes to the root; part ``trees_rehashed``);
  (b) every WHOLE-TREE BUILD triggered by any other check -- today an opening of ONE sampled word goes through ``TreeCache.tree()`` /
      ``e2e.interior.tree.load_tree()``, which read every leaf of that tree and (with ``recompute_leaves=True``) rehash every leaf before one
      path can be taken; that walk is archive work by the user's rule ("anything else that walks all leaves") even when a deployed check
      (``V5.*``, ``V6.*``) asked for it (part ``whole_tree_rebuild_for_openings``, attributed to the check that triggered it).  MEASURED
      2026-09-12 (n4): the largest such build is ``expected.committed_trees`` taking the weights tree's id list by ``cache.tree("weights").ids``
      (270 MB of leaf bytes read + hashed for a list that ``weights/index.json`` holds), inside ``V5.coverage_inputs``;
  (c) every ID-LIST WALK -- ``committed_trees`` reads every leaf id of a cited tree (``index.json``) and recomputes ``ids_digest``; it walks all
      leaves' metadata, not their bytes (part ``ids_walk``).  EXCEPTION to (b)/(c): a ``boundary/<rid>`` tree is the request's INPUT/OUTPUT (x, y),
      which the protocol requires the verifier to read in full (v_in(C) = x; y equality); its whole-tree work is DEPLOYED, part ``boundary_io``.
* **DEPLOYED** (``deployed_sparse``) = the work a verifier does authenticating the |Q| sampled units against the pinned roots:
  ``proofs`` (SP1 proof + statement bytes + the typed object, ``V4.proof.*`` / ``V4.family.*`` / the host verify loop), ``statements`` (the
  expected-statement derivation and the position / coverage checks, ``V5.*``, ``V3.interior_layout``, ``V3.literal_table``),
  ``openings`` (path-sized: one leaf's bytes + its sibling path, counted per opening), ``direct_checks`` (``V6.gather``, ``V6.inputs``),
  ``boundary_io`` (the boundary trees, ``V8.step2`` inputs, ``D6.request_commitment``), ``pins`` (instance / chronology / protocol / commitment
  digests, the population file and its derivation, ``V1.*`` / ``V2.*`` / ``V3.boundary`` / ``V3.interior`` / ``V3.weights_root`` / ``V8.*`` /
  ``V7.*`` / ``B.*`` / ``D6.vk_once`` / ``V4.guest`` / ``V4.elf_archived``).
* **CHECKPOINT** (``checkpoint``) = the verifier's OWN ``model.safetensors`` (``V6.checkpoint`` sha256 + the words it reads): the model, not the
  record (E3's rule) -- reported apart from both buckets.
* **UNCLASSIFIED** = any event outside every named check / section (before the first check, after the last, or in a section the map does not
  name).  It must be ~0 on an honest run; a non-zero value is a finding about the instrument, printed as such.

A row's ``whole_tree`` / ``ids_walk`` sub-counters are the parts (b)/(c) it triggered; ``own`` is the rest.  The bucket totals are sums of parts,
so a DEPLOYED check that triggered a whole-tree rebuild contributes its ``own`` to DEPLOYED and its ``whole_tree`` to ARCHIVE -- the split the
user asked for.  ``ratio_bytes_read_to_record`` = bucket bytes read / ``record_bytes`` (sum of file sizes under the record directory the verifier
was pointed at; on a slim copy that is the slim size, and the row says so).

HOW THE COUNTS ARE TAKEN (MEASURED in the verifier process; nothing modeled):
* reads: ``builtins.open`` / ``io.open`` in a read mode return a counting proxy (``read`` / ``readline`` / ``readlines`` / ``readinto`` /
  iteration); the capture reader's ``np.memmap`` slices of ``values.bin`` / ``leaves.bin`` (``RUReader.read``, ``WeightsReader._read_rows_mm`` /
  ``_stored_leaves``) are counted by slice length (a syscall counter cannot see mmap page faults).  Cross-check on Linux: ``/proc/self/io``
  ``rchar`` over the whole run vs the sum of attributed reads (``cross_check.unattributed_read_bytes``).  Python's own imports use
  ``io.open_code`` and are not counted.  SP1 proof files are read by the host SUBPROCESS: counted by file size (``stat``) under ``proofs``.
* hashes: ``hashlib.sha256`` is replaced by a counting constructor (bytes per ``update``); ``verity_vllm.hashing.leaf_hash`` / ``node_hash`` /
  ``lift_hash`` / ``ids_digest`` count leaves / nodes / id lists.  Every Merkle hash of the POC goes through ``hashing.H`` -> ``hashlib.sha256``.
* openings: ``verity_vllm.commit.MerkleTree.open`` / ``.path`` / ``TreeCache.open`` and ``e2e.interior.tree.InteriorTree.open`` -- each records
  (tree, leaf id, leaf bytes, path depth) so ``deployed_ideal`` can state what a verifier with STORED openings would transfer for |Q|.
* whole-tree builds: ``TreeCache._build`` and ``e2e.interior.tree.load_tree`` open a nested ``whole_tree`` section; ``expected.committed_trees``
  an ``ids_walk`` section.
* wall: ``perf_counter`` per row; RSS: ``ru_maxrss`` (high-water) after each row and the current RSS (``/proc/self/statm`` on Linux).
"""
from __future__ import annotations

import builtins
import hashlib
import io
import os
import re
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = "veritor/verify-accounting/v1"

_RAW_OPEN = io.open  # the un-patched opener (the instrument's own /proc reads go through it and are never counted)

#: (bucket, part) per check id / section name.  A check id not listed here is matched by prefix (``V4.proof.chunk07`` -> ``V4.proof.``); a name
#: matching nothing is UNCLASSIFIED (never silently a bucket).  ``whole_tree`` / ``ids_walk`` sub-counters override this per the rule above.
BUCKET_MAP: dict[str, tuple[str, str]] = {
    "V3.trees_rehashed": ("archive_check", "trees_rehashed"),
    # deployed mode (reverify --deployed-openings, e2e/verifier_v3/deployed.py): roots folded from the STORED leaf hashes (32 B x N, no leaf bytes)
    # and the per-opening authentication -- the work a deployed verifier does against the pinned roots; O(N) in COUNT, which stored paths would remove
    "V3.trees_from_leaf_hashes": ("deployed_sparse", "openings"),
    "V5.public_trees": ("deployed_sparse", "statements"),  # MUFU table trees rebuilt from the verifier's fixtures (a verifier input) + literal trees
    "V3.deployed_openings": ("deployed_sparse", "openings"),
    # proofs
    "V4.host_verify": ("deployed_sparse", "proofs"),
    "V4.vk_derivation_once": ("deployed_sparse", "proofs"),
    "V4.proof.": ("deployed_sparse", "proofs"),
    "V4.family.": ("deployed_sparse", "proofs"),
    "V4.proofs_binding": ("deployed_sparse", "proofs"),
    "V4.prover_summary": ("deployed_sparse", "proofs"),
    "V4.obligations_dir": ("deployed_sparse", "proofs"),
    "D6.vk_once": ("deployed_sparse", "proofs"),
    # statements: the verifier's own derivation of what must have been proved, and the position / coverage checks
    "V5.coverage_inputs": ("deployed_sparse", "statements"),
    "V5.": ("deployed_sparse", "statements"),
    "V3.interior_layout": ("deployed_sparse", "statements"),
    "V3.literal_table": ("deployed_sparse", "statements"),
    "V8.interior_regions": ("deployed_sparse", "statements"),
    # direct checks (the verifier's own openings + equality; no proof object)
    "V6.gather": ("deployed_sparse", "direct_checks"),
    "V6.inputs": ("deployed_sparse", "direct_checks"),
    "V6.input_checks_file": ("deployed_sparse", "direct_checks"),
    # the verifier's own model copy
    "V6.checkpoint": ("checkpoint", "checkpoint"),
    # boundary = the request's x / y (read in full by the protocol)
    "V8.step2": ("deployed_sparse", "boundary_io"),
    "V8.output_copies": ("deployed_sparse", "boundary_io"),
    "D6.request_commitment": ("deployed_sparse", "boundary_io"),
    "V7.output_domain": ("deployed_sparse", "boundary_io"),
    # pins
    "V1.": ("deployed_sparse", "pins"),
    "V2.": ("deployed_sparse", "pins"),
    "V3.boundary": ("deployed_sparse", "pins"),
    "V3.interior": ("deployed_sparse", "pins"),
    "V3.weights_root": ("deployed_sparse", "pins"),
    "V3.no_recommit": ("deployed_sparse", "pins"),
    "V4.guest": ("deployed_sparse", "proofs"),  # its row is the SP1 host's `info` subprocess (a vk derivation, ~30 s on the pod path), not a pin read
    "V4.elf_archived": ("deployed_sparse", "pins"),
    "V7.": ("deployed_sparse", "pins"),
    "V8.": ("deployed_sparse", "pins"),
    "B.": ("deployed_sparse", "pins"),
    "record": ("deployed_sparse", "pins"),
}

BUCKETS = ("archive_check", "deployed_sparse", "checkpoint", "unclassified")

#: path class of a read, by the path RELATIVE to the record directory (else by absolute-path hints)
_PATH_CLASSES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"adapter/selective/interior/[^/]+/[^/]+/leaf-\d+\.bin$"), "interior_leaf"),
    (re.compile(r"adapter/selective/interior/[^/]+/[^/]+/index\.json$"), "interior_index"),
    (re.compile(r"adapter/selective/interior/[^/]+/[^/]+/gates\.bin$"), "interior_gates"),
    (re.compile(r"adapter/selective/interior/"), "interior_other"),
    (re.compile(r"adapter/selective/ru/[^/]+/\d+/values\.bin$"), "ru_values"),
    (re.compile(r"adapter/selective/ru/[^/]+/\d+/index\.json$"), "ru_index"),
    (re.compile(r"adapter/selective/ru/[^/]+/boundary\.json$"), "boundary_json"),
    (re.compile(r"adapter/selective/weights/values\.bin$"), "weights_values"),
    (re.compile(r"adapter/selective/weights/leaves\.bin$"), "weights_leaves"),
    (re.compile(r"adapter/selective/weights/index\.json$"), "weights_index"),
    (re.compile(r"adapter/selective/boundary/"), "boundary_store"),
    (re.compile(r"adapter/selective/(commitments|commitments_phase0|exemption_request|manifest|description|capture|phase0|required)[^/]*\.json$"), "pins_meta"),
    (re.compile(r"adapter/production/production\.jsonl$"), "served_outputs"),
    (re.compile(r"obligations/chunk\d+/composed\.json$"), "typed_object"),
    (re.compile(r"obligations/chunk\d+/statement\.bin$"), "statement"),
    (re.compile(r"obligations/"), "obligations_other"),
    (re.compile(r"proofs/chunk\d+-compressed\.bin$"), "proof"),
    (re.compile(r"proofs/"), "proof_meta"),
    (re.compile(r"(chronology/|protocol|instance|weights_authorized|mapping\.units|verify/|ru_map|maps\.json|verifier_exit|expressibility)"), "pins_meta"),
]
_ABS_CLASSES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"model\.safetensors$"), "checkpoint"),
    (re.compile(r"population[^/]*\.json$"), "population"),
    (re.compile(r"\.elf$|veritor-zk-guest"), "elf"),
    (re.compile(r"fixtures/W11|mufu_.*\.xz$"), "verifier_fixture"),
    (re.compile(r"approved_guests\.json$|manifests/checkpoints\.json$"), "verifier_fixture"),
    (re.compile(r"expressibility"), "pins_meta"),
]


def _counter() -> dict[str, Any]:
    return {"bytes_read": 0, "bytes_hashed": 0, "hash_calls": 0, "leaves_hashed": 0, "nodes_hashed": 0, "ids_digests": 0, "openings": 0, "trees_built": 0,
            "files": set(), "reads_by_class": Counter()}


def _add(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k in ("bytes_read", "bytes_hashed", "hash_calls", "leaves_hashed", "nodes_hashed", "ids_digests", "openings", "trees_built"):
        dst[k] += src[k]
    dst["files"] |= src["files"]
    dst["reads_by_class"].update(src["reads_by_class"])


def _plain(c: dict[str, Any]) -> dict[str, Any]:
    return {"bytes_read": int(c["bytes_read"]), "bytes_hashed": int(c["bytes_hashed"]), "hash_calls": int(c["hash_calls"]), "leaves_hashed": int(c["leaves_hashed"]),
            "nodes_hashed": int(c["nodes_hashed"]), "ids_digests": int(c["ids_digests"]), "openings": int(c["openings"]), "trees_built": int(c["trees_built"]),
            "files_opened": len(c["files"]), "reads_by_class": dict(sorted(c["reads_by_class"].items()))}


class Segment:
    """One row: a named check / section.  ``own`` excludes the nested whole-tree / id-walk children; ``whole_tree`` / ``ids_walk`` sum them."""

    __slots__ = ("checks", "children", "ids_walk", "kind", "name", "own", "parent", "peak_rss_mb", "rss_mb_after", "section_wall_s", "t0", "tree_ids", "wall_s", "whole_tree")

    def __init__(self, name: str, kind: str, parent: Segment | None) -> None:
        self.name, self.kind, self.parent = name, kind, parent
        self.own, self.whole_tree, self.ids_walk = _counter(), _counter(), _counter()
        self.children: list[Segment] = []
        self.checks: list[str] = []
        self.tree_ids: list[str] = []
        self.t0 = time.perf_counter()
        self.wall_s = 0.0
        self.section_wall_s = 0.0  # wall of explicit sections that ran inside this segment (rows of their own: excluded from this row's wall)
        self.peak_rss_mb: float | None = None
        self.rss_mb_after: float | None = None


def _rss_mb() -> tuple[float | None, float | None]:
    """(peak high-water MB, current MB | None)."""
    peak = cur = None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = round(rss / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)
    except Exception:  # noqa: BLE001
        pass
    try:
        with _RAW_OPEN("/proc/self/statm", "r") as f:  # Linux only; the instrument's own read is not counted (raw io.open kept in _RAW_OPEN)
            cur = round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024 ** 2, 1)
    except Exception:  # noqa: BLE001
        cur = None
    return peak, cur


def _proc_rchar() -> int | None:
    try:
        with _RAW_OPEN("/proc/self/io", "r") as f:
            for ln in f:
                if ln.startswith("rchar:"):
                    return int(ln.split()[1])
    except Exception:  # noqa: BLE001
        return None
    return None


class _ReadProxy:
    """A file object wrapper that counts what is read; everything else is delegated."""

    __slots__ = ("_acct", "_f", "_path")

    def __init__(self, f: Any, path: str, acct: Accounting) -> None:
        self._f, self._path, self._acct = f, path, acct

    def _count(self, data: Any) -> Any:
        if data:
            self._acct.read(self._path, len(data))
        return data

    def read(self, *a: Any) -> Any:
        return self._count(self._f.read(*a))

    def read1(self, *a: Any) -> Any:
        return self._count(self._f.read1(*a))

    def readline(self, *a: Any) -> Any:
        return self._count(self._f.readline(*a))

    def readlines(self, *a: Any) -> Any:
        out = self._f.readlines(*a)
        self._acct.read(self._path, sum(len(x) for x in out))
        return out

    def readinto(self, b: Any) -> Any:
        n = self._f.readinto(b)
        if n:
            self._acct.read(self._path, int(n))
        return n

    def __iter__(self) -> Any:
        for ln in self._f:
            self._acct.read(self._path, len(ln))
            yield ln

    def __enter__(self) -> _ReadProxy:
        self._f.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._f.__exit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._f, name)


class Accounting:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.root = Segment("(unattributed)", "root", None)
        self.stack: list[Segment] = [self.root]
        self.closed: list[Segment] = []
        self.implicit: Segment | None = None
        self.openings: list[tuple[str, str, int, int]] = []  # (tree, leaf id, leaf bytes, depth)
        self.installed = False
        self.record_dir: str | None = None
        self.extra_paths: dict[str, str] = {}  # absolute path prefix -> class (population, checkpoint, ...)
        self._class_cache: dict[str, str] = {}
        self.rchar0 = _proc_rchar()
        self.t_start = time.perf_counter()
        self._suspended: list[Segment] = []

    # -- attribution ---------------------------------------------------------------------------------------------------------------------
    @property
    def active(self) -> Segment:
        return self.stack[-1]

    def _classify_path(self, path: str) -> str:
        cls = self._class_cache.get(path)
        if cls is None:
            cls = self._class_cache[path] = self._classify_uncached(path)
        return cls

    def _classify_uncached(self, path: str) -> str:
        try:
            p = os.path.realpath(path).replace("\\", "/")  # the record may be addressed through a symlink (/tmp -> /private/tmp on macOS)
        except OSError:
            p = path.replace("\\", "/")
        rel = None
        if self.record_dir:
            rd = os.path.realpath(self.record_dir).rstrip("/") + "/"
            if p.startswith(rd):
                rel = p[len(rd):]
        for prefix, cls in self.extra_paths.items():
            if p.startswith(os.path.realpath(prefix)):
                return cls
        if rel is not None:
            for rx, cls in _PATH_CLASSES:
                if rx.search(rel):
                    return cls
            return "record_other:" + "/".join(rel.split("/")[:2])
        for rx, cls in _ABS_CLASSES:
            if rx.search(p):
                return cls
        return "outside_record"

    def read(self, path: str, n: int) -> None:
        if n <= 0:
            return
        with self.lock:
            seg = self.active
            seg.own["bytes_read"] += n
            seg.own["files"].add(path)
            seg.own["reads_by_class"][self._classify_path(path)] += n

    def hashed(self, n: int) -> None:
        with self.lock:
            seg = self.active
            seg.own["bytes_hashed"] += n
            seg.own["hash_calls"] += 1

    def count(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.active.own[key] += n

    def opening(self, tree: str, leaf: str, nbytes: int, depth: int) -> None:
        with self.lock:
            self.active.own["openings"] += 1
            self.openings.append((str(tree), str(leaf), int(nbytes), int(depth)))

    @contextmanager
    def section(self, name: str, kind: str = "section", tree_id: str | None = None):
        """An explicit row.  ``kind`` = ``section`` (a named row of its own) | ``whole_tree`` | ``ids_walk`` (nested: rolled up into the parent's
        sub-counter and listed as a child)."""
        with self.lock:
            parent = self.active
            seg = Segment(name, kind, parent)
            if tree_id:
                seg.tree_ids.append(tree_id)
            self.stack.append(seg)
        try:
            yield seg
        finally:
            self._close(seg)

    def _close(self, seg: Segment) -> None:
        with self.lock:
            seg.wall_s = time.perf_counter() - seg.t0
            seg.peak_rss_mb, seg.rss_mb_after = _rss_mb()
            if self.stack and self.stack[-1] is seg:
                self.stack.pop()
            parent = seg.parent or self.root
            if seg.kind in ("whole_tree", "ids_walk"):
                parent.children.append(seg)  # a named section is a row of its own (closed below), never a child of the next check's row
            if seg.kind == "whole_tree":
                boundary = all(t.startswith(("boundary/", "hashes:")) for t in seg.tree_ids) and bool(seg.tree_ids)
                # the boundary tree is x / y: DEPLOYED boundary_io work, counted as the parent's own (with the tree named) -- rule exception;
                # so is a tree built from its STORED leaf hashes (`hashes:` -- deployed mode): 32 B per leaf, no leaf bytes walked
                _add(parent.own if boundary else parent.whole_tree, seg.own)
                _add(parent.own if boundary else parent.whole_tree, seg.whole_tree)
                _add(parent.ids_walk, seg.ids_walk)
                if not boundary:
                    parent.whole_tree["trees_built"] += 1
                    parent.tree_ids.extend(seg.tree_ids)
                else:
                    parent.own["trees_built"] += 1
            elif seg.kind == "ids_walk":
                _add(parent.ids_walk, seg.own)
                _add(parent.whole_tree, seg.whole_tree)  # a cache.tree() inside the walk is a whole-tree build
                _add(parent.ids_walk, seg.ids_walk)
                parent.tree_ids.extend(seg.tree_ids)
            else:
                seg.wall_s = max(0.0, seg.wall_s - seg.section_wall_s)  # nested explicit sections are their own rows
                parent.section_wall_s += seg.wall_s + seg.section_wall_s
                self.closed.append(seg)

    def check_boundary(self, code: str) -> None:
        """Called by ``Log.check(code, ...)``: the work since the previous boundary is this check's row (implicit attribution) unless an explicit
        section is open, in which case the check is listed on that section's row."""
        with self.lock:
            seg = self.active
            if seg.kind == "section":
                seg.checks.append(code)
                return
            if seg is self.root:
                # name the root's accumulated work as this check's row and start a fresh implicit accumulator
                row = Segment(code, "check", None)
                row.t0 = self.root.t0
                row.own, row.whole_tree, row.ids_walk = self.root.own, self.root.whole_tree, self.root.ids_walk
                row.children, row.tree_ids = self.root.children, self.root.tree_ids
                row.checks = [code]
                self.root.own, self.root.whole_tree, self.root.ids_walk = _counter(), _counter(), _counter()
                self.root.children, self.root.tree_ids = [], []
                row.wall_s = max(0.0, time.perf_counter() - row.t0 - self.root.section_wall_s)  # minus the explicit sections' own rows
                self.root.section_wall_s = 0.0
                self.root.t0 = time.perf_counter()
                row.peak_rss_mb, row.rss_mb_after = _rss_mb()
                self.closed.append(row)
                return
            seg.checks.append(code)

    # -- installation ----------------------------------------------------------------------------------------------------------------------
    def install(self) -> None:
        """Patch the hash constructor, the file openers, the memmap readers, the tree builders and the openers.  Idempotent."""
        if self.installed:
            return
        self.installed = True
        acct = self

        real_sha256 = hashlib.sha256

        class _Sha256:
            __slots__ = ("_h",)

            def __init__(self, data: Any = b"", *, usedforsecurity: bool = True) -> None:
                self._h = real_sha256()
                if data:
                    acct.hashed(len(data))
                    self._h.update(data)

            def update(self, data: Any) -> None:
                acct.hashed(len(data))
                self._h.update(data)

            def digest(self) -> bytes:
                return self._h.digest()

            def hexdigest(self) -> str:
                return self._h.hexdigest()

            def copy(self) -> _Sha256:
                c = _Sha256.__new__(_Sha256)
                c._h = self._h.copy()
                return c

            @property
            def name(self) -> str:
                return "sha256"

            digest_size = 32
            block_size = 64

            def __getattr__(self, item: str) -> Any:
                return getattr(self._h, item)

        hashlib.sha256 = _Sha256  # type: ignore[assignment]

        def counting_open(file: Any, mode: str = "r", *a: Any, **kw: Any):
            f = _RAW_OPEN(file, mode, *a, **kw)
            if isinstance(file, (str, bytes, os.PathLike)) and "r" in mode and "+" not in mode and not any(c in mode for c in "wax"):
                return _ReadProxy(f, os.fspath(file) if not isinstance(file, bytes) else file.decode(errors="replace"), acct)
            return f

        builtins.open = counting_open  # type: ignore[assignment]
        io.open = counting_open  # type: ignore[assignment]

        # the Merkle-hash counters (every Merkle hash of the POC is a verity_vllm.hashing call)
        try:
            from verity_vllm import hashing

            real_leaf, real_node, real_lift, real_ids = hashing.leaf_hash, hashing.node_hash, hashing.lift_hash, hashing.ids_digest

            def leaf_hash(*a: Any, **k: Any) -> bytes:
                acct.count("leaves_hashed")
                return real_leaf(*a, **k)

            def node_hash(*a: Any, **k: Any) -> bytes:
                acct.count("nodes_hashed")
                return real_node(*a, **k)

            def lift_hash(*a: Any, **k: Any) -> bytes:
                acct.count("nodes_hashed")
                return real_lift(*a, **k)

            def ids_digest(*a: Any, **k: Any) -> bytes:
                acct.count("ids_digests")
                return real_ids(*a, **k)

            hashing.leaf_hash, hashing.node_hash, hashing.lift_hash, hashing.ids_digest = leaf_hash, node_hash, lift_hash, ids_digest
        except Exception:  # noqa: BLE001
            pass
        # the memmap readers (RU values.bin, weights values.bin / leaves.bin)
        try:
            from verity_vllm import capture

            real_ru_read = capture.RUReader.read

            def ru_read(self: Any, canonical_id: str) -> bytes:
                out = real_ru_read(self, canonical_id)
                acct.read(str(self._values_path), len(out))
                return out

            capture.RUReader.read = ru_read  # type: ignore[method-assign]
            real_rows_mm = capture.WeightsReader._read_rows_mm

            def rows_mm(self: Any, entry: Any, start: int, end: int) -> bytes:
                out = real_rows_mm(self, entry, start, end)
                acct.read(str(self._values_path), len(out))
                return out

            capture.WeightsReader._read_rows_mm = rows_mm  # type: ignore[method-assign]
            real_stored = capture.WeightsReader._stored_leaves

            def stored_leaves(self: Any, entry: Any):
                out = real_stored(self, entry)
                if out:
                    acct.read(str(self._leaves_path), 32 * len(out))
                return out

            capture.WeightsReader._stored_leaves = stored_leaves  # type: ignore[method-assign]
        except Exception:  # noqa: BLE001
            pass
        # whole-tree builds + openings (capture trees)
        try:
            from verity_vllm import commit

            real_build = commit.TreeCache._build

            def build(self: Any, tree_id: str):
                # [deployed mode] a tree built from the STORED leaf hashes (recompute_leaves=False) walks 32 B per leaf, never the leaf bytes:
                # tagged `hashes:` so the rule files it as the caller's own (deployed) work, not a whole-tree rebuild
                tag = tree_id if getattr(self, "recompute_leaves", True) else f"hashes:{tree_id}"
                with acct.section(f"tree:{tag}", kind="whole_tree", tree_id=tag):
                    t = real_build(self, tree_id)
                    acct.count("trees_built")
                    return t

            commit.TreeCache._build = build  # type: ignore[method-assign]
            real_open = commit.MerkleTree.open

            def m_open(self: Any, value: Any):
                o = real_open(self, value)
                acct.opening(self.tree_id, value.canonical_id, len(o.value), len(o.path))
                return o

            commit.MerkleTree.open = m_open  # type: ignore[method-assign]
            real_open_range = commit.MerkleTree.open_range

            def m_open_range(self: Any, values: Any, aggregate: Any):
                o = real_open_range(self, values, aggregate)
                acct.opening(self.tree_id, aggregate.canonical_id, len(o.value), len(o.path))
                return o

            commit.MerkleTree.open_range = m_open_range  # type: ignore[method-assign]
        except Exception:  # noqa: BLE001
            pass
        # whole-tree builds + openings (E2's interior trees)
        try:
            from e2e.interior import tree as itree

            real_load = itree.load_tree

            def load_tree(directory: Any, *a: Any, **k: Any):
                d = Path(directory)
                tid = f"interior/{d.parent.name}/{d.name}"
                if not k.get("recompute", True):
                    tid = f"hashes:{tid}"  # [deployed mode] built from index.json's per-leaf sha256, see TreeCache._build above
                with acct.section(f"tree:{tid}", kind="whole_tree", tree_id=tid):
                    t = real_load(directory, *a, **k)
                    acct.count("trees_built")
                    return t

            itree.load_tree = load_tree  # type: ignore[assignment]
            real_iopen = itree.InteriorTree.open

            def i_open(self: Any, rank: int):
                o = real_iopen(self, rank)
                acct.opening(self.tree_id, o.canonical_id, len(o.value), len(o.path))
                return o

            itree.InteriorTree.open = i_open  # type: ignore[method-assign]
        except Exception:  # noqa: BLE001
            pass
        # the id-list walk
        try:
            from e2e.verifier_v3 import expected as vex

            real_ct = vex.committed_trees

            def committed_trees(sel: Any, commitments: Any, tree_ids: Any, cache: Any):
                with acct.section("ids_walk:committed_trees", kind="ids_walk") as seg:
                    seg.tree_ids.extend(str(t) for t in tree_ids)
                    return real_ct(sel, commitments, tree_ids, cache)

            vex.committed_trees = committed_trees  # type: ignore[assignment]
            # `Deriver.committed` looks the function up on the module at call time (module global) -- patched by the assignment above
        except Exception:  # noqa: BLE001
            pass

    # -- the report -------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def bucket_of(name: str) -> tuple[str, str]:
        if name in BUCKET_MAP:
            return BUCKET_MAP[name]
        best = None
        for k, v in BUCKET_MAP.items():
            if k.endswith(".") and name.startswith(k) and (best is None or len(k) > len(best[0])):
                best = (k, v)
        return best[1] if best else ("unclassified", "unclassified")

    def report(self, *, record_dir: Path | str, sampled_units: int | None = None, proofs: list[dict[str, Any]] | None = None, note: str | None = None) -> dict[str, Any]:
        with self.lock:
            # anything still accumulating at the root is unattributed
            if any(self.root.own[k] for k in ("bytes_read", "bytes_hashed", "hash_calls")) or self.root.children:
                self.check_boundary("(after last check)")
                self.closed[-1].name = "(after last check)"
            rows = list(self.closed)
        record_dir = Path(record_dir)
        record_bytes = 0
        for dp, _dn, fns in os.walk(record_dir):
            for fn in fns:
                try:
                    record_bytes += os.stat(os.path.join(dp, fn)).st_size
                except OSError:
                    pass

        def zero() -> dict[str, Any]:
            return {"bytes_read": 0, "bytes_hashed": 0, "hash_calls": 0, "leaves_hashed": 0, "nodes_hashed": 0, "ids_digests": 0, "openings": 0, "trees_built": 0,
                    "files": set(), "reads_by_class": Counter(), "wall_s": 0.0, "peak_rss_mb": None, "checks": []}

        buckets: dict[str, dict[str, Any]] = {b: zero() | {"parts": {}} for b in BUCKETS}

        def part(b: str, p: str) -> dict[str, Any]:
            return buckets[b]["parts"].setdefault(p, zero())

        def pour(dst: dict[str, Any], c: dict[str, Any], wall: float, name: str, peak: float | None) -> None:
            _add(dst, c)
            dst["wall_s"] += wall
            if name not in dst["checks"]:
                dst["checks"].append(name)
            if peak is not None:
                dst["peak_rss_mb"] = max(dst["peak_rss_mb"] or 0.0, peak)

        per_check: dict[str, Any] = {}
        bucket_map: dict[str, Any] = {}
        merged: dict[str, Segment] = {}
        for s in rows:  # merge rows of the same name (a loop logging the same check id twice)
            if s.name in merged:
                m = merged[s.name]
                _add(m.own, s.own)
                _add(m.whole_tree, s.whole_tree)
                _add(m.ids_walk, s.ids_walk)
                m.wall_s += s.wall_s
                m.checks.extend(s.checks)
                m.tree_ids.extend(s.tree_ids)
                m.peak_rss_mb = max(m.peak_rss_mb or 0, s.peak_rss_mb or 0)
                m.rss_mb_after = s.rss_mb_after
                m.children.extend(s.children)
            else:
                merged[s.name] = s
        for name, s in merged.items():
            b, p = self.bucket_of(name)
            total = _counter()
            _add(total, s.own)
            _add(total, s.whole_tree)
            _add(total, s.ids_walk)
            # the row's own work goes to its bucket; whole-tree / id-walk work to ARCHIVE (except inside V3.trees_rehashed, which is archive anyway);
            # the wall of the nested builds goes with them (the row's own wall is what remains)
            child_wall = sum(c.wall_s for c in s.children if c.kind in ("whole_tree", "ids_walk"))
            if b == "archive_check":
                pour(part(b, p), total, s.wall_s, name, s.peak_rss_mb)
                pour(buckets[b], total, s.wall_s, name, s.peak_rss_mb)
                split = {"own": (b, p), "whole_tree": (b, p), "ids_walk": (b, p)}
            else:
                own_wall = max(0.0, s.wall_s - child_wall)
                pour(part(b, p), s.own, own_wall, name, s.peak_rss_mb)
                pour(buckets[b], s.own, own_wall, name, s.peak_rss_mb)
                wt_wall = sum(c.wall_s for c in s.children if c.kind == "whole_tree" and not all(t.startswith(("boundary/", "hashes:")) for t in c.tree_ids))
                iw_wall = sum(c.wall_s for c in s.children if c.kind == "ids_walk")
                if any(s.whole_tree[k] for k in ("bytes_read", "bytes_hashed", "leaves_hashed", "nodes_hashed")):
                    pour(part("archive_check", "whole_tree_rebuild_for_openings"), s.whole_tree, wt_wall, name, None)
                    pour(buckets["archive_check"], s.whole_tree, wt_wall, name, None)
                if any(s.ids_walk[k] for k in ("bytes_read", "bytes_hashed", "ids_digests")):
                    pour(part("archive_check", "ids_walk"), s.ids_walk, iw_wall, name, None)
                    pour(buckets["archive_check"], s.ids_walk, iw_wall, name, None)
                split = {"own": (b, p), "whole_tree": ("archive_check", "whole_tree_rebuild_for_openings"), "ids_walk": ("archive_check", "ids_walk"),
                         "children_wall_s": round(child_wall, 3)}
            per_check[name] = {**_plain(total), "own": _plain(s.own), "whole_tree": _plain(s.whole_tree) | {"trees": sorted(set(s.tree_ids))[:64], "n_trees": len(set(s.tree_ids))},
                               "ids_walk": _plain(s.ids_walk), "wall_s": round(s.wall_s, 3), "peak_rss_mb": s.peak_rss_mb, "rss_mb_after": s.rss_mb_after,
                               "bucket": b, "part": p, "checks": s.checks,
                               "children": [{"name": c.name, "kind": c.kind, "wall_s": round(c.wall_s, 3), **{k: int(c.own[k]) for k in ("bytes_read", "bytes_hashed", "leaves_hashed", "nodes_hashed")}}
                                            for c in s.children[:200]]}
            bucket_map[name] = split
        # proofs read by the host subprocess: by file size
        proof_bytes = statement_bytes = typed_object_bytes = composed_file_bytes = 0
        for pr in proofs or []:
            proof_bytes += int(pr.get("proof_bytes") or 0)
            statement_bytes += int(pr.get("statement_bytes") or 0)
            composed_file_bytes += int(pr.get("composed_json_bytes") or 0)
            # [item 3, 23:30Z] the typed object's CANONICAL bytes (h_O's preimage: composed.json#canonical, what check_binding hashes) when
            # reverify measured them; the whole composed.json file otherwise (the pre-23:30Z reading: the format's size, not the object's)
            typed_object_bytes += int(pr.get("typed_object_canonical_bytes") if pr.get("typed_object_canonical_bytes") is not None else (pr.get("composed_json_bytes") or 0))
        if proofs:
            pp = part("deployed_sparse", "proofs")
            pp["bytes_read"] += proof_bytes
            pp["reads_by_class"]["proof (host subprocess, by stat)"] += proof_bytes
            pp["n_proofs"] = len(proofs)
            pp["proof_bytes"], pp["statement_bytes"], pp["typed_object_bytes"] = proof_bytes, statement_bytes, typed_object_bytes
            buckets["deployed_sparse"]["bytes_read"] += proof_bytes
            buckets["deployed_sparse"]["reads_by_class"]["proof (host subprocess, by stat)"] += proof_bytes

        def fin(d: dict[str, Any]) -> dict[str, Any]:
            out = _plain(d) | {"wall_s": round(d["wall_s"], 3), "peak_rss_mb": d["peak_rss_mb"], "checks": d["checks"],
                               "ratio_bytes_read_to_record": (round(d["bytes_read"] / record_bytes, 6) if record_bytes else None),
                               "ratio_bytes_hashed_to_record": (round(d["bytes_hashed"] / record_bytes, 6) if record_bytes else None)}
            for k in ("n_proofs", "proof_bytes", "statement_bytes", "typed_object_bytes"):
                if k in d:
                    out[k] = d[k]
            if "parts" in d:
                out["parts"] = {k: fin(v) for k, v in d["parts"].items()}
            # no null leaves: a part with no RSS sample (nested whole-tree / ids_walk work) or an empty bucket OMITS the key -- E3's
            # costlog `attach_verify_accounting` wraps every leaf it finds as MEASURED and refuses a null (a null is UNKNOWN, by name)
            return {k: v for k, v in out.items() if v is not None}

        out_buckets = {b: fin(d) for b, d in buckets.items()}
        out_buckets["deployed_sparse"]["sampled_units"] = sampled_units
        # deployed_ideal: distinct leaves the VERIFIER itself opened (direct checks, table ports, inputs) -- what stored openings would transfer
        distinct: dict[tuple[str, str], tuple[int, int]] = {}
        for t, leaf, nb, depth in self.openings:
            distinct[(t, leaf)] = (nb, depth)
        leaf_bytes = sum(nb for nb, _ in distinct.values())
        path_bytes = sum(32 * d for _, d in distinct.values())
        rchar1 = _proc_rchar()
        attributed = sum(int(s.own["bytes_read"]) + int(s.whole_tree["bytes_read"]) + int(s.ids_walk["bytes_read"]) for s in merged.values())
        return {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename, "record": str(record_dir), "record_bytes": record_bytes,
                "record_bytes_note": "sum of file sizes under the record directory the verifier was pointed at (a slim copy = the slim size)",
                "rule": __doc__.split("CLASSIFICATION RULE")[1].split("HOW THE COUNTS")[0].strip(),
                "buckets": out_buckets, "per_check": per_check, "bucket_map": bucket_map,
                "cross_check": {"proc_io_rchar_delta": (None if (self.rchar0 is None or rchar1 is None) else rchar1 - self.rchar0), "attributed_read_bytes": attributed,
                                "unattributed_read_bytes": (None if (self.rchar0 is None or rchar1 is None) else max(0, (rchar1 - self.rchar0) - attributed)),
                                "note": "rchar counts read(2) bytes of THIS process incl. pipes and the instrument's own writes' reads; mmap page faults are NOT in rchar (the memmap slices are counted by the instrument directly)"},
                "deployed_ideal": {"openings": len(self.openings), "distinct_leaves": len(distinct), "leaf_bytes": leaf_bytes, "path_bytes": path_bytes,
                                   "proof_bytes": proof_bytes, "statement_bytes": statement_bytes, "typed_object_bytes": typed_object_bytes,
                                   "composed_json_file_bytes": composed_file_bytes, "composed_json_not_needed_bytes": max(0, composed_file_bytes - typed_object_bytes),
                                   "bytes_transferred_to_verifier_with_stored_openings": leaf_bytes + path_bytes + proof_bytes + statement_bytes + typed_object_bytes,
                                   "measured_deployed_bytes_read": int(buckets["deployed_sparse"]["bytes_read"]),
                                   "note": "MEASURED from the openings this run took: leaf bytes + 32 B x path depth per distinct (tree, leaf) the verifier itself opened, plus the "
                                           "proof bytes, statement.bin bytes and the typed objects' CANONICAL bytes (composed.json#canonical = h_O's preimage; the rest of composed.json -- "
                                           "the prover's witness hex, the statement's hex copies, provenance -- is composed_json_not_needed_bytes; e2e.verifier_v3.composed_bytes decomposes it by key). "
                                           "measured_deployed_bytes_read is the deployed bucket beside it (E3 Table 4: measured vs ideal)"},
                "wall_s_total": round(time.perf_counter() - self.t_start, 3), "note": note}


ACCT: Accounting | None = None


def install() -> Accounting:
    global ACCT
    if ACCT is None:
        ACCT = Accounting()
        ACCT.install()
    return ACCT


def current() -> Accounting | None:
    return ACCT

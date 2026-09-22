"""Independent GEMM *coordinate* checks on authenticated bundle values (contract section 16.2).

A coordinate VU is one output word ``out[t, j]`` of a GEMM node instance.  The verifier opens,
and authenticates against ``commitments.json``:

* the committed output row leaf ``v=.../<gemm>/p=0`` at position ``t`` (the word is ``[2j, 2j+2)``
  of it -- outputs are committed per row, so the *output* opening costs the row);
* the input row ``x`` = the producer's row at ``t`` (with W1's ``arg_cols`` slice when the
  producer is wider);
* ONE row block of the parameter, ``w=<param>/c=<j>`` = column ``j`` of ``W^T`` (capture format
  v2, one leaf per row of the ``[N, K]`` parameter; the bytes are ``2K`` + one Merkle path);
* the bias/epilogue operands if the node has any (the POC's ``verity.bi_mm`` never does: the qkv
  bias is a separate ``aten.add`` node).

It then runs W4b's ``independent_gemm`` (C++ chain model, ``verity_vllm_numerics``) on the
``[1, K] x [1, K]`` problem -- the coverage gate is told the launch's ``N`` so the family lookup is
the description's, not the slice's -- and compares the predicted word to the committed one.  A
subset is also run through the pure-Python scalar gate ``independent_gemm_coordinate``.

The same node instances can be checked with the same-runtime whole-row relation
(``gemm.same_runtime.v1``: GPU, whole ``W`` opened -- under v2 as a ``RangeOpening`` over all row
blocks) so the two checker classes are measured on the same VUs and kept apart in Table 2.

Negatives: N02 (the committed output word at ``(t, j)`` is wrong but consistently committed:
the RU tree is rebuilt over the mutation, the opening authenticates, the relation rejects) and
section 15.2b (one authenticated operand word of ``x`` or of the ``W`` row is mutated: the
predicted output must change).

    PYTHONPATH=. python -m verity_vllm_adapter.coordinate_check <bundle> --out <dir> --checks 1000
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import identity, relations as R
from verity_vllm.capture import WEIGHTS_TREE_ID, CaptureReader
from verity_vllm.commit import (
    Opening,
    RangeOpening,
    TreeCache,
    load_commitments,
    verify_opening_in,
)
from verity_vllm.description import Description, NodeInstance
from verity_vllm.negatives import PatchedReader


class _LazyNumerics:
    """``verity_vllm_numerics`` needs ``veritor`` (Python >= 3.12); the same-runtime path does not."""

    def __getattr__(self, name: str) -> Any:
        from verity_vllm_numerics import relations as _nr

        return getattr(_nr, name)


NR: Any = _LazyNumerics()

FAMILIES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head")
GEMM_TYPES = ("gemm", "lm_head")


def family_of(node: Any) -> str:
    if node.op_type == "lm_head":
        return "lm_head"
    mp = str(node.extra.get("module_path") or node.op)
    for f in FAMILIES:
        if f in mp:
            return f
    return "other"


@dataclass(slots=True)
class CoordinateResult:
    canonical_id: str
    j: int
    op: str
    op_type: str
    family: str
    phase: str
    layer: str
    K: int
    N: int
    status: str = "structural"  # exact | mismatch | unresolved | auth_failed | missing | error
    detail: str = ""
    committed_word: int | None = None
    predicted_word: int | None = None
    x_id: str = ""
    w_id: str = ""
    opened_bytes: dict[str, int] = field(default_factory=dict)  # x, w, out_row, paths, total
    checker_seconds: float = 0.0
    python_gate_seconds: float | None = None
    python_gate_word: int | None = None
    coverage: str = ""
    kernel_path: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _open(cache: TreeCache, commitments: dict[str, Any], cid: str) -> tuple[Opening | RangeOpening | None, str | None]:
    try:
        o = cache.open(cid)
    except KeyError as e:
        return None, f"missing: {e}"
    if not verify_opening_in(commitments, o):
        return None, f"auth_failed: {cid}"
    return o, None


def _path_bytes(o: Opening | RangeOpening) -> int:
    if isinstance(o, RangeOpening):
        return o.path_bytes()
    return sum(len(p) for p in o.path if p is not None)


def _resolve_x(description: Description, inst: NodeInstance, boundary: Any, profile_id: str) -> tuple[str, str | None]:
    """(x value id, bias id | None) for a GEMM instance."""
    resolved = description.resolve_args(inst, boundary, profile_id)
    x = resolved.get("x")
    if not isinstance(x, str):
        raise R.RelationError(f"{inst.node.op}: 'x' does not resolve to one value id ({x!r})")
    bias = resolved.get("bias") or resolved.get("b")
    return x, (bias if isinstance(bias, str) else None)


def check_coordinate(
    description: Description,
    reader: CaptureReader,
    commitments: dict[str, Any],
    cache: TreeCache,
    canonical_id: str,
    j: int,
    *,
    python_gate: bool = False,
    arch: str = "sm_89",
) -> CoordinateResult:
    cid = identity.parse(canonical_id)
    boundary = reader.boundary(cid.request_id or "")
    inst = description.instance_of(canonical_id, boundary)
    node = inst.node
    N = int(description.resolve_shape(node.port_shape(inst.port))[0])
    res = CoordinateResult(canonical_id, int(j), node.op, node.op_type, family_of(node), inst.phase, node.layer, 0, N)
    if not (0 <= j < N):
        res.status, res.detail = "structural", f"j={j} outside [0, {N})"
        return res
    out, problem = _open(cache, commitments, canonical_id)
    if out is None:
        res.status, res.detail = (problem or "").split(":")[0], problem or ""
        return res
    res.committed_word = int(np.frombuffer(out.value, dtype="<u2")[j])
    try:
        x_id, bias_id = _resolve_x(description, inst, boundary, reader.profile_id)
    except Exception as e:  # noqa: BLE001
        res.status, res.detail = "missing", str(e)
        return res
    warg = node.arg("w")
    w_whole = str(identity.weight_id(warg.param or "", warg.shard))
    try:
        w_leaf = reader.weights.leaf_id_for_row(w_whole, j)
    except KeyError as e:
        res.status, res.detail = "missing", f"weight row leaf: {e}"
        return res
    if w_leaf == w_whole:
        res.status, res.detail = "structural", "weights are whole-tensor leaves (format v1): no coordinate-addressable opening"
        return res
    xo, problem = _open(cache, commitments, x_id)
    if xo is None:
        res.status, res.detail = (problem or "").split(":")[0], problem or ""
        return res
    wo, problem = _open(cache, commitments, w_leaf)
    if wo is None:
        res.status, res.detail = (problem or "").split(":")[0], problem or ""
        return res
    res.x_id, res.w_id = x_id, w_leaf
    inputs: dict[str, Any] = {"x": xo.as_value(), "w": wo.as_value(), "N": N, "M": 1, "arch": arch}
    bo = None
    if bias_id is not None:
        bo, problem = _open(cache, commitments, bias_id)
        if bo is None:
            res.status, res.detail = (problem or "").split(":")[0], problem or ""
            return res
        inputs["bias"] = bo.as_value()
    res.opened_bytes = {
        "x": len(xo.value),
        "w": len(wo.value),
        "bias": 0 if bo is None else len(bo.value),
        "out_row": len(out.value),
        "paths": _path_bytes(xo) + _path_bytes(wo) + _path_bytes(out) + (0 if bo is None else _path_bytes(bo)),
    }
    res.opened_bytes["total"] = sum(res.opened_bytes.values())
    res.K = int(np.frombuffer(wo.value, dtype="<u2").size)
    # a one-row block of the [N, K] parameter: its rows_per_leaf must be 1 for a single column
    if tuple(wo.shape)[0] != 1:
        res.status, res.detail = "structural", f"weight block has {wo.shape[0]} rows; coordinate checks need rows_per_leaf=1"
        return res
    try:
        cov = NR.coverage(inst, inputs)
    except NR.GemmRelationError as e:
        res.status, res.detail = "error", str(e)
        return res
    res.coverage, res.kernel_path = cov["reason"], cov["kernel_path"]
    if not cov["covered"]:
        res.status, res.detail = "unresolved", cov["reason"]
        return res
    t0 = time.perf_counter()
    try:
        predicted = NR.independent_gemm(inst, inputs)
    except NR.GemmUnresolved as e:
        res.status, res.detail = "unresolved", str(e)
        return res
    except NR.GemmRelationError as e:
        res.status, res.detail = "error", str(e)
        return res
    res.checker_seconds = time.perf_counter() - t0
    if len(predicted) != 2:
        res.status, res.detail = "error", f"model returned {len(predicted)} bytes for one coordinate"
        return res
    res.predicted_word = int(np.frombuffer(predicted, dtype="<u2")[0])
    res.status = "exact" if res.predicted_word == res.committed_word else "mismatch"
    if res.status == "mismatch":
        res.detail = f"predicted 0x{res.predicted_word:04x} != committed 0x{res.committed_word:04x}"
    if python_gate:
        t1 = time.perf_counter()
        try:
            pg = NR.independent_gemm_coordinate(inst, inputs, j=None)
            res.python_gate_word = int(np.frombuffer(pg, dtype="<u2")[0])
        except NR.GemmRelationError as e:
            res.python_gate_word = None
            res.detail = (res.detail + "; " if res.detail else "") + f"python gate: {e}"
        res.python_gate_seconds = time.perf_counter() - t1
    return res


@dataclass(slots=True)
class CoordinateOpenings:
    """The three authenticated openings of one GEMM coordinate ``out[t, j]`` (the seam the
    typed-obligation exporter consumes; INT).  ``x_cols`` is W1's ``arg_cols`` slice of the
    producer's row as an explicit column list, or None when the GEMM consumes the whole row."""

    canonical_id: str
    j: int
    op: str
    op_type: str
    family: str
    phase: str
    layer: str
    K: int
    N: int
    request_id: str
    x_id: str
    w_id: str
    param: str
    x: Opening
    w_row: Opening
    out_row: Opening
    x_cols: list[int] | None
    committed_word: int

    def meta(self) -> dict[str, Any]:
        return {"canonical_id": self.canonical_id, "j": self.j, "op": self.op, "op_type": self.op_type, "family": self.family,
                "phase": self.phase, "layer": self.layer, "K": self.K, "N": self.N, "request_id": self.request_id,
                "x_id": self.x_id, "w_id": self.w_id, "param": self.param, "x_cols": None if self.x_cols is None else [self.x_cols[0], self.x_cols[-1] + 1],
                "committed_word": self.committed_word}


def coordinate_openings(
    description: Description, reader: CaptureReader, commitments: dict[str, Any], cache: TreeCache, canonical_id: str, j: int,
) -> CoordinateOpenings:
    """Open and authenticate ``x`` (the producer's row), ``W[j, :]`` (the weights-v2 row leaf) and the
    committed output row of the GEMM instance ``canonical_id``; raise :class:`verity_vllm.relations.RelationError`
    when any opening is missing, fails to authenticate, or the bundle's weights are not row-addressable."""
    cid = identity.parse(canonical_id)
    inst, x_id, w_leaf, x_cols, N = describe_coordinate(description, reader, canonical_id, j)
    node = inst.node
    lo = open_coordinate(reader, commitments, cache, canonical_id, j, x_id=x_id, w_leaf=w_leaf, x_cols=x_cols)
    return CoordinateOpenings(canonical_id, int(j), node.op, node.op_type, family_of(node), inst.phase, node.layer, lo.K, N,
                              cid.request_id or "", x_id, w_leaf, node.arg("w").param or "", lo.x, lo.w_row, lo.out_row, x_cols, lo.committed_word)


def describe_coordinate(description: Description, reader: CaptureReader, canonical_id: str, j: int) -> tuple[NodeInstance, str, str, list[int] | None, int]:
    """What the (prover-written) Description says the operands of ``out[t, j]`` are: ``(instance, x id, W[j, :] leaf id, x column list |
    None, N)``.  A verifier COMPARES this against its own derivation from the pinned program (``e2e.verifier_v3.expected.Deriver``); the
    prover's exporter uses it as the address book.  ``RelationError`` when the description cannot name them."""
    cid = identity.parse(canonical_id)
    boundary = reader.boundary(cid.request_id or "")
    inst = description.instance_of(canonical_id, boundary)
    node = inst.node
    N = int(description.resolve_shape(node.port_shape(inst.port))[0])
    if not (0 <= j < N):
        raise R.RelationError(f"{canonical_id}: j={j} outside [0, {N})")
    x_id, bias_id = _resolve_x(description, inst, boundary, reader.profile_id)
    if bias_id is not None:
        raise R.RelationError(f"{canonical_id}: GEMM with a bias operand; the coordinate subcircuit has none")
    warg = node.arg("w")
    w_whole = str(identity.weight_id(warg.param or "", warg.shard))
    w_leaf = reader.weights.leaf_id_for_row(w_whole, j)
    if w_leaf == w_whole:
        raise R.RelationError(f"{canonical_id}: weights are whole-tensor leaves (format v1): no coordinate-addressable opening")
    cols = (node.extra.get("arg_cols") or {}).get("x")
    return inst, x_id, w_leaf, (None if not cols else list(range(int(cols[0]), int(cols[1])))), N


@dataclass(slots=True)
class LocatedOpenings:
    """The three authenticated openings of one GEMM coordinate at CALLER-NAMED positions (no Description involved)."""

    canonical_id: str
    j: int
    K: int
    x_id: str
    w_id: str
    x: Opening
    w_row: Opening
    out_row: Opening
    x_cols: list[int] | None
    committed_word: int


def open_coordinate(reader: CaptureReader, commitments: dict[str, Any], cache: TreeCache, canonical_id: str, j: int, *,
                    x_id: str, w_leaf: str, x_cols: list[int] | None) -> LocatedOpenings:
    """Open ``x_id`` (the producer's row), ``w_leaf`` (a weights-v2 row leaf) and the committed output row ``canonical_id`` and authenticate
    each against the committed roots; K is the weight row's width and must equal the x row's width (or the ``x_cols`` slice length).
    :class:`verity_vllm.relations.RelationError` when any opening is missing, fails to authenticate or has the wrong shape."""
    out, problem = _open(cache, commitments, canonical_id)
    if out is None or isinstance(out, RangeOpening):
        raise R.RelationError(problem or f"{canonical_id}: range opening where one row was expected")
    if not (0 <= j < int(np.frombuffer(out.value, dtype="<u2").size)):
        raise R.RelationError(f"{canonical_id}: j={j} outside the committed output row")
    xo, problem = _open(cache, commitments, x_id)
    if xo is None or isinstance(xo, RangeOpening):
        raise R.RelationError(problem or f"{x_id}: range opening where one row was expected")
    wo, problem = _open(cache, commitments, w_leaf)
    if wo is None or isinstance(wo, RangeOpening):
        raise R.RelationError(problem or f"{w_leaf}: range opening where one row was expected")
    K = int(np.frombuffer(wo.value, dtype="<u2").size)
    if tuple(wo.shape)[0] != 1:
        raise R.RelationError(f"{canonical_id}: weight block has {wo.shape[0]} rows; coordinate units need rows_per_leaf=1")
    x_width = int(np.frombuffer(xo.value, dtype="<u2").size)
    if x_cols is None and x_width != K:
        raise R.RelationError(f"{canonical_id}: x row has {x_width} words, K={K}, and no arg_cols slice")
    if x_cols is not None and (len(x_cols) != K or not (0 <= x_cols[0] and x_cols[-1] < x_width)):
        raise R.RelationError(f"{canonical_id}: arg_cols selects {len(x_cols)} columns of a {x_width}-word row, K={K}")
    return LocatedOpenings(canonical_id, int(j), K, x_id, w_leaf, xo, wo, out, x_cols, int(np.frombuffer(out.value, dtype="<u2")[j]))


# --- sampling -------------------------------------------------------------------------------------


def gemm_rows(description: Description, reader: CaptureReader) -> dict[tuple[str, str], list[tuple[str, NodeInstance]]]:
    """Every present GEMM/lm_head row VU of the bundle, bucketed by (family, phase)."""
    buckets: dict[tuple[str, str], list[tuple[str, NodeInstance]]] = defaultdict(list)
    ops = [n.op for n in description.nodes if n.op_type in GEMM_TYPES]
    for rid in reader.request_ids():
        b = reader.boundary(rid)
        for inst in description.instances(b, ops):
            cid = inst.output_id(reader.profile_id)
            if reader.locate(cid) is None:
                continue
            buckets[(family_of(inst.node), inst.phase)].append((cid, inst))
    return buckets


def stratified_sample(
    buckets: dict[tuple[str, str], list[tuple[str, NodeInstance]]],
    description: Description,
    n_total: int,
    seed: int,
) -> list[tuple[str, int, str, str]]:
    """``n_total`` (value id, j, family, phase) coordinates, equal share per (family, phase) bucket,
    rows drawn uniformly within the bucket (spreading over layers), ``j`` uniform in ``[0, N)``."""
    rng = random.Random(seed)
    keys = sorted(buckets)
    if not keys:
        return []
    share = max(1, n_total // len(keys))
    out: list[tuple[str, int, str, str]] = []
    for key in keys:
        rows = buckets[key]
        for _ in range(share):
            cid, inst = rng.choice(rows)
            N = int(description.resolve_shape(inst.node.port_shape(inst.port))[0])
            out.append((cid, rng.randrange(N), key[0], key[1]))
    # top up to n_total round-robin
    i = 0
    while len(out) < n_total:
        key = keys[i % len(keys)]
        cid, inst = rng.choice(buckets[key])
        N = int(description.resolve_shape(inst.node.port_shape(inst.port))[0])
        out.append((cid, rng.randrange(N), key[0], key[1]))
        i += 1
    return out


# --- negatives ------------------------------------------------------------------------------------


def patched_commitments(reader: CaptureReader, commitments: dict[str, Any], patched: PatchedReader, base_cache: TreeCache | None = None) -> tuple[dict[str, Any], TreeCache]:
    """Rebuild only the trees a mutation touches (roots are independent per tree); the returned
    cache serves the mutated capture, reusing ``base_cache``'s trees for every untouched tree (the
    928k-leaf weights tree would otherwise be rehashed for every negative)."""
    doc = json.loads(json.dumps(commitments))
    touched: set[str] = set()
    for cid in list(patched._patch_values) + list(patched._patch_entries):
        tid = reader.tree_of(cid)
        if tid:
            touched.add(tid)
    if patched._patch_weights:
        touched.add(WEIGHTS_TREE_ID)
    cache = TreeCache(patched, recompute_leaves=True)
    if base_cache is not None:
        for tid, tree in base_cache._trees.items():
            if tid not in touched:
                cache._trees[tid] = tree
    for tid in touched:
        old = doc["trees"].get(tid)
        if old is None:
            continue
        tree = cache.tree(tid)
        meta = {k: v for k, v in old.items() if k not in ("kind", "root", "leaf_count", "ids_digest")}
        doc["trees"][tid] = tree.record(old["kind"], **meta)
    return doc, cache


def n02_coordinate(description: Description, reader: CaptureReader, commitments: dict[str, Any], canonical_id: str, j: int, base_cache: TreeCache | None = None) -> dict[str, Any]:
    """Commit a wrong word at (t, j), consistently: the RU tree is rebuilt over the mutation."""
    original = reader.value(canonical_id)
    words = np.frombuffer(original.data, dtype="<u2").copy()
    words[j] ^= 0x0001  # one ulp of the committed word
    patched = PatchedReader(reader, values={canonical_id: words.tobytes()})
    doc, cache = patched_commitments(reader, commitments, patched, base_cache)
    res = check_coordinate(description, patched, doc, cache, canonical_id, j)
    return {
        "negative_id": "N02",
        "canonical_id": canonical_id,
        "j": j,
        "authenticated": res.status in ("exact", "mismatch"),
        "rejected": res.status == "mismatch",
        "status": res.status,
        "detail": res.detail,
        "committed_word": res.committed_word,
        "predicted_word": res.predicted_word,
    }


def mutate_operand(description: Description, reader: CaptureReader, commitments: dict[str, Any], canonical_id: str, j: int, which: str, seed: int, base_cache: TreeCache | None = None) -> dict[str, Any]:
    """Section 15.2b: mutate one authenticated operand word (``x`` or the ``W`` row); the relation's
    output must differ from the unmutated prediction (and from the committed word)."""
    cache0 = base_cache if base_cache is not None else TreeCache(reader)
    base = check_coordinate(description, reader, commitments, cache0, canonical_id, j)
    if base.status != "exact":
        return {"negative_id": "15.2b", "canonical_id": canonical_id, "j": j, "operand": which, "status": "skipped", "detail": f"baseline {base.status}: {base.detail}"}
    target = base.x_id if which == "x" else base.w_id
    value = reader.value(target)
    rng = random.Random(seed)
    words = np.frombuffer(value.data, dtype="<u2").copy()
    if which == "x" and inst_cols(description, canonical_id, reader):
        c0, c1 = inst_cols(description, canonical_id, reader)  # type: ignore[misc]
        k = rng.randrange(c0, c1)
    else:
        k = rng.randrange(words.size)
    # a strong, finite perturbation of one word: flip the sign and move the exponent by 3
    w0 = int(words[k])
    exp = (w0 >> 7) & 0xFF
    new_exp = exp + 3 if exp < 0xF0 else exp - 3
    if exp == 0:
        new_exp = 0x7F  # zero -> ~1.0
    first = ((w0 ^ 0x8000) & 0x807F) | (new_exp << 7)
    # The output is a bf16 word (8 significant bits): a perturbation of one operand word can be
    # absorbed by the output rounding when its partner is small (or exactly zero).  Escalate to a
    # 2^40-magnitude word when the first perturbation is invisible; the partner operand's word is
    # recorded so an unchanged output can be explained (partner zero -> provably invariant).
    attempts: list[dict[str, Any]] = []
    res = None
    for stage, new_word in (("exp+3,sign", first), ("2^40", 0x5380 | ((w0 & 0x8000) ^ 0x8000))):
        words[k] = new_word
        if which == "x":
            patched = PatchedReader(reader, values={target: words.tobytes()})
        else:
            patched = PatchedReader(reader, weights={target: words.tobytes()})
        doc, cache = patched_commitments(reader, commitments, patched, base_cache)
        res = check_coordinate(description, patched, doc, cache, canonical_id, j)
        changed = res.predicted_word is not None and res.predicted_word != base.predicted_word
        attempts.append({"stage": stage, "word_after": int(new_word), "predicted_after": res.predicted_word, "output_changed": changed, "status": res.status})
        if changed:
            break
    assert res is not None
    partner_word: int | None = None
    try:
        partner_id = base.w_id if which == "x" else base.x_id
        partner = np.frombuffer(reader.value(partner_id).data, dtype="<u2")
        if which == "x":
            cols = inst_cols(description, canonical_id, reader)
            partner_word = int(partner[k - (cols[0] if cols else 0)])
        else:
            cols = inst_cols(description, canonical_id, reader)
            partner_word = int(partner[k + (cols[0] if cols else 0)])
    except Exception:  # noqa: BLE001 - diagnostic only
        partner_word = None
    changed = attempts[-1]["output_changed"]
    return {
        "negative_id": "15.2b",
        "canonical_id": canonical_id,
        "j": j,
        "operand": which,
        "mutated_id": target,
        "word_index": k,
        "word_before": w0,
        "word_after": attempts[-1]["word_after"],
        "attempts": attempts,
        "partner_word": partner_word,
        "partner_zero": partner_word is not None and (partner_word & 0x7FFF) == 0,
        "authenticated": res.status in ("exact", "mismatch"),
        "output_changed": changed,
        "rejected": res.status == "mismatch",
        "status": res.status,
        "detail": res.detail,
        "predicted_before": base.predicted_word,
        "predicted_after": res.predicted_word,
        "committed_word": base.committed_word,
    }


def inst_cols(description: Description, canonical_id: str, reader: CaptureReader) -> tuple[int, int] | None:
    cid = identity.parse(canonical_id)
    inst = description.instance_of(canonical_id, reader.boundary(cid.request_id or ""))
    cols = (inst.node.extra.get("arg_cols") or {}).get("x")
    return (int(cols[0]), int(cols[1])) if cols else None


# --- same-runtime comparison on the same VUs -------------------------------------------------------


def same_runtime_rows(description: Description, reader: CaptureReader, commitments: dict[str, Any], ids: list[str]) -> R.CheckReport:
    """The whole-row same-runtime relation on the given row VUs (needs the adapter's GPU relations)."""
    import verity_vllm_adapter.relations_gpu  # noqa: F401  (registers gemm.same_runtime.v1)

    return R.check(description, reader, commitments, ids, cache=TreeCache(reader))


# --- driver ---------------------------------------------------------------------------------------


def summarize(results: list[CoordinateResult]) -> dict[str, Any]:
    by: dict[str, dict[str, Any]] = {}
    for r in results:
        key = f"{r.family}/{r.phase}"
        d = by.setdefault(key, {"attempted": 0, "exact": 0, "mismatch": 0, "unresolved": 0, "other": 0, "checker_seconds": 0.0, "opened_bytes": 0, "opened_bytes_w": 0, "opened_bytes_x": 0, "opened_bytes_out_row": 0, "opened_bytes_paths": 0, "layers": set(), "K": r.K, "N": r.N, "python_gate_seconds": 0.0, "python_gate_n": 0, "python_gate_agree": 0})
        d["attempted"] += 1
        if r.status in ("exact", "mismatch", "unresolved"):
            d[r.status] += 1
        else:
            d["other"] += 1
        d["checker_seconds"] += r.checker_seconds
        d["opened_bytes"] += r.opened_bytes.get("total", 0)
        d["opened_bytes_w"] += r.opened_bytes.get("w", 0)
        d["opened_bytes_x"] += r.opened_bytes.get("x", 0)
        d["opened_bytes_out_row"] += r.opened_bytes.get("out_row", 0)
        d["opened_bytes_paths"] += r.opened_bytes.get("paths", 0)
        d["layers"].add(r.layer)
        if r.python_gate_seconds is not None:
            d["python_gate_seconds"] += r.python_gate_seconds
            d["python_gate_n"] += 1
            d["python_gate_agree"] += int(r.python_gate_word == r.predicted_word)
    for d in by.values():
        d["layers"] = sorted(d["layers"], key=lambda s: (len(s), s))
        n = max(1, d["attempted"])
        d["checker_seconds_per_coordinate"] = d["checker_seconds"] / n
        d["opened_bytes_per_coordinate"] = d["opened_bytes"] / n
        if d["python_gate_n"]:
            d["python_gate_seconds_per_coordinate"] = d["python_gate_seconds"] / d["python_gate_n"]
    return by


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checks", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--python-gate", type=int, default=50, help="also run the pure-Python scalar gate on this many checks")
    ap.add_argument("--negatives", type=int, default=2, help="N02 and 15.2b cases per (family, phase)")
    ap.add_argument("--same-runtime", type=int, default=0, help="same-runtime whole-row checks per (family, phase) on the same VUs (GPU)")
    ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--skip-independent", action="store_true", help="draw the same sample but run only the same-runtime comparison (torch venv; W4b's module needs veritor/Python 3.12)")
    ap.add_argument("--negatives-only", action="store_true", help="draw the same sample but run only the negatives")
    args = ap.parse_args(argv)

    reader = CaptureReader(args.bundle)
    description = Description(reader.description)
    commitments = load_commitments(reader)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    buckets = gemm_rows(description, reader)
    sample = stratified_sample(buckets, description, args.checks, args.seed)
    cache = TreeCache(reader)
    results: list[CoordinateResult] = []
    t0 = time.perf_counter()
    for i, (cid, j, _fam, _phase) in enumerate(sample if not (args.skip_independent or args.negatives_only) else []):
        results.append(check_coordinate(description, reader, commitments, cache, cid, j, python_gate=i < args.python_gate, arch=args.arch))
        if (i + 1) % 100 == 0:
            done = sum(1 for r in results if r.status == "exact")
            print(f"[{i + 1}/{len(sample)}] exact={done} {time.perf_counter() - t0:.1f}s", flush=True)
    t_checks = time.perf_counter() - t0
    summary = summarize(results)
    statuses: dict[str, int] = defaultdict(int)
    for r in results:
        statuses[r.status] += 1
    print("statuses", dict(statuses), f"{t_checks:.1f}s", flush=True)

    # negatives per bucket
    negatives: list[dict[str, Any]] = []
    rng = random.Random(args.seed + 1)
    t_neg = time.perf_counter()
    for key in sorted(buckets):
        picks = [s for s in sample if (s[2], s[3]) == key]
        rng.shuffle(picks)
        for cid, j, _f, _p in picks[: (0 if args.skip_independent else args.negatives)]:
            negatives.append(n02_coordinate(description, reader, commitments, cid, j, cache))
            negatives.append(mutate_operand(description, reader, commitments, cid, j, "x", rng.randrange(1 << 30), cache))
            negatives.append(mutate_operand(description, reader, commitments, cid, j, "w", rng.randrange(1 << 30), cache))
        print(f"negatives {key}: {len(negatives)} done {time.perf_counter() - t_neg:.1f}s", flush=True)
    neg_summary = {
        "N02": {"attempted": sum(1 for n in negatives if n["negative_id"] == "N02"), "rejected": sum(1 for n in negatives if n["negative_id"] == "N02" and n["rejected"]), "authenticated": sum(1 for n in negatives if n["negative_id"] == "N02" and n.get("authenticated"))},
        "15.2b": {"attempted": sum(1 for n in negatives if n["negative_id"] == "15.2b" and n["status"] != "skipped"), "output_changed": sum(1 for n in negatives if n["negative_id"] == "15.2b" and n.get("output_changed")), "rejected": sum(1 for n in negatives if n["negative_id"] == "15.2b" and n.get("rejected")), "escalated": sum(1 for n in negatives if n["negative_id"] == "15.2b" and len(n.get("attempts", [])) > 1), "partner_zero": sum(1 for n in negatives if n["negative_id"] == "15.2b" and n.get("partner_zero"))},
    }
    print("negatives", neg_summary, flush=True)

    same_runtime: dict[str, Any] | None = None
    if args.same_runtime > 0:
        ids: list[str] = []
        picks_by: dict[tuple[str, str], list[str]] = {}
        for cid, _j, fam, phase in sample:
            picks_by.setdefault((fam, phase), [])
            if cid not in picks_by[(fam, phase)] and len(picks_by[(fam, phase)]) < args.same_runtime:
                picks_by[(fam, phase)].append(cid)
        for v in picks_by.values():
            ids.extend(v)
        t2 = time.perf_counter()
        rep = same_runtime_rows(description, reader, commitments, ids)
        same_runtime = {
            "seconds_total": time.perf_counter() - t2,
            "results": [r.to_json() for r in rep.results],
            "by_family": {},
        }
        for r in rep.results:
            inst_node = description.node(r.op)
            key = f"{family_of(inst_node)}/{identity.parse(r.canonical_id).phase}"
            d = same_runtime["by_family"].setdefault(key, {"attempted": 0, "exact": 0, "checker_seconds": 0.0, "input_bytes": 0, "output_bytes": 0})
            d["attempted"] += 1
            d["exact"] += int(r.status == "exact")
            d["checker_seconds"] += r.checker_seconds
            d["input_bytes"] += r.input_bytes
            d["output_bytes"] += r.output_bytes
        for d in same_runtime["by_family"].values():
            n = max(1, d["attempted"])
            d["checker_seconds_per_row"] = d["checker_seconds"] / n
            d["input_bytes_per_row"] = d["input_bytes"] / n
        print("same_runtime", {k: (v["exact"], v["attempted"], round(v["checker_seconds_per_row"], 4)) for k, v in same_runtime["by_family"].items()}, flush=True)

    doc = {
        "schema": "verity-vllm/coordinate-check/v1",
        "bundle": str(args.bundle),
        "profile_id": reader.profile_id,
        "weights_format_version": reader.weights.index.format_version,
        "weights_leaf_count": reader.weights.index.leaf_count(),
        "arch": args.arch,
        "model_version": None if args.skip_independent else NR.MODEL_VERSION,
        "skip_independent": bool(args.skip_independent),
        "seed": args.seed,
        "checks": len(results),
        "statuses": dict(statuses),
        "seconds": {"checks": t_checks, "total": time.perf_counter() - t_start},
        "by_family_phase": summary,
        "negatives_summary": neg_summary,
        "same_runtime": same_runtime,
        "buckets": {f"{k[0]}/{k[1]}": len(v) for k, v in sorted(buckets.items())},
    }
    (out_dir / "coordinate_summary.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    with open(out_dir / "coordinate_checks.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r.to_json(), sort_keys=True) + "\n")
    with open(out_dir / "coordinate_negatives.jsonl", "w") as f:
        for n in negatives:
            f.write(json.dumps(n, sort_keys=True) + "\n")
    print("RESULT", json.dumps({"checks": len(results), "statuses": dict(statuses), "negatives": neg_summary}), flush=True)
    return 0 if statuses.get("mismatch", 0) == 0 and not any(s in statuses for s in ("auth_failed",)) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Targeted negative tests N01-N07, N13, N14 (contract section 8, safeguards section 15.2/15.4).

Every test is a function of a finished capture plus its ``commitments.json`` and a fixed challenge
sample; nothing here needs vLLM or a GPU.  Mutations are applied through :class:`PatchedReader`
(an in-memory overlay on the capture: changed leaf bytes, changed index entries, substituted
weights, altered boundaries) so nothing on disk is touched and "commit the wrong value
consistently" is exactly a rebuilt tree over the overlay.

Each test returns a :class:`NegativeResult` with ``attempted``, ``rejected``,
``unexpectedly_accepted`` and a ``reproducer`` path (a JSON file naming the mutation, the ids
involved and the observed outcome).  ``attempted == rejected + unexpectedly_accepted``; variants
whose mutation is a no-op on the given capture (e.g. permuting two equal tokens) are recorded as
``inconclusive`` in the reproducer and counted in neither column.

Variants per id::

    N01  alter an opening's bytes, keep root and path                   -> authentication rejects
    N02  (result) mutate a committed output, rebuild a valid tree, keep inputs -> relation rejects
         (input_operand, 15.2b) mutate an authenticated input, rebuild   -> relation predicts
                                                                            another output
    N03  relabel: opening under another id / another tree; consistent copy of a value to another
         position                                                             -> binding rejects
    N04  reinterpret: dtype / shape in the opening; consistent dtype change in the index; a shard
         field the description does not predict                               -> schema/identity
    N05  stale buffer: second capture of an id (error and flag modes); buffer reused after add
    N06  substitute a weight without / with a rebuilt weights root, manifest unchanged
    N07  omit / duplicate a requested VU in the check set                     -> coverage rejects
    N13  change relation id, a bound value, the required set or the sample after the challenge
    N14  permute / drop protected output records (inconsistent and consistent forms)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import commit, dtypes, identity, relations
from .capture import (
    Boundary,
    CaptureError,
    CaptureReader,
    Entry,
    RunWriter,
    RUReader,
    StaleCaptureError,
    Value,
    WeightEntry,
    WeightsReader,
    _write_json,
)
from .commit import TreeCache, verify_opening_in
from .description import Description, DescriptionError

NEGATIVE_IDS = ("N01", "N02", "N03", "N04", "N05", "N06", "N07", "N13", "N14")


# --- in-memory overlay ---------------------------------------------------------------------------


class PatchedRU(RUReader):
    def __init__(self, base: RUReader, values: Mapping[str, bytes], entries: Mapping[str, Entry]):
        self.dir = base.dir
        self.request_id = base.request_id
        self.step = base.step
        self.index = base.index
        self._values_path = base._values_path
        self._mm = None
        self._base = base
        self._patch_values = dict(values)
        self._patch_entries = dict(entries)
        if entries:
            new_entries = [entries.get(e.canonical_id, e) for e in base.index.entries]
            self.index = type(base.index)(
                request_id=base.index.request_id,
                step=base.index.step,
                phase=base.index.phase,
                positions=base.index.positions,
                entries=new_entries,
                profile_id=base.index.profile_id,
                conflicts=base.index.conflicts,
                tainted=base.index.tainted,
                values_nbytes=base.index.values_nbytes,
            )
        self._entries = {e.canonical_id: e for e in self.index.entries}

    def read(self, canonical_id: str) -> bytes:
        if canonical_id in self._patch_values:
            self.entry(canonical_id)
            return self._patch_values[canonical_id]
        return self._base.read(canonical_id)

    def leaves(self, *, recompute: bool = True) -> dict[str, bytes]:
        return {v.canonical_id: v.leaf() for v in self.values()}


class PatchedWeights(WeightsReader):
    """Weights with in-memory substitutions keyed by whole-tensor id (``w=p``) or, under a
    format-v2 ``rows`` layout, by row-block id (``w=p/c=<row>``).  Leaves of a patched tensor
    are recomputed from the patched bytes; everything else comes from the base reader."""

    def __init__(self, base: WeightsReader, values: Mapping[str, bytes]):
        self.dir = base.dir
        self.index = base.index
        self._entries = base._entries
        self._values_path = base._values_path
        self._leaves_path = base._leaves_path
        self._mm = None
        self._leaves_mm = None
        self.resolver = base.resolver
        self._base = base
        self._patch = dict(values)

    def _block_patches(self, entry: WeightEntry) -> list[tuple[int, int, bytes]]:
        prefix = entry.canonical_id + "/c="
        out = []
        for k, v in self._patch.items():
            if k.startswith(prefix):
                start, end = entry.block_of(int(k[len(prefix):]))
                out.append((start, end, v))
        return out

    def _patched(self, entry: WeightEntry) -> bool:
        return entry.canonical_id in self._patch or bool(self._block_patches(entry))

    def entry_leaves(self, entry: WeightEntry, *, recompute: bool = False) -> dict[str, bytes]:
        """Patched tensors are always rehashed from their (patched) bytes; unpatched tensors keep
        the base reader's leaves (``recompute`` applies to them only when no leaves are stored),
        so a one-row mutation does not rehash every other tensor."""
        if self._patched(entry):
            return super().entry_leaves(entry, recompute=True)
        return self._base.entry_leaves(entry, recompute=recompute and self._base._stored_leaves(entry) is None)

    def _read_whole(self, entry: WeightEntry) -> bytes:
        if entry.canonical_id in self._patch:
            return self._patch[entry.canonical_id]
        raw = self._base._read_whole(entry)
        blocks = self._block_patches(entry)
        if blocks:
            buf = bytearray(raw)
            rb = entry.row_nbytes
            for start, end, v in blocks:
                if len(v) != (end - start) * rb:
                    raise CaptureError(f"{entry.canonical_id}: block patch of {len(v)} bytes for rows {start}..{end}")
                buf[start * rb : end * rb] = v
            raw = bytes(buf)
        return raw

    def _read_rows(self, entry: WeightEntry, start: int, end: int) -> bytes:
        if self._patched(entry):
            rb = entry.row_nbytes
            return self._read_whole(entry)[start * rb : end * rb]
        return self._base._read_rows(entry, start, end)


class PatchedReader(CaptureReader):
    """A capture with in-memory mutations; trees built from it are valid trees over the mutation."""

    def __init__(
        self,
        base: CaptureReader,
        *,
        values: Mapping[str, bytes] | None = None,
        entries: Mapping[str, Entry] | None = None,
        weights: Mapping[str, bytes] | None = None,
        boundaries: Mapping[str, Boundary] | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(base.run_dir)
        self._base = base
        self._patch_values = dict(values or {})
        self._patch_entries = dict(entries or {})
        self._patch_weights = dict(weights or {})
        self._patch_boundaries = dict(boundaries or {})
        if manifest is not None:
            self._manifest = dict(manifest)

    def ru(self, request_id: str, step: int) -> RUReader:
        key = (request_id, step)
        if key not in self._rus:
            base = self._base.ru(request_id, step)
            ids = {e.canonical_id for e in base.index.entries}
            values = {k: v for k, v in self._patch_values.items() if k in ids}
            entries = {k: v for k, v in self._patch_entries.items() if k in ids}
            self._rus[key] = PatchedRU(base, values, entries) if (values or entries) else base
        return self._rus[key]

    @property
    def weights(self) -> WeightsReader:
        if self._weights is None:
            base = self._base.weights
            self._weights = (
                PatchedWeights(base, self._patch_weights) if self._patch_weights else base
            )
        return self._weights

    def boundary(self, request_id: str) -> Boundary:
        if request_id in self._patch_boundaries:
            return self._patch_boundaries[request_id]
        return self._base.boundary(request_id)


# --- results -------------------------------------------------------------------------------------


@dataclass(slots=True)
class NegativeResult:
    negative_id: str
    attempted: int = 0
    rejected: int = 0
    unexpectedly_accepted: int = 0
    reproducer: str = ""
    variants: list[dict[str, Any]] = field(default_factory=list)

    @property
    def inconclusive(self) -> int:
        """Variants whose mutation could not be constructed (a no-op on this capture).

        Listed in the reproducer for transparency but not counted as attempts: a mutation that
        changes nothing observable is not applicable, so ``attempted == rejected +
        unexpectedly_accepted`` always holds and Table 1's ``rejected < attempted`` means exactly
        "something was accepted".
        """
        return sum(1 for v in self.variants if v["outcome"] == "inconclusive")

    def record(self, variant: str, outcome: str, **detail: Any) -> None:
        """``outcome`` is ``rejected``, ``accepted`` (unexpectedly) or ``inconclusive``."""
        if outcome == "rejected":
            self.attempted += 1
            self.rejected += 1
        elif outcome == "accepted":
            self.attempted += 1
            self.unexpectedly_accepted += 1
        elif outcome != "inconclusive":
            raise ValueError(outcome)
        self.variants.append({"variant": variant, "outcome": outcome, **detail})

    def to_json(self) -> dict[str, Any]:
        return {
            "negative_id": self.negative_id,
            "attempted": self.attempted,
            "rejected": self.rejected,
            "unexpectedly_accepted": self.unexpectedly_accepted,
            "reproducer": self.reproducer,
        }

    def to_reproducer(self, harness: NegativeHarness) -> dict[str, Any]:
        return {
            "negative_id": self.negative_id,
            "run_dir": str(harness.run_dir),
            "sample": list(harness.sample),
            "seed": harness.seed,
            "attempted": self.attempted,
            "rejected": self.rejected,
            "unexpectedly_accepted": self.unexpectedly_accepted,
            "inconclusive": self.inconclusive,
            "variants": self.variants,
            "rerun": (
                f"python -m verity_vllm.negatives {harness.run_dir} --only {self.negative_id} "
                f"--seed {harness.seed} --sample-size {len(harness.sample)}"
            ),
        }


# --- mutations -----------------------------------------------------------------------------------


def flip_bit(data: bytes, byte_index: int = 0, mask: int = 0x01) -> bytes:
    out = bytearray(data)
    out[byte_index % len(out)] ^= mask
    return bytes(out)


def perturb_value(value: Value, rng: np.random.Generator) -> bytes:
    """Bytes that decode to a different value of the same dtype/shape (never a NaN for floats)."""
    if value.dtype == "bfloat16":
        bits = value.array().reshape(-1).copy()
        i = int(rng.integers(len(bits)))
        # toggle the lowest mantissa bit, or the sign of a zero: both change the decoded number
        bits[i] ^= np.uint16(0x0001) if (bits[i] & 0x7F80) != 0x7F80 else np.uint16(0x8000)
        return dtypes.array_to_bytes(bits.reshape(value.shape))
    if value.dtype in ("int32", "int64", "int16", "int8", "uint8", "uint16", "uint32", "uint64"):
        arr = value.array().reshape(-1).copy()
        arr[0] = arr[0] + 1 if arr[0] < np.iinfo(arr.dtype).max else arr[0] - 1
        return dtypes.array_to_bytes(arr.reshape(value.shape))
    return flip_bit(value.data, int(rng.integers(len(value.data))), 0x01)


def perturb_input_strongly(value: Value) -> bytes:
    """A large change of an input (all signs flipped / +1 on ints), for the 15.2(b) test."""
    if value.dtype == "bfloat16":
        bits = value.array() ^ np.uint16(0x8000)
        zero = (bits & np.uint16(0x7FFF)) == 0
        bits[zero] = np.uint16(0x3F80)  # -0/+0 would not change a sum; make it 1.0
        return dtypes.array_to_bytes(bits)
    if value.dtype in ("int32", "int64"):
        arr = value.array() + 1
        return dtypes.array_to_bytes(arr)
    return bytes(b ^ 0xFF for b in value.data)


# --- the harness ---------------------------------------------------------------------------------


class NegativeHarness:
    def __init__(
        self,
        run_dir: str | os.PathLike[str],
        *,
        out_dir: str | os.PathLike[str] | None = None,
        sample: Sequence[str] | None = None,
        sample_size: int = 12,
        seed: int = 0,
        registry: Mapping[str, relations.Relation] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.reader = CaptureReader(self.run_dir)
        self.description = Description(self.reader.description)
        self.commitments = commit.load_commitments(self.reader)
        self.required = self.reader.required if self.reader.has("required.json") else None
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.registry = registry
        self.out_dir = Path(out_dir) if out_dir else self.run_dir / "negatives"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if sample is None:
            sample = relations.draw_sample(self.description, self.reader, sample_size, seed=seed)
        self.sample = list(sample)
        if not self.sample:
            raise ValueError("empty sample: nothing to test")
        self.cache = TreeCache(self.reader)
        self.challenge = relations.make_challenge(
            self.commitments, self.description, self.sample, self.required
        )

    # helpers
    def _check(
        self,
        reader: CaptureReader,
        commitments: Mapping[str, Any],
        sample: Sequence[str],
        **kw: Any,
    ) -> relations.CheckReport:
        return relations.check(
            self.description, reader, commitments, sample, registry=self.registry, **kw
        )

    def _rebuild(self, patched: CaptureReader) -> dict[str, Any]:
        return commit.build_commitments(patched, recompute_leaves=True, write=False)

    def _vus(self, limit: int) -> list[str]:
        return self.sample[:limit]

    def _instance(self, cid: str):
        c = identity.parse(cid)
        return self.description.instance_of(cid, self.reader.boundary(c.request_id or ""))

    def _leaf_inputs(self, cid: str) -> list[tuple[str, str]]:
        """(arg name, canonical id) of every leaf input of the VU (lists expanded)."""
        c = identity.parse(cid)
        boundary = self.reader.boundary(c.request_id or "")
        inst = self.description.instance_of(cid, boundary)
        out: list[tuple[str, str]] = []
        for name, ref in self.description.resolve_args(
            inst, boundary, self.reader.profile_id
        ).items():
            if isinstance(ref, dict):
                continue
            for r in ref if isinstance(ref, list) else [ref]:
                out.append((name, r))
        return out

    def _finish(self, result: NegativeResult) -> NegativeResult:
        path = self.out_dir / f"{result.negative_id}.json"
        _write_json(path, result.to_reproducer(self))
        result.reproducer = str(path)
        return result

    # --- N01 ---
    def n01(self, limit: int = 8) -> NegativeResult:
        result = NegativeResult("N01")
        for cid in self._vus(limit):
            opening = self.cache.open(cid)
            assert verify_opening_in(self.commitments, opening)
            forged = opening.replace(value=perturb_value(opening.as_value(), self.rng))
            ok = verify_opening_in(self.commitments, forged)
            result.record(
                "alter_value_keep_root_and_path",
                "accepted" if ok else "rejected",
                canonical_id=cid,
                tree=opening.tree,
            )
        return self._finish(result)

    # --- N02 ---
    def n02(self, limit: int = 8) -> NegativeResult:
        result = NegativeResult("N02")
        for cid in self._vus(limit):
            original = self.reader.value(cid)
            mutated = perturb_value(original, self.rng)
            patched = PatchedReader(self.reader, values={cid: mutated})
            commitments = self._rebuild(patched)
            report = self._check(patched, commitments, [cid])
            r = report.results[0]
            auth_ok = report.openings_verified == report.openings_attempted > 0
            if r.status == "mismatch" and auth_ok:
                outcome = "rejected"
            elif r.status == "exact":
                outcome = "accepted"
            else:
                outcome = "inconclusive"
            result.record(
                "result",
                outcome,
                canonical_id=cid,
                relation=r.relation_id,
                status=r.status,
                detail=r.detail,
                authentication=report.authentication(),
                mutated_leaf=commit.hashing.leaf_hash(
                    cid, original.dtype, original.shape, mutated
                ).hex(),
                new_root=commitments["trees"][self.reader.tree_of(cid) or ""]["root"],
            )
        # 15.2(b): mutate an authenticated input operand; the executable reference must move
        for cid in self._vus(limit):
            leaf_inputs = self._leaf_inputs(cid)
            if not leaf_inputs:
                continue
            if self.reader.value(cid).origin == "derived":
                # a derived leaf (an inactive expert slot's prescribed zero) depends on its inputs
                # only through the router's membership decision; mutating an operand it does not
                # consume is a no-op. N10 mutates the decision itself (W9).
                result.record(
                    "input_operand",
                    "inconclusive",
                    canonical_id=cid,
                    reason="derived leaf: prescribed value, operand dependence is membership only (see N10)",
                )
                continue
            name, input_id = leaf_inputs[0]
            kind = identity.parse(input_id).kind
            input_value = self.reader.value(input_id)
            strong = perturb_input_strongly(input_value)
            if kind == "boundary":
                patched = self._patch_boundary_token(input_id, input_value)
                if patched is None:
                    result.record(
                        "input_operand",
                        "inconclusive",
                        canonical_id=cid,
                        input=input_id,
                        reason="cannot mutate this boundary field",
                    )
                    continue
            elif kind == "weight":
                patched = PatchedReader(self.reader, weights={input_id: strong})
            else:
                patched = PatchedReader(self.reader, values={input_id: strong})
            commitments = self._rebuild(patched)
            report = self._check(patched, commitments, [cid])
            r = report.results[0]
            auth_ok = report.openings_verified == report.openings_attempted > 0
            if r.status == "mismatch" and auth_ok:
                outcome = "rejected"
            elif r.status == "exact":
                outcome = "accepted"  # the checker ignored its declared input
            else:
                outcome = "inconclusive"
            result.record(
                "input_operand",
                outcome,
                canonical_id=cid,
                input=input_id,
                arg=name,
                relation=r.relation_id,
                status=r.status,
                detail=r.detail,
            )
        return self._finish(result)

    def _patch_boundary_token(self, input_id: str, value: Value) -> PatchedReader | None:
        cid = identity.parse(input_id)
        boundary = Boundary.from_json(self.reader.boundary(cid.request_id or "").to_json())
        if cid.field == "prompt_token":
            assert cid.t is not None
            boundary.prompt_token_ids[cid.t] = (boundary.prompt_token_ids[cid.t] + 1) % max(
                2, self.description.symbols.get("V", 2**16)
            )
        elif cid.field == "output_token":
            assert cid.t is not None
            k = cid.t - boundary.prompt_len + 1
            new = (boundary.output_token_ids[k] + 1) % max(
                2, self.description.symbols.get("V", 2**16)
            )
            boundary.output_token_ids[k] = new
            boundary.output_records[k] = type(boundary.output_records[k])(
                t=cid.t, present=True, token_id=new, step=boundary.output_records[k].step
            )
        else:
            return None
        return PatchedReader(self.reader, boundaries={boundary.request_id: boundary})

    # --- N03 ---
    def n03(self, limit: int = 6) -> NegativeResult:
        result = NegativeResult("N03")
        vus = self._vus(limit)
        for cid in vus:
            opening = self.cache.open(cid)
            # (a) the same bytes under another id of the same op at another position
            other = self._other_position(cid)
            if other is not None:
                forged = opening.replace(canonical_id=other)
                ok = verify_opening_in(self.commitments, forged)
                result.record(
                    "relabel_id_keep_root",
                    "accepted" if ok else "rejected",
                    canonical_id=cid,
                    presented_as=other,
                )
            # (b) the opening presented under another tree's root
            other_tree = next((t for t in self.commitments["trees"] if t != opening.tree), None)
            if other_tree is not None:
                root = bytes.fromhex(self.commitments["trees"][other_tree]["root"])
                count = int(self.commitments["trees"][other_tree]["leaf_count"])
                forged = opening.replace(tree=other_tree, root=root, leaf_count=count)
                ok = verify_opening_in(self.commitments, forged)
                result.record(
                    "relabel_tree",
                    "accepted" if ok else "rejected",
                    canonical_id=cid,
                    presented_under=other_tree,
                )
        # (c) consistent relabel: copy the bytes of one position onto another and rebuild
        for cid in vus:
            other = self._other_position(cid)
            if other is None or self.reader.locate(other) is None:
                continue
            source = self.reader.value(cid)
            target = self.reader.value(other)
            if source.data == target.data or source.shape != target.shape:
                continue
            patched = PatchedReader(self.reader, values={other: source.data})
            commitments = self._rebuild(patched)
            report = self._check(patched, commitments, [other])
            r = report.results[0]
            outcome = (
                "rejected"
                if r.status == "mismatch"
                else ("accepted" if r.status == "exact" else "inconclusive")
            )
            result.record(
                "relabel_consistent",
                outcome,
                source=cid,
                target=other,
                status=r.status,
                detail=r.detail,
            )
            break
        return self._finish(result)

    def _other_position(self, cid: str) -> str | None:
        c = identity.parse(cid)
        boundary = self.reader.boundary(c.request_id or "")
        node = self.description.node(c.op or "")
        for t, phase in self.description.positions_of(node, boundary):
            if t != c.t:
                return str(c.with_(t=t, phase=phase))
        return None

    # --- N04 ---
    def n04(self, limit: int = 6) -> NegativeResult:
        result = NegativeResult("N04")
        for cid in self._vus(limit):
            opening = self.cache.open(cid)
            alt_dtype = _same_width_dtype(opening.dtype)
            if alt_dtype is not None:
                ok = verify_opening_in(self.commitments, opening.replace(dtype=alt_dtype))
                result.record(
                    "reinterpret_dtype_in_opening",
                    "accepted" if ok else "rejected",
                    canonical_id=cid,
                    dtype=opening.dtype,
                    presented_as=alt_dtype,
                )
            alt_shape = _other_shape(opening.shape)
            ok = verify_opening_in(self.commitments, opening.replace(shape=alt_shape))
            result.record(
                "reinterpret_shape_in_opening",
                "accepted" if ok else "rejected",
                canonical_id=cid,
                shape=list(opening.shape),
                presented_as=list(alt_shape),
            )
            # a shard field the description does not predict
            sharded = str(identity.parse(cid).with_(shard=0))
            ok = verify_opening_in(self.commitments, opening.replace(canonical_id=sharded))
            report = self._check(self.reader, self.commitments, [sharded])
            structural = report.results[0].status in ("structural", "missing")
            result.record(
                "reinterpret_shard",
                "rejected" if (not ok and structural) else "accepted",
                canonical_id=cid,
                presented_as=sharded,
                check_status=report.results[0].status,
            )
        # consistent reinterpretation: the index says another dtype, the tree is rebuilt
        for cid in self._vus(limit):
            entry = self.reader.ru(*self.reader.locate(cid)).entry(cid)  # type: ignore[misc]
            alt_dtype = _same_width_dtype(entry.dtype)
            if alt_dtype is None:
                continue
            new_entry = Entry(
                cid, alt_dtype, entry.shape, entry.offset, entry.nbytes, entry.origin, entry.leaf
            )
            patched = PatchedReader(self.reader, entries={cid: new_entry})
            commitments = self._rebuild(patched)
            report = self._check(patched, commitments, [cid])
            r = report.results[0]
            outcome = (
                "rejected"
                if r.status == "schema"
                else ("accepted" if r.status == "exact" else "inconclusive")
            )
            result.record(
                "reinterpret_dtype_consistent",
                outcome,
                canonical_id=cid,
                dtype=entry.dtype,
                presented_as=alt_dtype,
                status=r.status,
                detail=r.detail,
            )
            break
        return self._finish(result)

    # --- N05 ---
    def n05(self) -> NegativeResult:
        result = NegativeResult("N05")
        pid = self.reader.profile_id
        manifest = self.reader.manifest
        with tempfile.TemporaryDirectory(prefix="verity-n05-") as tmp:
            # (a) strict writer refuses a second capture with different bytes
            run = RunWriter(Path(tmp) / "a", manifest)
            cid = str(identity.value_id(pid, "n05", 0, "prefill", "embed", "embed_tokens", 0))
            ru = run.ru("n05", 0, phase="prefill", positions=[0])
            ru.add(cid, "bfloat16", [2], b"\x00\x00\x00\x00")
            try:
                ru.add(cid, "bfloat16", [2], b"\x01\x00\x00\x00")
                result.record("second_capture_strict", "accepted", canonical_id=cid)
            except StaleCaptureError as error:
                result.record(
                    "second_capture_strict", "rejected", canonical_id=cid, error=str(error)
                )
            ru.finish()
            # (b) flag mode keeps the first snapshot, taints the unit, and commitment refuses it
            run = RunWriter(Path(tmp) / "b", manifest)
            ru = run.ru("n05", 0, phase="prefill", positions=[0], on_duplicate="flag")
            ru.add(cid, "bfloat16", [2], b"\x00\x00\x00\x00")
            ru.add(cid, "bfloat16", [2], b"\x01\x00\x00\x00")
            index = ru.finish()
            stored = CaptureReader(Path(tmp) / "b").ru("n05", 0).read(cid)
            try:
                commit.build_commitments(Path(tmp) / "b", write=False)
                committed = True
            except commit.CommitError:
                committed = False
            ok = index.tainted and stored == b"\x00\x00\x00\x00" and not committed
            result.record(
                "second_capture_flagged",
                "rejected" if ok else "accepted",
                canonical_id=cid,
                tainted=index.tainted,
                first_kept=stored.hex(),
                committed=committed,
            )
            # (c) 15.4 buffer reuse: mutate the source buffer after add; the snapshot must hold
            run = RunWriter(Path(tmp) / "c", manifest)
            ru = run.ru("n05", 0, phase="prefill", positions=[0])
            buffer = np.array([0x3F80, 0x3F80], np.uint16)
            ru.add(cid, "bfloat16", [2], buffer)
            buffer[:] = 0xFFFF
            ru.finish()
            stored = CaptureReader(Path(tmp) / "c").ru("n05", 0).read(cid)
            ok = stored == b"\x80\x3f\x80\x3f"
            result.record(
                "buffer_reused_after_capture",
                "rejected" if ok else "accepted",
                canonical_id=cid,
                stored=stored.hex(),
            )
        return self._finish(result)

    # --- N06 ---
    def n06(self, limit: int = 6) -> NegativeResult:
        result = NegativeResult("N06")
        if not self.reader.has_weights():
            result.record("no_weights", "inconclusive", reason="capture has no weights index")
            return self._finish(result)
        weight_vus = [
            cid
            for cid in self.sample
            if any(identity.parse(i).kind == "weight" for _, i in self._leaf_inputs(cid))
        ]
        if not weight_vus:
            result.record("no_weight_reader", "inconclusive", reason="no sampled VU reads a weight")
            return self._finish(result)
        applicable = 0
        for cid in weight_vus:
            if applicable >= limit:
                break
            wid = next(i for _, i in self._leaf_inputs(cid) if identity.parse(i).kind == "weight")
            original = self.reader.weights.value(wid)
            substituted = perturb_input_strongly(original)
            # (a) bytes substituted, commitments unchanged -> the weight opening fails
            patched = PatchedReader(self.reader, weights={wid: substituted})
            report = self._check(patched, self.commitments, [cid])
            r = report.results[0]
            if r.status == "exact" and wid in r.inputs_unopened:
                # a declared lazy input the relation did not open: this instance's value does not
                # depend on the weight (an inactive expert slot's per-expert weights), so the
                # substitution is a no-op for this VU, not an acceptance (W9)
                result.record(
                    "substitute_weight_keep_root",
                    "inconclusive",
                    canonical_id=cid,
                    weight=wid,
                    reason="weight is a lazy input this instance never opens (value independent of it)",
                )
                continue
            applicable += 1
            result.record(
                "substitute_weight_keep_root",
                "rejected"
                if r.status == "auth_failed"
                else ("accepted" if r.status == "exact" else "inconclusive"),
                canonical_id=cid,
                weight=wid,
                status=r.status,
                detail=r.detail,
            )
            # (b) weights tree rebuilt over the substitute, manifest binding unchanged
            commitments = self._rebuild(patched)
            problems = commit.verify_source_linkage(patched, commitments)
            report = self._check(patched, commitments, [cid])
            r = report.results[0]
            rejected = bool(problems) or r.status == "mismatch"
            result.record(
                "substitute_weight_rebuild_root",
                "rejected" if rejected else "accepted",
                canonical_id=cid,
                weight=wid,
                linkage_problems=problems[:4],
                status=r.status,
            )
        return self._finish(result)

    # --- N07 ---
    def n07(self) -> NegativeResult:
        result = NegativeResult("N07")
        sample = self.sample
        report = self._check(self.reader, self.commitments, sample)
        base = relations.verify_sample_coverage(sample, report)
        if not base.ok:
            result.record("baseline", "inconclusive", reason="baseline coverage already fails")
        omitted = self._check(self.reader, self.commitments, sample[1:])
        cov = relations.verify_sample_coverage(sample, omitted)
        result.record(
            "omit_requested_vu",
            "rejected" if not cov.ok and cov.missing else "accepted",
            omitted=sample[0],
            coverage=cov.to_json(),
        )
        duplicated = self._check(self.reader, self.commitments, sample + [sample[0]])
        cov = relations.verify_sample_coverage(sample, duplicated)
        result.record(
            "duplicate_vu_in_report",
            "rejected" if cov.duplicated else "accepted",
            duplicated=sample[0],
            coverage=cov.to_json(),
        )
        cov = relations.verify_sample_coverage(sample + [sample[0]], report)
        result.record(
            "duplicate_vu_in_request",
            "rejected" if cov.duplicated else "accepted",
            duplicated=sample[0],
            coverage=cov.to_json(),
        )
        substituted = self._check(self.reader, self.commitments, sample[1:] + [sample[0]])
        substituted.results[-1].canonical_id = sample[0] + "/c=0"
        cov = relations.verify_sample_coverage(sample, substituted)
        result.record(
            "substitute_vu",
            "rejected" if cov.missing and cov.extra else "accepted",
            coverage=cov.to_json(),
        )
        return self._finish(result)

    # --- N13 ---
    def n13(self) -> NegativeResult:
        result = NegativeResult("N13")
        sample = self.sample
        challenge = self.challenge
        baseline = self._check(
            self.reader, self.commitments, sample, challenge=challenge, required_doc=self.required
        )
        if baseline.challenge_digest != challenge.digest().hex():
            result.record("baseline", "inconclusive", reason="challenge did not bind")
        # (a) change the relation of the sampled VU's node after the challenge
        cid = sample[0]
        node = self._instance(cid).node
        doc = self.description.to_json()
        for n in doc["nodes"]:
            if n["op"] == node.op:
                n["relation"] = (
                    "const.fill.v1" if n["relation"] != "const.fill.v1" else "token.equals.v1"
                )
        swapped = Description(doc)
        result.record(
            "relation_changed_after_challenge",
            self._binding_outcome(
                lambda: relations.check(
                    swapped,
                    self.reader,
                    self.commitments,
                    sample,
                    challenge=challenge,
                    required_doc=self.required,
                    registry=self.registry,
                )
            ),
            canonical_id=cid,
            node=node.op,
        )
        # (b) change a bound value (and its root) after the challenge
        patched = PatchedReader(
            self.reader, values={cid: perturb_value(self.reader.value(cid), self.rng)}
        )
        commitments = self._rebuild(patched)
        result.record(
            "bound_value_changed_after_challenge",
            self._binding_outcome(
                lambda: self._check(
                    patched, commitments, sample, challenge=challenge, required_doc=self.required
                )
            ),
            canonical_id=cid,
        )
        # (c) a different sample than the one the challenge fixed
        other = list(sample[1:]) + [sample[0]] if len(sample) > 1 else sample + [sample[0]]
        result.record(
            "sample_changed_after_challenge",
            self._binding_outcome(
                lambda: self._check(
                    self.reader,
                    self.commitments,
                    other,
                    challenge=challenge,
                    required_doc=self.required,
                )
            ),
        )
        # (d) the required set changed after the challenge
        if self.required is not None:
            shrunk = json.loads(json.dumps(self.required))
            shrunk["interfaces"] = shrunk["interfaces"][1:]
            result.record(
                "required_changed_after_challenge",
                self._binding_outcome(
                    lambda: self._check(
                        self.reader,
                        self.commitments,
                        sample,
                        challenge=challenge,
                        required_doc=shrunk,
                    )
                ),
            )
        return self._finish(result)

    @staticmethod
    def _binding_outcome(run: Callable[[], relations.CheckReport]) -> str:
        try:
            run()
        except relations.ChallengeBindingError:
            return "rejected"
        return "accepted"

    # --- N14 ---
    def n14(self) -> NegativeResult:
        result = NegativeResult("N14")
        for rid in self.reader.request_ids():
            boundary = self.reader.boundary(rid)
            present = [r for r in boundary.output_records if r.present]
            if len(present) < 2:
                continue
            # (a) permute two records, commitments unchanged -> authentication of the leaves fails.
            # Pick the first pair of produced positions whose tokens differ (swapping equal tokens
            # changes nothing and would be inconclusive).
            tokens = boundary.output_token_ids
            pair = next(
                (
                    (i, j)
                    for i in range(len(tokens))
                    for j in range(i + 1, len(tokens))
                    if tokens[i] != tokens[j]
                ),
                None,
            )
            if pair is None:
                result.record(
                    "permute_records",
                    "inconclusive",
                    request_id=rid,
                    reason="all produced tokens are equal; a permutation changes nothing",
                )
            else:
                i, j = pair
                permuted = Boundary.from_json(boundary.to_json())
                permuted.output_token_ids[i], permuted.output_token_ids[j] = tokens[j], tokens[i]
                permuted.output_records[i] = _record(permuted.output_records[i], tokens[j])
                permuted.output_records[j] = _record(permuted.output_records[j], tokens[i])
                patched = PatchedReader(self.reader, boundaries={rid: permuted})
                linkage = relations.check_output_linkage(
                    self.description, patched, self.commitments, rid
                )
                result.record(
                    "permute_records_keep_root",
                    "rejected" if not linkage.ok else "accepted",
                    request_id=rid,
                    problems=linkage.problems[:4],
                )
                # (b) consistent: boundary tree rebuilt over the permuted records
                commitments = self._rebuild(patched)
                linkage = relations.check_output_linkage(
                    self.description, patched, commitments, rid
                )
                result.record(
                    "permute_records_rebuild_root",
                    "rejected" if not linkage.ok else "accepted",
                    request_id=rid,
                    problems=linkage.problems[:4],
                )
            # (c) drop the last produced token consistently (claim a shorter output)
            dropped = Boundary.from_json(boundary.to_json())
            dropped.output_token_ids.pop()
            last = present[-1]
            k = boundary.output_records.index(last)
            dropped.output_records[k] = type(last)(
                t=last.t, present=False, token_id=None, step=None
            )
            if dropped.finish_reason == "length":
                dropped.finish_reason = "stop"
            patched = PatchedReader(self.reader, boundaries={rid: dropped})
            try:
                commitments = self._rebuild(patched)
                linkage = relations.check_output_linkage(
                    self.description, patched, commitments, rid
                )
                rejected, problems = not linkage.ok, linkage.problems[:4]
            except commit.CommitError as error:
                rejected, problems = True, [str(error)]
            result.record(
                "drop_last_record_rebuild_root",
                "rejected" if rejected else "accepted",
                request_id=rid,
                problems=problems,
            )
            # (c') ... and also drop the forward step that produced it, so the boundary is
            # internally consistent; the orphaned replay unit still exposes the claim
            if dropped.steps and dropped.steps[-1].positions == (last.t,):
                dropped.steps.pop()
                patched = PatchedReader(self.reader, boundaries={rid: dropped})
                try:
                    commitments = self._rebuild(patched)
                    linkage = relations.check_output_linkage(
                        self.description, patched, commitments, rid
                    )
                    rejected, problems = not linkage.ok, linkage.problems[:4]
                except commit.CommitError as error:
                    rejected, problems = True, [str(error)]
                result.record(
                    "drop_last_record_and_step",
                    "rejected" if rejected else "accepted",
                    request_id=rid,
                    problems=problems,
                )
            # (d) drop a record slot outright (max_tokens no longer matches)
            shorter = Boundary.from_json(boundary.to_json())
            shorter.output_records.pop()
            patched = PatchedReader(self.reader, boundaries={rid: shorter})
            try:
                commitments = self._rebuild(patched)
                linkage = relations.check_output_linkage(
                    self.description, patched, commitments, rid
                )
                rejected = not linkage.ok
            except commit.CommitError:
                rejected = True
            result.record(
                "drop_record_slot", "rejected" if rejected else "accepted", request_id=rid
            )
            break
        if not result.attempted:
            result.record("no_request_with_two_outputs", "inconclusive")
        return self._finish(result)

    # --- all ---
    def run(self, ids: Iterable[str] | None = None) -> list[NegativeResult]:
        wanted = list(ids) if ids is not None else list(NEGATIVE_IDS)
        table: dict[str, Callable[[], NegativeResult]] = {
            "N01": self.n01, "N02": self.n02, "N03": self.n03, "N04": self.n04, "N05": self.n05,
            "N06": self.n06, "N07": self.n07, "N13": self.n13, "N14": self.n14,
        }  # fmt: skip
        results = []
        for nid in wanted:
            if nid not in table:
                raise ValueError(f"{nid} is not a harness negative (have {sorted(table)})")
            results.append(table[nid]())
        _write_json(self.out_dir / "negatives.json", {
            "run_dir": str(self.run_dir),
            "sample": self.sample,
            "seed": self.seed,
            "challenge": self.challenge.to_json(),
            "results": [r.to_json() for r in results],
            "inconclusive": {r.negative_id: r.inconclusive for r in results if r.inconclusive},
        })  # fmt: skip
        return results


def _record(record, token_id: int):
    return type(record)(t=record.t, present=True, token_id=token_id, step=record.step)


def _same_width_dtype(dtype: str) -> str | None:
    table = {
        "bfloat16": "float16",
        "float16": "bfloat16",
        "int32": "float32",
        "float32": "int32",
        "int64": "float64",
        "int8": "uint8",
        "uint8": "int8",
        "float8_e4m3fn": "uint8",
    }
    return table.get(dtype)


def _other_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return (1,)
    if shape[0] % 2 == 0 and shape[0] > 0:
        return (2, shape[0] // 2) + tuple(shape[1:])
    return (1,) + tuple(shape)


def run_all(
    run_dir: str | os.PathLike[str],
    *,
    out_dir: str | os.PathLike[str] | None = None,
    sample: Sequence[str] | None = None,
    sample_size: int = 12,
    seed: int = 0,
    ids: Iterable[str] | None = None,
) -> list[NegativeResult]:
    harness = NegativeHarness(
        run_dir, out_dir=out_dir, sample=sample, sample_size=sample_size, seed=seed
    )
    return harness.run(ids)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the targeted negative tests on a capture")
    parser.add_argument("run_dir")
    parser.add_argument(
        "--out", default=None, help="where reproducers go (default <run>/negatives)"
    )
    parser.add_argument("--only", default=None, help="comma-separated negative ids")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--sample", default=None, help="JSON file with the challenge sample ids")
    args = parser.parse_args(argv)
    sample = json.loads(Path(args.sample).read_text()) if args.sample else None
    try:
        results = run_all(
            args.run_dir,
            out_dir=args.out,
            sample=sample,
            sample_size=args.sample_size,
            seed=args.seed,
            ids=args.only.split(",") if args.only else None,
        )
    except (CaptureError, commit.CommitError, DescriptionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for r in results:
        line = r.to_json()
        line["inconclusive"] = r.inconclusive
        print(json.dumps(line))
    return (
        0 if all(r.unexpectedly_accepted == 0 and r.rejected == r.attempted for r in results) else 1
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "NEGATIVE_IDS",
    "NegativeHarness",
    "NegativeResult",
    "PatchedReader",
    "run_all",
]

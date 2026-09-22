"""The attention interior a per-primitive attention VU would open -- and the refuser that says it is not committed (A5/6, 20:40Z brief).

PROPOSED layout ``attn-interior/v0`` (Q-A56-5 to the integrator / A2; NOT adopted, nothing writes it): the attention operator
``layers.<L>.self_attn.attn.attn`` at ``(request, t)`` gets, beside its output row ``port=0`` (``[NH, D]`` bf16), interior output
ports in the record's own ``op/port`` id grammar, every word in the order the registered body ``verity_vllm.registry.b1.AttentionHead``
produces it (blocks visited last -> first, per block A / B1 / B2 / B3 / B4 / B5 / C, `b1.py:335-388`):

| port | leaf | dtype | shape (T = t+1 keys, nb = ceil(T/64)) | body words |
|---|---|---|---|---|
| 1 | ``scores`` | float32 | ``[NH, T]`` | ``s[c] = DotBf16(q, k[c])`` (`b1.py:346`) |
| 2 | ``rowmax`` | float32 | ``[NH, nb, 3]`` | ``bmax`` after the chain, ``m_new``, ``m_use`` (`:347-351`) |
| 3 | ``p`` | float32 | ``[NH, T]`` | ``p[c] = MufuEx2Ftz(F32FmaSubFtz(s[c], scale, max_scaled))`` (`:357`) |
| 4 | ``pb`` | bfloat16 | ``[NH, T]`` | ``F2fpBf16(p[c])`` (`:372`) |
| 5 | ``lanes`` | float32 | ``[NH, nb, 4]`` | the four lane sums after the block (`:359-370`) |
| 6 | ``o`` | float32 | ``[NH, nb, BN/16, D]`` | the accumulator after every PV k-step (`:378-388`) |
| 7 | ``qk_acc`` | float32 | ``[NH, T, D/16]`` | the chained ``tc_dot16`` accumulators inside each ``DotBf16`` (`b1.py:99-106`) |

Size per (t, layer): 4 bytes x NH x (T x (2 + D/16) + nb x (7 + D x BN/16)) + 2 x NH x T -- at B0 (NH 9, D 64) and t = 8: 9 x
(9 x 6 + 1 x 263) x 4 + 2 x 81 = 11.6 KB; at 6500 x 1599 the census's interior total is 389 GB per RU (`data/census/report.md:22`).

Per-primitive kinds and what each reads (``committed`` = in today's RU trees; ``interior`` = one of the ports above):

* ``DotBf16_v1{K=64}|call`` -- q slice (rope port 0, committed), K row (rope port 1 at position c, committed) -> ``scores[h, c]`` (interior);
* ``AmpereBF16TcDot16_v1|batch`` -- PV step: ``o[h, b, ks-1, d]`` / zero (interior), 16 ``pb`` words (interior), 16 V words (gemm port 0,
  committed) -> ``o[h, b, ks, d]`` (interior); QK step: ``qk_acc`` (interior), q, k words (committed) -> ``qk_acc`` / ``scores`` (interior);
* ``F2fpBf16_v1|primitive`` -- ``p[h, c]`` (interior) -> ``pb[h, c]`` (interior);
* ``F32MulFtz_v1|primitive``, ``MufuEx2Ftz_v1|primitive``, ``F32FmaSubFtz_v1|primitive``, ``F32Max_v1|primitive``, ``F32AddFtz_v1|primitive`` --
  FP32 interior words in, FP32 interior word out (ports 1-3, 5, 6); no approved checker (fa2 guest only, whole-head fold).
"""

from __future__ import annotations

from dataclasses import dataclass

from .builders import BuildRefusal, Built, LocatedUnit, Record

__all__ = ["INTERIOR_KINDS", "INTERIOR_PORTS", "InteriorRefuser", "attn_op", "interior_leaf_id"]

INTERIOR_PORTS: dict[int, tuple[str, str]] = {
    1: ("scores", "float32"), 2: ("rowmax", "float32"), 3: ("p", "float32"), 4: ("pb", "bfloat16"),
    5: ("lanes", "float32"), 6: ("o", "float32"), 7: ("qk_acc", "float32"),
}


@dataclass(frozen=True)
class KindReads:
    kind: str
    interior_in: tuple[int, ...]   # interior ports read
    interior_out: int | None       # interior port written (None: the output is committed)
    committed_in: tuple[str, ...]  # committed operator rows read (op suffix:port)
    approved_checker: str | None   # family in the approved guest, or None


INTERIOR_KINDS: dict[str, KindReads] = {
    "AmpereBF16TcDot16_v1|batch": KindReads("AmpereBF16TcDot16_v1|batch", (6, 4, 7), 6, ("self_attn.qkv_proj.gemm:0", "self_attn.rotary_emb.rope:0", "self_attn.rotary_emb.rope:1"),
                                            "veritor.tensor-core.ampere_bf16_m16n8k16@2"),
    "DotBf16_v1{K=64}|call": KindReads("DotBf16_v1{K=64}|call", (), 1, ("self_attn.rotary_emb.rope:0", "self_attn.rotary_emb.rope:1"), "veritor.tc-ampere-bf16@2"),
    "F2fpBf16_v1|primitive": KindReads("F2fpBf16_v1|primitive", (3,), 4, (), "veritor.elem-f32@1"),
    "F32MulFtz_v1|primitive": KindReads("F32MulFtz_v1|primitive", (2, 6), 6, (), None),
    "MufuEx2Ftz_v1|primitive": KindReads("MufuEx2Ftz_v1|primitive", (2, 3), 3, (), None),
    "F32FmaSubFtz_v1|primitive": KindReads("F32FmaSubFtz_v1|primitive", (1, 2), 3, (), None),
    "F32Max_v1|primitive": KindReads("F32Max_v1|primitive", (1, 2), 2, (), None),
    "F32AddFtz_v1|primitive": KindReads("F32AddFtz_v1|primitive", (3, 5), 5, (), None),
}


def attn_op(layer: int) -> str:
    return f"layers.{layer}.self_attn.attn.attn"


def interior_leaf_id(record: Record, unit: LocatedUnit, port: int) -> str:
    """The canonical id the interior port would have, spelled from the committed output row's id (same prefix, other port)."""

    row = record.row(unit.request_id, unit.t, attn_op(int(unit.layer or 0)), 0)
    if row is None:
        return f"r={unit.request_id}/t={unit.t}/l={unit.layer}/op={attn_op(int(unit.layer or 0))}/port={port}"
    head, _, _ = row.canonical_id.rpartition("/port=")
    return f"{head}/port={port}"


class InteriorRefuser:
    """The registered builder for a per-primitive attention kind while the interior is uncommitted: checks what the record commits
    and refuses naming the interior leaves (and any committed row that is also absent).  A record that commits ``attn-interior/v0``
    would replace this entry with a real builder for the three approved-checker kinds; the five FTZ kinds also need a checker."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.reads = INTERIOR_KINDS[kind]

    def build(self, unit: LocatedUnit, record: Record) -> Built:
        if unit.kind != self.kind:
            raise BuildRefusal(f"builder for {self.kind!r} given a unit of kind {unit.kind!r}")
        if unit.layer is None:
            raise BuildRefusal(f"{unit.vu_id}: {self.kind} needs the located layer (attention body); the locator gave none")
        missing: list[tuple[str, str, int | None]] = []
        ru = f"ru/{unit.request_id}"
        for spec in self.reads.committed_in:
            suffix, port_s = spec.split(":")
            if record.row(unit.request_id, unit.t, f"layers.{unit.layer}.{suffix}", int(port_s)) is None:
                missing.append((ru, f"layers.{unit.layer}.{suffix}/port={port_s} at t={unit.t}", None))
        ports = tuple(sorted({*self.reads.interior_in, *(() if self.reads.interior_out is None else (self.reads.interior_out,))}))
        interior = []
        for p in ports:
            if record.row(unit.request_id, unit.t, attn_op(unit.layer), p) is None:
                cid = interior_leaf_id(record, unit, p)
                interior.append((ru, cid, None))
        checker = self.reads.approved_checker
        why = [f"{unit.vu_id} ({self.kind}, request {unit.request_id[:8]} t={unit.t} layer {unit.layer} head {unit.head}):"]
        if interior:
            why.append(f"{len(interior)} attention-interior leaf(s) not committed by the record (PROPOSED attn-interior/v0 ports {ports}): "
                       + ", ".join(f"{cid} [{INTERIOR_PORTS[int(cid.rsplit('=', 1)[1])][0]}]" for _, cid, _ in interior))
        if missing:
            why.append(f"{len(missing)} committed row(s) absent in this profile: " + ", ".join(cid for _, cid, _ in missing))
        if checker is None:
            why.append("and no checker for this kind exists in the approved guest 9fd00e60 (FTZ/MUFU gates only in the fa2 guest 33cba058, whole-head fold, `In` 16-bit)")
        if interior or missing:
            raise BuildRefusal(" ".join(why), missing=tuple(interior + missing))
        raise BuildRefusal(" ".join(why) + " the record commits the interior but the per-primitive builder for this layout is not implemented (attn-interior/v0 is PROPOSED, no record writes it)")

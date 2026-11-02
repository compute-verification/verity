"""Vocabulary-range exactness: the range values the vocabulary-range source commits for a tensor-parallel rank are the circuit's
`EmbeddingShard_v1` values, and the fused embedding kernel applied exactly those.

`acquire.sources.vocab_range_source` computes, at each `torch.ops._C.vocab_parallel_embedding` launch inside a manifest embedding site,
int32 [rows, 3] = (tok - START, START <= tok, tok <= START + VS - 1) from the launch's token ids.  One GPU run checks, per rank of a TP2
engine (vLLM, eager, batch invariant):

  (a) edge ids (0, START - 1, START, START + VS - 1, START + VS, the vocabulary's last id, and seeded random ids), through vLLM's own
      `vocab_parallel_embedding` with int32 and int64 ids inside a committed step: every committed row equals the circuit's values (`ir_values`:
      the gates `query.vocab_range.slots` names, in the `verity.ml.circuits.evaluation.evaluate_call` trace) and the kernel's output agrees (the
      row is the shard's row `local` when both range bits are set, else zero), and the output is bit-identical to the launch without the
      source;
  (b) live: the engine generating with the source installed commits one `<site>/vocab_range` per step on every rank; every committed row
      equals the circuit's values for its token id and the kernel's partial agrees with it; the generated tokens equal an engine without the
      source; the source acquires nothing outside a step and restores the patched op.

This module is the property and its record `vocab_range_exactness.json`, sealed like the other properties' (`record.seal`);
`verify_record` re-derives the digest and the verdict without a GPU.  The GPU run is `tests/properties/vocab_range_exactness_gpu.py`.
"""
from __future__ import annotations

from typing import Any

from verity_vllm.properties.record import digest_of, seal as seal_record
from verity_vllm.query import vocab_range as VR

SCHEMA = "verity-vllm/vocab-range-exactness/v1"
RECORD = "vocab_range_exactness.json"
WORLD = 2
_CALLS: dict[tuple, Any] = {}


def edge_ids(start: int, vs: int, vocab: int) -> list[int]:
    """The ids at and around the rank's range and the vocabulary's ends."""
    ids = [0, 1, start - 1, start, start + 1, start + vs - 2, start + vs - 1, start + vs, start + vs + 1, vocab - 1]
    return sorted({i for i in ids if 0 <= i < vocab})


def ir_values(vs: int, start: int, tok: int) -> tuple[int, int, int]:
    """(local, ge, le) of one token id in the circuit: evaluate `EmbeddingShard_v1{VS,H=2,START}` as a standalone Call over a zero table and read
    its committed interior gates (`query.vocab_range.slots`).  The three values depend on the id alone, not on H or the table; H = 2 is the
    smallest width whose output words are separate units reading them, as at every served width."""
    from verity.ml.circuits.evaluation.reference import standalone_call
    from verity.ml.circuits.evaluation import evaluate_call

    key = (vs, start)
    if key not in _CALLS:
        _CALLS[key] = standalone_call(VR.shard_fn(vs, 2, start))
    prog, call = _CALLS[key]
    t: dict[int, int] = {}
    evaluate_call(prog.circuit, call, [int(tok) & 0xFFFFFFFF] + [0] * (2 * vs), trace=t)
    local, ge, le = (t[call.lo + j] for j in VR.slots(vs, 2, start))
    return VR._s32(local), int(ge), int(le)


def rank_ok(r: dict[str, Any]) -> bool:
    return (not r.get("error") and int(r.get("edge_rows") or 0) > 0 and r.get("edge_ne_ir") == 0 and r.get("edge_kernel_disagrees") == 0
            and r.get("edge_out_equal") is True and r.get("reference_ne_ir") == 0)


def live_ok(v: dict[str, Any]) -> bool:
    ranks = v.get("ranks") or []
    return (not v.get("error") and len(ranks) == WORLD and v.get("tokens_equal_bare") is True
            and all(int(r.get("rows") or 0) > 0 and r.get("rows_ne_ir") == 0 and r.get("kernel_disagrees") == 0 and r.get("one_per_step") is True
                    and r.get("outside_step_acquired") == 0 and r.get("restored") is True and r.get("errors") == 0 for r in ranks)
            and len({r.get("start") for r in ranks}) == WORLD)


def _verdict(doc: dict[str, Any]) -> bool:
    ranks = doc.get("ranks") or []
    return len(ranks) == WORLD and all(map(rank_ok, ranks)) and live_ok(doc.get("live") or {})


def seal(doc: dict[str, Any]) -> dict[str, Any]:
    """`doc` with `ok` (the edge check ok on both ranks and the live check ok) and its `digest`."""
    return seal_record(doc, _verdict(doc))


def verify_record(doc: dict[str, Any]) -> list[str]:
    problems = [] if doc.get("schema") == SCHEMA else [f"schema {doc.get('schema')!r}, not {SCHEMA}"]
    if doc.get("digest") != digest_of(doc):
        problems.append("digest does not match the record")
    for r in doc.get("ranks") or []:
        if bool(r.get("ok")) != rank_ok(r):
            problems.append(f"rank {r.get('rank')}: ok={r.get('ok')} but its fields say {rank_ok(r)}")
    v = doc.get("live") or {}
    if v and bool(v.get("ok")) != live_ok(v):
        problems.append(f"live: ok={v.get('ok')} but its fields say {live_ok(v)}")
    if bool(doc.get("ok")) != _verdict(doc):
        problems.append(f"ok={doc.get('ok')} but the entries say {_verdict(doc)}")
    return problems


__all__ = ["SCHEMA", "RECORD", "WORLD", "edge_ids", "ir_values", "rank_ok", "live_ok", "seal", "verify_record"]

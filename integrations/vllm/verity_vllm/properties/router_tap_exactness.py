"""Router-tap exactness: the tapped `topk_softmax` the router-softmax source launches computes exactly the installed op's outputs, and
each row's words are the ordered router's committed values in the circuit.

`acquire.sources.router_softmax_source` launches, in place of vLLM's `_moe_C.topk_softmax`, the tapped op
`verity_router_tap.topk_softmax_tap` (`ops/pod_router_tap.sh`).  One GPU process checks, per configuration (`CONFIGS`: OLMoE's E = 64
and Qwen3-30B-A3B's E = 128, TOPK = 8, each with and without renormalize) over rows (the GPU run's edge rows -- ties, NaN, +-inf, +-0,
subnormal, overflowing and underflowing logits -- plus seeded random rows, some sprinkled with specials, in one launch of more than
one block):

  (a) the outputs (weights, ids, token_expert_indices) are bit-identical to the installed op's and to the same build with the tap off;
  (b) every word of every row is written: two tapped launches over tap buffers filled with `FILLS` agree on every word;
  (c) every word equals the ordered router's committed value (`MoeRouterTopKOrdered_v1` / `MoeRouterTopKOrderedNorm_v1`, the gate
      `query.router_softmax.slots` names for it) in the circuit's own evaluation (`ir_row`: the gate's value in the
      `verity.ml.circuits.evaluation.evaluate_call` trace), and the circuit's outputs equal the op's, on every row;
  (d) the source (`router_softmax_source`) on vLLM's routing path -- `fused_topk` in a module the source knows as an MoE layer --
      leaves the outputs bit-identical, acquires nothing outside a committed step, acquires one `moe/L<k>/router_softmax` per launch
      inside one, equal to the op's words, refuses a launch of another geometry, and restores the patched op;
  (e) live: vLLM engines (eager, batch invariant) serving tiny random-weight models of both served router shapes -- OLMoE's (E = 64,
      top-8) and Qwen3-MoE's (E = 128, top-8, renormalized), vLLM's own MoE layers -- generating with the source installed commit every
      MoE layer's softmax at every step; each committed row equals the circuit's words over the logits the router received, and the generated
      tokens equal the same engine's without the source.

This module is the property and its record `router_tap_exactness.json` (environment, the build's sha256, per-configuration results,
`ok`), whose `digest` is the sha256 of the canonical JSON of every other field; `verify_record` re-derives the digest and the verdict
without a GPU.  The GPU run is `tests/properties/router_tap_exactness_gpu.py`.
"""
from __future__ import annotations

from typing import Any

from verity_vllm.properties.record import digest_of, seal as seal_record
from verity_vllm.query import router_softmax as RS

SCHEMA = "verity-vllm/router-tap-exactness/v1"
RECORD = "router_tap_exactness.json"
#: (E, TOPK, VPT, renormalize)
CONFIGS = ((64, 8, 8, False), (64, 8, 8, True), (128, 8, 8, False), (128, 8, 8, True))
#: two tap-buffer fills (NaN payloads the kernel never writes: its NaN words are 0x7FFFFFFF): a word both launches leave at their fill
#: was not written
FILLS = (0x7FA00001, 0x7FA00002)
#: random rows per configuration besides the edge rows (the full run); `QUICK_RANDOM` for --quick
RANDOM, QUICK_RANDOM = 1000, 40


# ------------------------------------------------------------------------------------------------------------------------ the circuit
_CALLS: dict[tuple, Any] = {}


def ir_row(E: int, TOPK: int, VPT: int, renormalize: bool, logits: list[int]) -> tuple[list[int], list[int], list[int]]:
    """(tap words, weights, ids) of one row in the circuit: evaluate the ordered router as a standalone Call and read its committed interior
    gates (`query.router_softmax.slots`) from the trace."""
    from verity.ml.circuits.evaluation.reference import standalone_call
    from verity.ml.circuits.evaluation import evaluate_call

    key = (E, TOPK, VPT, renormalize)
    if key not in _CALLS:
        _CALLS[key] = standalone_call(RS.router_fn(E, TOPK, VPT, renormalize))
    prog, call = _CALLS[key]
    t: dict[int, int] = {}
    out = evaluate_call(prog.circuit, call, [int(v) & 0xFFFF for v in logits], transcript=t)
    words = [t[call.lo + j] & 0xFFFFFFFF for j in RS.slots(E, TOPK, VPT, renormalize)]
    return words, [w & 0xFFFFFFFF for w in out[:TOPK]], [i & 0xFFFFFFFF for i in out[TOPK:]]


def classify(E: int, TOPK: int, renormalize: bool, i: int) -> str:
    """The value a word of a row holds (`query.router_softmax.layout`)."""
    at = 0
    for name, n in RS.layout(E, TOPK, renormalize):
        if i < at + n:
            return name
        at += n
    raise IndexError(i)


# ------------------------------------------------------------------------------------------------------------------------ record
def config_ok(e: dict[str, Any]) -> bool:
    return (not e.get("error") and int(e.get("rows") or 0) > 0 and e.get("out_eq_installed") is True and e.get("out_eq_untapped") is True
            and e.get("unwritten") == 0 and e.get("tap_ne_ir") == 0 and e.get("out_ne_ir") == 0 and int(e.get("ir_rows") or 0) == int(e.get("rows") or 0)
            and int(e.get("words_per_row") or 0) == RS.words(int(e["E"]), int(e["TOPK"]), bool(e["renormalize"])))


def source_ok(s: dict[str, Any]) -> bool:
    return (not s.get("error") and s.get("outputs_equal") is True and s.get("outside_step_acquired") == 0 and s.get("names_ok") is True
            and s.get("words_equal") is True and s.get("refused_other_geometry") is True and s.get("restored") is True)


#: the router shapes the live check serves: (E, TOPK, renormalize) of OLMoE and of Qwen3-30B-A3B
LIVE_SHAPES = ((64, 8, False), (128, 8, True))


def live_model_ok(m: dict[str, Any]) -> bool:
    return (not m.get("error") and int(m.get("rows_checked") or 0) > 0 and m.get("rows_ne_ir") == 0 and m.get("tokens_equal_bare") is True
            and m.get("every_layer_every_step") is True and m.get("errors") == 0)


def live_ok(v: dict[str, Any]) -> bool:
    ms = v.get("models") or []
    shapes = {(int(m.get("experts") or 0), int(m.get("topk") or 0), bool(m.get("renormalize"))) for m in ms}
    return not v.get("error") and shapes >= set(LIVE_SHAPES) and all(map(live_model_ok, ms))


def _verdict(doc: dict[str, Any]) -> bool:
    cs = doc.get("configs") or []
    have = {(int(c["E"]), int(c["TOPK"]), int(c["VPT"]), bool(c["renormalize"])) for c in cs if "E" in c}
    return (have >= set(CONFIGS) and all(map(config_ok, cs)) and source_ok(doc.get("source") or {})
            and (doc.get("quick") is True or live_ok(doc.get("live") or {})))


def seal(doc: dict[str, Any]) -> dict[str, Any]:
    """`doc` with `ok` (every configuration of `CONFIGS` ok, the source check ok, and -- unless the run was --quick -- the live check ok)
    and its `digest`."""
    return seal_record(doc, _verdict(doc))


def verify_record(doc: dict[str, Any]) -> list[str]:
    """What is wrong with a record: its schema, its digest, and every `ok` re-derived from the fields beside it (empty = intact and
    consistent)."""
    problems = [] if doc.get("schema") == SCHEMA else [f"schema {doc.get('schema')!r}, not {SCHEMA}"]
    if doc.get("digest") != digest_of(doc):
        problems.append("digest does not match the record")
    for e in doc.get("configs") or []:
        if bool(e.get("ok")) != config_ok(e):
            problems.append(f"config {e.get('name')}: ok={e.get('ok')} but its fields say {config_ok(e)}")
    for key, fn in (("source", source_ok), ("live", live_ok)):
        s = doc.get(key) or {}
        if s and bool(s.get("ok")) != fn(s):
            problems.append(f"{key}: ok={s.get('ok')} but its fields say {fn(s)}")
    if bool(doc.get("ok")) != _verdict(doc):
        problems.append(f"ok={doc.get('ok')} but the entries say {_verdict(doc)}")
    return problems


__all__ = ["SCHEMA", "RECORD", "CONFIGS", "FILLS", "RANDOM", "QUICK_RANDOM", "LIVE_SHAPES", "ir_row", "classify", "config_ok", "source_ok",
           "live_model_ok", "live_ok",
           "seal", "verify_record"]

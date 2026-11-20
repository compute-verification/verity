"""The stochastic selector is REPLAYED in form (A): `GumbelTopPTokenSelect_v1{V}` has a registered
evaluator in `sampled_replay` -- the registry's own composite (TemperatureScale -> TopPMask_v1 with the Program's `Const32[S]` as its
split-count operand -> GumbelSelectF32 from the committed seed + the Const64 pos) on the committed bf16 logits row -- so a stochastic
Commit's sampled ids are recomputed from committed inputs, not excluded as 'no registered evaluator'.  CPU only, small V.

What is pinned:
  * the evaluator IS the registry walk (`evaluate_spec` on `spec_registered`), bit-equal to the difftest adapter's static-S spec;
  * the six operands resolve by NAME: logits = the committed lm_head member of this step, top_p / pos / temperature / S = the Program's
    Const32/Const64 rows (read from the spec text), seed = committed runner.sampler/seed of THIS (step, request);
  * the committed seed word is LOAD-BEARING: a different committed seed replays to a different id -> FAIL by name;
  * a missing seed identity is PARTIAL (result None, named), never a pass; a selector reading S from a Program input is unresolved by name;
  * a flipped committed sampled id fails at (request, step, runner.sampler/sampled_token_ids).
"""
from __future__ import annotations

import numpy as np

from verity_vllm.check.replay import index as IX
from verity_vllm.definitions.numerics import rows as RK

V, H, T = 64, 8, 2          # vocab, hidden (the logits Gemm's K), decode steps after the arrival step


def _f32(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


TOP_P, TEMP, S, SEED = 0.95, 0.8, 4, 0x1234_5678_9ABC
LG_MEMBER = "out"              # the lm_head Gemm's committed member (required_manifest.members_for)


def _program(splits_input: bool = False):
    """Per step: a logits Gemm (lm_head, committed `logits` member) -> Const32 top_p, Const64 pos, Const32 temperature, Const32[S] -> the
    selector wired exactly as the fold wires it (registry signature order: logits, top_p, seed, pos, temp, S).  ``splits_input``: each
    selector reads S from a `splits` Program input instead (its step's entry)."""
    rows, i = [], 0

    def add(spec, name, args):
        nonlocal i
        rows.append({"i": i, "spec": spec, "form": "call", "name": name, "args": args})
        i += 1
        return i - 1

    for s in range(1 + T):
        sfx = "" if s == 0 else f"_{s}"
        lg = add(f"Gemm_v1{{K={H},N={V}}}", f"model.logits_processor/triton_launch_{s}[0]", [[["p", 0, 0, H]], [["p", 2, 0, H * V]]])
        p = add(f"Const32[0x{_f32(TOP_P):08x}]_v1", f"gumbel_topp_token_select{sfx}/top_p", [])
        pos = add(f"Const64[0x{(100 + s):016x}]_v1", f"gumbel_topp_token_select{sfx}/pos", [])
        tm = add(f"Const32[0x{_f32(TEMP):08x}]_v1", f"gumbel_topp_token_select{sfx}/temperature", [])
        split = [["p", 3, s, s + 1]] if splits_input else [["n", add(f"Const32[0x{S:08x}]_v1", f"gumbel_topp_token_select{sfx}/splits", []), 0, 1]]
        add(f"GumbelTopPTokenSelect_v1{{V={V}}}", f"gumbel_topp_token_select{sfx}[0]",
            [[["n", lg, 0, V]], [["n", p, 0, 1]], [["p", 1, 0, 1]], [["n", pos, 0, 1]], [["n", tm, 0, 1]], split])
    params = [["prompt", f"Array<{H},Value<32>>"], ["seed", "Array<1,Value<64>>"],
              ["weights", f"{{model.model.embed_tokens.weight: Array<{V},Array<{H},Value<16>>>}}"]]
    if splits_input:
        params.append(["splits", f"Array<{1 + T},Value<32>>"])
    return {"schema": "cb-a/instance-sequence/v1", "digest": "s" * 64, "params": params, "rows": rows}


class _Store:
    """Committed words by identity (rid, engine step, op_path, member) -> word array; a missing identity reads as (None, why)."""

    def __init__(self, blocks: dict[tuple[str, int, str, str], np.ndarray]):
        self.blocks = blocks

    def has(self, rid, step, op_path, member):
        return (rid, step, op_path, member) in self.blocks

    def words(self, rid, step, op_path, member, elo, ehi):
        b = self.blocks.get((rid, step, op_path, member))
        if b is None:
            return None, f"no committed identity ({rid}, step {step}, {op_path}/{member})"
        return np.asarray(b).reshape(-1)[elo:ehi].copy(), {2: "bf16", 4: "u32", 8: "i64"}[np.asarray(b).dtype.itemsize]


def _logits(rng, seed_row):
    return (np.asarray(rng.standard_normal(V) * 3, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _honest(arrive: int = 5, rid: str = "r0", seed: int = SEED):
    """The committed world of one honest request: logits rows, seed words per step, and sampled ids = the registry's own evaluation."""
    inst = _program()
    pi = IX.ProgramIndex(inst)
    rng = np.random.default_rng(7)
    blocks: dict[tuple[str, int, str, str], np.ndarray] = {}
    lg_op = pi.describe(pi.by_i[0])["op_path"]
    assert pi.describe(pi.by_i[0])["members"] == [(LG_MEMBER, V, "bf16")]
    for s in range(1 + T):
        lg = _logits(rng, s)
        blocks[(rid, arrive + s, lg_op, LG_MEMBER)] = lg
        blocks[(rid, arrive + s, IX.SAMPLER_OP, "seed")] = np.asarray([seed], dtype=np.uint64)
        tok = RK.gumbel_topp_token_select({"V": V}, [lg, np.asarray([_f32(TOP_P)], np.uint32), np.asarray([seed], np.uint64),
                                                    np.asarray([100 + s], np.uint64), np.asarray([_f32(TEMP)], np.uint32),
                                                    np.asarray([S], np.uint32)])
        blocks[(rid, arrive + s, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)] = np.asarray([tok], dtype=np.uint64)
    return inst, pi, blocks, lg_op


def _select_rows(pi):
    return [r for r in pi.rows if r["spec"].startswith("GumbelTopPTokenSelect_v1")]


def test_const_rows_read_from_the_program_text():
    assert IX.const_word({"spec": "Const32[0x3f733333]_v1"}).tolist() == [0x3f733333]
    w = IX.const_word({"spec": "Const64[0x00000000000000ff]_v1"})
    assert w.dtype == np.uint64 and w.tolist() == [255]
    assert IX.const_word({"spec": "Gemm_v1{K=8,N=64}"}) is None and IX.const_word(None) is None
    assert IX.const_word({"spec": "Const32[0x1ffffffff]_v1"}) is None      # not a 32-bit word: unmodelled, never truncated



"""The stochastic selector is REPLAYED in form (A): `GumbelTopPTokenSelect_v1{V}` has a registered
evaluator in `sampled_replay` -- the registry's own composite (TemperatureScale -> TopPMask_v1 with the committed `splits` S as its
operand -> GumbelSelectF32 from the committed seed + the Const64 pos) on the committed bf16 logits row -- so a stochastic Commit's
sampled ids are recomputed from committed inputs, not excluded as 'no registered evaluator'.  CPU only, small V.

What is pinned:
  * the evaluator IS the registry walk (`evaluate_spec` on `spec_registered`), bit-equal to the difftest adapter's static-S spec;
  * the six operands resolve by NAME: logits = the committed lm_head member of this step, top_p / pos / temperature = the Program's
    Const32/Const64 rows (read from the spec text), seed = committed runner.sampler/seed of THIS (step, request), splits = committed
    runner.sampler/splits of the step the Program's slice names;
  * the committed seed and splits words are LOAD-BEARING: a different committed seed replays to a different id -> FAIL by name;
  * a missing splits identity is PARTIAL (result None, named), never a pass; a splits slice that names another step is unresolved by name;
  * a flipped committed sampled id fails at (request, step, runner.sampler/sampled_token_ids).
"""
from __future__ import annotations

import numpy as np
import pytest

from verity_vllm.check.replay import evaluate as EV, index as IX, population as POP
from verity_vllm.check.replay import replay_codes as RC
from verity_vllm.query.manifest import format as RM
from verity.ml.definitions.numerics import rows as RK

V, H, T = 64, 8, 2          # vocab, hidden (the logits Gemm's K), decode steps after the arrival step


def _f32(x: float) -> int:
    return int(np.float32(x).view(np.uint32))


TOP_P, TEMP, S_OF_STEP, SEED = 0.95, 0.8, [4, 2, 1], 0x1234_5678_9ABC
LG_MEMBER = "out"              # the lm_head Gemm's committed member (required_manifest.members_for)


def _program():
    """Per step: a logits Gemm (lm_head, committed `logits` member) -> Const32 top_p, Const64 pos, Const32 temperature -> the selector wired
    exactly as the fold wires it (registry signature order: logits, top_p, seed, pos, temp, splits; `splits[s]` = the step's event)."""
    rows, i = [], 0

    def add(spec, name, args):
        nonlocal i
        rows.append({"i": i, "spec": spec, "form": "call", "name": name, "args": args})
        i += 1
        return i - 1

    for s in range(1 + T):
        sfx = "" if s == 0 else f"_{s}"
        lg = add(f"Gemm_v1{{K={H},N={V}}}", f"model.logits_processor/triton_launch_{s}[0]", [[["p", 0, 0, H]], [["p", 3, 0, H * V]]])
        p = add(f"Const32[0x{_f32(TOP_P):08x}]_v1", f"gumbel_topp_token_select{sfx}/top_p", [])
        pos = add(f"Const64[0x{(100 + s):016x}]_v1", f"gumbel_topp_token_select{sfx}/pos", [])
        tm = add(f"Const32[0x{_f32(TEMP):08x}]_v1", f"gumbel_topp_token_select{sfx}/temperature", [])
        add(f"GumbelTopPTokenSelect_v1{{V={V}}}", f"gumbel_topp_token_select{sfx}[0]",
            [[["n", lg, 0, V]], [["n", p, 0, 1]], [["p", 1, 0, 1]], [["n", pos, 0, 1]], [["n", tm, 0, 1]], [["p", 2, s, s + 1]]])
    params = [["prompt", f"Array<{H},Value<32>>"], ["seed", "Array<1,Value<64>>"], ["splits", f"Array<{1 + T},Value<32>>"],
              ["weights", f"{{model.model.embed_tokens.weight: Array<{V},Array<{H},Value<16>>>}}"]]
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


def _honest(arrive: int = 5, rid: str = "r0", seed: int = SEED, splits=S_OF_STEP):
    """The committed world of one honest request: logits rows, seed / splits words per step, and sampled ids = the registry's own evaluation."""
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
        blocks[(rid, arrive + s, IX.SAMPLER_OP, "splits")] = np.asarray([splits[s]], dtype=np.uint32)
        tok = RK.gumbel_topp_token_select({"V": V}, [lg, np.asarray([_f32(TOP_P)], np.uint32), np.asarray([seed], np.uint64),
                                                    np.asarray([100 + s], np.uint64), np.asarray([_f32(TEMP)], np.uint32),
                                                    np.asarray([splits[s]], np.uint32)])
        blocks[(rid, arrive + s, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)] = np.asarray([tok], dtype=np.uint64)
    return inst, pi, blocks, lg_op


def _select_rows(pi):
    return [r for r in pi.rows if r["spec"].startswith("GumbelTopPTokenSelect_v1")]


def test_the_stochastic_selector_has_a_registered_evaluator_equal_to_the_registry_difftest_walk():
    assert "GumbelTopPTokenSelect_v1" in RK.EVALUATORS and "GumbelTopPTokenSelect_v1" not in RC.NOT_YET
    from verity_vllm.properties.admission import evaluate_spec
    from verity.ml.definitions.registry import sampling_topp_difftest as DT
    rng = np.random.default_rng(11)
    lg = _logits(rng, 0)
    for S in (1, 2, 4, 8):
        ops = [lg, np.asarray([_f32(TOP_P)], np.uint32), np.asarray([SEED], np.uint64), np.asarray([9], np.uint64),
               np.asarray([_f32(TEMP)], np.uint32), np.asarray([S], np.uint32)]
        got = EV.evaluate("GumbelTopPTokenSelect_v1", {"V": V}, [[o] for o in ops], [None] * 6)
        ref = evaluate_spec(DT.spec({"V": V, "S": S}), [lg.astype(np.uint64)] + [o.astype(np.uint64) for o in ops[1:5]])
        assert list(got) == [RK.SAMPLED_MEMBER] and got[RK.SAMPLED_MEMBER].dtype == np.uint64
        assert int(got[RK.SAMPLED_MEMBER][0]) == int(ref[0]) & 0xFFFFFFFF
    with pytest.raises(ValueError, match="6 operands"):
        RK.gumbel_topp_token_select({"V": V}, ops[:5])
    with pytest.raises(ValueError, match="operand logits"):
        RK.gumbel_topp_token_select({"V": V}, [lg[:-1]] + ops[1:])


def test_const_rows_read_from_the_program_text():
    assert IX.const_word({"spec": "Const32[0x3f733333]_v1"}).tolist() == [0x3f733333]
    w = IX.const_word({"spec": "Const64[0x00000000000000ff]_v1"})
    assert w.dtype == np.uint64 and w.tolist() == [255]
    assert IX.const_word({"spec": "Gemm_v1{K=8,N=64}"}) is None and IX.const_word(None) is None
    assert IX.const_word({"spec": "Const32[0x1ffffffff]_v1"}) is None      # not a 32-bit word: unmodelled, never truncated


def test_honest_stochastic_request_replays_every_selector_from_committed_inputs_by_name():
    inst, pi, blocks, lg_op = _honest()
    store = _Store(blocks)
    for s, r in enumerate(_select_rows(pi)):
        d = pi.describe(r)
        assert d["op_path"] == IX.SAMPLER_OP and d["members"] == [(RK.SAMPLED_MEMBER, 1, "i64")] and d["step"] == s
        rec = EV.replay_vu(pi, store, lambda *_a: None, {"i": r["i"], "phase": "decode", "request_id": "r0"}, arrive=5)
        assert rec["result"] is True, rec
        kinds = [(x["kind"], x.get("identity", [None, None, None, None])[2:4] if x["kind"] == "committed" else x.get("spec")) for x in rec["inputs"]]
        assert kinds == [("committed", [lg_op, LG_MEMBER]), ("const", f"Const32[0x{_f32(TOP_P):08x}]_v1"),
                         ("committed", [IX.SAMPLER_OP, "seed"]), ("const", f"Const64[0x{(100 + s):016x}]_v1"),
                         ("const", f"Const32[0x{_f32(TEMP):08x}]_v1"), ("committed", [IX.SAMPLER_OP, "splits"])]
        # the seed and splits identities are THIS step's (engine step = arrive + s), request-local element 0
        assert [x["identity"][1] for x in rec["inputs"] if x["kind"] == "committed"] == [5 + s] * 3
        assert rec["outputs"][0]["committed"] == rec["outputs"][0]["evaluated"] == int(blocks[("r0", 5 + s, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)][0])


def test_a_flipped_committed_sampled_id_fails_named_at_the_sampler_identity():
    inst, pi, blocks, _ = _honest()
    r = _select_rows(pi)[1]
    key = ("r0", 6, IX.SAMPLER_OP, RK.SAMPLED_MEMBER)
    blocks[key] = (blocks[key] + 1) % V
    rec = EV.replay_vu(pi, _Store(blocks), lambda *_a: None, {"i": r["i"], "phase": "decode", "request_id": "r0"}, arrive=5)
    assert rec["result"] is False
    assert "runner.sampler/sampled_token_ids" in rec["why"] and "step 1" in rec["why"]


def test_the_committed_seed_and_splits_words_are_load_bearing():
    """A different committed seed (the sampler consumed another seed than the Program's) replays to a different id at some step -> FAIL
    by name; the committed sampled ids stay the honest ones.  Likewise the committed S: TopPMask_v1's split geometry is the operand."""
    inst, pi, blocks, _ = _honest()
    honest = {k: v for k, v in blocks.items()}
    for s in range(1 + T):
        blocks[("r0", 5 + s, IX.SAMPLER_OP, "seed")] = np.asarray([SEED ^ 0xFFFF], dtype=np.uint64)
    res = [EV.replay_vu(pi, _Store(blocks), lambda *_a: None, {"i": r["i"], "phase": "decode", "request_id": "r0"}, arrive=5)
           for r in _select_rows(pi)]
    assert all(x["result"] is not None for x in res)
    assert any(x["result"] is False for x in res), [x["outputs"] for x in res]
    bad = next(x for x in res if x["result"] is False)
    assert bad["outputs"][0]["committed"] != bad["outputs"][0]["evaluated"] and "runner.sampler/sampled_token_ids" in bad["why"]
    # honest words again -> all pass (the failure above was the seed, nothing else)
    assert all(EV.replay_vu(pi, _Store(honest), lambda *_a: None, {"i": r["i"], "phase": "decode", "request_id": "r0"}, arrive=5)["result"]
               for r in _select_rows(pi))


def test_a_missing_splits_identity_is_partial_by_name_never_a_pass():
    inst, pi, blocks, _ = _honest()
    del blocks[("r0", 7, IX.SAMPLER_OP, "splits")]
    r = _select_rows(pi)[2]
    rec = EV.replay_vu(pi, _Store(blocks), lambda *_a: None, {"i": r["i"], "phase": "decode", "request_id": "r0"}, arrive=5)
    assert rec["result"] is None and "runner.sampler/splits" in rec["why"] and "step 7" in rec["why"]
    del blocks[("r0", 6, IX.SAMPLER_OP, "seed")]
    r = _select_rows(pi)[1]
    rec = EV.replay_vu(pi, _Store(blocks), lambda *_a: None, {"i": r["i"], "phase": "decode", "request_id": "r0"}, arrive=5)
    assert rec["result"] is None and "runner.sampler/seed" in rec["why"] and "step 6" in rec["why"]


def test_a_splits_slice_naming_another_step_is_unresolved_by_name():
    inst = _program()
    sel = [r for r in inst["rows"] if r["spec"].startswith("GumbelTopPTokenSelect_v1")][2]
    sel["args"][5] = [["p", 2, 0, 1]]                                            # step 2's selector reading splits[0]
    pi = IX.ProgramIndex(inst)
    _, _, blocks, _ = _honest()
    rec = EV.replay_vu(pi, _Store(blocks), lambda *_a: None, {"i": sel["i"], "phase": "decode", "request_id": "r0"}, arrive=5)
    assert rec["result"] is None and "splits slice [0, 1) names step 0" in rec["why"] and "step 2" in rec["why"]


def test_the_selector_is_a_replayed_stratum_of_the_population_not_a_no_evaluator_gap():
    inst, pi, blocks, _ = _honest()
    pop = POP.population(pi, "r0", arrive=5) if hasattr(POP, "population") else None
    if pop is None:
        pytest.skip("population API not exposed by name")
    vus, not_ev = pop[0], pop[1]
    fams = {RM.spec_family(pi.by_i[v["i"]]["spec"]) for v in vus}
    assert "GumbelTopPTokenSelect_v1" in fams
    assert not any("GumbelTopPTokenSelect_v1" in k for k in not_ev)

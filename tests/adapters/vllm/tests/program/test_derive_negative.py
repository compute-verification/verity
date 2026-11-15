"""Negative tests of the importer that read the vLLM adapter's served Definitions: the cache-ownership oracle (the K/V rows
an attention call reads must be produced by THIS request at positions <= t) on a hand-built and a derived `Serve`, and the
capture-to-program correspondence primitives. The importer's plain-PyTorch negatives are the PyTorch adapter's
`tests/frontend/test_derive_negative.py`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from verity_torch.frontend.provenance import registry_version
from verity_vllm.definitions.registry import b1
from verity_vllm.definitions.registry.prims import BF16, I32
from verity_vllm.program.frontend import vllm_derive as _D

from tests.program.capture_check import LazyTranscript, Segments, check_call
from verity.circuits import Program, bind, program_digest
from verity.circuits.evaluation import evaluate_region
from verity.circuits.refs import Coll, concat
from verity.circuits.types import Array


def _input_gate_ranges(P: Program) -> list[tuple[int, int]]:
    out = []
    for n_idx, n in enumerate(P.root.body.nodes):
        if getattr(n.fn, "family", None) == "Input":
            lo = P.root.body.offsets[n_idx]
            out.append((lo, lo + n.gates))
    return out


# -- cache ownership: K/V rows an attention call reads must be produced by THIS request at positions <= t --

TINY: dict[str, Any] = {
    "model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
    "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 8, "BIAS": True, "tie_word_embeddings": True, "profile": {},
}


_LOS_CACHE: dict[int, list[int]] = {}


def _producer_root_call(C, g: int, roots: list):
    import bisect
    los = _LOS_CACHE.get(id(roots))
    if los is None or len(los) != len(roots):
        los = _LOS_CACHE[id(roots)] = [c.lo for c in roots]
    i = bisect.bisect_right(los, g) - 1
    if i < 0 or not (roots[i].lo <= g < roots[i].hi):
        return None
    return roots[i]


def check_cache_ownership(P: Program) -> list[str]:
    """For every root call named `t=<t>/layers.<L>/attn`: its K/V argument gates must be produced by the calls
    `t=<i>/layers.<L>/pre` for i in 0..t (same layer, this request, no future position) or be Input gates
    (never: KV rows are computed). Returns the list of violations (empty = owned correctly)."""
    C = P.circuit
    roots = sorted(C.root_calls(), key=lambda c: c.lo)
    named = {c.node.name: c for c in roots if c.node.name}
    bad: list[str] = []
    import re
    for name, call in named.items():
        m = re.match(r"^(?:r=(?P<r>[^/]+)/)?t=(?P<t>\d+)/layers\.(?P<L>\d+)/attn$", name)
        if not m:
            continue
        t, L, r = int(m["t"]), int(m["L"]), m["r"]
        prefix = f"r={r}/" if r else ""
        args = call.args
        nq = call.arg_types()[0].leaves
        kv = args[nq:]
        T = call.fn.bindings.get("T") if hasattr(call.fn, "bindings") else None
        if T is not None and t + 1 != T:
            bad.append(f"{name}: attends over T={T} rows at position t={t} (expected t+1={t + 1})")
        owners = set()
        for g in kv:
            pc = _producer_root_call(C, g, roots)
            if pc is None:
                bad.append(f"{name}: K/V gate {g} is not produced by any root call (Input?)")
                continue
            owners.add(pc.node.name)
        want = {f"{prefix}t={i}/layers.{L}/pre" for i in range(t + 1)}
        extra = owners - want
        missing = want - owners
        if extra:
            bad.append(f"{name}: K/V rows read from {sorted(extra)[:4]} (other layer / future position / other request)")
        if missing:
            bad.append(f"{name}: K/V rows of {sorted(missing)[:4]} are not read (a position is skipped)")
    return bad


def test_cache_ownership_oracle_accepts_hand_built_serve():
    P = Program(bind(b1.Serve, C=TINY, LP=2, STEPS=1))
    assert check_cache_ownership(P) == []


def _miswired_serve(kind: str):
    """A Serve-shaped program whose decode attention reads K/V rows from the wrong owner: `layer` (rows of layer
    L+1), `future` (the row this position itself produces is replaced by the next step's row), `stale` (a
    position is dropped)."""
    from verity_vllm.definitions.registry.b1 import (
        Embedding,
        Final,
        LayerPost,
        LayerPre,
        kv_type,
    )

    from verity.circuits.defs import CompositeDefinition
    C, LP = TINY, 2

    def body(B, S, prompt, weights):
        NL, Hh = C["NL"], C["H"]
        hs = [B.call(bind(Embedding, V=C["V"], H=Hh), prompt[i], weights["embed"], name=f"t={i}/embed") for i in range(LP)]
        res = [None] * LP
        pre_by_layer = []
        for L in range(NL):
            w = weights["layers"][L]
            pre = [B.call(bind(LayerPre, C=C, L0=(L == 0)), hs[i], hs[i] if L == 0 else res[i], w, weights["cos_sin"][i],
                          name=f"t={i}/layers.{L}/pre") for i in range(LP)]
            pre_by_layer.append(pre)
            for i in range(LP):
                T = i + 1
                if kind == "layer" and L == NL - 1 and i == LP - 1:
                    src = pre_by_layer[L - 1]  # the previous layer's rows
                elif kind == "future" and i == 0:
                    src = [pre[1], pre[1]][:1]  # position 0 reads position 1's row instead of its own
                elif kind == "stale" and i == LP - 1:
                    src = pre[1:]  # drops position 0
                    T = LP - 1
                else:
                    src = pre[:T]
                kc = Coll(kv_type(C, T), concat([p[2].refs for p in src[:T]]))
                vc = Coll(kv_type(C, T), concat([p[3].refs for p in src[:T]]))
                attn = B.call(bind(b1.Attention, T=T, NH=C["NH"], KVH=C["KVH"], D=C["D"]), pre[i][1], kc, vc, name=f"t={i}/layers.{L}/attn")
                post = B.call(bind(LayerPost, C=C), attn, pre[i][0], w, name=f"t={i}/layers.{L}/post")
                hs[i], res[i] = post[0], post[1]
        return B.call(bind(Final, C=C), hs[-1], res[-1], weights["norm"], weights["embed"], name=f"t={LP - 1}/final")

    D = CompositeDefinition(f"MiswiredServe_{kind}", 1, (), lambda S: ((("prompt", Array(LP, I32)), ("weights", b1.weights_type(C))), I32),
                            body, register=False)
    return Program(bind(D))


@pytest.mark.parametrize("kind", ["layer", "future", "stale"])
def test_cache_ownership_oracle_rejects_miswired_kv(kind):
    P = _miswired_serve(kind)
    bad = check_cache_ownership(P)
    assert bad, f"mis-wired ({kind}) K/V accepted by the ownership check"


# -- the same property without names: a derived FLAT program carries no `t=/layers./attn` names (hints only) --

def _base_name(fn) -> str:
    d = getattr(fn, "defn", fn)
    return getattr(d, "name", None) or str(getattr(fn, "id", fn)).split("_v")[0]


def _kv_rows(call):
    """(k_rows, v_rows) of an `Attention{T,...}` call: per cached position, the tuple of its argument gates."""
    types = call.arg_types()
    nq = types[0].leaves
    kt, vt = types[1], types[2]
    T, per = kt.n, kt.leaves // kt.n
    args = call.args
    k = args[nq:nq + kt.leaves]
    v = args[nq + kt.leaves:nq + kt.leaves + vt.leaves]
    return [tuple(k[i * per:(i + 1) * per]) for i in range(T)], [tuple(v[i * per:(i + 1) * per]) for i in range(T)]


def check_cache_ownership_by_kind(P: Program, NL: int, LP: int, STEPS: int) -> list[str]:
    """Name-independent K/V ownership. Every root call of kind `Attention` reads T cached rows of K and of V; each
    row must be produced by a root call (never an Input / unowned gate), rows of one call must have pairwise distinct
    producers (no duplicated or dropped position), and the calls must form exactly NL chains in which each call's
    first T-1 rows are EXACTLY the rows the previous call of that chain read (the cache is extended, never replaced,
    never borrowed from another layer / request / future position), covering positions 1..LP+STEPS. Returns the
    violations; an empty attention selection is itself a violation (boundary 6)."""
    C = P.circuit
    roots = sorted(C.root_calls(), key=lambda c: c.lo)
    attn = [c for c in roots if _base_name(c.fn) == "Attention"]
    bad: list[str] = []
    if len(attn) != NL * (LP + STEPS):
        bad.append(f"{len(attn)} Attention calls at root; expected NL*(LP+STEPS) = {NL * (LP + STEPS)}")
    if not attn:
        return bad

    def prod(row):
        s = set()
        for g in row:
            pc = _producer_root_call(C, g, roots)
            if pc is None:
                return None
            s.add(pc.lo)
        return frozenset(s)

    chains: list[list[tuple]] = []
    for c in attn:
        krows, vrows = _kv_rows(c)
        T = len(krows)
        kp, vp = [prod(r) for r in krows], [prod(r) for r in vrows]
        if any(p is None for p in kp + vp):
            bad.append(f"Attention@{c.lo} (T={T}): a K/V row is not produced by any root call (Input or unowned)")
            continue
        if len(set(kp)) != T or len(set(vp)) != T:
            bad.append(f"Attention@{c.lo} (T={T}): K/V rows share a producer (a position is duplicated / dropped)")
        for ch in chains:
            Tl, kl, vl = ch[-1]
            if Tl < T and kl == kp[:Tl] and vl == vp[:Tl]:
                ch.append((T, kp, vp))
                break
        else:
            if T != 1:
                bad.append(f"Attention@{c.lo} (T={T}): no earlier attention of any layer read exactly its first {T - 1} rows "
                           f"(rows borrowed from another layer / request / future position)")
            chains.append([(T, kp, vp)])
    if len(chains) != NL:
        bad.append(f"{len(chains)} K/V chains (layers); expected NL={NL}")
    for k, ch in enumerate(chains):
        Ts = [t for t, _, _ in ch]
        if Ts != list(range(1, LP + STEPS + 1)):
            bad.append(f"chain {k} attends over T={Ts}; expected 1..{LP + STEPS}")
    return bad


def test_cache_ownership_by_kind_accepts_hand_built_serve():
    P = Program(bind(b1.Serve, C=TINY, LP=2, STEPS=1))
    assert check_cache_ownership_by_kind(P, TINY["NL"], 2, 1) == []


@pytest.mark.parametrize("kind", ["layer", "future", "stale"])
def test_cache_ownership_by_kind_rejects_miswired_kv(kind):
    bad = check_cache_ownership_by_kind(_miswired_serve(kind), TINY["NL"], 2, 1)
    assert bad, f"mis-wired ({kind}) K/V accepted by the name-independent ownership check"


def _derive_examples():
    return pytest.importorskip("verity_vllm.program.frontend.derive_examples", reason="DERIVE's derive_examples.py not merged yet")


def _derive_serve(steps: int = 1, **kw):
    dex = _derive_examples()
    return _D.derive_report(dex.GreedyServe(TINY, steps, **kw), (torch.zeros(2, dtype=torch.int32),), mode="profile", profile="b1-eager", config=TINY)


def test_derived_tiny_serve_owns_its_kv_cache_and_matches_the_registered_program():
    """The derived program of DERIVE's tiny `GreedyServe` (vLLM-iface attention, KV as forward-context state):
    K/V ownership holds by NAME (if the family matched) and by KIND (always); and if the family rule claims
    `Serve_v1{C,LP,STEPS}`, the digest equals the hand-built program's (an independent check of that claim)."""
    rep = _derive_serve(1)
    if not rep.ok:
        pytest.skip(f"tiny GreedyServe does not derive: {rep.unsupported}")
    P = rep.program
    assert check_cache_ownership_by_kind(P, TINY["NL"], 2, 1) == []
    hand = Program(bind(b1.Serve, C=TINY, LP=2, STEPS=1))
    if rep.family and rep.family.get("matched"):
        assert check_cache_ownership(P) == []
        assert rep.digest == program_digest(hand), "family rule claimed Serve_v1 but the digest differs from the registered program's"
        assert P.gates == hand.gates
    else:
        assert rep.digest != program_digest(hand)


def test_a_different_circuit_is_not_relabelled_as_serve():
    """Boundary 4: `final_norm_all_rows=True` (vLLM's model forward — final norm and lm_head over EVERY prefill row,
    of which only the last is sampled) is a legitimately different circuit with extra dead work. It must derive
    under its OWN identity, never be relabelled `Serve_v1`, and still own its KV cache."""
    rep = _derive_serve(1, final_norm_all_rows=True)
    if not rep.ok:
        pytest.skip(f"all-rows GreedyServe does not derive: {rep.unsupported}")
    hand = Program(bind(b1.Serve, C=TINY, LP=2, STEPS=1))
    assert not (rep.family and rep.family.get("matched")), "a different circuit was relabelled Serve_v1"
    assert rep.digest != program_digest(hand)
    assert rep.gates != hand.gates
    assert check_cache_ownership_by_kind(rep.program, TINY["NL"], 2, 1) == []
    why = rep.provenance.get("families_not_matched") or []
    assert any("Serve" in str(f.get("rule", "")) for f in why), "the family rule must record WHY Serve_v1 was not claimed"


# ---------------------------------------------------------------------------------------------------------
# 6. capture-to-program mappings, corrupted values, missing obligations, mismatched versions
#    (pure verity.circuits: the correspondence primitives that any derive_corr.py must rest on; the derive_corr.py
#     specific tests are appended when CAPCORR's script lands)
# ---------------------------------------------------------------------------------------------------------


def _segments(d: dict[int, int]) -> Segments:
    """A value source over an explicit gate->word dict (stands in for weights + captured leaves), one segment
    per maximal run of consecutive gates."""
    s = Segments()
    ks = sorted(d)
    i = 0
    while i < len(ks):
        j = i
        while j + 1 < len(ks) and ks[j + 1] == ks[j] + 1:
            j += 1
        s.add(ks[i], np.array([d[g] for g in ks[i:j + 1]], dtype=np.uint32), {"leaf_id": f"xfer:{ks[i]}"})
        i = j + 1
    return s


def _tiny_program_with_truth():
    """A root call of `BiasAdd{N=8}` over prescribed inputs, fully evaluated: (program, truth transcript, the call)."""
    from verity.circuits.defs import CompositeDefinition
    W = CompositeDefinition("XferWrapBiasAdd", 1, (), lambda S: ((("y", Array(8, BF16)), ("b", Array(8, BF16))), Array(8, BF16)),
                            lambda B, S, y, b: B.call(bind(b1.BiasAdd, N=8), y, b, name="ba"), register=False)
    P = Program(bind(W))
    C = P.circuit
    rng = np.random.default_rng(0)
    truth: dict[int, int] = {}
    for lo, hi in _input_gate_ranges(P):
        for g in range(lo, hi):
            truth[g] = int(rng.integers(0x3C00, 0x4000))  # bf16 words in [1, 2)
    evaluate_region(C, C[0:C.size], truth)
    call = [c for c in C.root_calls() if not c.is_primitive][0]
    return P, truth, call


def _check(call, gates, source: dict[int, int], captured: np.ndarray) -> dict:
    return check_call(call, gates, LazyTranscript(_segments(source)), captured)


def test_correspondence_exact_on_true_values():
    P, truth, call = _tiny_program_with_truth()
    gates = call.returns
    inputs_only = {g: v for g, v in truth.items() if any(lo <= g < hi for lo, hi in _input_gate_ranges(P))}
    r = _check(call, gates, inputs_only, np.array([truth[g] for g in gates], dtype=np.uint16))
    assert r["status"] == "exact" and r["mismatches"] == 0, r


def test_corrupted_captured_value_fails():
    P, truth, call = _tiny_program_with_truth()
    gates = call.returns
    inputs_only = {g: v for g, v in truth.items() if any(lo <= g < hi for lo, hi in _input_gate_ranges(P))}
    captured = np.array([truth[g] for g in gates], dtype=np.uint16)
    captured[3] ^= 0x0001  # one flipped mantissa bit in one word
    r = _check(call, gates, inputs_only, captured)
    assert r["status"] == "mismatch" and r["mismatches"] == 1 and r["first_mismatches"][0]["word"] == 3, r


def test_wrong_address_map_fails():
    """A capture mapped to the wrong gates (an address map off by one call / a permutation) must not verify."""
    P, truth, call = _tiny_program_with_truth()
    gates = call.returns
    inputs_only = {g: v for g, v in truth.items() if any(lo <= g < hi for lo, hi in _input_gate_ranges(P))}
    true_words = np.array([truth[g] for g in gates], dtype=np.uint16)
    rolled = np.roll(true_words, 1)  # right values, wrong positions
    assert not np.array_equal(rolled, true_words)
    r = _check(call, gates, inputs_only, rolled)
    assert r["status"] == "mismatch" and r["mismatches"] > 0, r
    # the map points at Input gates instead of the call's returns: shape/identity check must fail closed
    wrong_gates = tuple(range(call.lo - 8, call.lo))  # the bias inputs, not the outputs
    r2 = _check(call, wrong_gates, inputs_only, true_words)
    assert r2["status"] != "exact" or r2["words_compared"] == 0, r2


def test_corrupted_prescribed_input_fails():
    P, truth, call = _tiny_program_with_truth()
    gates = call.returns
    inputs_only = {g: v for g, v in truth.items() if any(lo <= g < hi for lo, hi in _input_gate_ranges(P))}
    k = min(inputs_only)
    inputs_only[k] ^= 0x0100  # a corrupted weight word
    r = _check(call, gates, inputs_only, np.array([truth[g] for g in gates], dtype=np.uint16))
    assert r["status"] == "mismatch", r


def test_missing_operand_is_reported_not_invented():
    """A value the record does not carry (and nothing recomputes) is `missing-operand`, never silently zero."""
    P, truth, call = _tiny_program_with_truth()
    gates = call.returns
    partial = {g: v for g, v in truth.items() if any(lo <= g < hi for lo, hi in _input_gate_ranges(P))}
    del partial[min(partial)]
    r = _check(call, gates, partial, np.array([truth[g] for g in gates], dtype=np.uint16))
    assert r["status"] == "missing-operand", r


def test_wrong_dtype_or_shape_in_record_fails_shape_check():
    from tests.program.capture_check import shape_ok as _shape_ok
    P, truth, call = _tiny_program_with_truth()
    ok, _ = _shape_ok(call, call.returns, {"shape": [8], "dtype": "bfloat16"})
    assert ok
    ok, why = _shape_ok(call, call.returns, {"shape": [8], "dtype": "float32"})
    assert not ok and "dtype" in why
    ok, why = _shape_ok(call, call.returns, {"shape": [4], "dtype": "bfloat16"})
    assert not ok and "words" in why


def test_program_identity_differs_between_versions_and_statics():
    """A record names a program by digest; any tool must refuse a record whose digest is not the program's.
    Here: the digest separates statics and registry versions (the mismatch the tool must detect)."""
    d8 = program_digest(Program(bind(b1.BiasAdd, N=8)))
    d16 = program_digest(Program(bind(b1.BiasAdd, N=16)))
    assert d8 != d16
    rv = registry_version()
    assert rv["digest"] and rv["primitives"] > 0
    # the registry-version record is what pins the primitive vocabulary: two records that disagree on it
    # are not comparable even when the program digest agrees (frontend §8.1)
    rec_a = {"program": d8, "registry": rv["digest"]}
    rec_b = {"program": d8, "registry": "0" * 64}
    assert rec_a["program"] == rec_b["program"] and rec_a["registry"] != rec_b["registry"]

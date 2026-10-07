"""Stage DENSE B1, Program side (CPU) — **padded baseline (experimental)**, lane `pad` (R16, BRIEF §7B).

The lifted fixed Program of one real-shape B1 request — SmolLM2-135M (`B0_CONFIG`, the registry's expansion oracle, the model of the
`smollm2-135m__…__b1__i256__o32` panel row) under the D54-corrected FA2 record (`attention_fa2(64)`, `Serve_v4`'s binding record) at
`prompt_len = 256`, `max_tokens = cap = 32` (`D_MAX = cap − 1 = 31` decode steps, SPEC §4 [R3]) with a declared EOS set — built from the
manifest alone and fixed before any execution.  What this module establishes (LIFTING-REVIEW T-1, T-3(ii) arithmetic, PAD-03, F-pad-06):

  * the builder consumes `(C, prompt_len, cap, eos_set)` and the registered Definitions only; a served length is refused by name;
  * the Program, its gate count and digest are functions of the manifest (differ with `cap`, `prompt_len`, `eos_set`; never with a stop);
  * PAD-03 address stability **modulo the guard nodes** against the frozen `Serve_v2{B0, LP=256, STEPS=31}`: same gates plus exactly
    `2·D_MAX + |E| + 1`, every named unlifted node maps monotonically to a lifted body node of the same gate count, and the root body is
    exactly `[prefill, eos[·], bot32, (live, tok_in, step) × D_MAX]`;
  * the sampling-event population is `cap` structurally, whatever the EOS set; every step's gate range is present;
  * the T-3(ii) exact count under `GUARD=in`: `Σ_{s ≥ a_r} (1 + 2·NL·KVH·D)` ⊥ return leaves — as arithmetic AND as the Program's own
    step return arity; the §7 rule on a served transcript with an early EOS reproduces `out[]` and names the shortfall failure;
  * PAD-02 / F-pad-06 leaf layout: `enc(⊥_32) = 2^32` and `Active(2^32 − 1)` survive the `⌈(w+1)/32⌉`-word pack → unpack; a one-word layout
    for a 33-bit leaf is refused by name, never read as Active.

What it does NOT establish (needs a real record with an early EOS on a pod — HELP M-0076): the active-step bit-for-bit replay against the
active-reference run and the C2 replay at active / transition / inactive steps.  Nothing here is inferred from an active-reference green.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.program import Program
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import lifted as L

C = b1.B0_CONFIG  # HuggingFaceTB/SmolLM2-135M: NL=30, H=576, NH=9, KVH=3, D=64, V=49152
LP, CAP = 256, 32
D_MAX = CAP - 1
EOS = (0,)  # the declared stop set of the request (SmolLM2 base: <|endoftext|> = 0); a declaration of the manifest, not a model fact
ATT = b1.attention_fa2(C["D"])  # D54: BN=128 at head dim 64 — the Serve_v4 binding record
RNG = np.random.default_rng(1601)


@pytest.fixture(scope="module")
def programs():
    p_u = Program(b1.serve_v2(C, LP, D_MAX, attention=ATT))
    p_l = Program(L.lifted_request(C, LP, CAP, EOS, attention=ATT, lag=0))
    return p_u, p_l


@pytest.fixture(scope="module")
def program_eos2():
    """The same request under the declared EOS set (0, 2) — built once (memory: the laptop guardian's 2.5 GB per-process cap)."""
    return Program(L.lifted_request(C, LP, CAP, (0, 2), lag=0, attention=ATT))


def _root_named(prog):
    return [(c.node.name, c) for c in prog.circuit.root_calls() if c.node.name]


def _body_named(prog):
    m = []
    for c in prog.circuit.root_calls():
        nm = c.node.name
        if nm == "prefill" or (nm and nm.endswith("/step")):
            for k in c.children():
                if k.node.name:
                    m.append((k.node.name, k))
    return m


# ---------------------------------------------------------------------------------------------------------
# T-1: built from the manifest only; fixed
# ---------------------------------------------------------------------------------------------------------


def test_builder_inputs_are_the_manifest_and_a_served_length_is_refused():
    import inspect
    params = inspect.signature(L.lifted_request).parameters
    assert set(params) == {"C", "prompt_len", "cap", "eos", "T", "gemm", "attention", "norm", "lag"}  # no record, no served length, no flag
    assert params["lag"].default is inspect.Parameter.empty  # DP-pad-02: the lag is declared, never defaulted
    sp = L.lifted_request(C, LP, CAP, EOS, attention=ATT, lag=0)
    assert sp is L.lifted_request(C, LP, CAP, EOS, T=CAP - 1, attention=ATT, lag=0)  # T = cap - 1 is the only admissible T
    assert sp.bindings["STEPS"] == D_MAX and "D_MAX" not in sp.bindings and sp.bindings["GUARD"] == "in" and sp.bindings["EOS"] == EOS  # F-pad-12
    with pytest.raises(ValueError, match="never from a served length"):
        L.lifted_request(C, LP, CAP, EOS, T=20, attention=ATT, lag=0)  # served 21 of 32 → rebuilt to the served length: refused
    with pytest.raises(ValueError, match="never from a served length"):
        L.lifted_request(C, LP, CAP, EOS, T=CAP, attention=ATT, lag=0)
    with pytest.raises(ValueError, match="at least one sampling event"):
        L.lifted_request(C, LP, 0, EOS, attention=ATT, lag=0)
    with pytest.raises(NotImplementedError, match="prefill-only"):
        L.lifted_request(C, LP, 1, EOS, attention=ATT, lag=0)  # declared, not faked


def test_refusals_fire_eagerly_in_the_builder_and_lazily_on_a_direct_bind_both_by_name():
    """F-pad-13: `lserve()`/`lifted_request()` validate BEFORE the lazy specialization exists (no `Program()` needed); a direct
    `bind(LServe, …)` that skips the builder is still refused, at `Program()`, by the same names.  F-pad-12: the static is `STEPS`
    (decode steps = cap − 1), so the spec string's population reads `STEPS + 1 = cap`."""
    from verity.ml.circuits import bind
    eager = [
        (ValueError, "non-empty declared EOS set", dict(eos=())),
        (ValueError, "1 <= STEPS and LP \\+ STEPS <= MAX_POS", dict(cap=C["MAX_POS"] - LP + 2)),
        (ValueError, "GUARD='out' is not implemented", dict(guard="out")),
        (L.BoundaryFailure, "eos id", dict(eos=(L.enc_bot(32),))),
    ]
    for exc, msg, kw in eager:
        guard = kw.pop("guard", "in")
        cap = kw.pop("cap", CAP)
        eos = kw.pop("eos", EOS)
        with pytest.raises(exc, match=msg):
            L.lserve(C, LP, cap - 1, eos, lag=0, attention=ATT, guard=guard)  # no Program(): the refusal is the builder's
    # the lazy path: bind() succeeds (it never runs the body), Program() refuses by the same name
    lazy = bind(L.LServe, C=C, LP=LP, STEPS=D_MAX, EOS=(), GUARD="in", LAG=0, GEMM=b1.GEMM_AMPERE, ATTENTION=ATT, NORM=b1.norm_bundle(C))
    assert lazy.complete and "EOS=()" in lazy.id
    with pytest.raises(ValueError, match="non-empty declared EOS set"):
        Program(lazy)
    lazy = bind(L.LServe, C=C, LP=LP, STEPS=0, EOS=EOS, GUARD="in", LAG=0, GEMM=b1.GEMM_AMPERE, ATTENTION=ATT, NORM=b1.norm_bundle(C))
    with pytest.raises(ValueError, match="1 <= STEPS"):
        Program(lazy)
    assert "STEPS" in L.LServe.statics and "D_MAX" not in L.LServe.statics
    assert f"STEPS={D_MAX}" in L.lifted_request(C, LP, CAP, EOS, attention=ATT, lag=0).id


@pytest.mark.slow
def test_program_and_digest_are_functions_of_the_manifest(programs, program_eos2):
    _, p_l = programs
    d = program_digest(p_l)
    again = Program(L.lifted_request(C, LP, CAP, EOS, attention=ATT, lag=0))
    assert again.gates == p_l.gates and program_digest(again) == d  # a second build (a second "execution") is the same Program
    assert program_digest(program_eos2) != d  # the EOS set is a manifest fact
    assert L.lifted_request(C, LP, CAP - 1, EOS, attention=ATT, lag=0).gates < p_l.gates  # the cap is structural
    assert L.lifted_request(C, LP - 1, CAP, EOS, attention=ATT, lag=0).gates < p_l.gates  # so is the prompt length
    assert L.lifted_request(C, LP, CAP, EOS, attention=ATT, lag=0).id != b1.serve_v2(C, LP, D_MAX, attention=ATT).id
    desc = encode_program(p_l)
    assert program_digest(desc) == d and desc["program_fn"].startswith("LServe_v2{") and "GUARD='in'" in desc["root"]
    del again  # keep the module under the laptop's per-process memory cap (mem_guardian: 2.5 GB)


# ---------------------------------------------------------------------------------------------------------
# PAD-03: address stability modulo the guard nodes, on the real shape
# ---------------------------------------------------------------------------------------------------------


def _lserve_names(d_max: int, n_eos: int) -> list[str]:
    """LServe_v1's root node names: prefill, the EOS consts, bot32, slot 1's controller, then per step s: (live, tok_in, step) and slot
    s+1's controller (is_eos, out, and the next emit while a slot follows)."""
    cap = d_max + 1
    names = ["enable/one", "prefill"] + [f"eos[{i}]" for i in range(n_eos)] + ["bot32", "slot=1/emit", "slot=1/is_eos", "slot=1/out"] + (["slot=2/emit"] if cap > 1 else [])
    for s in range(1, d_max + 1):
        t = s + 1
        names += [f"s={s}/live", f"s={s}/tok_in", f"s={s}/step", f"slot={t}/is_eos", f"slot={t}/out"] + ([f"slot={t + 1}/emit"] if t < cap else [])
    return names


def test_address_stability_modulo_guard_nodes_vs_serve_v2(programs):
    p_u, p_l = programs
    assert p_u.gates > 10**9  # a real shape, not a toy
    # the R17-1 guard (2 per step + |E| + bot) + the Continuation_v1 controller (R17-32: is_eos, out per slot; emit per slot) over cap = D_MAX + 1 slots
    assert p_l.gates == p_u.gates + 2 * D_MAX + len(EOS) + 2 + 3 * (D_MAX + 1) == p_u.gates + L.continuation_gate_count(D_MAX + 1, len(EOS), 0)
    assert p_l.input_gates == p_u.input_gates  # the same prescribed leaves (lifted widths, same count)
    names = [n for n, _ in _root_named(p_l)]
    assert names == _lserve_names(D_MAX, len(EOS))
    # every named unlifted root node ↦ a lifted body node of the same gate count, in the same order (monotone), covering the body
    u, l = _root_named(p_u), dict(_body_named(p_l))
    assert len(u) == len(l) and [n for n, _ in u] == [n for n, _ in _body_named(p_l)]
    covered, prev_hi = 0, -1
    for name, cu in u:
        cl = l[name]
        assert cl.hi - cl.lo == cu.hi - cu.lo, name
        assert cl.lo >= prev_hi  # monotone: the lifted map is the unlifted one with the guard nodes spliced in
        prev_hi = cl.hi
        covered += cu.hi - cu.lo
    assert covered == p_u.gates - p_u.input_gates
    # the guard nodes are exactly the difference: LStepLive (1 gate) + LSelect (1) per step, |E| consts, one LBot
    guard = [(n, c) for n, c in _root_named(p_l) if not (n == "prefill" or n.endswith("/step"))]
    assert sum(c.hi - c.lo for _, c in guard) == 2 * D_MAX + len(EOS) + 2 + 3 * (D_MAX + 1)
    assert {c.id.split("{")[0] for n, c in guard if n.endswith("/is_eos")} == {f"LIsEOS{len(EOS)}_v1"}
    assert {c.id.split("{")[0] for n, c in guard if n.endswith("/emit")} == {"Const2[0x1]_v1", "LEmitNext[H=1]_v1"}
    assert {c.id.split("{")[0] for n, c in guard if n.endswith("/out")} == {"LSelect32_v1"}
    assert {c.id.split("{")[0] for n, c in guard if n.endswith("/live")} == {f"LStepLive{len(EOS)}_v1"}
    assert {c.id.split("{")[0] for n, c in guard if n.endswith("/tok_in")} == {"LSelect32_v1"}


# ---------------------------------------------------------------------------------------------------------
# R3: the population is structural — cap sampling events, every step present, whatever the EOS set
# ---------------------------------------------------------------------------------------------------------


def test_sampling_event_population_is_cap_and_independent_of_the_eos_set(programs, program_eos2):
    _, p_l = programs
    events = [k for n, k in _body_named(p_l) if n.endswith("/final")]
    assert len(events) == CAP and all(k.id.startswith("Lifted[Final_v2]") for k in events)
    ts = [n for n, _ in _body_named(p_l)]
    assert sum(1 for n in ts if n.endswith("/embed")) == LP + D_MAX  # one embedding per position, decode ones included
    steps = {int(n.split("=")[1].split("/")[0]): c for n, c in _root_named(p_l) if n.endswith("/step")}
    assert sorted(steps) == list(range(1, D_MAX + 1))
    leaves = 1 + 2 * C["NL"] * C["KVH"] * C["D"]
    assert all(len(c.returns) == leaves for c in steps.values())  # y + the K/V rows of every layer: the step's committed leaves
    # a different declared EOS set: the same shape (names and gate counts of every body node), only the constants differ
    shape = lambda p: [(n, c.hi - c.lo) for n, c in _body_named(p)]  # noqa: E731
    assert shape(program_eos2) == shape(p_l) and program_eos2.gates == p_l.gates + 1


# ---------------------------------------------------------------------------------------------------------
# T-3(ii): the exact-count rule under GUARD=in, as arithmetic and on served transcripts
# ---------------------------------------------------------------------------------------------------------


def test_exact_count_rule_and_the_served_prefix_rule_on_the_real_shape(programs):
    _, p_l = programs
    per_step = 1 + 2 * C["NL"] * C["KVH"] * C["D"]
    assert per_step == 11521
    steps = {int(n.split("=")[1].split("/")[0]): c for n, c in _root_named(p_l) if n.endswith("/step")}
    for a_r in (1, 5, 20, 31, 32):
        want = L.expected_bot_return_leaves(C, CAP, a_r, lag=0)
        assert want == (CAP - a_r) * per_step == sum(len(steps[s].returns) for s in range(a_r, D_MAX + 1))
    assert L.expected_bot_return_leaves(C, CAP, CAP, lag=0) == 0  # no EOS: nothing is ⊥
    with pytest.raises(ValueError):
        L.expected_bot_return_leaves(C, CAP, 0, lag=0)
    # a served transcript that stops at event 4 with the EOS: a_r = 5; the fixed out[] is 5 Active + 27 ⊥, from the served tokens alone
    served = [int(x) for x in RNG.integers(1, C["V"], 4)] + [EOS[0]]
    out = L.assign_inactive("smol/r0", served, EOS, CAP, lag=0)
    assert len(out) == CAP and out[:5] == served and out[5:] == [L.enc_bot(32)] * 27
    assert L.activity_prefix(out, EOS, CAP) == 5 == L.served_count(out)
    with pytest.raises(L.ServedPrefixFailure, match=r"n_r=4 a_r=32: declared activity not executed"):
        L.assign_inactive("smol/r1", served[:4], EOS, CAP, lag=0)  # stopped without an EOS below the cap (F-pm-24's shape)
    with pytest.raises(L.ServedPrefixFailure, match=r"n_r=6 a_r=5: execution past EOS"):
        L.assign_inactive("smol/r2", served + [7], EOS, CAP, lag=0)
    full = [int(x) for x in RNG.integers(1, C["V"], CAP)]
    assert L.assign_inactive("smol/r3", full, EOS, CAP, lag=0) == full  # ran to the cap: every event Active


# ---------------------------------------------------------------------------------------------------------
# PAD-02 / F-pad-06: the leaf layout round trip (attack A5)
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("w", [1, 2, 16, 31, 32])
def test_leaf_layout_round_trip(w):
    n = L.leaf_words(w)
    assert n == (2 if w == 32 else 1)
    words = [L.enc_bot(w), L.inv(w), L.enc_active(w, (1 << w) - 1), L.enc_active(w, 1 << (w - 1)), 0] + \
        [int(x) for x in RNG.integers(0, 1 << (w + 1), 16)]
    for x in words:
        packed = L.pack_leaf(w, x)
        assert len(packed) == n and all(0 <= p < 2**32 for p in packed)
        assert L.unpack_leaf(w, packed) == x  # the tag survives; Active(2^w − 1) is not confused with anything
    assert L.pack_leaf(32, L.enc_bot(32)) == [0, 1] and L.pack_leaf(32, (1 << 32) - 1) == [0xFFFFFFFF, 0]
    with pytest.raises(ValueError, match="not a"):
        L.pack_leaf(w, 1 << (w + 1))


def test_one_word_layout_for_a_33_bit_leaf_is_refused_not_read_as_active():
    x = L.enc_bot(32)
    truncated = [x & 0xFFFFFFFF]  # what a 32-bit-only serialiser would store: the tag is gone, the word reads Active(0)
    assert truncated == [0] and L.is_active(32, truncated[0])  # the attack: silently Active
    with pytest.raises(L.LiftedLayoutError, match="tag bit is not representable"):
        L.unpack_leaf(32, truncated)
    with pytest.raises(L.LiftedLayoutError, match="above the lifted width"):
        L.unpack_leaf(16, [1 << 17])
    with pytest.raises(L.LiftedLayoutError):
        L.unpack_leaf(1, [])

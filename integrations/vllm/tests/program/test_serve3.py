"""Lane PAD-IMPL: `Serve@3` at TINY-PAD against PAD-SPEC v1's frozen semantics — the prefix property
(§13, `n ∈ {0, 3, D_MAX}`), the absorbing property and the inactive step-boundary
relation (§6), output interpretation (§8), digest distinctness (§10), the measured gate budget (§14.1), and
the vectorized reference evaluator (§11/§12, `verity_vllm/program/kernels/serve3_reference.py`) bit-for-bit against
the gate-level evaluation.

Fixture: TINY-PAD dims (spec §14: the test-suite TINY with `MAX_POS` 8 → 16), random bf16 weights
(`default_rng(1001)`, scale 0.05) and prompt (`default_rng(2002)`), chosen so the greedy raw samples of the
frozen `Serve_v1{TINY, LP=4, STEPS=8}` program are `[15, 23, 26, 17, 26, 17, 26, 17, 26]` — first
occurrences 15@0, 23@1, 26@2, 17@3, so binding `EOS_ID = 15 / 17 / 63` realizes exactly `ℓ = 0 / 3 / 8`
live decode steps (`EOS_ID` is a prescribed static; each value is its own `Serve_v3` binding, spec §2/§10).
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity.ml.circuits import Program, bind  # noqa: E402
from verity.ml.circuits.codec import program_digest  # noqa: E402
from verity_vllm.program.registry import b1, serve3  # noqa: E402
from verity_vllm.program.kernels import serve3_reference as ref  # noqa: E402
from verity_vllm.program.registry.b1 import weights_type  # noqa: E402

TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
        "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True,
        "tie_word_embeddings": True, "profile": {}}
LP, D_MAX = 4, 8
#: raw greedy samples of `Serve_v1{TINY, LP=4, STEPS=8}` on the fixture inputs (pinned; y[3]=17 is fresh)
Y_RAW = [15, 23, 26, 17, 26, 17, 26, 17, 26]
#: EOS_ID -> realized live decode-step count ℓ on the fixture (spec §13 targets: n ∈ {0, 3, D_MAX})
EOS_FOR = {0: 15, 3: 17, 8: 63}


def c3(eos: int) -> dict:
    return {**TINY, "EOS_ID": eos}  # spec §10: C₃ = {**C, "EOS_ID": id} at the binding site


@functools.lru_cache(maxsize=None)
def _inputs() -> tuple[tuple[int, ...], tuple[int, ...]]:
    rng = np.random.default_rng(1001)
    w = ((rng.standard_normal(weights_type(TINY).leaves) * 0.05).astype(np.float32).view(np.uint32) >> 16)
    prompt = np.random.default_rng(2002).integers(0, TINY["V"], LP)
    return tuple(int(x) for x in prompt), tuple(int(x) for x in w)


@functools.lru_cache(maxsize=None)
def _v1(steps: int):
    """(program, transcript, outputs) of the frozen `Serve_v1{TINY, LP, STEPS=steps}`."""
    prompt, w = _inputs()
    prog = Program(bind(b1.Serve, C=TINY, LP=LP, STEPS=steps))
    t = ref.materialize(prog, [list(prompt), list(w)])
    return prog, t, [t[g] for g in prog.circuit.root_outputs()]


@functools.lru_cache(maxsize=None)
def _v3(eos: int):
    """(program, transcript, out, live) of `Serve_v3{TINY_PAD(eos), LP, D_MAX, Ampere}`."""
    prompt, w = _inputs()
    prog = Program(serve3.serve_v3(c3(eos), LP, D_MAX))
    t = ref.materialize(prog, [list(prompt), list(w)])
    o = [t[g] for g in prog.circuit.root_outputs()]
    return prog, t, o[:D_MAX + 1], o[D_MAX + 1:]


def _named_v1(prog):
    """Named root calls of a `Serve_v1` program: every non-input gate lies in exactly one of them."""
    return {c.node.name: c for c in prog.circuit.root_calls() if c.node.name}


def _steps_v3(prog):
    """The `s=<s>/step` calls of a `Serve_v3` program, by step index."""
    return {int(c.node.name.split("=")[1].split("/")[0]): c
            for c in prog.circuit.root_calls() if c.node.name and c.node.name.endswith("/step")}


def _named_v3(prog):
    """The static gate map φ's target intervals (spec §13): body-node names -> calls, prefill/final/guard at
    root and each decode step's body nodes inside its `PadDecodeStep` call."""
    m = {}
    for c in prog.circuit.root_calls():
        nm = c.node.name
        if nm and nm.endswith("/step"):
            for k in c.children():
                if k.node.name:
                    m[k.node.name] = k
        elif nm:
            m[nm] = c
    return m


def test_fixture_raw_samples_are_pinned():
    _, _, y = _v1(D_MAX)
    assert y == Y_RAW
    assert Y_RAW[3] not in Y_RAW[:3] and EOS_FOR[8] not in Y_RAW[:8]


def test_gate_budget_matches_the_spec():
    """Spec §14.1 (measured): 56,936 = 56,361 body + 575 guard gates, the same in every execution; the
    guard is 6 gates; a decode step carries 71 guard/mux gates."""
    assert bind(serve3.PadGuard, EOS=7).gates == 6
    sp3 = serve3.serve_v3(c3(7), LP, D_MAX)
    sp1 = bind(b1.Serve, C=TINY, LP=LP, STEPS=D_MAX)
    assert (sp3.gates, sp1.gates) == (56936, 56361)
    step = bind(serve3.PadDecodeStep, C=c3(7), T=LP + 5, GEMM=b1.GEMM_AMPERE, ATTENTION=b1.ATTENTION_FA2,
                NORM=b1.norm_bundle(TINY))
    assert step.gates == 4967 + 71  # step 5 body (spec §14.1) + 6 guard + 1 zero + 2*NL*KVH*D selects


@pytest.mark.parametrize("n", [0, 3, D_MAX])
def test_prefix_property(n):
    """Spec §13: the padded circuit's live region equals `Serve_v1{TINY, LP, STEPS=n}` **gate-for-gate**
    under the static map φ_n (prefill nodes -> prefill nodes, step-s body nodes -> the same-named nodes of
    `PadDecodeStep{T=LP+s}`), when the realized live count is ℓ = n."""
    p1, t1, y1 = _v1(n)
    p3, t3, out, live = _v3(EOS_FOR[n])
    assert sum(live[:D_MAX]) == n  # the fixture realizes ℓ = n
    m1, m3 = _named_v1(p1), _named_v3(p3)
    covered = 0
    for name, c1 in m1.items():
        c3_ = m3[name]
        assert c3_.hi - c3_.lo == c1.hi - c1.lo, name  # position-for-position (same bodies, CF1)
        assert [t3[g] for g in range(c3_.lo, c3_.hi)] == [t1[g] for g in range(c1.lo, c1.hi)], name
        covered += c1.hi - c1.lo
    # the named nodes tile all of Serve_v1's computed gates: the comparison really is gate-for-gate
    assert covered == p1.gates - p1.input_gates
    # wire caveat (spec §13): on the live region the guard's SelectI32 is the identity, so the padded
    # circuit's token slots out[0..n] equal Serve_v1{STEPS=n}'s n+1 raw samples
    assert out[:n + 1] == y1


def test_absorbing_property_and_inactive_boundary():
    """Spec §6: once `live_s = 0` — every later token slot is the pad constant 0, every appended k/v word
    is the zero row, the flag stays 0; and the `PadDecodeStep` boundary relation holds: `live_in = 0` ⟹
    boundary outputs are constants, while live steps' muxes are the identity on the computed rows."""
    KVD = TINY["KVH"] * TINY["D"]
    H, NHD = TINY["H"], TINY["NH"] * TINY["D"]
    for ell in (0, 3):
        p3, t3, out, live = _v3(EOS_FOR[ell])
        assert live == [1] * ell + [0] * (D_MAX + 1 - ell)  # non-increasing; flips at event ℓ
        assert out[ell + 1:] == [0] * (D_MAX - ell)  # pad constant on the whole suffix
        for s, c in sorted(_steps_v3(p3).items()):
            r = c.returns  # (tok_out, live_out, k_new[NL*KVH*D], v_new[NL*KVH*D]) in leaf order
            tok_out, live_out = t3[r[0]], t3[r[1]]
            k_new = [t3[g] for g in r[2:2 + 2 * KVD]]
            v_new = [t3[g] for g in r[2 + 2 * KVD:2 + 4 * KVD]]
            if s > ell:  # inactive: the §6 boundary relation, constants only
                bt, bl, bk, bv = ref.inactive_boundary(TINY)
                assert (tok_out, live_out) == (bt, bl), s
                assert k_new == [int(x) for x in bk.ravel()] and v_new == [int(x) for x in bv.ravel()], s
            else:  # live: the mux forwards the step's computed rows (identity on the live branch)
                assert live_out == (0 if s == ell else 1) and tok_out == out[s], s
                for L in range(TINY["NL"]):
                    pre = {k.node.name: k for k in c.children()}[f"t={LP + s - 1}/layers.{L}/pre"]
                    pr = pre.returns  # Tuple(x[H], q[NH*D], k[KVH*D], v[KVH*D]) leaves
                    krow = [t3[g] for g in pr[H + NHD:H + NHD + KVD]]
                    vrow = [t3[g] for g in pr[H + NHD + KVD:H + NHD + 2 * KVD]]
                    assert k_new[L * KVD:(L + 1) * KVD] == krow and v_new[L * KVD:(L + 1) * KVD] == vrow, (s, L)


@pytest.mark.parametrize("ell", [0, 3, 8])
def test_output_interpretation(ell):
    """Spec §8: `n = Σ live[0..D_MAX-1]`; EOS-exclusive transcript `out[0..n-1]`; the sampled EOS committed
    at slot `n` as stop evidence (when stopping by EOS); suffix slots equal `PAD_TOKEN_ID = 0`."""
    eos = EOS_FOR[ell]
    _, _, out, live = _v3(eos)
    n, transcript = ref.interpret(out, live)
    assert n == ell and list(transcript) == Y_RAW[:ell]
    if ell < D_MAX:
        assert out[ell] == eos  # stop evidence: the live event ℓ samples and commits the EOS
        assert out[ell + 1:] == [0] * (D_MAX - ell)
    else:
        assert out[D_MAX] == Y_RAW[D_MAX]  # length cap: slot D_MAX committed, never a semantic output
    # well-formedness (spec §8) — theorems of the circuit, checked on its outputs
    assert all(live[g] <= live[g - 1] for g in range(1, D_MAX + 1))
    assert all(live[g] == (live[g - 1] and out[g] != eos) for g in range(1, D_MAX + 1) if live[g - 1] == 1)


def test_reference_evaluator_matches_the_gates():
    """The vectorized padding-layer reference (spec §11: inactive branches are validated against the
    registered specification): `pad_chain` on the raw samples reproduces the committed `(out, live)`
    bit-for-bit, and `mux_rows` reproduces every appended row of every step."""
    KVD = TINY["KVH"] * TINY["D"]
    for ell in (0, 3, 8):
        p3, t3, out, live = _v3(EOS_FOR[ell])
        steps = _steps_v3(p3)
        # the raw samples y_g: the TokenSelect outputs, pre-mux (inactive ones are canonical garbage)
        finals = {c.node.name: c for c in p3.circuit.root_calls() if c.node.name}
        y = [t3[finals[f"t={LP - 1}/final"].returns[0]]]
        for s in range(1, D_MAX + 1):
            fc = {k.node.name: k for k in steps[s].children()}[f"t={LP + s - 1}/final"]
            y.append(t3[fc.returns[0]])
        r_out, r_live = ref.pad_chain(y, EOS_FOR[ell])
        assert [int(x) for x in r_out] == out and [int(x) for x in r_live] == live
        # the row muxes, vectorized over steps and words, against the gate-level appended rows
        rows, muxed = [], []
        for s in sorted(steps):
            c = steps[s]
            r = c.returns
            muxed.append([t3[g] for g in r[2:2 + 4 * KVD]])
            pres = {k.node.name: k for k in c.children()}
            row = []
            for half in (0, 1):  # k then v, layer-major as in the step's return
                for L in range(TINY["NL"]):
                    pr = pres[f"t={LP + s - 1}/layers.{L}/pre"].returns
                    base = TINY["H"] + TINY["NH"] * TINY["D"] + half * KVD
                    row.append([t3[g] for g in pr[base:base + KVD]])
            rows.append([w for L in range(TINY["NL"]) for w in row[L]] +
                        [w for L in range(TINY["NL"]) for w in row[2 + L]])
        live_before = [1 if s <= ell else 0 for s in range(1, D_MAX + 1)]
        r_rows = ref.mux_rows(live_before, np.asarray(rows, dtype=np.uint16))
        assert [[int(w) for w in row] for row in r_rows] == muxed


def test_digest_distinctness():
    """Spec §10: `Serve_v3` digests are distinct from the frozen `Serve_v2{STEPS=D_MAX}` binding (the guard
    gates alone force it) and distinct across `D_MAX` and across `EOS_ID`; `Serve_v1/v2` stay untouched."""
    d = lambda sp: program_digest(Program(sp))  # noqa: E731
    v3 = d(serve3.serve_v3(c3(17), LP, D_MAX))
    assert v3 != d(b1.serve_v2(TINY, LP, D_MAX))
    assert v3 != d(serve3.serve_v3(c3(17), LP, D_MAX - 1))  # across D_MAX
    assert v3 != d(serve3.serve_v3(c3(63), LP, D_MAX))  # across EOS_ID
    # and the readable ids are distinct too (the specialization identity, spec §10)
    assert serve3.serve_v3(c3(17), LP, D_MAX).id != serve3.serve_v3(c3(63), LP, D_MAX).id


def test_validity_and_config_discipline():
    """Spec §2/§10: v3 rejects a config without EOS_ID and LP + D_MAX > MAX_POS; binding never mutates the
    base config dict."""
    base = dict(TINY)
    with pytest.raises(ValueError, match="EOS_ID"):
        serve3.serve_v3(base, LP, D_MAX).gates  # noqa: B018 — elaboration is lazy; force it
    with pytest.raises(ValueError, match="MAX_POS"):
        serve3.serve_v3(c3(7), LP, TINY["MAX_POS"] - LP + 1).gates  # noqa: B018
    serve3.serve_v3(c3(7), LP, D_MAX).gates  # noqa: B018
    assert base == TINY and "EOS_ID" not in TINY

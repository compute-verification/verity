"""The private-operand-safe folds (lane fold-private): the v3 units (``privsel``) and the v1 units (``unit.py`` / ``fp8``)
with four instruction steps per column -- ``relations._folded`` on the eight base relations.

What is checked here (CPU): the eight base systems are byte-identical to their pinned fixtures (folding is additive); each
folded fixture is the compiled folded system (the Rust pin's bytes); the folded unit accepts exactly the word of four
chained base steps and refuses the base's one-, two- and three-step intermediates (the fold's only new boundary); the
chain runner proves the same synthetic VUs as the base relation with a quarter of the columns, and the negative battery
-- incl. the folded-boundary family of ``relchain.negatives`` -- is rejected.  ``FOLD_DIFF_N`` scales the differential
(default 300 chains per relation; the lane ran 1e5 at the model level and 2e4 at the unit level, see its note).
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from verity.ml.circuits.silicon.silicon import tc_dot

from . import protocol as protocol_mod
from .compile import P
from .fp8.census import census
from .relations import RELATIONS, Relation
from .relchain import RelationChainRunner, instances, negatives
from .serialize import system_bytes
from .witness import apply_sparse, build_witness, tables

FIX = Path(__file__).resolve().parents[2] / "ligero-verify" / "fixtures"
V3X4 = ["bf16-hopper-v3x4", "bf16-ampere-v3x4", "fp8-hopper-v3x4", "fp8-ada-v3x4"]
V1X4 = ["bf16-hopper-x4", "bf16-ampere-x4", "fp8-hopper-x4", "fp8-ada-x4"]
FOLDED = [RELATIONS[n] for n in V3X4 + V1X4]
BASE_OF = {"bf16-hopper-v3x4": "bf16-hopper-v3", "bf16-ampere-v3x4": "bf16-ampere-v3", "fp8-hopper-v3x4": "fp8-hopper-v3",
           "fp8-ada-v3x4": "fp8-ada-v3", "bf16-hopper-x4": "bf16-hopper", "bf16-ampere-x4": "bf16-ampere",
           "fp8-hopper-x4": "fp8-hopper", "fp8-ada-x4": "fp8-ada"}
BASE_FIXTURE = {"bf16-hopper": FIX / "systems/bf16-hopper.system.bin", "fp8-hopper": FIX / "systems/fp8-hopper.system.bin",
                "fp8-ada": FIX / "fp8-ada/system.bin", "bf16-ampere": FIX / "system.bin",
                **{n: FIX / f"systems/{n}.system.bin" for n in ("bf16-hopper-v3", "bf16-ampere-v3", "fp8-hopper-v3", "fp8-ada-v3")}}
N_DIFF = int(os.environ.get("FOLD_DIFF_N", "48"))


def _base(rel: Relation) -> Relation:
    return RELATIONS[BASE_OF[rel.name]]


def _synthetic(rel: Relation) -> Relation:
    """The relation on synthetic instances (bf16-ampere / bf16-ampere-x4 read the frozen set, absent on most machines)."""
    if not rel.frozen_tier:
        return rel
    return replace(rel, frozen_tier=None, instances_dataset="bench-instances-bf16-ampere/v1", instances_tier="vu-k1536-bf16-ampere",
                   instance_seed=20260922)


@pytest.mark.parametrize("name", sorted(BASE_FIXTURE), ids=str)
def test_base_systems_byte_identical_to_pinned_fixtures(name):
    rel = RELATIONS[name]
    assert system_bytes(rel.compile(rel.params, True)) == BASE_FIXTURE[name].read_bytes(), f"{name}: base system changed"


@pytest.mark.parametrize("rel", FOLDED, ids=lambda r: r.name)
def test_folded_fixture_is_the_compiled_system(rel):
    base = _base(rel)
    sys4 = rel.compile(rel.params, True)
    assert system_bytes(sys4) == (FIX / f"systems/{rel.name}.system.bin").read_bytes(), f"{rel.name}: fixture != compile"
    assert system_bytes(rel.compile(rel.params, True)) == system_bytes(sys4), "compile is deterministic"
    c4, c1 = census(sys4), census(base.compile(base.params, True))
    assert rel.k == 4 * base.k and rel.steps * 4 == base.steps and rel.fold == 4 and base.fold == 1
    # rows per unit: sub-linear in the fold for the Hopper / Ada units (the accumulator decode is shared), but the Ampere v1
    # unit's rows are almost all per-group bit rows, and its fold costs 3.6 % MORE rows per unit than four units (14576 vs
    # 4 x 3516 at x4, ratio 4.1456, fixed by the pinned fixture above) -- there the fold's only saving is the 4x fewer sub-batches
    ratio = c4["rows_per_unit"] / c1["rows_per_unit"]
    assert ratio < (4.15 if rel.name == "bf16-ampere-x4" else 4.0), (c4["rows_per_unit"], c1["rows_per_unit"])
    words_per_pin = 3 if (rel.name.endswith("-v3x4") and rel.word_dtype == "<u1") else 1
    if rel.name.endswith("-v3x4"):
        assert c4["pin_rows"] == 2 * -(-rel.k // words_per_pin)          # the packed word runs of both operands
    else:
        assert c4["pin_rows"] == 4 * c1["pin_rows"]                     # v1: one pin per operand component
    # the tag convention of the base family: v3 absorbs its name for every operand format, v1 only for FP8 (D12 / D13)
    assert rel.tag == (b"" if base.tag == b"" else f"|rel={rel.name}|v1".encode())
    assert (rel.tag == b"") == (base.tag == b"")
    assert rel.model.groups == tuple(base.model.groups) * 4 and rel.y_public is base.y_public and rel.operands is base.operands


def _accepts(sys, tb, W: torch.Tensor, pub: dict) -> torch.Tensor:
    """Per column: every linear and quadratic constraint holds (as ``pubsel/relation_test._accepts``)."""
    lin = apply_sparse((tb.lin_q, tb.lin_j, tb.lin_coef), tb.lin_c, tb.L, W)
    ok = torch.ones(W.shape[1], dtype=torch.bool)
    for q, name in enumerate(tb.lin_pub):
        rhs = pub[name] % P if name is not None else torch.zeros_like(lin[q])
        ok &= lin[q] == rhs
    A = apply_sparse(tb.quad["a"][:3], tb.quad["a"][3], tb.Q, W)
    B = apply_sparse(tb.quad["b"][:3], tb.quad["b"][3], tb.Q, W)
    C = apply_sparse(tb.quad["c"][:3], tb.quad["c"][3], tb.Q, W)
    ok &= ~(((A * B) % P != C).any(dim=0))
    return ok


def _unit_accepts(rel: Relation, sys, tb, c, a, b, y_claim):
    """Honest witness (hints from the true unit) against the claimed FP32 words ``y_claim``; a claim the verifier's public
    decode refuses (``ValueError``: e.g. an FP8 word with nonzero low bits) counts as rejected everywhere."""
    p = rel.params
    try:
        pub = rel.public_vectors(p, c, a, b, y_claim, device="cpu")
    except ValueError:
        return torch.zeros(len(y_claim), dtype=torch.bool)
    hints = rel.hints(p, sys, c, a, b, None, "cpu")
    return _accepts(sys, tb, build_witness(sys, pub, hints, "cpu"), pub)


@pytest.mark.parametrize("rel", FOLDED, ids=lambda r: r.name)
def test_folded_unit_is_exactly_four_chained_base_steps(rel):
    """Unit mode, ``N_DIFF`` random chains: the folded model's word == four chained base steps (from +-0, from the model's
    own state, and from words near a one-step result); the folded unit accepts that word and refuses the one-, two- and
    three-step intermediates whenever they differ, plus one random bit flip of the claim."""
    base = _base(rel)
    k, n = base.k, N_DIFF
    rng = np.random.default_rng(97)
    a, b = rel.operands(rng, (n, 4 * k)), rel.operands(rng, (n, 4 * k))
    a0, b0 = rel.operands(rng, (n, k)), rel.operands(rng, (n, k))
    c = np.zeros(n, dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    inter = np.zeros((n, 3), dtype=np.int64)
    for i in range(n):
        y1 = tc_dot(base.model, 0, a0[i].tolist(), b0[i].tolist())
        q = i % 4
        cc = (0 if i % 2 else 0x80000000) if q == 0 else y1 if q == 1 else tc_dot(base.model, y1, b0[i].tolist(), a0[i].tolist()) if q == 2 \
            else ((y1 & 0x807FFFFF) | (int(np.clip(((y1 >> 23) & 0xFF) + rng.integers(-12, 13), 1, 254)) << 23) if (y1 >> 23) & 0xFF not in (0, 0xFF) else y1)
        if (cc >> 23) & 0xFF == 0xFF:
            cc = 0
        st = cc
        for s in range(4):
            st = tc_dot(base.model, st, a[i, k * s:k * s + k].tolist(), b[i, k * s:k * s + k].tolist())
            if s < 3:
                inter[i, s] = st
        assert tc_dot(rel.model, cc, a[i].tolist(), b[i].tolist()) == st, f"{rel.name}: chain {i}: fold != 4 steps"
        c[i], y[i] = cc, st
    sys = rel.compile(rel.params, False)
    tb = tables(sys, "cpu")
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.int64)  # noqa: E731
    pub_word = np.vectorize(lambda v: rel.y_public(int(v)), otypes=[np.int64])
    ok = _unit_accepts(rel, sys, tb, T(c), T(a), T(b), T(y))                 # the claim is the FP32 word (public_vectors packs it)
    assert bool(ok.all()), f"{rel.name}: honest folded units rejected {torch.nonzero(~ok).flatten().tolist()[:5]}"
    for s in range(3):
        differs = T(pub_word(inter[:, s]) != pub_word(y))
        if bool(differs.any()):
            bad = _unit_accepts(rel, sys, tb, T(c), T(a), T(b), T(inter[:, s])) & differs
            assert not bool(bad.any()), f"{rel.name}: {s + 1}-step intermediate accepted as the fold's word {torch.nonzero(bad).flatten().tolist()[:5]}"
    # one flipped bit of the FP32 claim: FP8 packs bits 10..31 (bit 31 = sign), BF16 keeps bits 16..31 -- flips the public word
    bits = rng.integers(10 if rel.word_dtype == "<u1" else 16, 32, size=n)
    y_bad = y ^ (np.int64(1) << bits)
    differs = T(pub_word(y_bad) != pub_word(y))
    bad = _unit_accepts(rel, sys, tb, T(c), T(a), T(b), T(y_bad)) & differs
    assert not bool(bad.any()), f"{rel.name}: flipped claims accepted {torch.nonzero(bad).flatten().tolist()[:5]}"
    assert int(differs.sum()) >= n // 2


@pytest.mark.parametrize("rel", FOLDED, ids=lambda r: r.name)
def test_folded_chain_is_the_base_claim_cpu(rel):
    """The chain runner: the same synthetic VUs (operands, final words) as the base relation with a quarter of the columns
    (every fourth base accumulator); two VUs proved + verified (interactive, non-ZK), the negative battery -- with the
    folded-boundary family -- rejected."""
    base = _synthetic(_base(rel))
    rel = _synthetic(rel)
    vus, vb = instances(rel, 2, seed=5), instances(base, 2, seed=5)
    assert all(v[0] == w[0] and v[1] == w[1] and v[3] == w[3] for v, w in zip(vus, vb))
    assert all(v[2][3::4] == w[2] for v, w in zip(vb, vus)), "the folded accumulators are every fourth base accumulator"
    R = RelationChainRunner(rel, "cpu", -128.0, zk=False, mode=protocol_mod.MODE_INTERACTIVE)
    proof, pubs, lay = R.prove_vus(vus, check=True)
    ok, why = R.verify_vus(proof, pubs, lay)
    assert ok, why
    _, _, ypub = pubs
    assert int(ypub[rel.steps - 1]) == rel.y_public(vus[0][3]) and int(ypub[2 * rel.steps - 1]) == rel.y_public(vus[1][3])
    R = RelationChainRunner(rel, "cpu", -128.0, zk=False, mode=protocol_mod.MODE_FIAT_SHAMIR)
    negs = negatives(R, vus, row_negatives=12)
    accepted = [name for name, (ok, _) in negs if ok]
    assert accepted == [], f"{rel.name}: negatives accepted: {accepted}"
    boundary = [(name, why) for name, (_, why) in negs if name.startswith("folded boundary")]
    assert len(boundary) >= 2, boundary
    assert all("challenge" not in why for _, why in boundary), f"relation-level rejections expected: {boundary}"


def test_frozen_instances_take_every_fourth_accumulator():
    """bf16-ampere-x4 reads the frozen vu-k1536 set like bf16-ampere (skipped where the set is absent)."""
    from verity_numerical.bench import instances as frozen
    rel, base = RELATIONS["bf16-ampere-x4"], RELATIONS["bf16-ampere"]
    assert rel.frozen_tier == base.frozen_tier == "vu-k1536"
    try:
        vus, vb = instances(rel, 2), instances(base, 2)
    except FileNotFoundError:
        pytest.skip(f"frozen bench-instances/v1 not built under {frozen.DEFAULT_ROOT}")
    assert all(v[0] == w[0] and v[1] == w[1] and v[3] == w[3] and v[2] == w[2][3::4] for v, w in zip(vus, vb))
